"""Deterministic inferred-Watch runtime tests with fake vision, Jev, SAM, and watch adapters."""

import asyncio
import time
from io import BytesIO

from PIL import Image

from visionbrain import mission_runtime as runtime_module
from visionbrain.mission_autotarget import (
    AutotargetError,
    JevChoice,
    VisionCandidate,
    VisionProposal,
    VisionResponse,
)
from visionbrain.mission_contracts import (
    GeometryItem,
    Principal,
    SourceBinding,
    SourceFrame,
    ToolResult,
    WatchLease,
    WatchLeaseRelease,
)
from visionbrain.mission_runtime import DEFAULT_GOAL, MissionRuntime, _brief_sha256
from visionbrain.mission_store import MissionStore

SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}
BINDING = SourceBinding("scout-1", "41")
COOLER = VisionCandidate(
    "c1",
    "cooler",
    "Inspect the cooler's placement and access clearance",
    "The cooler partly blocks the access path",
    "A cooler lid is visible beside the door",
    "Contents cannot be seen",
)


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


def _usable(*candidates):
    return VisionResponse(VisionProposal(True, "", tuple(candidates)), "raw", None, "checkpoint", "gemma")


class _Vision:
    def __init__(self, response):
        self.response = response
        self.calls = 0

    def available(self, model_key):
        return model_key == "gemma"

    def propose_candidates(self, context):
        self.calls += 1
        return self.response

    def close(self):
        pass


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def now_ms(self):
        return int(self.now * 1000)

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class _Sam:
    def __init__(self, items=(), before=None):
        self.items = tuple(items)
        self.before = before
        self.requests = []

    def available_tools(self):
        return ("detect_objects", "segment_objects")

    def execute(self, request, context):
        if self.before is not None:
            self.before()
        self.requests.append(request)
        return ToolResult(status="ok" if self.items else "empty", items=self.items)


class _Jev:
    def __init__(self, choice=None, error=None, configured=True, before=None):
        self.choice = choice
        self.error = error
        self.configured_value = configured
        self.before = before
        self.calls = []

    def configured(self):
        return self.configured_value

    async def decide(self, *, state, criteria, timeout_seconds):
        self.calls.append({"criteria": dict(criteria), "timeout": timeout_seconds})
        if self.before is not None:
            self.before()
        if self.error is not None:
            raise self.error
        return self.choice


class _Watch:
    def __init__(self):
        self.applied = []
        self.released = []
        self.active = None

    def current_configuration_revision(self):
        return 0

    def apply(self, request):
        self.applied.append(request)
        self.active = WatchLease(
            lease_id=f"lease-{len(self.applied)}",
            mission_id=request.mission_id,
            source_binding=request.source_binding,
            configuration_revision=request.expected_configuration_revision + 1,
            targets=request.targets,
            task=request.task,
            expires_at_ms=request.expires_at_ms,
        )
        return self.active

    def release(self, lease, reason):
        self.released.append((lease.lease_id, reason))
        if self.active is not None and self.active.lease_id == lease.lease_id:
            self.active = None
        return WatchLeaseRelease(True, lease.configuration_revision + 1)


class _Source:
    def __init__(self, jpeg, clock):
        self.jpeg = jpeg
        self.clock = clock

    def __call__(self, requested):
        if requested != BINDING:
            return None
        return SourceFrame(self.jpeg, BINDING.source_id, BINDING.source_epoch, 15, self.clock.monotonic())


def _runtime(directory, *, vision, sam, jev, watch=None, clock=None):
    directory.mkdir(parents=True, exist_ok=True)
    clock = clock or _Clock()
    store = MissionStore(directory / "missions.sqlite3", directory / "evidence")
    runtime = MissionRuntime(
        store,
        vision,
        sam,
        _Source(_jpeg(), clock),
        watch or _Watch(),
        lambda event: None,
        qualified_models={"gemma": ["inspect", "watch"]},
        approved_source_ids=(BINDING.source_id,),
        decision_client=jev,
        clock=clock,
    )
    return runtime, store


def _command(name, *, request_id, mission_id=None, revision=None, args=None):
    command = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "command": name,
        "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        command["expected_revision"] = revision
    return command


def _inferred_args(**overrides):
    args = {
        "profile": {"id": "visual_inspection", "version": 1},
        "expertise": "Storage inspection technician",
        "goal_mode": "inferred",
        "mode": "watch",
        "reasoning_model": "gemma",
        "source_binding": {"source_id": BINDING.source_id, "source_epoch": BINDING.source_epoch},
    }
    args.update(overrides)
    return args


def _binding_args():
    return {"source_binding": {"source_id": BINDING.source_id, "source_epoch": BINDING.source_epoch}}


async def _start(runtime, args):
    created = await runtime.handle(_command("create", request_id="create-1", args=args), Principal("installation"), SCOPES)
    assert created["ok"], created
    snapshot = created["result"]["snapshot"]
    resumed = await runtime.handle(
        _command("resume", request_id="resume-1", mission_id=snapshot["mission_id"], revision=snapshot["revision"], args=_binding_args()),
        Principal("installation"),
        SCOPES,
    )
    assert resumed["ok"], resumed
    return snapshot["mission_id"]


async def _wait_for(predicate, timeout=3.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.01)
    raise AssertionError("condition was not met before timeout")


def _when(store, mission_id, predicate):
    snapshot = store.get_mission(mission_id)
    return snapshot if snapshot is not None and predicate(snapshot) else None


def _item(label):
    return GeometryItem("item-1", label, 0.9, (0.1, 0.1, 0.5, 0.5), None, "sam")


def test_capabilities_advertise_inferred_targeting_only_when_it_is_configured(tmp_path):
    unconfigured_jev = _Jev(configured=False)
    for name, jev, available, reason in (
        ("no-client", None, False, "the Jev decision client is not configured on this runtime"),
        ("no-key", unconfigured_jev, False, "the TypeSafe API key is not set on this host"),
        ("ready", _Jev(), True, None),
    ):
        runtime, store = _runtime(tmp_path / name, vision=_Vision(_usable()), sam=_Sam(), jev=jev)
        try:
            descriptor = runtime.capabilities(integrated=True)["expertise_autotarget"]
        finally:
            store.close()
        assert descriptor == {
            "supported": True,
            "available": available,
            "decision_model": "jev-1.13.0",
            "reason": reason,
            "goal_modes": ["inferred", "operator"],
        }


def test_inferred_create_is_bounded_and_rejects_untyped_or_contradictory_input_without_type_errors(tmp_path):
    cases = [
        ({"goal_mode": ["inferred"]}, "goal_mode must be inferred or operator"),
        ({"goal_mode": {"mode": "inferred"}}, "goal_mode must be inferred or operator"),
        ({"goal": "Find the cooler"}, "goal to be exactly empty"),
        ({"goal": "   "}, "goal to be exactly empty"),
        ({"mode": "inspect", "source_binding": None}, "available only for Watch"),
        ({"expertise": "   "}, "expertise is empty"),
    ]

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable()), sam=_Sam(), jev=_Jev())
        try:
            for index, (overrides, fragment) in enumerate(cases):
                args = _inferred_args(**overrides)
                if "source_binding" in overrides and overrides["source_binding"] is None:
                    args.pop("source_binding")
                reply = await runtime.handle(_command("create", request_id=f"bad-{index}", args=args), Principal("installation"), SCOPES)
                assert reply["ok"] is False, overrides
                assert reply["error"]["code"] == "invalid_request", overrides
                assert fragment in reply["error"]["message"], overrides
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_expertise_uses_only_the_approved_nonblank_bound_and_operator_requires_a_goal(tmp_path):
    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable()), sam=_Sam(), jev=_Jev())
        try:
            accepted = await runtime.handle(
                _command("create", request_id="ai-qa", args=_inferred_args(expertise="AI/QA")),
                Principal("installation"),
                SCOPES,
            )
            assert accepted["ok"], accepted
            assert accepted["result"]["snapshot"]["goal"] == ""
            assert accepted["result"]["snapshot"]["goal_mode"] == "inferred"

            too_long = await runtime.handle(
                _command("create", request_id="too-long", args=_inferred_args(expertise="x" * 1001)),
                Principal("installation"),
                SCOPES,
            )
            assert too_long["ok"] is False and too_long["error"]["code"] == "invalid_request"

            blank_operator = await runtime.handle(
                _command("create", request_id="operator-blank", args=_inferred_args(goal_mode="operator", goal="")),
                Principal("installation"),
                SCOPES,
            )
            assert blank_operator["ok"] is False and blank_operator["error"]["code"] == "invalid_request"

            operator = await runtime.handle(
                _command("create", request_id="operator-ok", args=_inferred_args(goal_mode="operator", goal="Highlight the cooler")),
                Principal("installation"),
                SCOPES,
            )
            assert operator["ok"], operator
            assert operator["result"]["snapshot"]["goal"] == "Highlight the cooler"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_omitted_goal_mode_keeps_legacy_semantics_and_hash(tmp_path):
    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable()), sam=_Sam(), jev=_Jev())
        try:
            legacy = await runtime.handle(
                _command(
                    "create",
                    request_id="legacy",
                    args={"profile": {"id": "visual_inspection", "version": 1}, "expertise": "Technician", "mode": "inspect", "reasoning_model": "gemma"},
                ),
                Principal("installation"),
                SCOPES,
            )
            assert legacy["ok"], legacy
            snapshot = legacy["result"]["snapshot"]
            assert snapshot["goal_mode"] == "legacy"
            assert snapshot["goal"] == DEFAULT_GOAL
            assert snapshot["brief_sha256"] == _brief_sha256("Technician", DEFAULT_GOAL)
            assert snapshot["autotarget_status"] is None
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_update_brief_mode_change_must_carry_goal_and_clears_objective_state(tmp_path):
    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable()), sam=_Sam(), jev=_Jev())
        try:
            prepared = await runtime.handle(
                _command(
                    "prepare",
                    request_id="prepare-1",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "Storage inspection technician",
                        "goal_mode": "operator",
                        "goal": "Highlight the cooler",
                        "mode": "watch",
                        "reasoning_model": "gemma",
                        "source_id": BINDING.source_id,
                    },
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert prepared["ok"], prepared
            snapshot = prepared["result"]["snapshot"]

            missing_goal = await runtime.handle(
                _command("update_brief", request_id="mode-no-goal", mission_id=snapshot["mission_id"], revision=snapshot["revision"], args={"goal_mode": "inferred"}),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert missing_goal["ok"] is False
            assert missing_goal["error"]["code"] == "invalid_request"
            assert "same request" in missing_goal["error"]["message"]

            switched = await runtime.handle(
                _command("update_brief", request_id="mode-inferred", mission_id=snapshot["mission_id"], revision=snapshot["revision"], args={"goal_mode": "inferred", "goal": ""}),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert switched["ok"], switched
            updated = switched["result"]["snapshot"]
            assert updated["goal_mode"] == "inferred"
            assert updated["goal"] == ""
            assert updated["brief_version"] == 2
            assert updated["working_objective"] is None
            assert updated["brief_history"][-1]["goal_mode"] == "inferred"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_activation_without_a_configured_key_is_refused_before_any_mission_starts(tmp_path):
    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable()), sam=_Sam(), jev=_Jev(configured=False))
        try:
            prepared = await runtime.handle(
                _command(
                    "prepare",
                    request_id="prepare-key",
                    args={
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "Storage inspection technician",
                        "goal_mode": "inferred",
                        "mode": "watch",
                        "reasoning_model": "gemma",
                        "source_id": BINDING.source_id,
                    },
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert prepared["ok"], prepared
            snapshot = prepared["result"]["snapshot"]
            activated = await runtime.handle(
                _command("activate", request_id="activate-key", mission_id=snapshot["mission_id"], revision=snapshot["revision"], args=_binding_args()),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert activated["ok"] is False
            assert activated["error"]["code"] == "autotarget_unavailable"
            assert store.get_mission(snapshot["mission_id"])["state"] == "created"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_inferred_watch_applies_the_ai_selected_target_and_operator_pause_clears_it(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    jev = _Jev(choice=JevChoice("c1", {"c1": 0.9, "none": 0.1}, 0.8, "jev-1.13.0"))
    sam = _Sam(items=[_item("cooler")])
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=sam, jev=jev, watch=watch)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(
                lambda: _when(store, mission_id, lambda s: s.get("autotarget_status") == "watching" and s.get("cycle_history"))
            )
            assert persisted["goal"] == ""
            assert persisted["goal_mode"] == "inferred"
            assert persisted["working_objective"] == COOLER.objective
            assert persisted["working_objective_reason"] == COOLER.reason
            assert persisted["watch_lease"]["targets"] == ["cooler"]
            assert persisted["decision_record"]["outcome"] == "watching"
            assert persisted["cycle_history"][-1]["cycle_id"] == persisted["decision_record"]["cycle_id"]
            assert persisted["decision_record"]["selected_label"] == "cooler"
            assert persisted["decision_record"]["grounding"]["matched_count"] == 1
            assert set(jev.calls[0]["criteria"]) == {"c1", "none"}
            assert watch.applied[0].targets == ("cooler",)
            assert persisted["budget"]["window_generations_used"] == 1
            assert persisted["budget"]["window_tools_used"] == 2

            paused = await runtime.handle(
                _command("pause", request_id="pause-1", mission_id=mission_id, revision=persisted["revision"]),
                Principal("installation"),
                SCOPES,
            )
            assert paused["ok"], paused
            after = store.get_mission(mission_id)
            assert after["working_objective"] is None
            assert after["autotarget_status"] is None
            assert watch.released == [("lease-1", "operator_paused")]
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_none_choice_releases_the_discretionary_target_and_keeps_watch_running(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    jev = _Jev(choice=JevChoice("none", {"c1": 0.1, "none": 0.9}, 0.9, "jev-1.13.0"))
    sam = _Sam(items=[_item("cooler")])
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=sam, jev=jev, watch=watch)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("autotarget_status") == "none"))
            assert persisted["state"] == "running"
            assert persisted["watch_lease"] is None
            assert persisted["targets"] == []
            assert persisted["working_objective"] is None
            assert persisted["decision_record"]["selected_candidate_id"] == "none"
            assert persisted["budget"]["window_tools_used"] == 1
            assert sam.requests == []
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_unusable_scene_waits_without_asking_jev_or_applying_a_target(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    unusable = VisionResponse(VisionProposal(False, "Frame is dark", ()), "raw", None, "checkpoint", "gemma")
    jev = _Jev(choice=JevChoice("c1", {"c1": 1.0, "none": 0.0}, 0.9, "jev-1.13.0"))
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(unusable), sam=_Sam(), jev=jev, watch=watch)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("autotarget_status") == "waiting_evidence"))
            assert persisted["state"] == "running"
            assert persisted["decision_record"]["outcome"] == "waiting_evidence"
            assert jev.calls == []
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_jev_failure_pauses_with_its_actual_code_and_never_reports_none(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    jev = _Jev(error=AutotargetError("jev_timeout", "decision exceeded its timeout"))
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=_Sam(items=[_item("cooler")]), jev=jev, watch=watch)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("state") == "paused"))
            assert persisted["reason"] == "autotarget_decision_failed"
            assert persisted["autotarget_status"] == "failed"
            assert persisted["decision_record"]["error_code"] == "jev_timeout"
            assert persisted["watch_lease"] is None
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_sam_that_does_not_localize_the_selected_label_is_unconfirmed_not_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    jev = _Jev(choice=JevChoice("c1", {"c1": 0.9, "none": 0.1}, 0.8, "jev-1.13.0"))
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=_Sam(items=[_item("bottle")]), jev=jev, watch=watch)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("autotarget_status") == "unconfirmed"))
            assert persisted["state"] == "running"
            assert persisted["working_objective"] is None
            assert persisted["decision_record"]["grounding"]["matched_count"] == 0
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_input_that_expires_while_jev_decides_pauses_instead_of_committing_none(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    clock = _Clock()
    jev = _Jev(choice=JevChoice("none", {"c1": 0.1, "none": 0.9}, 0.9, "jev-1.13.0"), before=lambda: clock.advance(16))
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=_Sam(), jev=jev, watch=watch, clock=clock)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("state") == "paused"))
            assert persisted["reason"] == "input_expired"
            assert persisted["autotarget_status"] == "failed"
            assert persisted["decision_record"]["error_code"] == "input_expired"
            assert watch.released == []
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_input_that_expires_while_sam_grounds_pauses_instead_of_reporting_unconfirmed(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.01)
    clock = _Clock()
    jev = _Jev(choice=JevChoice("c1", {"c1": 0.9, "none": 0.1}, 0.8, "jev-1.13.0"))
    sam = _Sam(items=[_item("bottle")], before=lambda: clock.advance(16))
    watch = _Watch()

    async def scenario():
        runtime, store = _runtime(tmp_path, vision=_Vision(_usable(COOLER)), sam=sam, jev=jev, watch=watch, clock=clock)
        try:
            mission_id = await _start(runtime, _inferred_args())
            persisted = await _wait_for(lambda: _when(store, mission_id, lambda s: s.get("state") == "paused"))
            assert persisted["reason"] == "input_expired"
            assert persisted["autotarget_status"] == "failed"
            assert persisted["decision_record"]["error_code"] == "input_expired"
            assert watch.applied == []
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
