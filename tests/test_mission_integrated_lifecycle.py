"""First-start and source-loss semantics for integrated mission controls."""

import asyncio
import time
from io import BytesIO

from PIL import Image

from visionbrain.mission_contracts import Decision, Principal, SourceBinding, SourceFrame, WatchLease, WatchLeaseRelease, WatchProposal
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self):
        self.calls = 0

    def available(self, _model):
        return True

    def plan(self, _context):
        self.calls += 1
        return Decision(1, "finish", watch=WatchProposal(("container",), "detect"))

    def close(self):
        return None


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, _request, _context):
        raise AssertionError("finish should not invoke a perception tool")


class _Watch:
    def __init__(self):
        self.active = None
        self.released = []

    def current_configuration_revision(self):
        return 0

    def apply(self, request):
        self.active = WatchLease(
            lease_id="lease-1",
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
        self.active = None
        return WatchLeaseRelease(True, lease.configuration_revision + 1)


class _Source:
    def __init__(self):
        self.available = False
        self.epoch = "41"
        self.calls = 0

    def __call__(self, binding):
        self.calls += 1
        if not self.available:
            return None
        return SourceFrame(
            _jpeg(), binding.source_id, self.epoch, self.calls, time.monotonic()
        )


def _command(command, *, request_id, mission_id=None, revision=None, args=None):
    message = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "mission_id": mission_id,
        "command": command,
        "args": args or {},
    }
    if revision is not None:
        message["expected_revision"] = revision
    return message


async def _prepare(runtime):
    return await runtime.handle(
        _command(
            "prepare",
            request_id="prepare",
            args={
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "container maintenance technician",
                "goal": "Find defects",
                "mode": "watch",
                "reasoning_model": "gemma",
                "source_id": "scout-1",
            },
        ),
        Principal("installation"),
        SCOPES,
        integrated=True,
    )


async def _wait_for(predicate, *, timeout=2.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def _runtime(tmp_path, source, planner=None, watch=None):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    planner = planner or _Planner()
    runtime = MissionRuntime(
        store,
        planner,
        _Tools(),
        source,
        watch or _Watch(),
        lambda _event: None,
        qualified_models={"gemma": ["watch"]},
        approved_source_ids={"scout-1"},
    )
    return runtime, store, planner


def test_failed_first_activation_keeps_on_intent_and_later_frame_does_not_start_planner(tmp_path):
    async def scenario():
        source, planner = _Source(), _Planner()
        runtime, store, _planner = _runtime(tmp_path, source, planner)
        try:
            prepared = await _prepare(runtime)
            snapshot = prepared["result"]["snapshot"]
            assert snapshot["state"] == "created"
            assert snapshot["activation_intent"] == "when_source_starts"

            failed = await runtime.handle(
                _command(
                    "activate",
                    request_id="activate-before-source",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"source_binding": {"source_id": "scout-1", "source_epoch": "41"}},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert not failed["ok"]
            assert failed["result"]["execution_outcome"] == "waiting_for_source"
            assert failed["result"]["snapshot"]["state"] == "created"
            assert failed["result"]["snapshot"]["activation_intent"] == "when_source_starts"

            source.available = True
            await asyncio.sleep(0.03)
            current = store.get_mission(snapshot["mission_id"])
            assert current["state"] == "created"
            assert current["activation_intent"] == "when_source_starts"
            assert planner.calls == 0

            activated = await runtime.handle(
                _command(
                    "activate",
                    request_id="activate-after-source",
                    mission_id=snapshot["mission_id"],
                    revision=current["revision"],
                    args={"source_binding": {"source_id": "scout-1", "source_epoch": "41"}},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert activated["ok"]
            assert activated["result"]["execution_outcome"] == "planning"
            assert activated["result"]["snapshot"]["state"] == "running"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_update_brief_keeps_prepared_mission_waiting_without_resolving_a_source(tmp_path):
    async def scenario():
        source, planner = _Source(), _Planner()
        runtime, store, _planner = _runtime(tmp_path, source, planner)
        try:
            prepared = await _prepare(runtime)
            snapshot = prepared["result"]["snapshot"]
            calls_before = source.calls
            updated = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="brief-before-source",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"goal": "Find loose fasteners"},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert updated["ok"]
            assert updated["result"]["execution_outcome"] == "waiting_for_source"
            assert updated["result"]["snapshot"]["state"] == "created"
            assert updated["result"]["snapshot"]["activation_intent"] == "when_source_starts"
            assert source.calls == calls_before
            assert planner.calls == 0
            assert store.get_mission(snapshot["mission_id"])["goal"] == "Find loose fasteners"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_brief_update_after_source_loss_reports_paused_and_never_restarts_on_frame(tmp_path):
    async def scenario():
        source, planner, watch = _Source(), _Planner(), _Watch()
        runtime, store, _planner = _runtime(tmp_path, source, planner, watch)
        try:
            prepared = await _prepare(runtime)
            snapshot = prepared["result"]["snapshot"]
            source.available = True
            activated = await runtime.handle(
                _command(
                    "activate",
                    request_id="activate-source",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"source_binding": {"source_id": "scout-1", "source_epoch": "41"}},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert activated["ok"]
            await _wait_for(lambda: bool(watch.active))
            await runtime.source_changed("scout-1", None, available=False)
            paused = store.get_mission(snapshot["mission_id"])
            assert paused["state"] == "paused"
            calls_before_update = planner.calls

            updated = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="brief-after-source-loss",
                    mission_id=snapshot["mission_id"],
                    revision=paused["revision"],
                    args={"goal": "Find corrosion"},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert updated["ok"]
            assert updated["result"]["execution_outcome"] == "paused"
            assert updated["result"]["snapshot"]["state"] == "paused"
            assert planner.calls == calls_before_update

            source.available = True
            source.epoch = "42"
            await asyncio.sleep(0.03)
            after_later_frame = store.get_mission(snapshot["mission_id"])
            assert after_later_frame["state"] == "paused"
            assert planner.calls == calls_before_update
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_noop_brief_update_preserves_active_watch_revision_generation_events_and_lease(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 60.0)

    async def scenario():
        source, planner, watch = _Source(), _Planner(), _Watch()
        source.available = True
        runtime, store, _planner = _runtime(tmp_path, source, planner, watch)
        try:
            prepared = await _prepare(runtime)
            snapshot = prepared["result"]["snapshot"]
            activated = await runtime.handle(
                _command(
                    "activate",
                    request_id="activate-for-noop",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"source_binding": {"source_id": "scout-1", "source_epoch": "41"}},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert activated["ok"]
            mission_id = snapshot["mission_id"]
            await _wait_for(lambda: bool(watch.active))
            await _wait_for(
                lambda: any(
                    event["kind"] == "cycle_finished"
                    for event in store.events_since(mission_id, 0)["events"]
                )
            )

            before = store.get_mission(mission_id)
            events_before = store.events_since(mission_id, 0)
            lease_before = watch.active
            released_before = list(watch.released)
            no_op = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="noop-active-brief",
                    mission_id=mission_id,
                    revision=before["revision"],
                    args={"expertise": before["expertise"], "goal": before["goal"]},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )

            assert no_op["ok"]
            assert no_op["result"]["execution_outcome"] == "unchanged"
            after = store.get_mission(mission_id)
            assert after["revision"] == before["revision"]
            assert after["execution_generation"] == before["execution_generation"]
            assert after["watch_lease"] == before["watch_lease"]
            assert store.events_since(mission_id, 0) == events_before
            assert watch.active is lease_before
            assert watch.released == released_before
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_brief_history_is_bounded_and_preserves_only_newest_eight_full_texts(tmp_path):
    async def scenario():
        source = _Source()
        runtime, store, _planner = _runtime(tmp_path, source)
        try:
            prepared = await _prepare(runtime)
            snapshot = prepared["result"]["snapshot"]
            for version in range(2, 65):
                expertise = f"E{version}" + "🧭" * (1_000 - len(f"E{version}"))
                goal = f"G{version}" + "🔎" * (1_000 - len(f"G{version}"))
                updated = await runtime.handle(
                    _command(
                        "update_brief",
                        request_id=f"brief-version-{version}",
                        mission_id=snapshot["mission_id"],
                        revision=snapshot["revision"],
                        args={"expertise": expertise, "goal": goal},
                    ),
                    Principal("installation"),
                    SCOPES,
                    integrated=True,
                )
                assert updated["ok"]
                snapshot = updated["result"]["snapshot"]

            history = snapshot["brief_history"]
            assert len(history) == 64
            assert [entry["version"] for entry in history] == list(range(1, 65))
            for entry in history[:-8]:
                assert set(entry) == {"version", "updated_at_ms", "brief_sha256"}
            for entry in history[-8:]:
                assert set(entry) == {
                    "version", "updated_at_ms", "brief_sha256", "expertise", "goal"
                }
            retained_text_bytes = sum(
                len(entry["expertise"].encode("utf-8"))
                + len(entry["goal"].encode("utf-8"))
                for entry in history[-8:]
            )
            assert retained_text_bytes <= 64 * 1024

            events_before = store.events_since(snapshot["mission_id"], 0)
            no_op = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="noop-at-history-limit",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"goal": snapshot["goal"]},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert no_op["ok"]
            assert no_op["result"]["execution_outcome"] == "unchanged"
            assert no_op["result"]["snapshot"]["revision"] == snapshot["revision"]
            assert store.events_since(snapshot["mission_id"], 0) == events_before

            rejected = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="brief-version-65",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"goal": "Inspection goal 65"},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert not rejected["ok"]
            assert rejected["error"]["code"] == "brief_history_full"
            assert store.get_mission(snapshot["mission_id"]) == snapshot
            assert store.events_since(snapshot["mission_id"], 0) == events_before
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


class _RecordingWatch(_Watch):
    def __init__(self):
        super().__init__()
        self.tasks = []

    def apply(self, request):
        self.tasks.append(request.task)
        return super().apply(request)


async def _prepare_with(runtime, request_id="prepare-watch-task", **extra):
    return await runtime.handle(
        _command(
            "prepare",
            request_id=request_id,
            args={
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "container maintenance technician",
                "goal": "Find defects",
                "mode": "watch",
                "reasoning_model": "gemma",
                "source_id": "scout-1",
                **extra,
            },
        ),
        Principal("installation"),
        SCOPES,
        integrated=True,
    )


async def _activate(runtime, snapshot):
    return await runtime.handle(
        _command(
            "activate",
            request_id="activate-watch-task",
            mission_id=snapshot["mission_id"],
            revision=snapshot["revision"],
            args={"source_binding": {"source_id": "scout-1", "source_epoch": "41"}},
        ),
        Principal("installation"),
        SCOPES,
        integrated=True,
    )


def test_watch_task_segment_overrides_planner_task_on_lease(tmp_path):
    async def scenario():
        source, watch = _Source(), _RecordingWatch()
        source.available = True
        runtime, store, _planner = _runtime(tmp_path, source, watch=watch)
        try:
            prepared = await _prepare_with(runtime, watch_task="segment")
            snapshot = prepared["result"]["snapshot"]
            assert snapshot["watch_task"] == "segment"
            assert (await _activate(runtime, snapshot))["ok"]
            await _wait_for(lambda: watch.tasks)
            assert watch.tasks[0] == "segment"
            mission = await _wait_for(lambda: (m := store.get_mission(snapshot["mission_id"])) and m.get("watch_lease") and m)
            assert mission["task"] == "segment"
            assert mission["watch_task"] == "segment"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_absent_watch_task_keeps_planner_task(tmp_path):
    async def scenario():
        source, watch = _Source(), _RecordingWatch()
        source.available = True
        runtime, store, _planner = _runtime(tmp_path, source, watch=watch)
        try:
            prepared = await _prepare_with(runtime)
            snapshot = prepared["result"]["snapshot"]
            assert "watch_task" not in snapshot
            assert (await _activate(runtime, snapshot))["ok"]
            await _wait_for(lambda: watch.tasks)
            assert watch.tasks[0] == "detect"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_invalid_watch_task_is_rejected_by_prepare_and_update_brief(tmp_path):
    async def scenario():
        source = _Source()
        runtime, store, _planner = _runtime(tmp_path, source)
        try:
            for bad in ("track", "", None, 1):
                rejected = await _prepare_with(runtime, request_id=f"bad-{bad!r}", watch_task=bad)
                assert not rejected["ok"]
                assert rejected["error"]["code"] == "invalid_request"
            prepared = await _prepare_with(runtime)
            snapshot = prepared["result"]["snapshot"]
            rejected = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="bad-watch-task",
                    mission_id=snapshot["mission_id"],
                    revision=snapshot["revision"],
                    args={"watch_task": "track"},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert not rejected["ok"]
            assert rejected["error"]["code"] == "invalid_request"
            assert store.get_mission(snapshot["mission_id"]) == snapshot
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_update_brief_with_only_watch_task_applies_at_next_cycle(tmp_path, monkeypatch):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "WATCH_MIN_INTERVAL_SECONDS", 0.05)

    async def scenario():
        source, watch = _Source(), _RecordingWatch()
        source.available = True
        runtime, store, _planner = _runtime(tmp_path, source, watch=watch)
        try:
            prepared = await _prepare_with(runtime, watch_task="detect")
            snapshot = prepared["result"]["snapshot"]
            mission_id = snapshot["mission_id"]
            assert (await _activate(runtime, snapshot))["ok"]
            await _wait_for(lambda: watch.tasks)
            assert watch.tasks[0] == "detect"
            before = store.get_mission(mission_id)
            updated = await runtime.handle(
                _command(
                    "update_brief",
                    request_id="watch-task-only",
                    mission_id=mission_id,
                    revision=before["revision"],
                    args={"watch_task": "segment"},
                ),
                Principal("installation"),
                SCOPES,
                integrated=True,
            )
            assert updated["ok"]
            after = updated["result"]["snapshot"]
            assert after["watch_task"] == "segment"
            assert after["brief_version"] == before["brief_version"]
            assert after["execution_generation"] == before["execution_generation"]
            assert after["state"] == "running"
            await _wait_for(lambda: "segment" in watch.tasks)
            assert store.get_mission(mission_id)["watch_task"] == "segment"
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
