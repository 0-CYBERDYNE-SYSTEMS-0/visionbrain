"""Close-up recovery behavior when the evidence root quota is exhausted."""

import asyncio
import base64
import hashlib
import time
from io import BytesIO

from PIL import Image
import pytest

from visionbrain.mission_contracts import Decision, Principal, ToolResult
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore, QuotaAccountingIncomplete


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg(color):
    output = BytesIO()
    Image.new("RGB", (32, 24), color).save(output, format="JPEG")
    return output.getvalue()


def _command(name, request_id, mission_id=None, revision=None, args=None):
    message = {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "command": name,
        "mission_id": mission_id,
        "args": args or {},
    }
    if revision is not None:
        message["expected_revision"] = revision
    return message


def _attachment(jpeg, *, closeup_request_id=None):
    args = {"jpeg_b64": base64.b64encode(jpeg).decode(), "sha256": hashlib.sha256(jpeg).hexdigest()}
    if closeup_request_id is not None:
        args["closeup_request_id"] = closeup_request_id
    return args


async def _wait_for(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before bounded timeout")


class _CloseupPlanner:
    def __init__(self):
        self.calls = 0

    def available(self, model_key):
        return model_key == "gemma"

    def plan(self, _context):
        self.calls += 1
        if self.calls == 1:
            return Decision(1, "request_closeup", {"description": "surface detail", "reason": "Need a closer image."})
        return Decision(1, "finish")

    def close(self):
        pass


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, _request, _context):
        return ToolResult("empty")


class _Watch:
    def current_configuration_revision(self):
        return 0


@pytest.mark.parametrize(
    ("quota_case", "expected_reason"),
    [
        ("full", "evidence_quota_full"),
        ("accounting_unavailable", "evidence_quota_accounting_unavailable"),
    ],
)
def test_closeup_quota_recovery_retains_waiting_request_and_failed_reply_identity(
    tmp_path, monkeypatch, quota_case, expected_reason
):
    from visionbrain import mission_runtime as runtime_module

    monkeypatch.setattr(runtime_module, "MAX_HUMAN_WAIT_SECONDS", 60)
    initial_jpeg = _jpeg((12, 90, 175))
    closeup_jpeg = _jpeg((175, 90, 12))
    orphan = b"unindexed root evidence" * 4
    evidence_root = tmp_path / "evidence"
    root_quota = (
        len(initial_jpeg) + len(closeup_jpeg) + len(orphan) - 1
        if quota_case == "full" else 1_000_000
    )
    store = MissionStore(
        tmp_path / "missions.sqlite3",
        evidence_root,
        root_quota_bytes=root_quota,
    )
    orphan_path = None
    if quota_case == "full":
        orphan_path = evidence_root / "unindexed-orphan.bin"
        orphan_path.write_bytes(orphan)
    else:
        original_root_usage = store._root_evidence_usage
        scans = 0

        def incomplete_on_closeup_scan():
            nonlocal scans
            scans += 1
            if scans == 2:
                raise QuotaAccountingIncomplete("injected bounded root scan failure")
            return original_root_usage()

        monkeypatch.setattr(store, "_root_evidence_usage", incomplete_on_closeup_scan)
    runtime = MissionRuntime(
        store,
        _CloseupPlanner(),
        _Tools(),
        lambda _binding: None,
        _Watch(),
        lambda _event: None,
        qualified_models={"gemma": ["inspect"]},
    )

    async def scenario():
        try:
            created = await runtime.handle(
                _command("create", "quota-closeup-create", args={
                    "profile": {"id": "visual_inspection", "version": 1},
                    "expertise": "site inspector",
                    "mode": "inspect",
                    "reasoning_model": "gemma",
                }),
                Principal("installation"),
                SCOPES,
            )
            assert created["ok"], created
            snapshot = created["result"]["snapshot"]
            attached = await runtime.handle(
                _command(
                    "attach_evidence", "quota-closeup-input", snapshot["mission_id"], snapshot["revision"],
                    _attachment(initial_jpeg),
                ),
                Principal("installation"),
                SCOPES,
            )
            assert attached["ok"], attached
            snapshot = attached["result"]["snapshot"]
            resumed = await runtime.handle(
                _command("resume", "quota-closeup-first-resume", snapshot["mission_id"], snapshot["revision"]),
                Principal("installation"),
                SCOPES,
            )
            assert resumed["ok"], resumed
            waiting = await _wait_for(
                lambda: (
                    current
                    if (current := store.get_mission(snapshot["mission_id"]))["state"] == "waiting_evidence"
                    else None
                )
            )
            closeup = waiting["closeup_request"]
            assert closeup is not None

            original_attachment = _command(
                "attach_evidence",
                "quota-closeup-original-photo",
                waiting["mission_id"],
                waiting["revision"],
                _attachment(closeup_jpeg, closeup_request_id=closeup["request_id"]),
            )
            refused = await runtime.handle(original_attachment, Principal("installation"), SCOPES)
            assert not refused["ok"]
            assert refused["error"]["code"] == expected_reason
            waiting_after_refusal = refused["result"]["snapshot"]
            assert waiting_after_refusal["state"] == "waiting_evidence"
            assert waiting_after_refusal["reason"] == expected_reason
            assert waiting_after_refusal["closeup_request"] == closeup
            assert waiting_after_refusal["execution_generation"] == waiting["execution_generation"] + 1
            assert waiting_after_refusal["cycle_id"] is None
            assert waiting_after_refusal["watch_lease"] is None
            assert "retry the requested close-up" in waiting_after_refusal["activity"]
            assert refused["result"]["execution_outcome"] == "waiting_evidence"
            assert store.get_mission(waiting["mission_id"]) == waiting_after_refusal

            if orphan_path is not None:
                orphan_path.unlink()
            root_has_headroom = (
                (orphan_path is None or not orphan_path.exists())
                and store._root_evidence_usage() + len(closeup_jpeg) <= store.root_quota_bytes
            )

            stale = await runtime.handle(
                _command(
                    "attach_evidence", "quota-closeup-stale-revision", waiting["mission_id"], waiting["revision"],
                    _attachment(closeup_jpeg, closeup_request_id=closeup["request_id"]),
                ),
                Principal("installation"),
                SCOPES,
            )
            after_stale = store.get_mission(waiting["mission_id"])
            wrong = await runtime.handle(
                _command(
                    "attach_evidence", "quota-closeup-wrong-request", waiting["mission_id"],
                    after_stale["revision"], _attachment(closeup_jpeg, closeup_request_id="wrong-closeup-id"),
                ),
                Principal("installation"),
                SCOPES,
            )
            after_wrong = store.get_mission(waiting["mission_id"])
            retry = await runtime.handle(
                _command(
                    "attach_evidence",
                    "quota-closeup-retry-photo",
                    waiting["mission_id"],
                    after_wrong["revision"],
                    _attachment(closeup_jpeg, closeup_request_id=closeup["request_id"]),
                ),
                Principal("installation"),
                SCOPES,
            )
            replay = await runtime.handle(original_attachment, Principal("installation"), SCOPES)

            assert (
                root_has_headroom
                and stale["error"]["code"] == "revision_conflict"
                and after_stale == waiting_after_refusal
                and wrong["error"]["code"] == "closeup_mismatch"
                and after_wrong == waiting_after_refusal
                and retry["ok"]
                and retry["result"]["snapshot"]["evidence"][-1]["closeup_request_id"] == closeup["request_id"]
                and replay == refused
            ), (
                "expected corrected storage condition + a new matching close-up request to succeed, "
                "stale/wrong requests to leave the wait unchanged, and original replay to stay exact; actual "
                f"reason={refused['error']['code']}, root_headroom={root_has_headroom}, "
                f"stale={stale.get('error')}, wrong={wrong.get('error')}, retry={retry.get('error')}, "
                f"replay_identical={replay == refused}"
            )
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
