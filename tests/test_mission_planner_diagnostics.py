"""Private bounded records for invalid planner responses."""

import asyncio
import base64
import hashlib
import time
from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from visionbrain.mission_contracts import Principal
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MAX_PLANNER_DIAGNOSTIC_BYTES, MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


class _InvalidResponsePlanner:
    def __init__(self, raw):
        self.raw = raw
        self.calls = 0

    def available(self, _model):
        return True

    def plan_with_response(self, _context):
        self.calls += 1
        return SimpleNamespace(
            valid=False,
            decision=None,
            raw_text=self.raw,
            error="nested reason is not a valid detect_objects argument",
            model_key="gemma",
            checkpoint="cached-gemma-revision",
            prompt_version="mission.v1-local-json-7",
        )

    def close(self):
        return None


class _Tools:
    def available_tools(self):
        return ("detect_objects", "segment_objects", "read_text", "inspect_crop")

    def execute(self, *_args):
        raise AssertionError("invalid planner response must not call a tool")


class _Watch:
    def current_configuration_revision(self):
        return 0

    def release(self, *_args):
        raise AssertionError("Inspect has no Watch lease")


async def _wait_for(predicate, timeout=2.0):
    until = time.monotonic() + timeout
    while time.monotonic() < until:
        value = predicate()
        if value:
            return value
        await asyncio.sleep(0.005)
    raise AssertionError("condition was not met before timeout")


def test_invalid_planner_raw_output_is_stored_privately_and_bounded(tmp_path):
    async def scenario():
        raw = "RAW_PLANNER_SECRET:" + ("x" * (MAX_PLANNER_DIAGNOSTIC_BYTES + 10))
        planner = _InvalidResponsePlanner(raw)
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store,
            planner,
            _Tools(),
            lambda _binding: None,
            _Watch(),
            lambda _event: None,
            qualified_models={"gemma": ["inspect"]},
        )
        jpeg = _jpeg()
        try:
            created = await runtime.handle(
                {
                    "type": "mission_command", "schema_version": 1,
                    "request_id": "create-diagnostic", "mission_id": None,
                    "command": "create", "args": {
                        "profile": {"id": "visual_inspection", "version": 1},
                        "expertise": "equipment inspector", "goal": "Find visible defects",
                        "mode": "inspect", "reasoning_model": "gemma",
                    },
                },
                Principal("installation"), SCOPES,
            )
            snapshot = created["result"]["snapshot"]
            attached = await runtime.handle(
                {
                    "type": "mission_command", "schema_version": 1,
                    "request_id": "attach-diagnostic", "mission_id": snapshot["mission_id"],
                    "expected_revision": snapshot["revision"], "command": "attach_evidence",
                    "args": {
                        "jpeg_b64": base64.b64encode(jpeg).decode(),
                        "sha256": hashlib.sha256(jpeg).hexdigest(),
                    },
                },
                Principal("installation"), SCOPES,
            )
            resumed = await runtime.handle(
                {
                    "type": "mission_command", "schema_version": 1,
                    "request_id": "resume-diagnostic", "mission_id": snapshot["mission_id"],
                    "expected_revision": attached["result"]["snapshot"]["revision"],
                    "command": "resume", "args": {},
                },
                Principal("installation"), SCOPES,
            )
            assert resumed["ok"]
            final = await _wait_for(
                lambda: (
                    store.get_mission(snapshot["mission_id"])
                    if store.get_mission(snapshot["mission_id"])["state"] == "paused"
                    else None
                )
            )
            assert final["reason"] == "planner_repair_exhausted"
            records = store.planner_diagnostics(snapshot["mission_id"])
            assert len(records) == 2
            assert all(record["raw_text_truncated"] for record in records)
            assert all(len(record["raw_text"].encode("utf-8")) <= MAX_PLANNER_DIAGNOSTIC_BYTES for record in records)
            assert records[0]["raw_text"].startswith("RAW_PLANNER_SECRET:")
            assert records[0]["model_key"] == "gemma"
            assert records[0]["checkpoint"] == "cached-gemma-revision"
            assert records[0]["prompt_version"] == "mission.v1-local-json-7"
            assert "RAW_PLANNER_SECRET" not in str(final)

            exported = await runtime.handle(
                {
                    "type": "mission_command", "schema_version": 1,
                    "request_id": "export-diagnostic", "mission_id": snapshot["mission_id"],
                    "command": "export", "args": {},
                },
                Principal("installation"), SCOPES,
            )
            assert exported["ok"]
            assert "RAW_PLANNER_SECRET" not in str(exported["result"]["packet"])
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


class _InvalidFinishPlanner(_InvalidResponsePlanner):
    def plan_with_response(self, _context):
        from visionbrain.mission_contracts import Decision, FindingProposal

        self.calls += 1
        finding = FindingProposal("a cooler", "localized_object", ())
        return SimpleNamespace(
            valid=True,
            decision=Decision(1, "finish", findings=(finding,)),
            raw_text=self.raw,
            model_key="gemma",
            checkpoint="cached-gemma-revision",
            prompt_version="mission.v1-local-json-7",
        )


def test_finish_validation_failure_stores_the_raw_planner_output(tmp_path):
    async def scenario():
        planner = _InvalidFinishPlanner("RAW_FINISH_OUTPUT")
        store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
        runtime = MissionRuntime(
            store, planner, _Tools(), lambda _binding: None, _Watch(),
            lambda _event: None, qualified_models={"gemma": ["inspect"]},
        )
        jpeg = _jpeg()

        async def command(name, mission_id=None, revision=None, args=None):
            message = {
                "type": "mission_command", "schema_version": 1,
                "request_id": f"{name}-finish", "mission_id": mission_id,
                "command": name, "args": args or {},
            }
            if revision is not None:
                message["expected_revision"] = revision
            return await runtime.handle(message, Principal("installation"), SCOPES)

        try:
            created = await command("create", args={
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "equipment inspector", "goal": "Find visible defects",
                "mode": "inspect", "reasoning_model": "gemma",
            })
            snapshot = created["result"]["snapshot"]
            attached = await command("attach_evidence", snapshot["mission_id"], snapshot["revision"], {
                "jpeg_b64": base64.b64encode(jpeg).decode(),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            })
            await command("resume", snapshot["mission_id"], attached["result"]["snapshot"]["revision"])
            final = await _wait_for(
                lambda: (
                    store.get_mission(snapshot["mission_id"])
                    if store.get_mission(snapshot["mission_id"])["state"] == "paused"
                    else None
                )
            )
            assert final["reason"] == "planner_repair_exhausted"
            records = store.planner_diagnostics(snapshot["mission_id"])
            assert len(records) == 2
            assert records[0]["raw_text"] == "RAW_FINISH_OUTPUT"
            assert "grounded item" in records[0]["error"]
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
