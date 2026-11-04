"""Model-free read, evidence-integrity, and reply-retention tests."""

import asyncio
import base64
import hashlib
from io import BytesIO

from PIL import Image

from visionbrain.mission_contracts import Principal
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


SCOPES = {"mission:read", "mission:control", "mission:evidence", "mission:review"}


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (16, 12), (25, 110, 180)).save(output, format="JPEG")
    return output.getvalue()


class _Planner:
    def __init__(self):
        self.calls = 0

    def available(self, _model_key):
        return True

    def plan(self, _context):
        self.calls += 1
        raise AssertionError("read tests must not dispatch native planning")

    def close(self):
        pass


class _Tools:
    def available_tools(self):
        return ()

    def execute(self, *_args):
        raise AssertionError("read tests must not dispatch native tools")


class _Watch:
    def current_configuration_revision(self):
        return 0

    def apply(self, _request):
        raise AssertionError("read tests must not change Watch state")

    def release(self, _lease, _reason):
        raise AssertionError("read tests must not release Watch state")


def _runtime(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    events = []
    planner = _Planner()
    runtime = MissionRuntime(
        store,
        planner,
        _Tools(),
        lambda _binding: None,
        _Watch(),
        events.append,
        qualified_models={"gemma": ["inspect"]},
    )
    return runtime, store, planner, events


async def _create_with_evidence(runtime):
    created = await runtime.handle(
        {
            "type": "mission_command",
            "schema_version": 1,
            "request_id": "create-1",
            "command": "create",
            "mission_id": None,
            "args": {
                "profile": {"id": "visual_inspection", "version": 1},
                "expertise": "container maintenance technician",
                "mode": "inspect",
                "reasoning_model": "gemma",
            },
        },
        Principal("installation"),
        SCOPES,
    )
    assert created["ok"]
    snapshot = created["result"]["snapshot"]
    jpeg = _jpeg()
    attached = await runtime.handle(
        {
            "type": "mission_command",
            "schema_version": 1,
            "request_id": "attach-1",
            "command": "attach_evidence",
            "mission_id": snapshot["mission_id"],
            "expected_revision": snapshot["revision"],
            "args": {
                "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                "sha256": hashlib.sha256(jpeg).hexdigest(),
            },
        },
        Principal("installation"),
        SCOPES,
    )
    assert attached["ok"]
    return attached["result"]["snapshot"], attached["result"]["evidence_id"], jpeg


async def _read(runtime, request_id, command, mission_id=None, args=None):
    return await runtime.handle(
        {
            "type": "mission_command",
            "schema_version": 1,
            "request_id": request_id,
            "command": command,
            "mission_id": mission_id,
            "args": args or {},
        },
        Principal("installation"),
        SCOPES,
    )


def _replace_file(path, data):
    replacement = path.with_suffix(".replacement")
    replacement.write_bytes(data)
    replacement.replace(path)


def test_read_commands_do_not_persist_replies_but_mutations_remain_durable(tmp_path):
    async def scenario():
        runtime, store, _planner, _events = _runtime(tmp_path)
        try:
            snapshot, evidence_id, _jpeg_bytes = await _create_with_evidence(runtime)
            mission_id = snapshot["mission_id"]
            reads = [
                ("read-get", "get", mission_id, {}),
                ("read-list", "list", None, {}),
                ("read-events", "events_since", mission_id, {"cursor": 0}),
                ("read-export", "export", mission_id, {}),
                ("read-evidence", "get_evidence", mission_id, {"evidence_id": evidence_id}),
            ]
            for request_id, command, target, args in reads:
                reply = await _read(runtime, request_id, command, target, args)
                assert reply["ok"]

            placeholders = ",".join("?" for _ in reads)
            read_count = store._connection.execute(
                f"SELECT COUNT(*) FROM requests WHERE request_id IN ({placeholders})",
                tuple(item[0] for item in reads),
            ).fetchone()[0]
            mutation_count = store._connection.execute(
                "SELECT COUNT(*) FROM requests WHERE request_id IN ('create-1', 'attach-1')"
            ).fetchone()[0]
            assert read_count == 0
            assert mutation_count == 2
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_get_and_export_use_stat_checks_and_report_replaced_evidence(tmp_path, monkeypatch):
    async def scenario():
        runtime, store, _planner, _events = _runtime(tmp_path)
        try:
            snapshot, evidence_id, jpeg = await _create_with_evidence(runtime)
            mission_id = snapshot["mission_id"]

            def fail_full_read(*_args, **_kwargs):
                raise AssertionError("get/export must not read and hash every JPEG")

            monkeypatch.setattr(store, "read_evidence", fail_full_read)
            assert (await _read(runtime, "get-fast", "get", mission_id))["ok"]
            assert (await _read(runtime, "export-fast", "export", mission_id))["ok"]

            path = store.evidence_root / f"{evidence_id}.jpg"
            _replace_file(path, b"x" * len(jpeg))
            reply = await _read(runtime, "get-changed", "get", mission_id)
            assert reply["ok"]
            evidence = reply["result"]["snapshot"]["evidence"][0]
            assert evidence["available"] is False
            assert reply["revision"] == snapshot["revision"] + 1
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_get_evidence_failure_commits_unavailable_downgrade_after_read_rollback(tmp_path):
    async def scenario():
        runtime, store, _planner, events = _runtime(tmp_path)
        try:
            snapshot, evidence_id, _jpeg_bytes = await _create_with_evidence(runtime)
            mission_id = snapshot["mission_id"]

            def add_supported_finding(tx):
                current = tx.get_mission(mission_id)
                updated = dict(current)
                updated["findings"] = [{
                    "status": "supported",
                    "evidence_id": evidence_id,
                    "evidence_refs": [evidence_id],
                    "localization": {"status": "supported", "evidence_id": evidence_id},
                }]
                return tx.update_mission(
                    updated,
                    expected_revision=current["revision"],
                    updated_at_ms=current["updated_at_ms"] + 1,
                )

            store.transact(add_supported_finding)
            (store.evidence_root / f"{evidence_id}.jpg").unlink()
            reply = await _read(
                runtime,
                "missing-evidence-read",
                "get_evidence",
                mission_id,
                {"evidence_id": evidence_id},
            )

            assert reply["ok"] is False
            assert reply["error"]["code"] == "evidence_unavailable"
            current = store.get_mission(mission_id)
            assert current["evidence"][0]["available"] is False
            assert current["findings"][0]["status"] == "unresolved"
            assert current["findings"][0]["localization"]["status"] == "unresolved"
            assert [event.kind for event in events].count("evidence_unavailable") == 1
            assert store.events_since(mission_id, 0)["events"][-1]["kind"] == "evidence_unavailable"
            assert store._connection.execute(
                "SELECT COUNT(*) FROM requests WHERE request_id = 'missing-evidence-read'"
            ).fetchone()[0] == 0
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())


def test_resume_hashes_exact_evidence_before_planner_dispatch(tmp_path):
    async def scenario():
        runtime, store, planner, _events = _runtime(tmp_path)
        try:
            snapshot, evidence_id, jpeg = await _create_with_evidence(runtime)
            path = store.evidence_root / f"{evidence_id}.jpg"
            path.write_bytes(b"x" * len(jpeg))
            reply = await runtime.handle(
                {
                    "type": "mission_command",
                    "schema_version": 1,
                    "request_id": "resume-tampered",
                    "command": "resume",
                    "mission_id": snapshot["mission_id"],
                    "expected_revision": snapshot["revision"],
                    "args": {},
                },
                Principal("installation"),
                SCOPES,
            )

            assert reply["ok"] is False
            assert reply["error"]["code"] == "evidence_unavailable"
            assert planner.calls == 0
            current = store.get_mission(snapshot["mission_id"])
            assert current["state"] == "created"
            assert current["evidence"][0]["available"] is False
        finally:
            await runtime.close()
            store.close()

    asyncio.run(scenario())
