"""CPU-only tests for the internal metadata-only packet preview."""

import asyncio
import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from visionbrain.mission_record_projection import MissionRecordProjectionError
from visionbrain.mission_records import (
    EvidenceRecord,
    Finding,
    InspectionPacket,
    Observation,
    validate_record_bundle,
)
from visionbrain.mission_runtime import MissionRuntime
from visionbrain.mission_store import MissionStore


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), (12, 90, 175)).save(output, format="JPEG")
    return output.getvalue()


def _seed_mission(store: MissionStore) -> tuple[str, str, str]:
    mission_id = "packet-preview-mission"
    initial = {
        "mission_id": mission_id,
        "revision": 1,
        "state": "completed",
        "mode": "inspect",
        "updated_at_ms": 1_000,
    }

    def operation(tx):
        tx.insert_mission(initial)
        primary = tx.save_evidence(
            mission_id,
            _jpeg(),
            kind="image",
            created_at_ms=10,
        )
        recorded_missing = tx.save_evidence(
            mission_id,
            _jpeg(),
            kind="image",
            created_at_ms=11,
        )
        tx.set_evidence_unavailable(mission_id, recorded_missing["evidence_id"], "missing")
        final = dict(initial)
        final.update({
            "execution_generation": 2,
            "cycle_id": None,
            "cycle_history": [{
                "cycle_id": "cycle-1",
                "outcome": "completed",
                "execution_generation": 1,
                "evidence_refs": [{"evidence_id": primary["evidence_id"]}],
                "finding_ids": ["finding-unresolved"],
            }],
            "evidence": [primary, recorded_missing],
            "findings": [{
                "finding_id": "finding-unresolved",
                "claim": "A visible item needs a closer look.",
                "claim_type": "visual_hypothesis",
                "status": "unresolved",
                "reason": "insufficient visual detail",
                "evidence_id": primary["evidence_id"],
                "evidence_refs": [primary["evidence_id"]],
                "review": {
                    "decision": "accepted",
                    "actor": "operator-1",
                    "time_ms": 1_100,
                    "mission_revision": 2,
                    "note": "Recorded review attribution.",
                },
            }],
        })
        tx.add_tool_record(
            mission_id,
            "cycle-1",
            1,
            {
                "tool_result_id": "tool-empty",
                "tool": "detect_objects",
                "status": "empty",
                "input_evidence_id": primary["evidence_id"],
                "items": [],
                "evidence_ids": [],
                "text": "",
            },
            current=True,
            created_at_ms=12,
        )
        saved = tx.update_mission(final, expected_revision=1, updated_at_ms=1_001)
        tx.append_event(mission_id, saved["revision"], "fixture_ready", {}, 1_001)
        return primary["evidence_id"], recorded_missing["evidence_id"]

    primary_id, missing_id = store.transact(operation).value
    return mission_id, primary_id, missing_id


def _seed_paused_watch(store: MissionStore) -> tuple[str, str, str]:
    mission_id = "packet-preview-paused-watch"
    source_binding = {"source_id": "scout-1", "source_epoch": "41"}
    initial = {
        "mission_id": mission_id,
        "revision": 1,
        "state": "created",
        "mode": "watch",
        "updated_at_ms": 1_000,
    }

    def operation(tx):
        tx.insert_mission(initial)
        primary = tx.save_evidence(
            mission_id,
            _jpeg(),
            kind="frame",
            origin="watch_frame",
            source_id=source_binding["source_id"],
            source_epoch=source_binding["source_epoch"],
            frame_id=17,
            capture_time_ms=None,
            created_at_ms=1_010,
        )
        recorded_missing = tx.save_evidence(
            mission_id,
            _jpeg(),
            kind="frame",
            origin="watch_frame",
            source_id=source_binding["source_id"],
            source_epoch=source_binding["source_epoch"],
            frame_id=18,
            capture_time_ms=None,
            created_at_ms=1_020,
        )
        tx.set_evidence_unavailable(mission_id, recorded_missing["evidence_id"], "missing")
        finding = {
            "finding_id": "finding-watch-unresolved",
            "claim": "A visible item needs a closer look.",
            "claim_type": "visual_hypothesis",
            "status": "unresolved",
            "reason": "insufficient visual detail",
            "evidence_id": primary["evidence_id"],
            "evidence_refs": [primary["evidence_id"]],
            "source_binding": source_binding,
            "frame_id": 17,
            "observations": [{
                "evidence_id": primary["evidence_id"],
                "frame_id": 17,
                "source_binding": source_binding,
            }],
            "review": {
                "decision": "accepted",
                "actor": "operator-1",
                "time_ms": 1_100,
                "mission_revision": 2,
                "note": "Recorded review attribution.",
            },
        }
        tx.add_tool_record(
            mission_id,
            "cycle-watch-1",
            1,
            {
                "tool_result_id": "tool-watch-empty",
                "tool": "detect_objects",
                "status": "empty",
                "input_evidence_id": primary["evidence_id"],
                "items": [],
                "evidence_ids": [],
                "text": "",
            },
            current=True,
            created_at_ms=1_030,
        )
        final = dict(initial)
        final.update({
            "revision": 2,
            "state": "paused",
            "reason": "operator_paused",
            "source_binding": source_binding,
            "execution_generation": 2,
            "cycle_id": None,
            "watch_lease": None,
            "cycle_history": [{
                "cycle_id": "cycle-watch-1",
                "outcome": "completed",
                "execution_generation": 1,
                "evidence_refs": [{"evidence_id": primary["evidence_id"]}],
                "finding_ids": [finding["finding_id"]],
            }],
            "evidence": [primary, recorded_missing],
            "findings": [finding],
        })
        saved = tx.update_mission(final, expected_revision=1, updated_at_ms=1_100)
        tx.append_event(mission_id, saved["revision"], "fixture_ready", {}, 1_100)
        return primary["evidence_id"], recorded_missing["evidence_id"]

    primary_id, missing_id = store.transact(operation).value
    return mission_id, primary_id, missing_id


def _runtime(store: MissionStore, event_sink=None) -> MissionRuntime:
    return MissionRuntime(
        store, object(), object(), lambda _binding: None, object(), event_sink or (lambda _event: None)
    )


def _preview_command(mission_id: str, revision: int, request_id: str = "preview-1") -> dict:
    return {
        "type": "mission_command",
        "schema_version": 1,
        "request_id": request_id,
        "mission_id": mission_id,
        "expected_revision": revision,
        "command": "preview_packet",
        "args": {},
    }


def _store_state(store: MissionStore, mission_id: str) -> tuple:
    with store._lock:
        mission = store._connection.execute(
            "SELECT revision, state, snapshot_json FROM missions WHERE mission_id = ?", (mission_id,)
        ).fetchone()
        evidence = store._connection.execute(
            "SELECT evidence_id, available, metadata_json FROM evidence WHERE mission_id = ? ORDER BY evidence_id",
            (mission_id,),
        ).fetchall()
        events = store._connection.execute(
            "SELECT sequence, revision, kind, data_json, timestamp_ms FROM events WHERE mission_id = ? ORDER BY sequence",
            (mission_id,),
        ).fetchall()
        pins = store._connection.execute(
            "SELECT evidence_id FROM exported_evidence WHERE mission_id = ? ORDER BY evidence_id",
            (mission_id,),
        ).fetchall()
        requests = store._connection.execute(
            "SELECT request_id, payload_sha256, reply_json FROM requests ORDER BY request_id"
        ).fetchall()
        file_state = store._connection.execute(
            "SELECT evidence_id, device, inode, size, mtime_ns, ctime_ns FROM evidence_file_state ORDER BY evidence_id"
        ).fetchall()
        packet_tables = store._connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name LIKE '%packet%' ORDER BY name"
        ).fetchall()
    return tuple(tuple(row) for row in (mission,)) + tuple(
        tuple(tuple(row) for row in rows)
        for rows in (evidence, events, pins, requests, file_state, packet_tables)
    )


def test_packet_preview_is_deterministic_and_metadata_only(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, primary_id, missing_id = _seed_mission(store)
        # Physical loss after the last persisted reconciliation remains unobserved by preview.
        (store.evidence_root / f"{primary_id}.jpg").unlink()
        before = _store_state(store, mission_id)
        runtime = _runtime(store)

        first = asyncio.run(runtime.preview_packet(mission_id))
        second = asyncio.run(runtime.preview_packet(mission_id))

        assert first == second
        assert isinstance(first.packet, InspectionPacket)
        assert first.packet.state == "draft"
        assert first.packet.mission_id == mission_id
        assert first.packet.mission_revision == 2
        assert first.packet.packet_id.startswith("packet-")
        assert first.canonical_json == json.dumps(
            json.loads(first.canonical_json),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        assert first.content_sha256 == hashlib.sha256(first.canonical_json.encode("utf-8")).hexdigest()
        validate_record_bundle(first.records)
        assert first.records[-1] == first.packet

        finding = next(record for record in first.records if isinstance(record, Finding))
        assert finding.finding_id in first.packet.finding_ids
        assert finding.visual_state == "unresolved"
        assert finding.review is not None
        assert (finding.review.state, finding.review.actor, finding.review.note) == (
            "accepted", "operator-1", "Recorded review attribution."
        )
        observation = next(record for record in first.records if isinstance(record, Observation))
        assert observation.status == "stale"
        evidence = {
            record.evidence.evidence_id: record
            for record in first.records
            if isinstance(record, EvidenceRecord)
        }
        assert evidence[primary_id].availability == "available"
        assert evidence[missing_id].availability == "missing"
        assert first.packet.evidence_ids == tuple(sorted(evidence))
        assert _store_state(store, mission_id) == before
    finally:
        store.close()


def test_packet_preview_identity_changes_with_mission_revision(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)
        runtime = _runtime(store)
        first = asyncio.run(runtime.preview_packet(mission_id))

        def advance_revision(tx):
            snapshot = tx.get_mission(mission_id)
            return tx.update_mission(
                snapshot,
                expected_revision=snapshot["revision"],
                updated_at_ms=snapshot["updated_at_ms"] + 1,
            )

        updated = store.transact(advance_revision).value
        second = asyncio.run(runtime.preview_packet(mission_id))

        assert updated["revision"] == first.packet.mission_revision + 1
        assert second.packet.mission_revision == updated["revision"]
        assert second.packet.packet_id != first.packet.packet_id
        assert second.content_sha256 != first.content_sha256
        assert second.packet.state == "draft"
    finally:
        store.close()


def test_packet_preview_propagates_malformed_projection_error(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)

        def corrupt_projection(tx):
            snapshot = tx.get_mission(mission_id)
            snapshot["findings"][0]["status"] = "pending"
            return tx.update_mission(
                snapshot,
                expected_revision=snapshot["revision"],
                updated_at_ms=snapshot["updated_at_ms"] + 1,
            )

        store.transact(corrupt_projection)
        with pytest.raises(MissionRecordProjectionError, match="unsupported_visual_state"):
            asyncio.run(_runtime(store).preview_packet(mission_id))
    finally:
        store.close()


def test_preview_packet_command_returns_read_only_paused_watch_history(tmp_path, monkeypatch):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, primary_id, missing_id = _seed_paused_watch(store)
        (store.evidence_root / f"{primary_id}.jpg").unlink()
        before = _store_state(store, mission_id)
        events = []
        runtime = _runtime(store, events.append)
        reads = []
        read_rows = store.read_mission_record_rows

        def counted_read(mission_id, *, tx=None):
            reads.append(mission_id)
            return read_rows(mission_id, tx=tx)

        monkeypatch.setattr(store, "read_mission_record_rows", counted_read)
        command = _preview_command(mission_id, 2)
        first = asyncio.run(runtime.handle(command, "operator-1", {"mission:read"}))
        second = asyncio.run(runtime.handle(command, "operator-1", {"mission:read"}))

        assert first == second
        assert first["ok"] is True, first
        assert first["revision"] == 2
        result = first["result"]
        assert result["envelope_version"] == 1
        assert result["kind"] == "inspection_packet_preview"
        assert result["verification"] == {
            "basis": "persisted_metadata_only",
            "fresh_media_verified": False,
        }
        assert result["mission_id"] == mission_id
        assert result["mission_revision"] == 2
        assert result["packet"]["state"] == "draft"
        assert result["packet"] == result["bundle"][-1]
        canonical = json.loads(result["canonical_json"])
        assert set(canonical) == {"format", "records"}
        assert canonical["format"] == "visionbrain.packet-preview.v1"
        assert canonical["records"] == result["bundle"]
        assert result["content_sha256"] == hashlib.sha256(
            result["canonical_json"].encode("utf-8")
        ).hexdigest()
        assert reads == [mission_id, mission_id]
        assert events == []

        observation = next(row for row in result["bundle"] if row["record_type"] == "observation")
        assert observation["status"] == "stale"
        assert (observation["source_id"], observation["source_epoch"], observation["frame_id"]) == (
            "scout-1", "41", 17
        )
        finding = next(row for row in result["bundle"] if row["record_type"] == "finding")
        assert finding["visual_state"] == "unresolved"
        assert finding["review"]["actor"] == "operator-1"
        evidence = {row["evidence"]["evidence_id"]: row for row in result["bundle"] if row["record_type"] == "evidence"}
        assert evidence[primary_id]["availability"] == "available"
        assert evidence[missing_id]["availability"] == "missing"
        assert not (store.evidence_root / f"{primary_id}.jpg").exists()
        assert _store_state(store, mission_id) == before
    finally:
        store.close()


def test_preview_packet_command_reports_revision_conflict_without_writes(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)
        before = _store_state(store, mission_id)
        reply = asyncio.run(_runtime(store).handle(
            _preview_command(mission_id, 1), "operator-1", {"mission:read"}
        ))
        assert reply["ok"] is False
        assert reply["error"]["code"] == "revision_conflict"
        assert reply["revision"] == 2
        assert reply["result"]["snapshot"]["revision"] == 2
        assert _store_state(store, mission_id) == before
    finally:
        store.close()


@pytest.mark.parametrize(
    ("inconsistency", "expected_reason"),
    [("active_cycle", "active_cycle_not_projectable"), ("invalid_finding", "unsupported_visual_state")],
)
def test_preview_packet_command_reports_projection_failure(tmp_path, inconsistency, expected_reason):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_paused_watch(store)

        def corrupt(tx):
            snapshot = tx.get_mission(mission_id)
            if inconsistency == "active_cycle":
                snapshot["cycle_id"] = "cycle-still-active"
            else:
                snapshot["findings"][0]["status"] = "pending"
            return tx.update_mission(
                snapshot,
                expected_revision=snapshot["revision"],
                updated_at_ms=snapshot["updated_at_ms"] + 1,
            )

        changed = store.transact(corrupt).value
        reply = asyncio.run(_runtime(store).handle(
            _preview_command(mission_id, changed["revision"]), "operator-1", {"mission:read"}
        ))
        assert reply["ok"] is False
        assert reply["error"]["code"] == "packet_projection_error"
        assert reply["result"]["projection_error"]["reason"] == expected_reason
        assert reply["result"]["mission_revision"] == changed["revision"]
        assert reply["revision"] == changed["revision"]
    finally:
        store.close()


def test_preview_packet_command_rejects_non_ready_state(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)

        def mark_running(tx):
            snapshot = tx.get_mission(mission_id)
            snapshot["state"] = "running"
            return tx.update_mission(
                snapshot,
                expected_revision=snapshot["revision"],
                updated_at_ms=snapshot["updated_at_ms"] + 1,
            )

        changed = store.transact(mark_running).value
        reply = asyncio.run(_runtime(store).handle(
            _preview_command(mission_id, changed["revision"]), "operator-1", {"mission:read"}
        ))
        assert reply["ok"] is False
        assert reply["error"]["code"] == "packet_not_ready"
        assert reply["result"]["mission_state"] == "running"
        assert reply["revision"] == changed["revision"]
    finally:
        store.close()


def test_preview_packet_command_rejects_blank_mission_id_and_missing_args(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)
        before = _store_state(store, mission_id)
        valid = _preview_command(mission_id, 2)
        invalid_commands = [
            {**valid, "mission_id": " "},
            {key: value for key, value in valid.items() if key != "args"},
        ]
        for command in invalid_commands:
            reply = asyncio.run(_runtime(store).handle(
                command, "operator-1", {"mission:read"}
            ))
            assert reply["ok"] is False
            assert reply["error"]["code"] == "invalid_request"
        assert _store_state(store, mission_id) == before
    finally:
        store.close()


def test_preview_packet_oversize_reply_is_explicit_and_read_only(tmp_path, monkeypatch):
    from dataclasses import replace

    import visionbrain.mission_runtime as runtime_module
    from visionbrain.mission_contracts import MAX_OUTBOUND_MESSAGE_BYTES

    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        mission_id, _primary_id, _missing_id = _seed_mission(store)
        before = _store_state(store, mission_id)
        real_builder = runtime_module.build_packet_preview

        def oversized(rows):
            return replace(real_builder(rows), canonical_json="x" * (MAX_OUTBOUND_MESSAGE_BYTES + 1))

        monkeypatch.setattr(runtime_module, "build_packet_preview", oversized)
        reply = asyncio.run(_runtime(store).handle(
            _preview_command(mission_id, 2), "operator-1", {"mission:read"}
        ))
        assert reply["ok"] is False
        assert reply["error"]["code"] == "result_too_large"
        assert reply["error"]["retryable"] is False
        assert reply["error"]["outcome_unknown"] is False
        assert reply["mission_id"] == mission_id
        assert reply["revision"] == 2
        assert reply["result"] == {
            "recovery": "get",
            "mission_id": mission_id,
            "revision": 2,
        }
        assert _store_state(store, mission_id) == before
    finally:
        store.close()
