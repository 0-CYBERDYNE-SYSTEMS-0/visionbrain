"""Durability, dedupe, evidence integrity, and event-cursor tests."""

import hashlib
import json
from io import BytesIO

import pytest
from PIL import Image

from visionbrain.mission_store import EvidenceUnavailable, IdempotencyConflict, MissionStore


def _jpeg():
    output = BytesIO()
    Image.new("RGB", (12, 9), (25, 110, 180)).save(output, format="JPEG")
    return output.getvalue()


def _mission(mission_id="mission-1"):
    return {
        "mission_id": mission_id,
        "revision": 1,
        "state": "created",
        "updated_at_ms": 1,
        "last_sequence": 0,
    }


def test_request_dedupe_is_principal_scoped_and_rejects_changed_payload(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    calls = []

    def operation(tx):
        calls.append("run")
        return {"ok": True, "value": len(calls)}

    try:
        first = store.perform_request("installation-a", "r1", {"command": "get"}, operation, now_ms=1)
        replay = store.perform_request("installation-a", "r1", {"command": "get"}, operation, now_ms=2)
        second_principal = store.perform_request("installation-b", "r1", {"command": "get"}, operation, now_ms=3)
        assert first.reply == replay.reply
        assert replay.replayed is True
        assert second_principal.replayed is False
        assert calls == ["run", "run"]
        with pytest.raises(IdempotencyConflict):
            store.perform_request("installation-a", "r1", {"command": "cancel"}, operation, now_ms=4)
    finally:
        store.close()


def test_events_page_returns_last_item_cursor_while_global_sequence_is_ahead(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        def setup(tx):
            tx.insert_mission(_mission())
            for index in range(1, 4):
                tx.append_event("mission-1", 1, "activity", {"index": index}, index)

        store.transact(setup)
        page = store.events_since("mission-1", 0, limit=1)
        assert [event["sequence"] for event in page["events"]] == [1]
        assert page["next_cursor"] == 1
        assert page["last_sequence"] == 3
        assert page["has_more"] is True
    finally:
        store.close()


def test_evidence_chunks_revalidate_the_whole_jpeg_before_serving(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    jpeg = _jpeg()
    try:
        def add(tx):
            tx.insert_mission(_mission())
            return tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)

        evidence = store.transact(add).value
        first = store.get_evidence_chunk(evidence["evidence_id"], offset=0, length=8)
        assert first["jpeg_bytes"] == jpeg[:8]
        assert first["sha256"] == hashlib.sha256(jpeg).hexdigest()
        path = store.evidence_root / f"{evidence['evidence_id']}.jpg"
        # Keep the stored byte length so serving must catch the content hash change.
        path.write_bytes(b"x" * len(jpeg))
        with pytest.raises(EvidenceUnavailable):
            store.get_evidence_chunk(evidence["evidence_id"], offset=0, length=8)
        with pytest.raises(ValueError, match="offset exceeds"):
            # Restore valid media to reach the strict range check.
            path.write_bytes(jpeg)
            store.get_evidence_chunk(evidence["evidence_id"], offset=len(jpeg) + 1, length=1)
    finally:
        store.close()


def test_event_pages_are_byte_bounded_and_advance_only_through_returned_rows(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        def setup(tx):
            tx.insert_mission(_mission())
            for index in range(1, 4):
                tx.append_event("mission-1", index, "activity", {"blob": "x" * 5_000}, index)

        store.transact(setup)
        page = store.events_since("mission-1", 0, limit=3, max_bytes=2_048)
        assert len(json.dumps(page, separators=(",", ":")).encode()) <= 2_048
        assert page["events"][0]["sequence"] == 1
        assert page["events"][0]["data"]["truncated"] is True
        assert page["next_cursor"] == 1
        assert page["last_sequence"] == 3
        assert page["has_more"] is True
        next_page = store.events_since("mission-1", page["next_cursor"], limit=1, max_bytes=2_048)
        assert next_page["events"][0]["sequence"] == 2
        assert next_page["next_cursor"] == 2
    finally:
        store.close()


def test_event_history_retention_requires_resync_for_pruned_cursor(tmp_path):
    from visionbrain.mission_store import MAX_EVENT_HISTORY

    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        def setup(tx):
            tx.insert_mission(_mission())
            for index in range(1, MAX_EVENT_HISTORY + 6):
                tx.append_event("mission-1", 1, "activity", {"index": index}, index)

        store.transact(setup)
        stale = store.events_since("mission-1", 0)
        assert stale["resync_required"] is True
        current = store.events_since("mission-1", MAX_EVENT_HISTORY + 4)
        assert current["events"][0]["sequence"] == MAX_EVENT_HISTORY + 5
        assert current["next_cursor"] == MAX_EVENT_HISTORY + 5
    finally:
        store.close()


def test_active_mission_query_is_not_limited_to_the_recent_mission_page(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        store.transact(lambda tx: [
            tx.insert_mission({**_mission(f"mission-{index}"), "state": "running"})
            for index in range(125)
        ])
        assert len(store.list_missions(limit=100)) == 100
        assert len(store.list_active_missions()) == 125
    finally:
        store.close()


def test_retired_media_unlinks_only_after_successful_request_commit(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    jpeg = _jpeg()
    try:
        def seed(tx):
            tx.insert_mission(_mission())
            evidence = tx.save_evidence(
                "mission-1", jpeg, kind="frame", created_at_ms=1
            )
            tx.add_tool_record(
                "mission-1",
                "cycle-1",
                1,
                {
                    "tool_result_id": "tool-1",
                    "input_evidence_id": evidence["evidence_id"],
                    "evidence_ids": [],
                },
                current=True,
                created_at_ms=1,
            )
            return evidence

        evidence = store.transact(seed).value
        evidence_id = evidence["evidence_id"]
        path = store.evidence_root / f"{evidence_id}.jpg"

        def rollback(tx):
            assert tx.retire_evidence(evidence_id)["available"] is False
            raise RuntimeError("force rollback")

        with pytest.raises(RuntimeError, match="force rollback"):
            store.transact(rollback)
        assert path.exists()
        assert store.get_evidence_chunk(evidence_id, offset=0, length=8)["jpeg_bytes"] == jpeg[:8]
        assert store.evidence_refs("mission-1")[0]["available"] is True

        result = store.perform_request(
            "installation",
            "roll-off",
            {"command": "retire"},
            lambda tx: {
                "evidence": tx.retire_evidence(evidence_id),
                "ok": True,
            },
            now_ms=2,
        )

        assert result.reply["ok"] is True
        assert not path.exists()
        tombstone = store.evidence_refs("mission-1")[0]
        assert tombstone["available"] is False
        assert tombstone["availability_reason"] == "rolled_off"
        record = store.tool_records("mission-1")[0]
        assert record["evidence_availability"][evidence_id] == {
            "available": False,
            "availability_reason": "rolled_off",
        }
        with pytest.raises(EvidenceUnavailable):
            store.get_evidence_chunk(evidence_id, offset=0, length=8)
    finally:
        store.close()
