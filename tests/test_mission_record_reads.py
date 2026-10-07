"""Coherent, metadata-only read inputs for released mission-record projection."""

import fcntl
import json
import os
import queue
import sqlite3
import threading
from io import BytesIO

import pytest
from PIL import Image

from visionbrain import mission_store
from visionbrain.mission_store import MissionStore, MissionStoreError


def _jpeg(color=(25, 110, 180)):
    output = BytesIO()
    Image.new("RGB", (12, 9), color).save(output, format="JPEG")
    return output.getvalue()


def _mission(mission_id="mission-read-1"):
    return {
        "mission_id": mission_id,
        "revision": 1,
        "state": "completed",
        "updated_at_ms": 1,
        "last_sequence": 0,
        "mode": "inspect",
        "cycle_history": [],
    }


def _seed_read_rows(store):
    def seed(tx):
        tx.insert_mission(_mission())
        tombstone = tx.save_evidence(
            "mission-read-1", _jpeg(), kind="original", created_at_ms=1
        )
        retained = tx.save_evidence(
            "mission-read-1", _jpeg((90, 120, 45)), kind="crop", created_at_ms=2
        )
        tx.add_tool_record(
            "mission-read-1",
            "cycle-current",
            3,
            {"tool_result_id": "tool-current", "status": "ok"},
            current=True,
            created_at_ms=3,
        )
        tx.add_tool_record(
            "mission-read-1",
            "cycle-stale",
            2,
            {"tool_result_id": "tool-stale", "status": "failed"},
            current=False,
            created_at_ms=2,
        )
        tx.retire_evidence(tombstone["evidence_id"])
        return tombstone, retained

    return store.transact(seed).value


def test_record_read_returns_snapshot_all_evidence_and_tool_envelopes(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        tombstone, retained = _seed_read_rows(store)
        retained_path = store.evidence_root / f"{retained['evidence_id']}.jpg"
        retained_path.unlink()

        result = store.read_mission_record_rows("mission-read-1")

        assert result["snapshot"]["mission_id"] == "mission-read-1"
        assert result["snapshot"]["revision"] == 1
        evidence = {row["evidence_id"]: row for row in result["evidence_rows"]}
        assert set(evidence) == {tombstone["evidence_id"], retained["evidence_id"]}
        assert evidence[tombstone["evidence_id"]]["mission_id"] == "mission-read-1"
        assert evidence[tombstone["evidence_id"]]["available"] is False
        assert evidence[tombstone["evidence_id"]]["availability_reason"] == "rolled_off"
        assert evidence[retained["evidence_id"]]["mission_id"] == "mission-read-1"
        assert evidence[retained["evidence_id"]]["available"] is True
        assert not retained_path.exists(), "the read returns persisted metadata without checking media"

        tools = {row["record"]["tool_result_id"]: row for row in result["tool_rows"]}
        assert tools["tool-current"] == {
            "mission_id": "mission-read-1",
            "cycle_id": "cycle-current",
            "execution_generation": 3,
            "is_current": True,
            "record": {"tool_result_id": "tool-current", "status": "ok"},
        }
        assert tools["tool-stale"]["cycle_id"] == "cycle-stale"
        assert tools["tool-stale"]["execution_generation"] == 2
        assert tools["tool-stale"]["is_current"] is False
        assert tools["tool-stale"]["record"]["status"] == "failed"
    finally:
        store.close()


def test_record_read_returns_none_for_absent_mission(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        assert store.read_mission_record_rows("absent") is None
        assert not store._connection.in_transaction
    finally:
        store.close()


def test_record_read_refuses_active_transaction_without_rolling_it_back(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))

        def operation(tx):
            with pytest.raises(MissionStoreError, match="active store transaction"):
                store.read_mission_record_rows("mission-read-1")
            assert store._connection.in_transaction
            tx.append_event("mission-read-1", 1, "read_refused", {}, 2)

        result = store.transact(operation)
        assert result.events[0]["kind"] == "read_refused"
        assert not store._connection.in_transaction
    finally:
        store.close()


def test_record_read_does_not_wait_for_root_file_lock(tmp_path, monkeypatch):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    try:
        _seed_read_rows(store)
        lock_fd = os.open(store._root_quota_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        monkeypatch.setattr(mission_store, "MAX_ROOT_QUOTA_LOCK_WAIT_SECONDS", 0.05)
        try:
            result = store.read_mission_record_rows("mission-read-1")
            assert result is not None
            assert len(result["evidence_rows"]) == 2
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)
    finally:
        store.close()


def test_record_read_uses_one_sqlite_snapshot_across_rows(tmp_path):
    database = tmp_path / "missions.sqlite3"
    store = MissionStore(database, tmp_path / "evidence")
    writer = None
    continue_reader = threading.Event()
    second_select_started = threading.Event()
    result_queue = queue.Queue()
    reader = None
    try:
        _seed_read_rows(store)
        evidence_id = store.evidence_refs("mission-read-1")[0]["evidence_id"]
        writer = sqlite3.connect(database, timeout=2)

        def set_generation(generation):
            snapshot = json.loads(
                writer.execute(
                    "SELECT snapshot_json FROM missions WHERE mission_id = ?",
                    ("mission-read-1",),
                ).fetchone()[0]
            )
            snapshot["read_generation"] = generation
            writer.execute(
                "UPDATE missions SET revision = ?, snapshot_json = ? WHERE mission_id = ?",
                (1 if generation == "old" else 2, json.dumps(snapshot), "mission-read-1"),
            )
            raw_evidence = writer.execute(
                "SELECT metadata_json FROM evidence WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()[0]
            evidence = json.loads(raw_evidence)
            evidence["read_generation"] = generation
            writer.execute(
                "UPDATE evidence SET metadata_json = ? WHERE evidence_id = ?",
                (json.dumps(evidence), evidence_id),
            )
            raw_tool = writer.execute(
                "SELECT record_json FROM tool_records WHERE record_id = ?", ("tool-current",)
            ).fetchone()[0]
            record = json.loads(raw_tool)
            record["read_generation"] = generation
            writer.execute(
                "UPDATE tool_records SET cycle_id = ?, execution_generation = ?, is_current = ?, record_json = ? "
                "WHERE record_id = ?",
                (
                    "cycle-current" if generation == "old" else "cycle-new",
                    3 if generation == "old" else 4,
                    1,
                    json.dumps(record),
                    "tool-current",
                ),
            )

        writer.execute("BEGIN IMMEDIATE")
        set_generation("old")
        writer.commit()

        select_count = 0

        def pause_before_evidence_select(statement):
            nonlocal select_count
            if statement.lstrip().upper().startswith("SELECT"):
                select_count += 1
                if select_count == 2:
                    second_select_started.set()
                    if not continue_reader.wait(timeout=3):
                        raise TimeoutError("reader was not released after concurrent commit")

        store._connection.set_trace_callback(pause_before_evidence_select)

        def read_snapshot():
            try:
                result_queue.put(("ok", store.read_mission_record_rows("mission-read-1")))
            except BaseException as exc:
                result_queue.put(("error", exc))

        reader = threading.Thread(target=read_snapshot, daemon=True)
        reader.start()
        assert second_select_started.wait(timeout=3), "reader did not reach evidence query"
        writer.execute("BEGIN IMMEDIATE")
        set_generation("new")
        writer.commit()
        continue_reader.set()
        reader.join(timeout=3)
        assert not reader.is_alive(), "record read did not finish"
        status, payload = result_queue.get(timeout=1)
        assert status == "ok", repr(payload)
        assert payload["snapshot"]["read_generation"] == "old"
        evidence = next(row for row in payload["evidence_rows"] if row["evidence_id"] == evidence_id)
        assert evidence["read_generation"] == "old"
        tool = next(row for row in payload["tool_rows"] if row["record"]["tool_result_id"] == "tool-current")
        assert tool["cycle_id"] == "cycle-current"
        assert tool["execution_generation"] == 3
        assert tool["record"]["read_generation"] == "old"
    finally:
        continue_reader.set()
        if reader is not None and reader.is_alive():
            reader.join(timeout=3)
        if writer is not None:
            writer.close()
        store._connection.set_trace_callback(None)
        store.close()
