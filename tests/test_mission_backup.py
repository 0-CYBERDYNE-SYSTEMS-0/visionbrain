"""Offline MissionStore backup and restore integrity tests."""

import json
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from PIL import Image

from visionbrain.mission_backup import (
    BACKUP_VERSION,
    MAX_MANIFEST_BYTES,
    MANIFEST_NAME,
    MissionBackupError,
    MissionBackupDurabilityUnconfirmed,
    restore_mission_store,
)
from visionbrain.mission_store import MissionStore


def _jpeg(color=(25, 110, 180)):
    from io import BytesIO

    output = BytesIO()
    Image.new("RGB", (12, 9), color).save(output, format="JPEG")
    return output.getvalue()


def _mission(mission_id="mission-1"):
    return {
        "mission_id": mission_id,
        "revision": 1,
        "state": "created",
        "updated_at_ms": 1,
        "last_sequence": 0,
        "evidence": [],
    }


def _seed_store(store, *, pin=True, origin="unknown", kind="imported"):
    jpeg = _jpeg()

    def seed(tx):
        tx.insert_mission(_mission())
        evidence = tx.save_evidence(
            "mission-1",
            jpeg,
            kind=kind,
            origin=origin,
            source_id="source-original",
            source_epoch="epoch-original",
            frame_id=42,
            capture_time_ms=123,
            created_at_ms=1,
        )
        tx.append_event("mission-1", 1, "created", {"kept": True}, 2)
        if pin:
            tx.pin_exported_evidence("mission-1", [evidence["evidence_id"]])
        return evidence

    evidence = store.transact(seed).value
    store.perform_request(
        "installation-a",
        "request-1",
        {"command": "get", "mission_id": "mission-1"},
        lambda _tx: {"ok": True, "reply": "memoized"},
        now_ms=3,
    )
    return evidence, jpeg


def _manifest(path):
    return json.loads(path.read_text())


def _write_manifest(path, manifest):
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")


def test_backup_restore_preserves_wal_records_pins_and_independent_media(tmp_path):
    source_root = tmp_path / "source"
    backup_root = tmp_path / "backup"
    restore_root = tmp_path / "restored"
    store = MissionStore(source_root / "missions.sqlite3", source_root / "evidence")
    try:
        evidence, jpeg = _seed_store(store, origin="watch_frame", kind="frame")
        with store._lock:
            store._connection.execute("PRAGMA wal_autocheckpoint = 0")
        store.transact(lambda tx: tx.update_mission_activity(
            {**tx.get_mission("mission-1"), "last_sequence": 7},
            expected_revision=1,
        ))
        wal_path = source_root / "missions.sqlite3-wal"
        assert wal_path.exists() and wal_path.stat().st_size > 0

        assert store.backup_to(backup_root) == backup_root
        manifest = _manifest(backup_root / MANIFEST_NAME)
        assert manifest["version"] == BACKUP_VERSION
        assert manifest["database"]["path"] == "missions.sqlite3"
        assert set(manifest) == {"format", "version", "database", "evidence"}
        assert restore_mission_store(backup_root, restore_root) == restore_root
    finally:
        store.close()

    shutil.rmtree(source_root)
    restored = MissionStore(restore_root / "missions.sqlite3", restore_root / "evidence")
    try:
        snapshot = restored.get_mission("mission-1")
        assert snapshot["revision"] == 1
        assert snapshot["last_sequence"] == 7
        assert restored.events_since("mission-1", 0)["events"][0]["data"] == {"kept": True}
        replay = restored.perform_request(
            "installation-a",
            "request-1",
            {"command": "get", "mission_id": "mission-1"},
            lambda _tx: pytest.fail("restored request memo was not used"),
            now_ms=4,
        )
        assert replay.replayed is True
        assert replay.reply == {"ok": True, "reply": "memoized"}
        assert restored.read_evidence(evidence["evidence_id"])["jpeg_bytes"] == jpeg
        assert restored._connection.execute(
            "SELECT origin FROM evidence WHERE evidence_id = ?", (evidence["evidence_id"],)
        ).fetchone()[0] == "watch_frame"
        restored.check_evidence_file(evidence["evidence_id"])
        evidence_ref = restored.evidence_refs("mission-1")[0]
        assert "origin" not in evidence_ref
        assert (
            evidence_ref["source_id"], evidence_ref["source_epoch"],
            evidence_ref["frame_id"], evidence_ref["capture_time_ms"],
        ) == ("source-original", "epoch-original", 42, 123)
        assert str(restore_root / "evidence" / f"{evidence['evidence_id']}.jpg") == restored._connection.execute(
            "SELECT path FROM evidence WHERE evidence_id = ?", (evidence["evidence_id"],)
        ).fetchone()[0]
        assert restored._connection.execute(
            "SELECT evidence_id FROM exported_evidence WHERE mission_id = 'mission-1'"
        ).fetchone()[0] == evidence["evidence_id"]
    finally:
        restored.close()


def test_backup_inside_store_transaction_fails_fast_and_rolls_back(tmp_path):
    source = tmp_path / "source"
    target = tmp_path / "backup"
    core_src = Path(__file__).resolve().parents[1] / "src"
    script = r'''
import json
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[3])
from visionbrain.mission_backup import MissionBackupError
from visionbrain.mission_store import MissionStore
source, target = Path(sys.argv[1]), Path(sys.argv[2])
store = MissionStore(source / "missions.sqlite3", source / "evidence")
try:
    def operation(tx):
        tx.insert_mission({"mission_id": "rolled-back", "revision": 1,
            "state": "created", "updated_at_ms": 1, "last_sequence": 0,
            "evidence": []})
        store.backup_to(target)
    try:
        store.transact(operation)
    except MissionBackupError as exc:
        result = {"error": str(exc), "mission": store.get_mission("rolled-back"),
            "target_exists": target.exists(),
            "stages": [str(path) for path in target.parent.glob(
                f".{target.name}.backup-*")]}
        print(json.dumps(result))
    else:
        raise SystemExit("backup unexpectedly succeeded inside transaction")
finally:
    store.close()
'''
    # Keep the regression itself bounded: the old SQLite backup call could hang.
    result = subprocess.run(
        [sys.executable, "-c", script, str(source), str(target), str(core_src)],
        capture_output=True,
        text=True,
        timeout=3,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout)
    assert "active store transaction" in outcome["error"]
    assert outcome["mission"] is None
    assert outcome["target_exists"] is False
    assert outcome["stages"] == []


def test_backup_reports_published_path_when_parent_fsync_fails(tmp_path, monkeypatch):
    import visionbrain.mission_backup as backup_module

    source = tmp_path / "source"
    target = tmp_path / "backup"
    restored_root = tmp_path / "verified-restore"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    _seed_store(store, pin=False)
    original_fsync = backup_module._fsync_directory

    def fail_parent_fsync(path):
        if path == tmp_path:
            raise OSError("injected parent fsync failure")
        original_fsync(path)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(backup_module, "_fsync_directory", fail_parent_fsync)
            with pytest.raises(MissionBackupDurabilityUnconfirmed) as caught:
                store.backup_to(target)
        assert caught.value.operation == "backup"
        assert caught.value.published_path == target
        assert "durability is unconfirmed" in str(caught.value)
        assert f"published at {target}" in str(caught.value)
        assert target.is_dir()
        assert not list(tmp_path.glob(".backup.backup-*"))

        with pytest.raises(MissionBackupError, match="already exists"):
            store.backup_to(target)
        restore_mission_store(target, restored_root)
        restored = MissionStore(restored_root / "missions.sqlite3", restored_root / "evidence")
        try:
            assert restored.get_mission("mission-1") is not None
            restored.check_evidence_file(restored.evidence_refs("mission-1")[0]["evidence_id"])
        finally:
            restored.close()
    finally:
        store.close()


def test_restore_reports_published_path_when_parent_fsync_fails(tmp_path, monkeypatch):
    import visionbrain.mission_backup as backup_module

    source = tmp_path / "source"
    backup = tmp_path / "backup"
    target = tmp_path / "restored"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    _seed_store(store, pin=False)
    try:
        store.backup_to(backup)
    finally:
        store.close()

    original_fsync = backup_module._fsync_directory

    def fail_parent_fsync(path):
        if path == tmp_path:
            raise OSError("injected parent fsync failure")
        original_fsync(path)

    with monkeypatch.context() as patch:
        patch.setattr(backup_module, "_fsync_directory", fail_parent_fsync)
        with pytest.raises(MissionBackupDurabilityUnconfirmed) as caught:
            restore_mission_store(backup, target)
    assert caught.value.operation == "restore"
    assert caught.value.published_path == target
    assert "durability is unconfirmed" in str(caught.value)
    assert f"published at {target}" in str(caught.value)
    assert target.is_dir()
    assert not list(tmp_path.glob(".restored.restore-*"))

    restored = MissionStore(target / "missions.sqlite3", target / "evidence")
    try:
        assert restored.get_mission("mission-1") is not None
        restored.check_evidence_file(restored.evidence_refs("mission-1")[0]["evidence_id"])
    finally:
        restored.close()
    with pytest.raises(MissionBackupError, match="already exists"):
        restore_mission_store(backup, target)


def test_unavailable_evidence_tombstone_survives_backup_restore(tmp_path):
    source = tmp_path / "source"
    backup = tmp_path / "backup"
    target = tmp_path / "restored"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    try:
        evidence, _jpeg_bytes = _seed_store(store, pin=False)
        store.transact(lambda tx: tx.retire_evidence(evidence["evidence_id"]))
        assert not (source / "evidence" / f"{evidence['evidence_id']}.jpg").exists()
        store.backup_to(backup)
    finally:
        store.close()

    manifest_record = _manifest(backup / MANIFEST_NAME)["evidence"][0]
    assert manifest_record["available"] is False
    assert manifest_record["path"] is None
    restore_mission_store(backup, target)
    restored = MissionStore(target / "missions.sqlite3", target / "evidence")
    try:
        tombstone = restored.evidence_refs("mission-1")[0]
        assert tombstone["evidence_id"] == evidence["evidence_id"]
        assert tombstone["available"] is False
        assert tombstone["availability_reason"] == "rolled_off"
        assert not (target / "evidence" / f"{evidence['evidence_id']}.jpg").exists()
        assert restored._connection.execute(
            "SELECT path, available FROM evidence WHERE evidence_id = ?",
            (evidence["evidence_id"],),
        ).fetchone()[:] == (str(target / "evidence" / f"{evidence['evidence_id']}.jpg"), 0)
    finally:
        restored.close()


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_backup_rejects_missing_or_corrupt_available_media_without_target(tmp_path, damage):
    source = tmp_path / "source"
    target = tmp_path / "backup"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    try:
        evidence, jpeg = _seed_store(store, pin=False)
        media = source / "evidence" / f"{evidence['evidence_id']}.jpg"
        if damage == "missing":
            media.unlink()
        else:
            media.write_bytes(bytes([jpeg[0] ^ 1]) + jpeg[1:])
        with pytest.raises(MissionBackupError):
            store.backup_to(target)
        assert not target.exists()
        assert not list(tmp_path.glob(".backup.backup-*"))
    finally:
        store.close()


@pytest.mark.parametrize(
    "attack", ["traversal", "unsupported_version", "database_digest", "symlink", "duplicate_key"]
)
def test_restore_rejects_adversarial_or_stale_bundle_without_target(tmp_path, attack):
    source = tmp_path / "source"
    backup = tmp_path / "backup"
    target = tmp_path / "restored"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    try:
        evidence, _jpeg_bytes = _seed_store(store, pin=False)
        store.backup_to(backup)
    finally:
        store.close()

    manifest_path = backup / MANIFEST_NAME
    manifest = _manifest(manifest_path)
    if attack == "traversal":
        manifest["evidence"][0]["path"] = "../../outside.jpg"
        _write_manifest(manifest_path, manifest)
    elif attack == "unsupported_version":
        manifest["version"] += 1
        _write_manifest(manifest_path, manifest)
    elif attack == "database_digest":
        database = backup / manifest["database"]["path"]
        encoded = bytearray(database.read_bytes())
        encoded[-1] ^= 1
        database.write_bytes(encoded)
    elif attack == "duplicate_key":
        manifest_path.write_text(
            manifest_path.read_text().replace('"version":1', '"version":1,"version":1')
        )
    else:
        media = backup / "evidence" / f"{evidence['evidence_id']}.jpg"
        media.unlink()
        media.symlink_to(tmp_path / "outside.jpg")
        (tmp_path / "outside.jpg").write_bytes(_jpeg())

    with pytest.raises(MissionBackupError):
        restore_mission_store(backup, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_restore_rejects_oversized_manifest_without_target(tmp_path):
    source = tmp_path / "source"
    backup = tmp_path / "backup"
    target = tmp_path / "restored"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    try:
        _seed_store(store, pin=False)
        store.backup_to(backup)
    finally:
        store.close()
    (backup / MANIFEST_NAME).write_bytes(b" " * (MAX_MANIFEST_BYTES + 1))
    with pytest.raises(MissionBackupError, match="size limit"):
        restore_mission_store(backup, target)
    assert not target.exists()
    assert not list(tmp_path.glob(".restored.restore-*"))


def test_restore_failure_after_staging_leaves_no_target_and_existing_target_is_never_replaced(tmp_path):
    source = tmp_path / "source"
    backup = tmp_path / "backup"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    try:
        evidence, _jpeg_bytes = _seed_store(store, pin=False)
        store.backup_to(backup)
    finally:
        store.close()

    media = backup / "evidence" / f"{evidence['evidence_id']}.jpg"
    media.write_bytes(b"corrupt")
    failed_target = tmp_path / "failed-restore"
    with pytest.raises(MissionBackupError):
        restore_mission_store(backup, failed_target)
    assert not failed_target.exists()
    assert not list(tmp_path.glob(".failed-restore.restore-*"))

    existing_target = tmp_path / "existing"
    existing_target.mkdir()
    marker = existing_target / "keep.txt"
    marker.write_text("do not replace")
    with pytest.raises(MissionBackupError, match="already exists"):
        restore_mission_store(backup, existing_target)
    assert marker.read_text() == "do not replace"


def test_backup_holds_store_lock_until_evidence_copy_finishes(tmp_path, monkeypatch):
    import visionbrain.mission_backup as backup_module

    source = tmp_path / "source"
    target = tmp_path / "backup"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    _seed_store(store, pin=False)
    copying = threading.Event()
    allow_copy = threading.Event()
    writer_started = threading.Event()
    writer_finished = threading.Event()
    errors = []
    original_copy = backup_module._copy_bounded_file

    def blocked_copy(*args, **kwargs):
        copying.set()
        if not allow_copy.wait(3):
            raise TimeoutError("test copy barrier timed out")
        return original_copy(*args, **kwargs)

    monkeypatch.setattr(backup_module, "_copy_bounded_file", blocked_copy)

    def run_backup():
        try:
            store.backup_to(target)
        except BaseException as exc:
            errors.append(exc)

    def run_writer():
        writer_started.set()
        try:
            store.transact(lambda tx: tx.insert_mission(_mission("concurrent")))
        except BaseException as exc:
            errors.append(exc)
        finally:
            writer_finished.set()

    backup_thread = threading.Thread(target=run_backup)
    writer_thread = threading.Thread(target=run_writer)
    try:
        backup_thread.start()
        assert copying.wait(2)
        writer_thread.start()
        assert writer_started.wait(2)
        assert not writer_finished.wait(0.1)
        allow_copy.set()
        backup_thread.join(3)
        writer_thread.join(3)
        assert not backup_thread.is_alive()
        assert not writer_thread.is_alive()
        assert errors == []
        assert store.get_mission("concurrent") is not None
        restored_root = tmp_path / "restored"
        restore_mission_store(target, restored_root)
        restored = MissionStore(restored_root / "missions.sqlite3", restored_root / "evidence")
        try:
            assert restored.get_mission("concurrent") is None
        finally:
            restored.close()
    finally:
        allow_copy.set()
        backup_thread.join(3)
        if writer_thread.ident is not None:
            writer_thread.join(3)
        store.close()


def test_backup_refuses_existing_or_overlapping_destination(tmp_path):
    source = tmp_path / "source"
    existing = tmp_path / "existing-backup"
    store = MissionStore(source / "missions.sqlite3", source / "evidence")
    _seed_store(store, pin=False)
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("do not replace")
    try:
        with pytest.raises(MissionBackupError, match="already exists"):
            store.backup_to(existing)
        with pytest.raises(MissionBackupError, match="separate"):
            store.backup_to(source / "nested-backup")
        assert marker.read_text() == "do not replace"
        assert not (source / "nested-backup").exists()
    finally:
        store.close()
