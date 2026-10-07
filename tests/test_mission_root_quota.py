"""Temporary-root reproductions for aggregate quota and post-commit recovery review."""

from io import BytesIO
import fcntl
import multiprocessing
import os
import sqlite3
from pathlib import Path

import pytest
from PIL import Image

from visionbrain import mission_store
from visionbrain.mission_store import (
    DEFAULT_ROOT_QUOTA_BYTES,
    MissionStore,
    QuotaAccountingIncomplete,
    QuotaExceeded,
    RootQuotaExceeded,
)


def _jpeg(color=(25, 110, 180)):
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
    }


def _stored_origin(store, evidence_id):
    with store._lock:
        row = store._connection.execute(
            "SELECT origin FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
    return row[0] if row else None


def _concurrent_root_save(database_path, evidence_root, jpeg, barrier, results):
    store = MissionStore(
        database_path,
        evidence_root,
        quota_bytes=1_000_000,
        root_quota_bytes=2 * len(jpeg) - 1,
    )
    try:
        barrier.wait(timeout=5)

        def add(tx):
            return tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)

        try:
            store.transact(add)
            results.put("saved")
        except QuotaExceeded:
            results.put("quota")
    finally:
        store.close()


def test_separate_stores_share_injected_root_quota(tmp_path):
    """A second database cannot spend bytes already present in a shared root."""
    evidence_root = tmp_path / "shared-evidence"
    jpeg = _jpeg()
    root_limit = 2 * len(jpeg) - 1
    first_db = MissionStore(tmp_path / "first.sqlite3", evidence_root, quota_bytes=1_000_000, root_quota_bytes=root_limit)
    second_db = MissionStore(tmp_path / "second.sqlite3", evidence_root, quota_bytes=1_000_000, root_quota_bytes=root_limit)
    try:
        def add_one(store):
            def operation(tx):
                tx.insert_mission(_mission())
                return tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)

            return store.transact(operation).value

        first = add_one(first_db)
        with pytest.raises(QuotaExceeded, match="evidence root quota exceeded"):
            add_one(second_db)
        aggregate_files = sum(path.stat().st_size for path in evidence_root.glob("*.jpg"))
        first_usage = first_db.transact(lambda tx: tx.total_evidence_usage()).value[1]
        second_usage = second_db.transact(lambda tx: tx.total_evidence_usage()).value[1]
        assert first_usage == len(jpeg)
        assert second_usage == 0
        assert aggregate_files == len(jpeg)
        assert (evidence_root / f"{first['evidence_id']}.jpg").exists()
    finally:
        first_db.close()
        second_db.close()


def test_unindexed_orphan_bytes_remain_charged_after_store_restart(tmp_path):
    """A crash-left file counts even when no database row refers to it."""
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    orphan = evidence_root / "interrupted-write.jpg"
    jpeg = _jpeg()
    root_limit = len(jpeg) + 64 - 1
    database = tmp_path / "missions.sqlite3"
    store = MissionStore(database, evidence_root, quota_bytes=1_000_000, root_quota_bytes=root_limit)
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
    finally:
        store.close()

    orphan.write_bytes(b"o" * 64)
    store = MissionStore(database, evidence_root, quota_bytes=1_000_000, root_quota_bytes=root_limit)
    try:
        with pytest.raises(QuotaExceeded, match="evidence root quota exceeded"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
            )
        actual_root_bytes = sum(path.stat().st_size for path in evidence_root.iterdir() if path.is_file())
        assert actual_root_bytes == 64
        assert orphan.exists()
        assert len(store.evidence_refs("mission-1")) == 0
    finally:
        store.close()


def test_root_quota_refusal_carries_measured_usage_and_incoming_bytes(tmp_path):
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    orphan = evidence_root / "unindexed-orphan"
    orphan.write_bytes(b"orphan-bytes")
    jpeg = _jpeg()
    store = MissionStore(
        tmp_path / "missions.sqlite3",
        evidence_root,
        quota_bytes=1_000_000,
        root_quota_bytes=len(jpeg) + orphan.stat().st_size - 1,
    )
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        with pytest.raises(RootQuotaExceeded) as raised:
            store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
            )
        assert raised.value.root_usage_bytes == orphan.stat().st_size
        assert raised.value.incoming_bytes == len(jpeg)
        assert not list(evidence_root.glob("*.jpg"))
    finally:
        store.close()


def test_evidence_origins_persist_across_reopen_without_public_metadata(tmp_path):
    database = tmp_path / "missions.sqlite3"
    root = tmp_path / "evidence"
    jpeg = _jpeg()
    store = MissionStore(database, root, quota_bytes=1_000_000, root_quota_bytes=1_000_000)
    try:
        def seed(tx):
            tx.insert_mission(_mission())
            return {
                "watch": tx.save_evidence(
                    "mission-1", jpeg, kind="frame", origin="watch_frame", created_at_ms=1
                ),
                "explicit": tx.save_evidence(
                    "mission-1", jpeg, kind="frame", origin="explicit_attachment", created_at_ms=2
                ),
                "crop": tx.save_evidence(
                    "mission-1", jpeg, kind="crop", origin="generated_crop", created_at_ms=3
                ),
                "unknown": tx.save_evidence("mission-1", jpeg, kind="frame", created_at_ms=4),
            }

        records = store.transact(seed).value
        assert all("origin" not in record for record in records.values())
    finally:
        store.close()

    reopened = MissionStore(database, root, quota_bytes=1_000_000, root_quota_bytes=1_000_000)
    try:
        origins = {
            name: _stored_origin(reopened, record["evidence_id"])
            for name, record in records.items()
        }
        assert origins == {
            "watch": "watch_frame",
            "explicit": "explicit_attachment",
            "crop": "generated_crop",
            "unknown": "unknown",
        }
        assert reopened.transact(lambda tx: (
            tx._is_evidence_rolloff_eligible(records["watch"]["evidence_id"]),
            tx._is_evidence_rolloff_eligible(records["explicit"]["evidence_id"]),
            tx._is_evidence_rolloff_eligible(records["crop"]["evidence_id"]),
            tx._is_evidence_rolloff_eligible(records["unknown"]["evidence_id"]),
        )).value == (True, False, True, False)
        assert all("origin" not in ref for ref in reopened.evidence_refs("mission-1"))
    finally:
        reopened.close()


def test_legacy_evidence_origin_migrates_to_protected_unknown(tmp_path):
    database = tmp_path / "missions.sqlite3"
    root = tmp_path / "evidence"
    store = MissionStore(database, root, quota_bytes=1_000_000, root_quota_bytes=1_000_000)
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        evidence = store.transact(lambda tx: tx.save_evidence(
            "mission-1", _jpeg(), kind="frame", origin="watch_frame", created_at_ms=1
        )).value
    finally:
        store.close()

    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE evidence DROP COLUMN origin")

    migrated = MissionStore(database, root, quota_bytes=1_000_000, root_quota_bytes=1_000_000)
    try:
        assert _stored_origin(migrated, evidence["evidence_id"]) == "unknown"
        assert not migrated.transact(
            lambda tx: tx._is_evidence_rolloff_eligible(evidence["evidence_id"])
        ).value
    finally:
        migrated.close()


def test_default_root_quota_is_decimal_100_gb():
    assert DEFAULT_ROOT_QUOTA_BYTES == 100_000_000_000


def test_held_root_lock_fails_with_bounded_accounting_error(tmp_path, monkeypatch):
    jpeg = _jpeg()
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence", root_quota_bytes=100_000)
    store.transact(lambda tx: tx.insert_mission(_mission()))
    fd = os.open(store._root_quota_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setattr(mission_store, "MAX_ROOT_QUOTA_LOCK_WAIT_SECONDS", 0.05)
    try:
        with pytest.raises(QuotaAccountingIncomplete, match="root lock busy.*wait limit"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
            )
        assert not list(store.evidence_root.glob("*.jpg"))
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        store.close()


def test_metadata_transactions_do_not_wait_for_root_lock(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", root)
    fd = os.open(store._root_quota_lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setattr(mission_store, "MAX_ROOT_QUOTA_LOCK_WAIT_SECONDS", 0.05)
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        result = store.perform_read(lambda tx: {"mission": tx.get_mission("mission-1")})
        assert result.reply["mission"]["mission_id"] == "mission-1"
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        store.close()


def test_interrupted_root_lock_acquisition_closes_lock_fd(tmp_path, monkeypatch):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    store.transact(lambda tx: tx.insert_mission(_mission()))
    real_open = os.open
    real_close = os.close
    lock_fds = set()
    closed_lock_fds = set()

    def track_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if os.fspath(path) == os.fspath(store._root_quota_lock_path):
            lock_fds.add(fd)
        return fd

    def track_close(fd):
        if fd in lock_fds:
            closed_lock_fds.add(fd)
        return real_close(fd)

    def interrupt_flock(_fd, _operation):
        raise KeyboardInterrupt("injected lock acquisition interruption")

    jpeg = _jpeg()
    with monkeypatch.context() as patch:
        patch.setattr(mission_store.os, "open", track_open)
        patch.setattr(mission_store.os, "close", track_close)
        patch.setattr(mission_store.fcntl, "flock", interrupt_flock)
        with pytest.raises(KeyboardInterrupt, match="lock acquisition interruption"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
            )
    try:
        assert lock_fds and closed_lock_fds == lock_fds
        assert not list(store.evidence_root.glob("*.jpg"))
    finally:
        store.close()


def test_concurrent_processes_cannot_overadmit_shared_root(tmp_path):
    jpeg = _jpeg()
    root = tmp_path / "evidence"
    databases = [tmp_path / "one.sqlite3", tmp_path / "two.sqlite3"]
    for database in databases:
        store = MissionStore(database, root, quota_bytes=1_000_000, root_quota_bytes=2 * len(jpeg) - 1)
        store.transact(lambda tx: tx.insert_mission(_mission()))
        store.close()

    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(target=_concurrent_root_save, args=(db, root, jpeg, barrier, results))
        for db in databases
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=12)
        assert all(not process.is_alive() for process in processes)
        assert [process.exitcode for process in processes] == [0, 0]
        assert sorted([results.get(timeout=2), results.get(timeout=2)]) == ["quota", "saved"]
        assert sum(path.stat().st_size for path in root.glob("*.jpg")) == len(jpeg)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        results.close()


def test_lexical_root_aliases_share_one_quota_lock(tmp_path):
    root = tmp_path / "evidence"
    (tmp_path / "alias-parent").mkdir()
    alias = tmp_path / "alias-parent" / ".." / "evidence"
    jpeg = _jpeg()
    limit = 2 * len(jpeg) - 1
    first = MissionStore(tmp_path / "one.sqlite3", root, root_quota_bytes=limit)
    second = MissionStore(tmp_path / "two.sqlite3", alias, root_quota_bytes=limit)
    try:
        first.transact(lambda tx: tx.insert_mission(_mission()))
        second.transact(lambda tx: tx.insert_mission(_mission()))
        def add(store):
            return store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
            ).value

        add(first)
        with pytest.raises(QuotaExceeded, match="evidence root quota exceeded"):
            add(second)
    finally:
        first.close()
        second.close()


def test_scan_ignores_hostile_symlinks_without_following_targets(tmp_path):
    root = tmp_path / "evidence"
    root.mkdir()
    external = tmp_path / "external.bin"
    external.write_bytes(b"x" * 4096)
    (root / "external-link").symlink_to(external)
    (root / "external-dir").symlink_to(tmp_path, target_is_directory=True)
    jpeg = _jpeg()
    store = MissionStore(tmp_path / "missions.sqlite3", root, root_quota_bytes=len(jpeg))
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        result = store.transact(
            lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)
        )
        assert (root / f"{result.value['evidence_id']}.jpg").is_file()
        assert external.stat().st_size == 4096
    finally:
        store.close()


def test_incomplete_bounded_scan_fails_closed_before_creating_media(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "orphan-a").write_bytes(b"a")
    (root / "orphan-b").write_bytes(b"b")
    store = MissionStore(tmp_path / "missions.sqlite3", root, root_quota_bytes=100_000)
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        monkeypatch.setattr(mission_store, "MAX_ROOT_QUOTA_SCAN_ENTRIES", 1)
        with pytest.raises(QuotaExceeded, match="accounting incomplete.*entry limit"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", _jpeg(), kind="imported", created_at_ms=1)
            )
        assert not list(root.glob("*.jpg"))
        assert (root / "orphan-a").exists() and (root / "orphan-b").exists()
    finally:
        store.close()


def test_scan_time_limit_fails_closed(tmp_path, monkeypatch):
    root = tmp_path / "evidence"
    root.mkdir()
    (root / "orphan").write_bytes(b"x")
    store = MissionStore(tmp_path / "missions.sqlite3", root, root_quota_bytes=100_000)
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        monkeypatch.setattr(mission_store, "MAX_ROOT_QUOTA_SCAN_SECONDS", 0)
        with pytest.raises(QuotaExceeded, match="accounting incomplete.*time limit"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", _jpeg(), kind="imported", created_at_ms=1)
            )
        assert not list(root.glob("*.jpg"))
    finally:
        store.close()


def test_rollback_removes_new_media_before_releasing_root_quota(tmp_path):
    jpeg = _jpeg()
    root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", root, root_quota_bytes=len(jpeg))
    created = {}
    try:
        def fail(tx):
            tx.insert_mission(_mission())
            created.update(tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1))
            raise RuntimeError("rollback after media write")

        with pytest.raises(RuntimeError, match="rollback after media write"):
            store.transact(fail)
        assert not (root / f"{created['evidence_id']}.jpg").exists()
        store.transact(lambda tx: tx.insert_mission(_mission()))
        saved = store.transact(
            lambda tx: tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=2)
        )
        assert (root / f"{saved.value['evidence_id']}.jpg").exists()
    finally:
        store.close()


def test_failed_retired_unlink_remains_charged_until_removed(tmp_path, monkeypatch):
    jpeg = _jpeg()
    root = tmp_path / "evidence"
    store = MissionStore(tmp_path / "missions.sqlite3", root, root_quota_bytes=len(jpeg))
    try:
        store.transact(lambda tx: tx.insert_mission(_mission()))
        original = store.transact(
            lambda tx: tx.save_evidence("mission-1", jpeg, kind="original", created_at_ms=1)
        ).value
        retired_path = root / f"{original['evidence_id']}.jpg"
        unlink = Path.unlink

        def fail_retirement(path, *args, **kwargs):
            if path == retired_path:
                raise OSError("cleanup deferred")
            return unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_retirement)
            store.transact(lambda tx: tx.retire_evidence(original["evidence_id"]))

        with pytest.raises(QuotaExceeded, match="evidence root quota exceeded"):
            store.transact(
                lambda tx: tx.save_evidence("mission-1", jpeg, kind="replacement", created_at_ms=2)
            )
        assert retired_path.exists()
        retired_path.unlink()
        saved = store.transact(
            lambda tx: tx.save_evidence("mission-1", jpeg, kind="replacement", created_at_ms=3)
        )
        assert (root / f"{saved.value['evidence_id']}.jpg").exists()
    finally:
        store.close()


@pytest.mark.parametrize("entrypoint", ["transact", "perform_request"])
def test_post_commit_retired_unlink_error_does_not_delete_new_committed_media(
    tmp_path, monkeypatch, caplog, entrypoint
):
    """A failed post-commit retirement unlink is deferred without harming new media."""
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    jpeg = _jpeg()
    fresh_jpeg = _jpeg((180, 60, 25))
    try:
        def seed(tx):
            tx.insert_mission(_mission())
            return tx.save_evidence("mission-1", jpeg, kind="original", created_at_ms=1)

        retired = store.transact(seed).value
        retired_path = store.evidence_root / f"{retired['evidence_id']}.jpg"
        created = {}

        def mutate(tx):
            created.update(tx.save_evidence("mission-1", fresh_jpeg, kind="new", created_at_ms=2))
            tx.retire_evidence(retired["evidence_id"])
            return created

        original_unlink = Path.unlink
        injected = []

        def fail_retired_unlink_once(path, *args, **kwargs):
            if path == retired_path and not injected:
                injected.append(path)
                raise OSError("injected retired-media unlink failure")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", fail_retired_unlink_once)
            if entrypoint == "transact":
                result = store.transact(mutate)
                assert result.value["evidence_id"] == created["evidence_id"]
            else:
                first_request = store.perform_request(
                    "installation-a",
                    "retire-and-add",
                    {"command": "replace_evidence"},
                    lambda tx: {"new": mutate(tx)},
                    now_ms=2,
                )
                replay = store.perform_request(
                    "installation-a",
                    "retire-and-add",
                    {"command": "replace_evidence"},
                    lambda _tx: pytest.fail("committed request reply was not replayed"),
                    now_ms=3,
                )
                assert replay.replayed is True
                assert replay.reply == first_request.reply

        assert injected == [retired_path]
        assert any("retired evidence cleanup deferred" in record.message for record in caplog.records)
        rows = store.evidence_refs("mission-1")
        fresh_row = next(row for row in rows if row["evidence_id"] == created["evidence_id"])
        assert fresh_row["available"] is True
        fresh_path = store.evidence_root / f"{created['evidence_id']}.jpg"
        assert fresh_path.exists(), "committed evidence metadata must not point to deleted media"
        assert retired_path.exists(), "the injected failed unlink leaves old media for later recovery"
    finally:
        store.close()


@pytest.mark.parametrize("entrypoint", ["transact", "perform_request"])
def test_post_commit_keyboard_interrupt_does_not_delete_new_committed_media(
    tmp_path, monkeypatch, entrypoint
):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    retired_jpeg = _jpeg()
    fresh_jpeg = _jpeg((50, 170, 70))
    try:
        def seed(tx):
            tx.insert_mission(_mission())
            return tx.save_evidence("mission-1", retired_jpeg, kind="original", created_at_ms=1)

        retired = store.transact(seed).value
        retired_path = store.evidence_root / f"{retired['evidence_id']}.jpg"
        created = {}

        def mutate(tx):
            created.update(tx.save_evidence("mission-1", fresh_jpeg, kind="new", created_at_ms=2))
            tx.retire_evidence(retired["evidence_id"])
            return created

        original_unlink = Path.unlink
        injected = []

        def interrupt_retired_unlink(path, *args, **kwargs):
            if path == retired_path and not injected:
                injected.append(path)
                raise KeyboardInterrupt("injected post-commit interruption")
            return original_unlink(path, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(Path, "unlink", interrupt_retired_unlink)
            with pytest.raises(KeyboardInterrupt, match="post-commit interruption"):
                if entrypoint == "transact":
                    store.transact(mutate)
                else:
                    store.perform_request(
                        "installation-a",
                        "interrupt-after-commit",
                        {"command": "replace_evidence"},
                        lambda tx: {"new": mutate(tx)},
                        now_ms=2,
                    )

        assert injected == [retired_path]
        fresh_path = store.evidence_root / f"{created['evidence_id']}.jpg"
        assert fresh_path.exists()
        fresh_row = next(row for row in store.evidence_refs("mission-1") if row["evidence_id"] == created["evidence_id"])
        assert fresh_row["available"] is True
        assert retired_path.exists()
        if entrypoint == "perform_request":
            replay = store.perform_request(
                "installation-a",
                "interrupt-after-commit",
                {"command": "replace_evidence"},
                lambda _tx: pytest.fail("committed request reply was not replayed"),
                now_ms=3,
            )
            assert replay.replayed is True
            assert replay.reply["new"]["evidence_id"] == created["evidence_id"]
    finally:
        store.close()


def test_precommit_failure_still_rolls_back_and_removes_new_media(tmp_path):
    store = MissionStore(tmp_path / "missions.sqlite3", tmp_path / "evidence")
    jpeg = _jpeg()
    created = {}
    try:
        def fail_after_save(tx):
            tx.insert_mission(_mission())
            created.update(tx.save_evidence("mission-1", jpeg, kind="new", created_at_ms=1))
            raise RuntimeError("injected pre-commit failure")

        with pytest.raises(RuntimeError, match="pre-commit failure"):
            store.transact(fail_after_save)
        assert store.get_mission("mission-1") is None
        assert store.evidence_refs("mission-1") == []
        assert not (store.evidence_root / f"{created['evidence_id']}.jpg").exists()
    finally:
        store.close()
