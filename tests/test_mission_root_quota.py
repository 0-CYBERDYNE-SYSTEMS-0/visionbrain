"""Temporary-root reproductions for aggregate quota and post-commit recovery review."""

from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from visionbrain.mission_store import MissionStore


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


def test_separate_store_quotas_do_not_bound_a_shared_evidence_root(tmp_path):
    """Demonstrate that two DB-local usage sums can overrun a chosen root limit."""
    evidence_root = tmp_path / "shared-evidence"
    first_db = MissionStore(tmp_path / "first.sqlite3", evidence_root, quota_bytes=1_000_000)
    second_db = MissionStore(tmp_path / "second.sqlite3", evidence_root, quota_bytes=1_000_000)
    jpeg = _jpeg()
    root_limit = 2 * len(jpeg) - 1
    first_db.quota_bytes = root_limit
    second_db.quota_bytes = root_limit
    try:
        def add_one(store):
            def operation(tx):
                tx.insert_mission(_mission())
                return tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)

            return store.transact(operation).value

        first = add_one(first_db)
        second = add_one(second_db)
        aggregate_files = sum(path.stat().st_size for path in evidence_root.glob("*.jpg"))
        first_usage = first_db.transact(lambda tx: tx.total_evidence_usage()).value[1]
        second_usage = second_db.transact(lambda tx: tx.total_evidence_usage()).value[1]
        assert first_usage == len(jpeg)
        assert second_usage == len(jpeg)
        assert aggregate_files == len(jpeg) * 2
        assert aggregate_files > root_limit
        assert (evidence_root / f"{first['evidence_id']}.jpg").exists()
        assert (evidence_root / f"{second['evidence_id']}.jpg").exists()
    finally:
        first_db.close()
        second_db.close()


def test_unindexed_orphan_bytes_are_ignored_by_current_quota_check(tmp_path):
    """A crash-left media file is absent from SQLite usage and does not reserve bytes."""
    evidence_root = tmp_path / "evidence"
    evidence_root.mkdir()
    orphan = evidence_root / "interrupted-write.jpg"
    orphan.write_bytes(b"o" * 64)
    jpeg = _jpeg()
    root_limit = len(jpeg) + orphan.stat().st_size - 1
    store = MissionStore(tmp_path / "missions.sqlite3", evidence_root, quota_bytes=root_limit)
    try:
        def add(tx):
            tx.insert_mission(_mission())
            return tx.save_evidence("mission-1", jpeg, kind="imported", created_at_ms=1)

        saved = store.transact(add).value
        actual_root_bytes = sum(path.stat().st_size for path in evidence_root.iterdir() if path.is_file())
        store_usage = store.transact(lambda tx: tx.total_evidence_usage()).value[1]
        assert store_usage == len(jpeg)
        assert actual_root_bytes == len(jpeg) + 64
        assert actual_root_bytes > root_limit
        assert orphan.exists()
        assert (evidence_root / f"{saved['evidence_id']}.jpg").exists()
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
