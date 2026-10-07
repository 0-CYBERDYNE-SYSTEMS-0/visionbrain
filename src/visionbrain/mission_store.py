"""Durable SQLite metadata and immutable media store for missions.

The module uses only the Python standard library at import time. JPEG parsing
uses Pillow lazily, and model/runtime dependencies are deliberately absent.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import sqlite3
import stat
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Mapping

from .mission_contracts import (
    MAX_DECODED_JPEG_BYTES,
    MAX_MISSION_EVIDENCE,
    MAX_OUTBOUND_MESSAGE_BYTES,
)

DEFAULT_QUOTA_BYTES = 2 * 1024 * 1024 * 1024
DEFAULT_ROOT_QUOTA_BYTES = 100_000_000_000
MAX_ROOT_QUOTA_SCAN_ENTRIES = 100_000
MAX_ROOT_QUOTA_SCAN_SECONDS = 2.0
MAX_ROOT_QUOTA_SCAN_DEPTH = 64
MAX_ROOT_QUOTA_LOCK_WAIT_SECONDS = 1.0
_LOGGER = logging.getLogger(__name__)
MAX_PLANNER_DIAGNOSTIC_BYTES = 32 * 1024
MAX_PLANNER_DIAGNOSTICS_PER_MISSION = 64
MAX_PLANNER_DIAGNOSTICS_TOTAL = 1_024
MAX_EVIDENCE_CHUNK_BYTES = 49_152
MAX_EVENTS_PAGE = 200
MAX_MISSIONS_PAGE = 100
MAX_EVENT_HISTORY = 200
MAX_EVENT_PAGE_BYTES = MAX_OUTBOUND_MESSAGE_BYTES - 2_048
_EVIDENCE_ORIGINS = frozenset({"unknown", "watch_frame", "explicit_attachment", "generated_crop"})


class MissionStoreError(Exception):
    """Base exception for durable mission-store failures."""


class IdempotencyConflict(MissionStoreError):
    """A principal reused a request ID with a different payload."""


class RevisionConflict(MissionStoreError):
    """A mutation used a stale mission revision."""

    def __init__(self, snapshot: Mapping[str, Any]) -> None:
        super().__init__("mission revision changed")
        self.snapshot = dict(snapshot)


class EvidenceUnavailable(MissionStoreError):
    """Evidence metadata exists but the immutable file cannot be read."""

    def __init__(self, message: str, *, availability_reason: str = "missing") -> None:
        super().__init__(message)
        if availability_reason not in {"missing", "corrupt", "rolled_off"}:
            raise ValueError("evidence availability reason is invalid")
        self.availability_reason = availability_reason


class InvalidEvidence(MissionStoreError):
    """The supplied bytes are not a valid bounded JPEG with the given hash."""


class QuotaExceeded(MissionStoreError):
    """Persisting evidence would exceed the configured durable media quota."""


class RootQuotaExceeded(QuotaExceeded):
    """The measured evidence root usage plus an incoming item exceeds its quota."""

    def __init__(self, root_usage_bytes: int, incoming_bytes: int) -> None:
        super().__init__("mission evidence root quota exceeded")
        self.root_usage_bytes = int(root_usage_bytes)
        self.incoming_bytes = int(incoming_bytes)


class QuotaAccountingIncomplete(QuotaExceeded):
    """Evidence-root usage could not be measured safely within configured bounds."""


@dataclass(frozen=True)
class RequestResult:
    reply: dict[str, Any]
    replayed: bool
    events: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class TransactionResult:
    value: Any
    events: tuple[dict[str, Any], ...] = ()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _payload_hash(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()


def _file_fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _cleanup_retired_paths(paths: list[Path]) -> None:
    """Delete retired media after commit; report failures without undoing committed state."""
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            _LOGGER.warning("retired evidence cleanup deferred for %s: %s", path.name, exc)


def _jpeg_dimensions(data: bytes) -> tuple[int, int]:
    if not data or len(data) > MAX_DECODED_JPEG_BYTES:
        raise InvalidEvidence("JPEG must be non-empty and at most 2 MiB")
    try:
        from PIL import Image

        with Image.open(BytesIO(data)) as image:
            if image.format != "JPEG":
                raise InvalidEvidence("evidence must be a normalized JPEG")
            width, height = image.size
            image.verify()
    except InvalidEvidence:
        raise
    except Exception as exc:
        raise InvalidEvidence("invalid JPEG image") from exc
    if width <= 0 or height <= 0 or width * height > 80_000_000:
        raise InvalidEvidence("JPEG dimensions are invalid or exceed the pixel limit")
    return int(width), int(height)


class StoreTransaction:
    """Short-lived operations over one ``BEGIN IMMEDIATE`` transaction."""

    def __init__(self, store: "MissionStore", connection: sqlite3.Connection) -> None:
        self._store = store
        self._connection = connection
        self.events: list[dict[str, Any]] = []
        self._created_paths: list[Path] = []
        self._retired_paths: list[Path] = []
        self._root_quota_guard = None

    def _ensure_root_quota_guard(self) -> None:
        if self._root_quota_guard is None:
            guard = self._store._root_quota_lock()
            guard.__enter__()
            self._root_quota_guard = guard

    def _release_root_quota_guard(self) -> None:
        guard, self._root_quota_guard = self._root_quota_guard, None
        if guard is not None:
            guard.__exit__(None, None, None)

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT snapshot_json FROM missions WHERE mission_id = ?", (mission_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def read_mission_record_rows(self, mission_id: str) -> dict[str, Any] | None:
        """Read all typed-record metadata from this transaction's SQLite view."""
        mission = self._connection.execute(
            "SELECT snapshot_json FROM missions WHERE mission_id = ?", (mission_id,)
        ).fetchone()
        if mission is None:
            return None
        evidence_rows = self._connection.execute(
            "SELECT mission_id, metadata_json, available FROM evidence "
            "WHERE mission_id = ? ORDER BY created_at_ms, evidence_id",
            (mission_id,),
        ).fetchall()
        tool_rows = self._connection.execute(
            "SELECT mission_id, cycle_id, execution_generation, is_current, record_json "
            "FROM tool_records WHERE mission_id = ? ORDER BY created_at_ms, record_id",
            (mission_id,),
        ).fetchall()
        result = {"snapshot": json.loads(mission[0]), "evidence_rows": [], "tool_rows": []}
        for row in evidence_rows:
            evidence = dict(json.loads(row[1]))
            evidence["mission_id"] = str(row[0])
            evidence["available"] = bool(row[2])
            if not evidence["available"]:
                evidence.setdefault("availability_reason", "rolled_off")
            result["evidence_rows"].append(evidence)
        for row in tool_rows:
            result["tool_rows"].append(
                {
                    "mission_id": str(row[0]),
                    "cycle_id": str(row[1]),
                    "execution_generation": int(row[2]),
                    "is_current": bool(row[3]),
                    "record": json.loads(row[4]),
                }
            )
        return result

    def list_active_missions(self) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT snapshot_json FROM missions "
            "WHERE state IN ('running', 'waiting_evidence', 'waiting_approval') "
            "ORDER BY updated_at_ms DESC, mission_id"
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_running_missions(self, *, excluding: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT snapshot_json FROM missions WHERE state = 'running'"
        params: tuple[Any, ...] = ()
        if excluding is not None:
            sql += " AND mission_id != ?"
            params = (excluding,)
        rows = self._connection.execute(sql + " ORDER BY updated_at_ms DESC, mission_id", params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def insert_mission(self, snapshot: Mapping[str, Any]) -> None:
        mission_id = str(snapshot["mission_id"])
        if self._connection.execute(
            "SELECT 1 FROM missions WHERE mission_id = ?", (mission_id,)
        ).fetchone():
            raise MissionStoreError("mission ID already exists")
        self._connection.execute(
            "INSERT INTO missions(mission_id, revision, state, snapshot_json, updated_at_ms) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                mission_id,
                int(snapshot["revision"]),
                str(snapshot["state"]),
                _json(snapshot),
                int(snapshot.get("updated_at_ms", 0)),
            ),
        )

    def update_mission(
        self,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int,
        updated_at_ms: int,
    ) -> dict[str, Any]:
        mission_id = str(snapshot["mission_id"])
        current = self.get_mission(mission_id)
        if current is None:
            raise KeyError(mission_id)
        if int(current["revision"]) != expected_revision:
            raise RevisionConflict(current)
        updated = dict(snapshot)
        updated["revision"] = expected_revision + 1
        updated["updated_at_ms"] = int(updated_at_ms)
        self._connection.execute(
            "UPDATE missions SET revision = ?, state = ?, snapshot_json = ?, updated_at_ms = ? "
            "WHERE mission_id = ? AND revision = ?",
            (
                updated["revision"],
                updated["state"],
                _json(updated),
                updated["updated_at_ms"],
                mission_id,
                expected_revision,
            ),
        )
        return updated

    def update_mission_activity(
        self,
        snapshot: Mapping[str, Any],
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Persist activity fields without changing the command revision."""
        mission_id = str(snapshot["mission_id"])
        current = self.get_mission(mission_id)
        if current is None:
            raise KeyError(mission_id)
        if int(current["revision"]) != expected_revision:
            raise RevisionConflict(current)
        updated = dict(snapshot)
        if (
            str(updated.get("mission_id")) != mission_id
            or int(updated.get("revision", -1)) != expected_revision
            or updated.get("state") != current.get("state")
        ):
            raise ValueError("activity update cannot change mission identity, revision, or state")
        changed_fields = {
            key for key in current.keys() | updated.keys()
            if current.get(key) != updated.get(key)
        }
        if not changed_fields.issubset({"budget", "last_sequence"}):
            raise ValueError("activity update may change only budget and event sequence")
        cursor = self._connection.execute(
            "UPDATE missions SET snapshot_json = ? WHERE mission_id = ? AND revision = ?",
            (_json(updated), mission_id, expected_revision),
        )
        if cursor.rowcount != 1:
            latest = self.get_mission(mission_id)
            if latest is None:
                raise KeyError(mission_id)
            raise RevisionConflict(latest)
        return updated

    def append_event(
        self,
        mission_id: str,
        revision: int,
        kind: str,
        data: Mapping[str, Any],
        timestamp_ms: int,
    ) -> dict[str, Any]:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE mission_id = ?",
            (mission_id,),
        ).fetchone()
        sequence = int(row[0]) + 1
        event = {
            "mission_id": mission_id,
            "revision": int(revision),
            "sequence": sequence,
            "kind": kind,
            "data": dict(data),
            "timestamp_ms": int(timestamp_ms),
        }
        self._connection.execute(
            "INSERT INTO events(mission_id, sequence, revision, kind, data_json, timestamp_ms) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                mission_id,
                sequence,
                revision,
                kind,
                _json(data),
                timestamp_ms,
            ),
        )
        self._connection.execute(
            "DELETE FROM events WHERE mission_id = ? AND sequence <= ?",
            (mission_id, sequence - MAX_EVENT_HISTORY),
        )
        mission = self.get_mission(mission_id)
        if mission is not None:
            mission["last_sequence"] = sequence
            self._connection.execute(
                "UPDATE missions SET snapshot_json = ? WHERE mission_id = ?",
                (_json(mission), mission_id),
            )
        self.events.append(event)
        return event

    def add_tool_record(
        self,
        mission_id: str,
        cycle_id: str,
        execution_generation: int,
        record: Mapping[str, Any],
        *,
        current: bool,
        created_at_ms: int,
    ) -> None:
        self._connection.execute(
            "INSERT INTO tool_records(record_id, mission_id, cycle_id, execution_generation, "
            "is_current, record_json, created_at_ms) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                str(record["tool_result_id"]),
                mission_id,
                cycle_id,
                execution_generation,
                1 if current else 0,
                _json(record),
                created_at_ms,
            ),
        )

    def evidence_usage(self, mission_id: str) -> tuple[int, int]:
        row = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM evidence "
            "WHERE mission_id = ? AND available = 1",
            (mission_id,),
        ).fetchone()
        return int(row[0]), int(row[1])

    def total_evidence_usage(self) -> tuple[int, int]:
        row = self._connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(bytes), 0) FROM evidence WHERE available = 1"
        ).fetchone()
        return int(row[0]), int(row[1])

    def list_evidence(self, mission_id: str) -> list[dict[str, Any]]:
        rows = self._connection.execute(
            "SELECT metadata_json FROM evidence WHERE mission_id = ? AND available = 1 "
            "ORDER BY created_at_ms, evidence_id",
            (mission_id,),
        ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def evidence_is_exported(self, evidence_id: str) -> bool:
        return self._connection.execute(
            "SELECT 1 FROM exported_evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone() is not None

    def _is_evidence_rolloff_eligible(self, evidence_id: str) -> bool:
        """Check persisted kind and trusted origin for an evidence row."""
        row = self._connection.execute(
            "SELECT kind, origin FROM evidence WHERE evidence_id = ? AND available = 1",
            (evidence_id,),
        ).fetchone()
        return row is not None and (
            (row[0] == "frame" and row[1] == "watch_frame")
            or (row[0] == "crop" and row[1] == "generated_crop")
        )

    def pin_exported_evidence(self, mission_id: str, evidence_ids: list[str]) -> None:
        self._connection.executemany(
            "INSERT OR IGNORE INTO exported_evidence(evidence_id, mission_id) "
            "SELECT evidence_id, mission_id FROM evidence "
            "WHERE evidence_id = ? AND mission_id = ? AND available = 1",
            [(evidence_id, mission_id) for evidence_id in evidence_ids],
        )

    def retire_evidence(self, evidence_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT path, metadata_json, mission_id FROM evidence "
            "WHERE evidence_id = ? AND available = 1",
            (evidence_id,),
        ).fetchone()
        if row is None:
            return None
        self._ensure_root_quota_guard()
        metadata = json.loads(row[1])
        metadata["available"] = False
        metadata["availability_reason"] = "rolled_off"
        self._connection.execute(
            "UPDATE evidence SET available = 0, metadata_json = ? WHERE evidence_id = ?",
            (_json(metadata), evidence_id),
        )
        rows = self._connection.execute(
            "SELECT record_id, record_json FROM tool_records WHERE mission_id = ?",
            (row[2],),
        ).fetchall()
        for record_id, encoded in rows:
            record = json.loads(encoded)
            if (
                evidence_id not in record.get("evidence_ids", ())
                and evidence_id != record.get("input_evidence_id")
            ):
                continue
            availability = dict(record.get("evidence_availability", {}))
            availability[evidence_id] = {
                "available": False,
                "availability_reason": "rolled_off",
            }
            record["evidence_availability"] = availability
            record["unavailable_evidence_ids"] = sorted(
                set(record.get("unavailable_evidence_ids", ())) | {evidence_id}
            )
            self._connection.execute(
                "UPDATE tool_records SET record_json = ? WHERE record_id = ?",
                (_json(record), record_id),
            )
        self._retired_paths.append(Path(row[0]))
        return metadata

    def set_evidence_unavailable(self, mission_id: str, evidence_id: str, reason: str) -> bool:
        """Persist an unavailable reason in evidence metadata and tool records."""
        if reason not in {"missing", "corrupt", "rolled_off"}:
            raise ValueError("evidence availability reason is invalid")
        row = self._connection.execute(
            "SELECT mission_id, metadata_json, available FROM evidence "
            "WHERE mission_id = ? AND evidence_id = ?",
            (mission_id, evidence_id),
        ).fetchone()
        if row is None:
            return False
        metadata = json.loads(row[1])
        changed = bool(row[2]) or metadata.get("available") is not False
        changed = changed or metadata.get("availability_reason") != reason
        metadata["available"] = False
        metadata["availability_reason"] = reason
        if changed:
            self._connection.execute(
                "UPDATE evidence SET available = 0, metadata_json = ? "
                "WHERE mission_id = ? AND evidence_id = ?",
                (_json(metadata), mission_id, evidence_id),
            )
        records = self._connection.execute(
            "SELECT record_id, record_json FROM tool_records WHERE mission_id = ?",
            (row[0],),
        ).fetchall()
        for record_id, encoded in records:
            record = json.loads(encoded)
            if (
                evidence_id not in record.get("evidence_ids", ())
                and evidence_id != record.get("input_evidence_id")
            ):
                continue
            availability = dict(record.get("evidence_availability", {}))
            target = {"available": False, "availability_reason": reason}
            unavailable = set(record.get("unavailable_evidence_ids", ()))
            if availability.get(evidence_id) == target and evidence_id in unavailable:
                continue
            availability[evidence_id] = target
            record["evidence_availability"] = availability
            record["unavailable_evidence_ids"] = sorted(unavailable | {evidence_id})
            self._connection.execute(
                "UPDATE tool_records SET record_json = ? WHERE record_id = ?",
                (_json(record), record_id),
            )
            changed = True
        return changed

    def save_evidence(
        self,
        mission_id: str,
        jpeg_bytes: bytes,
        *,
        kind: str,
        created_at_ms: int,
        origin: str = "unknown",
        source_id: str | None = None,
        source_epoch: str | None = None,
        frame_id: int | None = None,
        capture_time_ms: int | None = None,
        parent_evidence_id: str | None = None,
        crop_box: tuple[float, float, float, float] | None = None,
        input_transform: Mapping[str, Any] | None = None,
        closeup_request_id: str | None = None,
        brief_version: int | None = None,
        brief_sha256: str | None = None,
        model_provenance: Mapping[str, Any] | None = None,
        expected_sha256: str | None = None,
    ) -> dict[str, Any]:
        if origin not in _EVIDENCE_ORIGINS:
            raise ValueError("evidence origin is invalid")
        width, height = _jpeg_dimensions(jpeg_bytes)
        digest = hashlib.sha256(jpeg_bytes).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256.lower():
            raise InvalidEvidence("JPEG SHA-256 does not match the supplied digest")
        used = self.total_evidence_usage()[1]
        if used + len(jpeg_bytes) > self._store.quota_bytes:
            raise QuotaExceeded("mission evidence store quota exceeded")
        self._ensure_root_quota_guard()
        root_used = self._store._root_evidence_usage()
        if root_used + len(jpeg_bytes) > self._store.root_quota_bytes:
            raise RootQuotaExceeded(root_used, len(jpeg_bytes))
        mission_count, mission_used = self.evidence_usage(mission_id)
        if mission_count >= MAX_MISSION_EVIDENCE:
            raise QuotaExceeded(
                f"mission evidence limit reached ({MAX_MISSION_EVIDENCE} records)"
            )
        del mission_used

        evidence_id = uuid.uuid4().hex
        path = self._store.evidence_root / f"{evidence_id}.jpg"
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        self._created_paths.append(path)
        try:
            with os.fdopen(fd, "wb") as media:
                media.write(jpeg_bytes)
                media.flush()
                os.fsync(media.fileno())
        except BaseException:
            path.unlink(missing_ok=True)
            self._created_paths.remove(path)
            raise

        ref = {
            "evidence_id": evidence_id,
            "sha256": digest,
            "width": width,
            "height": height,
            "bytes": len(jpeg_bytes),
            "capture_time_ms": capture_time_ms,
            "source_id": source_id,
            "source_epoch": source_epoch,
            "frame_id": frame_id,
            "created_at_ms": int(created_at_ms),
            "kind": kind,
            "parent_evidence_id": parent_evidence_id,
            "crop_box": list(crop_box) if crop_box is not None else None,
            "input_transform": dict(input_transform or {}),
            "closeup_request_id": closeup_request_id,
            "available": True,
        }
        if brief_version is not None:
            ref["brief_version"] = int(brief_version)
        if brief_sha256 is not None:
            ref["brief_sha256"] = str(brief_sha256)
        if model_provenance:
            ref["model_provenance"] = dict(model_provenance)
        self._connection.execute(
            "INSERT INTO evidence(evidence_id, mission_id, sha256, bytes, width, height, kind, "
            "parent_evidence_id, source_id, source_epoch, frame_id, capture_time_ms, "
            "created_at_ms, path, metadata_json, available, origin) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)",
            (
                evidence_id,
                mission_id,
                digest,
                len(jpeg_bytes),
                width,
                height,
                kind,
                parent_evidence_id,
                source_id,
                source_epoch,
                frame_id,
                capture_time_ms,
                created_at_ms,
                str(path),
                _json(ref),
                origin,
            ),
        )
        fingerprint = _file_fingerprint(path.stat())
        self._connection.execute(
            "INSERT INTO evidence_file_state(evidence_id, device, inode, size, mtime_ns, ctime_ns) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (evidence_id, *fingerprint),
        )
        return ref


class MissionStore:
    """Single-writer SQLite store with immutable evidence outside temp storage."""

    def __init__(
        self,
        database_path: str | Path | None = None,
        evidence_root: str | Path | None = None,
        *,
        quota_bytes: int = DEFAULT_QUOTA_BYTES,
        root_quota_bytes: int = DEFAULT_ROOT_QUOTA_BYTES,
    ) -> None:
        base = Path.home() / ".visionbrain" / "missions"
        self.database_path = Path(database_path) if database_path is not None else base / "missions.sqlite3"
        self.evidence_root = (
            Path(evidence_root) if evidence_root is not None else self.database_path.parent / "evidence"
        )
        if quota_bytes <= 0:
            raise ValueError("quota_bytes must be positive")
        if root_quota_bytes <= 0:
            raise ValueError("root_quota_bytes must be positive")
        self.quota_bytes = int(quota_bytes)
        self.root_quota_bytes = int(root_quota_bytes)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.database_path.parent, 0o700)
        self.evidence_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.evidence_root = self.evidence_root.resolve(strict=True)
        os.chmod(self.evidence_root, 0o700)
        root_key = hashlib.sha256(os.fsencode(self.evidence_root)).hexdigest()
        self._root_quota_lock_path = self.evidence_root.parent / f".mission-root-quota-{root_key}.lock"
        self._lock = threading.RLock()
        self._connection = sqlite3.connect(
            str(self.database_path), timeout=30, check_same_thread=False
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.execute("PRAGMA busy_timeout = 30000")
        if str(self.database_path) != ":memory:":
            self._connection.execute("PRAGMA journal_mode = WAL")
            self._connection.execute("PRAGMA synchronous = FULL")
        self._initialize()

    @contextmanager
    def _root_quota_lock(self):
        """Serialize cooperating writers that share this canonical evidence root."""
        flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
        fd = None
        try:
            fd = os.open(self._root_quota_lock_path, flags, 0o600)
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise OSError("quota lock is not a regular file")
            os.fchmod(fd, 0o600)
            deadline = time.monotonic() + MAX_ROOT_QUOTA_LOCK_WAIT_SECONDS
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise QuotaAccountingIncomplete(
                            "mission evidence root quota accounting incomplete: root lock busy (wait limit exceeded)"
                        )
                    time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))
                except InterruptedError as exc:
                    raise QuotaAccountingIncomplete(
                        "mission evidence root quota accounting incomplete: root lock acquisition interrupted"
                    ) from exc
        except BaseException as exc:
            if fd is not None:
                os.close(fd)
            if isinstance(exc, (QuotaAccountingIncomplete, KeyboardInterrupt, SystemExit)):
                raise
            if isinstance(exc, OSError):
                raise QuotaAccountingIncomplete(
                    f"mission evidence root quota accounting incomplete: root lock unavailable ({exc})"
                ) from exc
            raise
        try:
            yield
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def _root_evidence_usage(self) -> int:
        """Sum regular-file sizes without following links, within finite scan bounds."""
        directory_flag = getattr(os, "O_DIRECTORY", 0)
        nofollow_flag = getattr(os, "O_NOFOLLOW", 0)
        if not directory_flag or not nofollow_flag:
            raise QuotaAccountingIncomplete(
                "mission evidence root quota accounting incomplete: safe directory traversal unavailable"
            )
        deadline = time.monotonic() + MAX_ROOT_QUOTA_SCAN_SECONDS
        directory_flags = os.O_RDONLY | directory_flag | nofollow_flag
        entries_seen = 0
        total_bytes = 0

        def check_deadline() -> None:
            if time.monotonic() > deadline:
                raise QuotaAccountingIncomplete(
                    "mission evidence root quota accounting incomplete: scan time limit exceeded"
                )

        def scan_directory(directory_fd: int, depth: int) -> None:
            nonlocal entries_seen, total_bytes
            check_deadline()
            try:
                iterator = os.scandir(directory_fd)
            except OSError as exc:
                raise QuotaAccountingIncomplete(
                    f"mission evidence root quota accounting incomplete: directory scan failed ({exc})"
                ) from exc
            with iterator:
                for entry in iterator:
                    check_deadline()
                    entries_seen += 1
                    if entries_seen > MAX_ROOT_QUOTA_SCAN_ENTRIES:
                        raise QuotaAccountingIncomplete(
                            "mission evidence root quota accounting incomplete: entry limit exceeded"
                        )
                    try:
                        metadata = entry.stat(follow_symlinks=False)
                        if stat.S_ISREG(metadata.st_mode):
                            total_bytes += metadata.st_size
                        elif stat.S_ISDIR(metadata.st_mode):
                            if depth >= MAX_ROOT_QUOTA_SCAN_DEPTH:
                                raise QuotaAccountingIncomplete(
                                    "mission evidence root quota accounting incomplete: depth limit exceeded"
                                )
                            child_fd = os.open(entry.name, directory_flags, dir_fd=directory_fd)
                            try:
                                scan_directory(child_fd, depth + 1)
                            finally:
                                os.close(child_fd)
                    except QuotaAccountingIncomplete:
                        raise
                    except OSError as exc:
                        raise QuotaAccountingIncomplete(
                            f"mission evidence root quota accounting incomplete: entry inspection failed ({exc})"
                        ) from exc
            check_deadline()

        try:
            root_fd = os.open(self.evidence_root, directory_flags)
            try:
                scan_directory(root_fd, 0)
            finally:
                os.close(root_fd)
        except QuotaAccountingIncomplete:
            raise
        except OSError as exc:
            raise QuotaAccountingIncomplete(
                f"mission evidence root quota accounting incomplete: root scan failed ({exc})"
            ) from exc
        return total_bytes

    def _initialize(self) -> None:
        with self._lock:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS missions (
                    mission_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    state TEXT NOT NULL,
                    snapshot_json TEXT NOT NULL,
                    updated_at_ms INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS events (
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    sequence INTEGER NOT NULL,
                    revision INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    timestamp_ms INTEGER NOT NULL,
                    PRIMARY KEY (mission_id, sequence)
                );
                CREATE TABLE IF NOT EXISTS requests (
                    principal_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    payload_sha256 TEXT NOT NULL,
                    reply_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL,
                    PRIMARY KEY (principal_id, request_id)
                );
                CREATE TABLE IF NOT EXISTS evidence (
                    evidence_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    sha256 TEXT NOT NULL,
                    bytes INTEGER NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    parent_evidence_id TEXT,
                    source_id TEXT,
                    source_epoch TEXT,
                    frame_id INTEGER,
                    capture_time_ms INTEGER,
                    created_at_ms INTEGER NOT NULL,
                    path TEXT NOT NULL UNIQUE,
                    metadata_json TEXT NOT NULL,
                    available INTEGER NOT NULL DEFAULT 1,
                    origin TEXT NOT NULL DEFAULT 'unknown'
                );
                CREATE TABLE IF NOT EXISTS evidence_file_state (
                    evidence_id TEXT PRIMARY KEY REFERENCES evidence(evidence_id),
                    device INTEGER NOT NULL,
                    inode INTEGER NOT NULL,
                    size INTEGER NOT NULL,
                    mtime_ns INTEGER NOT NULL,
                    ctime_ns INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS exported_evidence (
                    evidence_id TEXT NOT NULL REFERENCES evidence(evidence_id),
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    PRIMARY KEY (evidence_id, mission_id)
                );
                CREATE INDEX IF NOT EXISTS evidence_mission_idx ON evidence(mission_id);
                CREATE TABLE IF NOT EXISTS tool_records (
                    record_id TEXT PRIMARY KEY,
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    cycle_id TEXT NOT NULL,
                    execution_generation INTEGER NOT NULL,
                    is_current INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS tool_records_cycle_idx
                    ON tool_records(mission_id, cycle_id, created_at_ms);
                CREATE TABLE IF NOT EXISTS planner_diagnostics (
                    diagnostic_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    mission_id TEXT NOT NULL REFERENCES missions(mission_id),
                    cycle_id TEXT NOT NULL,
                    execution_generation INTEGER NOT NULL,
                    attempt INTEGER NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at_ms INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS planner_diagnostics_mission_idx
                    ON planner_diagnostics(mission_id, diagnostic_id);
                """
            )
            evidence_columns = {
                row[1] for row in self._connection.execute("PRAGMA table_info(evidence)")
            }
            if "available" not in evidence_columns:
                self._connection.execute(
                    "ALTER TABLE evidence ADD COLUMN available INTEGER NOT NULL DEFAULT 1"
                )
            if "origin" not in evidence_columns:
                self._connection.execute(
                    "ALTER TABLE evidence ADD COLUMN origin TEXT NOT NULL DEFAULT 'unknown'"
                )
            self._connection.commit()

    def _run_transaction(self, operation: Callable[[StoreTransaction], Any]) -> TransactionResult:
        with self._lock:
            tx = StoreTransaction(self, self._connection)
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                value = operation(tx)
                self._connection.commit()
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                for path in tx._created_paths:
                    path.unlink(missing_ok=True)
                raise
            else:
                _cleanup_retired_paths(tx._retired_paths)
                return TransactionResult(value, tuple(tx.events))
            finally:
                tx._release_root_quota_guard()

    def perform_read(self, operation: Callable[[StoreTransaction], Mapping[str, Any]]) -> RequestResult:
        """Run a read command transaction without storing its reply for replay."""
        result = self._run_transaction(operation)
        return RequestResult(dict(result.value), False, result.events)

    def perform_request(
        self,
        principal_id: str,
        request_id: str,
        payload: Mapping[str, Any],
        operation: Callable[[StoreTransaction], Mapping[str, Any]],
        *,
        now_ms: int,
    ) -> RequestResult:
        """Run and durably memoize one principal/request pair atomically."""
        if not principal_id or not request_id:
            raise ValueError("principal_id and request_id are required")
        payload_digest = _payload_hash(payload)
        with self._lock:
            tx = StoreTransaction(self, self._connection)
            try:
                self._connection.execute("BEGIN IMMEDIATE")
                row = self._connection.execute(
                    "SELECT payload_sha256, reply_json FROM requests "
                    "WHERE principal_id = ? AND request_id = ?",
                    (principal_id, request_id),
                ).fetchone()
                if row is not None:
                    self._connection.rollback()
                    if row[0] != payload_digest:
                        raise IdempotencyConflict("request ID reused with a different payload")
                    return RequestResult(json.loads(row[1]), True)

                reply = dict(operation(tx))
                self._connection.execute(
                    "INSERT INTO requests(principal_id, request_id, payload_sha256, reply_json, created_at_ms) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (principal_id, request_id, payload_digest, _json(reply), int(now_ms)),
                )
                self._connection.commit()
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                for path in tx._created_paths:
                    path.unlink(missing_ok=True)
                raise
            else:
                _cleanup_retired_paths(tx._retired_paths)
                return RequestResult(reply, False, tuple(tx.events))
            finally:
                tx._release_root_quota_guard()

    def transact(self, operation: Callable[[StoreTransaction], Any]) -> TransactionResult:
        """Atomically mutate mission state and append any events in ``operation``."""
        return self._run_transaction(operation)

    def get_mission(self, mission_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT snapshot_json FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def read_mission_record_rows(
        self, mission_id: str, *, tx: StoreTransaction | None = None
    ) -> dict[str, Any] | None:
        """Read one coherent metadata snapshot and the complete rows used by the record projector.

        Evidence rows contain decoded persisted metadata plus their authoritative
        SQL ``mission_id``. Tool rows contain the SQL envelope and nested record.
        Availability is persisted metadata; this method does not inspect media.
        """
        if tx is not None:
            if tx._store is not self:
                raise ValueError("record-row transaction belongs to another mission store")
            return tx.read_mission_record_rows(mission_id)
        with self._lock:
            if self._connection.in_transaction:
                raise MissionStoreError(
                    "cannot read mission record rows inside an active store transaction"
                )
            try:
                self._connection.execute("BEGIN")
                result = StoreTransaction(self, self._connection).read_mission_record_rows(mission_id)
                self._connection.commit()
                return result
            except BaseException:
                if self._connection.in_transaction:
                    self._connection.rollback()
                raise

    def list_missions(self, *, limit: int = 50, before_updated_at_ms: int | None = None) -> list[dict[str, Any]]:
        limit = max(1, min(int(limit), MAX_MISSIONS_PAGE))
        with self._lock:
            if before_updated_at_ms is None:
                rows = self._connection.execute(
                    "SELECT snapshot_json FROM missions ORDER BY updated_at_ms DESC, mission_id LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT snapshot_json FROM missions WHERE updated_at_ms < ? "
                    "ORDER BY updated_at_ms DESC, mission_id LIMIT ?",
                    (int(before_updated_at_ms), limit),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_active_missions(self) -> list[dict[str, Any]]:
        """Return all active snapshots without a paginated mission scan."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT snapshot_json FROM missions "
                "WHERE state IN ('running', 'waiting_evidence', 'waiting_approval') "
                "ORDER BY updated_at_ms DESC, mission_id"
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def list_running_missions(self, *, excluding: str | None = None) -> list[dict[str, Any]]:
        """Return concurrent execution candidates within the current transaction."""
        sql = "SELECT snapshot_json FROM missions WHERE state = 'running'"
        params: tuple[Any, ...] = ()
        if excluding is not None:
            sql += " AND mission_id != ?"
            params = (excluding,)
        rows = self._connection.execute(sql + " ORDER BY updated_at_ms DESC, mission_id", params).fetchall()
        return [json.loads(row[0]) for row in rows]

    def events_since(
        self,
        mission_id: str,
        cursor: int,
        *,
        limit: int = 100,
        max_bytes: int = MAX_EVENT_PAGE_BYTES,
    ) -> dict[str, Any]:
        limit = max(1, min(int(limit), MAX_EVENTS_PAGE))
        max_bytes = max(1_024, min(int(max_bytes), MAX_EVENT_PAGE_BYTES))
        with self._lock:
            if not self._connection.execute(
                "SELECT 1 FROM missions WHERE mission_id = ?", (mission_id,)
            ).fetchone():
                raise KeyError(mission_id)
            last = int(
                self._connection.execute(
                    "SELECT COALESCE(MAX(sequence), 0) FROM events WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()[0]
            )
            if cursor < 0 or cursor > last:
                return {"events": [], "last_sequence": last, "resync_required": True}
            oldest = int(
                self._connection.execute(
                    "SELECT COALESCE(MIN(sequence), 0) FROM events WHERE mission_id = ?",
                    (mission_id,),
                ).fetchone()[0]
            )
            if oldest and cursor < oldest - 1:
                return {"events": [], "last_sequence": last, "resync_required": True}
            rows = self._connection.execute(
                "SELECT mission_id, revision, sequence, kind, data_json, timestamp_ms "
                "FROM events WHERE mission_id = ? AND sequence > ? ORDER BY sequence LIMIT ?",
                (mission_id, int(cursor), limit),
            ).fetchall()
        candidates = [
            {
                "mission_id": row[0],
                "revision": row[1],
                "sequence": row[2],
                "kind": row[3],
                "data": json.loads(row[4]),
                "timestamp_ms": row[5],
            }
            for row in rows
        ]
        events: list[dict[str, Any]] = []
        for event in candidates:
            candidate = events + [event]
            cursor_value = int(event["sequence"])
            page = {
                "events": candidate,
                "last_sequence": last,
                "next_cursor": cursor_value,
                "has_more": cursor_value < last,
                "resync_required": False,
            }
            if len(_json(page).encode("utf-8")) > max_bytes:
                break
            events.append(event)
        if not events and candidates:
            event = candidates[0]
            compact = {
                **{key: event[key] for key in ("mission_id", "revision", "sequence", "kind", "timestamp_ms")},
                "data": {"activity": event["kind"], "truncated": True},
            }
            events = [compact]
        next_cursor = int(events[-1]["sequence"]) if events else int(cursor)
        return {
            "events": events,
            "last_sequence": last,
            "next_cursor": next_cursor,
            "has_more": next_cursor < last,
            "resync_required": False,
        }

    def _read_verified_evidence(self, evidence_id: str) -> tuple[dict[str, Any], tuple[int, ...]]:
        with self._lock:
            row = self._connection.execute(
                "SELECT path, sha256, bytes, width, height, metadata_json, available "
                "FROM evidence WHERE evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise KeyError(evidence_id)
            if not bool(row[6]):
                metadata = json.loads(row[5])
                reason = metadata.get("availability_reason", "rolled_off")
                if reason not in {"missing", "corrupt", "rolled_off"}:
                    reason = "corrupt"
                raise EvidenceUnavailable(
                    f"evidence is unavailable ({reason})",
                    availability_reason=reason,
                )
            expected_bytes = int(row[2])
            if expected_bytes <= 0 or expected_bytes > MAX_DECODED_JPEG_BYTES:
                raise EvidenceUnavailable(
                    "stored evidence byte length exceeds the configured limit",
                    availability_reason="corrupt",
                )
            path = Path(row[0])
            try:
                path.resolve().relative_to(self.evidence_root.resolve())
                with path.open("rb") as media:
                    before = _file_fingerprint(os.fstat(media.fileno()))
                    if before[2] > MAX_DECODED_JPEG_BYTES:
                        raise EvidenceUnavailable(
                            "evidence media exceeds the configured size limit",
                            availability_reason="corrupt",
                        )
                    if before[2] != expected_bytes:
                        raise EvidenceUnavailable(
                            "evidence byte length changed", availability_reason="corrupt"
                        )
                    data = media.read(MAX_DECODED_JPEG_BYTES + 1)
                    after = _file_fingerprint(os.fstat(media.fileno()))
            except (OSError, ValueError) as exc:
                reason = "missing" if isinstance(exc, OSError) else "corrupt"
                raise EvidenceUnavailable("evidence media is unavailable", availability_reason=reason) from exc
            if before != after:
                raise EvidenceUnavailable("evidence changed while being read", availability_reason="corrupt")
            if len(data) > MAX_DECODED_JPEG_BYTES:
                raise EvidenceUnavailable(
                    "evidence media exceeds the configured size limit",
                    availability_reason="corrupt",
                )
            expected_width, expected_height = int(row[3]), int(row[4])
            if len(data) != expected_bytes:
                raise EvidenceUnavailable("evidence byte length changed", availability_reason="corrupt")
            if hashlib.sha256(data).hexdigest() != row[1]:
                raise EvidenceUnavailable("evidence SHA-256 changed", availability_reason="corrupt")
            try:
                width, height = _jpeg_dimensions(data)
            except InvalidEvidence as exc:
                raise EvidenceUnavailable(
                    "stored evidence is no longer a valid JPEG", availability_reason="corrupt"
                ) from exc
            if (width, height) != (expected_width, expected_height):
                raise EvidenceUnavailable("evidence dimensions changed", availability_reason="corrupt")
            return {"metadata": json.loads(row[5]), "jpeg_bytes": data}, after

    def read_evidence(self, evidence_id: str) -> dict[str, Any]:
        """Read and verify the complete immutable JPEG before any use or serving."""
        verified, _fingerprint = self._read_verified_evidence(evidence_id)
        return verified

    def check_evidence_file(self, evidence_id: str) -> None:
        """Check a saved file fingerprint without hashing its JPEG contents."""
        with self._lock:
            row = self._connection.execute(
                "SELECT e.path, e.bytes, s.device, s.inode, s.size, s.mtime_ns, s.ctime_ns, "
                "e.available, e.metadata_json "
                "FROM evidence e LEFT JOIN evidence_file_state s USING (evidence_id) "
                "WHERE e.evidence_id = ?",
                (evidence_id,),
            ).fetchone()
            if row is None:
                raise KeyError(evidence_id)
            if not bool(row[7]):
                metadata = json.loads(row[8])
                reason = metadata.get("availability_reason", "rolled_off")
                if reason not in {"missing", "corrupt", "rolled_off"}:
                    reason = "corrupt"
                raise EvidenceUnavailable(
                    f"evidence is unavailable ({reason})",
                    availability_reason=reason,
                )
            path = Path(row[0])
            try:
                path.resolve().relative_to(self.evidence_root.resolve())
                current = _file_fingerprint(path.stat())
            except (OSError, ValueError) as exc:
                reason = "missing" if isinstance(exc, OSError) else "corrupt"
                raise EvidenceUnavailable("evidence media is unavailable", availability_reason=reason) from exc
            if current[2] != int(row[1]):
                raise EvidenceUnavailable("evidence byte length changed", availability_reason="corrupt")
            if row[2] is None:
                # Evidence created before file fingerprints were recorded gets one
                # full verification before its baseline is saved.
                _verified, current = self._read_verified_evidence(evidence_id)
                started_transaction = not self._connection.in_transaction
                self._connection.execute(
                    "INSERT OR IGNORE INTO evidence_file_state "
                    "(evidence_id, device, inode, size, mtime_ns, ctime_ns) VALUES (?, ?, ?, ?, ?, ?)",
                    (evidence_id, *current),
                )
                if started_transaction:
                    self._connection.commit()
                return
            expected = tuple(int(row[index]) for index in range(2, 7))
            if current != expected:
                raise EvidenceUnavailable("evidence file changed", availability_reason="corrupt")

    def get_evidence_chunk(
        self, evidence_id: str, *, offset: int = 0, length: int = MAX_EVIDENCE_CHUNK_BYTES
    ) -> dict[str, Any]:
        length = max(1, min(int(length), MAX_EVIDENCE_CHUNK_BYTES))
        if offset < 0:
            raise ValueError("offset must be non-negative")
        verified = self.read_evidence(evidence_id)
        meta = verified["metadata"]
        if offset > int(meta["bytes"]):
            raise ValueError("offset exceeds evidence length")
        data = verified["jpeg_bytes"][offset : offset + length]
        # JSON base64 adds 4/3 overhead; leave ample room under 256 KiB.
        if len(data) > MAX_OUTBOUND_MESSAGE_BYTES * 3 // 4:
            raise AssertionError("evidence chunk exceeds outbound message budget")
        return {
            "evidence_id": evidence_id,
            "sha256": meta["sha256"],
            "total_bytes": int(meta["bytes"]),
            "offset": int(offset),
            "width": int(meta["width"]),
            "height": int(meta["height"]),
            "jpeg_bytes": data,
        }

    def evidence_owner(self, evidence_id: str) -> str | None:
        """Return the owning mission without relying on a paginated mission list."""
        with self._lock:
            row = self._connection.execute(
                "SELECT mission_id FROM evidence WHERE evidence_id = ?", (evidence_id,)
            ).fetchone()
        return str(row[0]) if row else None

    def evidence_refs(self, mission_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT metadata_json, available FROM evidence WHERE mission_id = ? "
                "ORDER BY created_at_ms, evidence_id",
                (mission_id,),
            ).fetchall()
        refs = []
        for row in rows:
            ref = json.loads(row[0])
            if not bool(row[1]):
                ref["available"] = False
                ref.setdefault("availability_reason", "rolled_off")
            else:
                ref["available"] = True
            refs.append(ref)
        return refs

    def tool_records(self, mission_id: str, *, cycle_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if cycle_id is None:
                rows = self._connection.execute(
                    "SELECT record_json FROM tool_records WHERE mission_id = ? ORDER BY created_at_ms, record_id",
                    (mission_id,),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT record_json FROM tool_records WHERE mission_id = ? AND cycle_id = ? "
                    "ORDER BY created_at_ms, record_id",
                    (mission_id, cycle_id),
                ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def add_planner_diagnostic(
        self,
        mission_id: str,
        *,
        cycle_id: str,
        execution_generation: int,
        attempt: int,
        model_key: str,
        checkpoint: str,
        prompt_version: str,
        raw_text: str,
        error: str,
        repair_feedback: str | None,
        created_at_ms: int,
    ) -> None:
        """Persist bounded raw planner failures privately, outside snapshots/exports."""
        encoded = str(raw_text).encode("utf-8", errors="replace")
        truncated = len(encoded) > MAX_PLANNER_DIAGNOSTIC_BYTES
        bounded_raw = encoded[:MAX_PLANNER_DIAGNOSTIC_BYTES].decode("utf-8", errors="ignore")
        record = {
            "cycle_id": str(cycle_id)[:128],
            "execution_generation": int(execution_generation),
            "attempt": max(0, int(attempt)),
            "model_key": str(model_key)[:32],
            "checkpoint": str(checkpoint)[:256],
            "prompt_version": str(prompt_version)[:64],
            "raw_text": bounded_raw,
            "raw_text_truncated": truncated,
            "error": str(error)[:1_000],
            "repair_feedback": str(repair_feedback)[:1_000] if repair_feedback else None,
        }
        with self._lock:
            self._connection.execute(
                "INSERT INTO planner_diagnostics "
                "(mission_id, cycle_id, execution_generation, attempt, record_json, created_at_ms) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    mission_id,
                    record["cycle_id"],
                    record["execution_generation"],
                    record["attempt"],
                    _json(record),
                    int(created_at_ms),
                ),
            )
            self._connection.execute(
                "DELETE FROM planner_diagnostics WHERE diagnostic_id IN ("
                "SELECT diagnostic_id FROM planner_diagnostics WHERE mission_id = ? "
                "ORDER BY diagnostic_id DESC LIMIT -1 OFFSET ?) ",
                (mission_id, MAX_PLANNER_DIAGNOSTICS_PER_MISSION),
            )
            self._connection.execute(
                "DELETE FROM planner_diagnostics WHERE diagnostic_id IN ("
                "SELECT diagnostic_id FROM planner_diagnostics "
                "ORDER BY diagnostic_id DESC LIMIT -1 OFFSET ?) ",
                (MAX_PLANNER_DIAGNOSTICS_TOTAL,),
            )
            self._connection.commit()

    def planner_diagnostics(self, mission_id: str) -> list[dict[str, Any]]:
        """Return private bounded planner diagnostics for local operator tooling."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT record_json FROM planner_diagnostics "
                "WHERE mission_id = ? ORDER BY diagnostic_id",
                (mission_id,),
            ).fetchall()
        return [json.loads(row[0]) for row in rows]

    def backup_to(self, destination: str | Path) -> Path:
        """Create an offline backup under this store's mutation lock."""
        from .mission_backup import backup_mission_store

        return backup_mission_store(self, destination)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
