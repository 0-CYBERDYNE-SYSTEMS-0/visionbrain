"""Offline, integrity-checked backup and restore for :mod:`mission_store`."""

from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import sqlite3
import stat
import sys
import tempfile
from pathlib import Path
from typing import Any

from .mission_contracts import MAX_DECODED_JPEG_BYTES
from .mission_store import MissionStoreError, _jpeg_dimensions

BACKUP_FORMAT = "visionbrain.mission-store-backup"
BACKUP_VERSION = 1
MANIFEST_NAME = "manifest.json"
DATABASE_NAME = "missions.sqlite3"
EVIDENCE_DIRECTORY = "evidence"
_CHUNK_BYTES = 64 * 1024
MAX_MANIFEST_BYTES = 16 * 1024 * 1024  # Bound JSON parsing for format version 1.
_EVIDENCE_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


class MissionBackupError(MissionStoreError):
    """The backup is invalid, unsafe, corrupt, or cannot be published."""


def backup_mission_store(store: Any, destination: str | Path) -> Path:
    """Create a new offline backup while holding the store's mutation lock."""
    target = _new_destination(destination)
    _reject_overlap(target, _store_roots(store))
    stage = _new_stage(target, "backup")
    try:
        with store._lock:
            _create_backup_locked(store, stage)
        _publish_no_replace(stage, target)
        _fsync_directory(target.parent)
        return target
    except MissionBackupError:
        raise
    except Exception as exc:
        raise MissionBackupError(f"backup failed: {exc}") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def restore_mission_store(
    backup_directory: str | Path,
    destination: str | Path,
) -> Path:
    """Restore a validated backup into a new directory without replacement."""
    backup_root = _existing_directory(backup_directory)
    target = _new_destination(destination)
    _reject_overlap(target, (backup_root,))
    stage = _new_stage(target, "restore")
    try:
        manifest = _read_manifest(backup_root)
        _validate_bundle_layout(backup_root, manifest)
        _copy_and_validate_restore(backup_root, stage, target, manifest)
        _publish_no_replace(stage, target)
        _fsync_directory(target.parent)
        return target
    except MissionBackupError:
        raise
    except Exception as exc:
        raise MissionBackupError(f"restore failed: {exc}") from exc
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _create_backup_locked(store: Any, stage: Path) -> None:
    evidence_dir = stage / EVIDENCE_DIRECTORY
    evidence_dir.mkdir(mode=0o700)
    database_path = stage / DATABASE_NAME
    destination_connection = sqlite3.connect(str(database_path))
    try:
        store._connection.backup(destination_connection)
    finally:
        destination_connection.close()
    checkpoint = sqlite3.connect(str(database_path))
    try:
        mode = checkpoint.execute("PRAGMA journal_mode = DELETE").fetchone()[0]
        if str(mode).lower() != "delete":
            raise MissionBackupError("could not make the backup database self-contained")
    finally:
        checkpoint.close()
    os.chmod(database_path, 0o600)

    connection = _connect_readonly(database_path)
    try:
        rows = connection.execute(
            "SELECT evidence_id, sha256, bytes, width, height, path, available "
            "FROM evidence ORDER BY evidence_id"
        ).fetchall()
        file_states = {
            str(row[0]): tuple(int(value) for value in row[1:])
            for row in connection.execute(
                "SELECT evidence_id, device, inode, size, mtime_ns, ctime_ns "
                "FROM evidence_file_state"
            )
        }
    finally:
        connection.close()

    manifest_evidence = []
    for row in rows:
        evidence_id = _safe_evidence_id(row[0])
        digest = str(row[1])
        size = _integer(row[2], "evidence byte length", minimum=1)
        width = _integer(row[3], "evidence width", minimum=1)
        height = _integer(row[4], "evidence height", minimum=1)
        available = _database_boolean(row[6], "evidence availability")
        relative_name = f"{EVIDENCE_DIRECTORY}/{evidence_id}.jpg" if available else None
        record = {
            "evidence_id": evidence_id,
            "available": available,
            "path": relative_name,
            "bytes": size,
            "sha256": digest,
            "width": width,
            "height": height,
        }
        if available:
            if size > MAX_DECODED_JPEG_BYTES:
                raise MissionBackupError(f"available evidence {evidence_id} exceeds the JPEG limit")
            source_path = _source_evidence_path(store.evidence_root, row[5])
            copied, source_stat = _copy_bounded_file(
                source_path,
                evidence_dir / f"{evidence_id}.jpg",
                max_bytes=MAX_DECODED_JPEG_BYTES,
            )
            if file_states.get(evidence_id) not in (None, _fingerprint(source_stat)):
                raise MissionBackupError(f"available evidence {evidence_id} file fingerprint changed")
            _verify_media_record(record, copied)
        manifest_evidence.append(record)

    database_size, database_sha = _hash_file(database_path)
    manifest = {
        "format": BACKUP_FORMAT,
        "version": BACKUP_VERSION,
        "database": {
            "path": DATABASE_NAME,
            "bytes": database_size,
            "sha256": database_sha,
        },
        "evidence": manifest_evidence,
    }
    _write_json(stage / MANIFEST_NAME, manifest)
    _fsync_directory(evidence_dir)
    _fsync_directory(stage)
    _validate_bundle(stage, manifest)


def _copy_and_validate_restore(
    backup_root: Path,
    stage: Path,
    target: Path,
    manifest: dict[str, Any],
) -> None:
    (stage / EVIDENCE_DIRECTORY).mkdir(mode=0o700)
    database_entry = manifest["database"]
    _copy_verified_file(
        backup_root / DATABASE_NAME,
        stage / DATABASE_NAME,
        expected_bytes=database_entry["bytes"],
        expected_sha256=database_entry["sha256"],
    )
    for record in manifest["evidence"]:
        if not record["available"]:
            continue
        _copy_verified_file(
            backup_root / record["path"],
            stage / record["path"],
            expected_bytes=record["bytes"],
            expected_sha256=record["sha256"],
            max_bytes=MAX_DECODED_JPEG_BYTES,
        )

    _write_json(stage / MANIFEST_NAME, manifest)
    _validate_bundle(stage, manifest)
    (stage / MANIFEST_NAME).unlink()
    database_path = stage / DATABASE_NAME
    final_evidence_root = target / EVIDENCE_DIRECTORY
    connection = sqlite3.connect(str(database_path))
    try:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA synchronous = FULL")
        connection.execute("BEGIN IMMEDIATE")
        rows = connection.execute(
            "SELECT evidence_id, available FROM evidence ORDER BY evidence_id"
        ).fetchall()
        if len(rows) != len(manifest["evidence"]):
            raise MissionBackupError("database evidence rows changed during restore")
        for evidence_id, available_value in rows:
            evidence_id = _safe_evidence_id(evidence_id)
            available = _database_boolean(available_value, "evidence availability")
            final_path = str(final_evidence_root / f"{evidence_id}.jpg")
            connection.execute(
                "UPDATE evidence SET path = ? WHERE evidence_id = ?",
                (final_path, evidence_id),
            )
            if available:
                staged_path = stage / EVIDENCE_DIRECTORY / f"{evidence_id}.jpg"
                fingerprint = _fingerprint(staged_path.stat())
                connection.execute(
                    "INSERT INTO evidence_file_state "
                    "(evidence_id, device, inode, size, mtime_ns, ctime_ns) "
                    "VALUES (?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT(evidence_id) DO UPDATE SET "
                    "device=excluded.device, inode=excluded.inode, size=excluded.size, "
                    "mtime_ns=excluded.mtime_ns, ctime_ns=excluded.ctime_ns",
                    (evidence_id, *fingerprint),
                )
            else:
                connection.execute(
                    "DELETE FROM evidence_file_state WHERE evidence_id = ?", (evidence_id,)
                )
        connection.commit()
    except BaseException:
        if connection.in_transaction:
            connection.rollback()
        raise
    finally:
        connection.close()

    _validate_restored_database(database_path, stage, target, manifest)
    _fsync_directory(stage / EVIDENCE_DIRECTORY)
    _fsync_directory(stage)


def _read_manifest(root: Path) -> dict[str, Any]:
    _require_plain_directory(root)
    path = root / MANIFEST_NAME
    data, _ = _read_plain_file(path, max_bytes=MAX_MANIFEST_BYTES)
    try:
        manifest = json.loads(data, object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MissionBackupError("backup manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or set(manifest) != {"format", "version", "database", "evidence"}:
        raise MissionBackupError("backup manifest has an unsupported shape")
    if (
        manifest.get("format") != BACKUP_FORMAT
        or isinstance(manifest.get("version"), bool)
        or not isinstance(manifest.get("version"), int)
        or manifest.get("version") != BACKUP_VERSION
    ):
        raise MissionBackupError("backup manifest format/version is unsupported or stale")
    database = manifest.get("database")
    if not isinstance(database, dict) or set(database) != {"path", "bytes", "sha256"}:
        raise MissionBackupError("backup database manifest is invalid")
    if database["path"] != DATABASE_NAME:
        raise MissionBackupError("backup database path is unsafe")
    database["bytes"] = _integer(database["bytes"], "database byte length", minimum=1)
    database["sha256"] = _sha256(database["sha256"], "database SHA-256")
    records = manifest.get("evidence")
    if not isinstance(records, list):
        raise MissionBackupError("backup evidence manifest is invalid")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or set(record) != {
            "evidence_id", "available", "path", "bytes", "sha256", "width", "height"
        }:
            raise MissionBackupError("backup evidence entry has an unsupported shape")
        record["evidence_id"] = _safe_evidence_id(record["evidence_id"])
        if record["evidence_id"] in seen:
            raise MissionBackupError("backup evidence IDs are duplicated")
        seen.add(record["evidence_id"])
        record["available"] = _boolean(record["available"], "evidence availability")
        record["bytes"] = _integer(record["bytes"], "evidence byte length", minimum=1)
        record["width"] = _integer(record["width"], "evidence width", minimum=1)
        record["height"] = _integer(record["height"], "evidence height", minimum=1)
        record["sha256"] = _sha256(record["sha256"], "evidence SHA-256")
        expected_path = f"{EVIDENCE_DIRECTORY}/{record['evidence_id']}.jpg" if record["available"] else None
        if record["path"] != expected_path:
            raise MissionBackupError("backup evidence path is unsafe or stale")
        if record["available"] and record["bytes"] > MAX_DECODED_JPEG_BYTES:
            raise MissionBackupError("backup evidence exceeds the JPEG limit")
    return manifest


def _validate_bundle(root: Path, manifest: dict[str, Any]) -> None:
    _validate_bundle_layout(root, manifest)
    database_path = root / DATABASE_NAME
    _verify_file_digest(
        database_path,
        expected_bytes=manifest["database"]["bytes"],
        expected_sha256=manifest["database"]["sha256"],
    )
    connection = _connect_readonly(database_path)
    try:
        _check_database_integrity(connection)
        rows = connection.execute(
            "SELECT evidence_id, sha256, bytes, width, height, available "
            "FROM evidence ORDER BY evidence_id"
        ).fetchall()
    finally:
        connection.close()
    if len(rows) != len(manifest["evidence"]):
        raise MissionBackupError("manifest evidence count does not match the database")
    for row, record in zip(rows, manifest["evidence"], strict=True):
        actual = (
            str(row[0]), str(row[1]), int(row[2]), int(row[3]), int(row[4]),
            _database_boolean(row[5], "evidence availability"),
        )
        expected = (
            record["evidence_id"], record["sha256"], record["bytes"],
            record["width"], record["height"], record["available"],
        )
        if actual != expected:
            raise MissionBackupError("manifest evidence references do not match the database")
        if record["available"]:
            data, _ = _read_plain_file(root / record["path"], max_bytes=MAX_DECODED_JPEG_BYTES)
            _verify_media_record(record, data)


def _validate_bundle_layout(root: Path, manifest: dict[str, Any]) -> None:
    _require_plain_directory(root)
    database = root / DATABASE_NAME
    _require_plain_file(database)
    _require_plain_directory(root / EVIDENCE_DIRECTORY)
    expected_evidence = {
        f"{record['evidence_id']}.jpg"
        for record in manifest["evidence"] if record["available"]
    }
    actual_evidence = set()
    for child in (root / EVIDENCE_DIRECTORY).iterdir():
        info = child.lstat()
        if not stat.S_ISREG(info.st_mode) or child.is_symlink():
            raise MissionBackupError("backup evidence directory contains a non-regular file")
        actual_evidence.add(child.name)
    if actual_evidence != expected_evidence:
        raise MissionBackupError("backup evidence files do not match the manifest")
    expected_root = {MANIFEST_NAME, DATABASE_NAME, EVIDENCE_DIRECTORY}
    if {child.name for child in root.iterdir()} != expected_root:
        raise MissionBackupError("backup directory contains unexpected or stale files")


def _validate_restored_database(
    database_path: Path,
    stage: Path,
    target: Path,
    manifest: dict[str, Any],
) -> None:
    connection = sqlite3.connect(str(database_path))
    try:
        _check_database_integrity(connection)
        rows = connection.execute(
            "SELECT evidence_id, path, bytes, width, height, available "
            "FROM evidence ORDER BY evidence_id"
        ).fetchall()
        for row, record in zip(rows, manifest["evidence"], strict=True):
            evidence_id = _safe_evidence_id(row[0])
            expected_path = str(target / EVIDENCE_DIRECTORY / f"{evidence_id}.jpg")
            if row[1] != expected_path or int(row[2]) != record["bytes"]:
                raise MissionBackupError("restored evidence path or size is inconsistent")
            available = _database_boolean(row[5], "evidence availability")
            if available != record["available"]:
                raise MissionBackupError("restored evidence availability is inconsistent")
            if available:
                media = stage / EVIDENCE_DIRECTORY / f"{evidence_id}.jpg"
                _verify_media_record(record, _read_plain_file(media, max_bytes=MAX_DECODED_JPEG_BYTES)[0])
    finally:
        connection.close()


def _check_database_integrity(connection: sqlite3.Connection) -> None:
    result = connection.execute("PRAGMA integrity_check").fetchone()
    if result is None or result[0] != "ok":
        raise MissionBackupError("backup database failed SQLite integrity_check")
    if connection.execute("PRAGMA foreign_key_check").fetchone() is not None:
        raise MissionBackupError("backup database contains broken foreign-key references")


def _connect_readonly(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def _verify_media_record(record: dict[str, Any], data: bytes) -> None:
    if len(data) != record["bytes"]:
        raise MissionBackupError(f"evidence {record['evidence_id']} byte length does not match")
    if hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise MissionBackupError(f"evidence {record['evidence_id']} SHA-256 does not match")
    try:
        dimensions = _jpeg_dimensions(data)
    except Exception as exc:
        raise MissionBackupError(f"evidence {record['evidence_id']} is not a valid bounded JPEG") from exc
    if dimensions != (record["width"], record["height"]):
        raise MissionBackupError(f"evidence {record['evidence_id']} dimensions do not match")


def _source_evidence_path(root: Path, stored_path: Any) -> Path:
    root_real = root.resolve(strict=True)
    _require_plain_directory(root)
    path = Path(str(stored_path))
    try:
        resolved = path.resolve(strict=True)
        if resolved.parent != root_real:
            raise MissionBackupError("stored evidence path escapes its evidence root")
        _require_plain_file(path)
    except (OSError, ValueError) as exc:
        raise MissionBackupError("available evidence file is missing or unsafe") from exc
    return path


def _copy_verified_file(
    source: Path,
    destination: Path,
    *,
    expected_bytes: int,
    expected_sha256: str,
    max_bytes: int | None = None,
) -> None:
    if max_bytes is not None:
        data, _ = _copy_bounded_file(source, destination, max_bytes=max_bytes)
        actual_bytes = len(data)
        actual_sha256 = hashlib.sha256(data).hexdigest()
    else:
        actual_bytes, actual_sha256 = _copy_streaming_file(source, destination)
    if actual_bytes != expected_bytes or actual_sha256 != expected_sha256:
        raise MissionBackupError(f"file verification failed for {source.name}")


def _copy_bounded_file(
    source: Path,
    destination: Path,
    *,
    max_bytes: int | None,
) -> tuple[bytes, os.stat_result]:
    data, source_stat = _read_plain_file(source, max_bytes=max_bytes)
    fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return data, source_stat


def _copy_streaming_file(source: Path, destination: Path) -> tuple[int, str]:
    _require_plain_file(source)
    source_fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    output_fd = None
    source_stream = None
    output_stream = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise MissionBackupError(f"not a regular file: {source.name}")
        output_fd = os.open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        digest = hashlib.sha256()
        length = 0
        source_stream = os.fdopen(source_fd, "rb")
        source_fd = -1
        output_stream = os.fdopen(output_fd, "wb")
        output_fd = -1
        with source_stream as stream, output_stream as output:
            while chunk := stream.read(_CHUNK_BYTES):
                length += len(chunk)
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
            after = os.fstat(stream.fileno())
        if _fingerprint(before) != _fingerprint(after):
            raise MissionBackupError(f"file changed while being copied: {source.name}")
        return length, digest.hexdigest()
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if source_stream is not None:
            source_stream.close()
        if output_stream is not None:
            output_stream.close()
        if source_fd >= 0:
            os.close(source_fd)
        if output_fd is not None and output_fd >= 0:
            os.close(output_fd)


def _read_plain_file(path: Path, *, max_bytes: int | None = None) -> tuple[bytes, os.stat_result]:
    _require_plain_file(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    stream = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise MissionBackupError(f"not a regular file: {path.name}")
        if max_bytes is not None and before.st_size > max_bytes:
            raise MissionBackupError(f"file exceeds its size limit: {path.name}")
        chunks = []
        length = 0
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            while True:
                chunk = stream.read(_CHUNK_BYTES if max_bytes is None else min(_CHUNK_BYTES, max_bytes + 1 - length))
                if not chunk:
                    break
                length += len(chunk)
                if max_bytes is not None and length > max_bytes:
                    raise MissionBackupError(f"file exceeds its size limit: {path.name}")
                chunks.append(chunk)
            after = os.fstat(stream.fileno())
        if _fingerprint(before) != _fingerprint(after):
            raise MissionBackupError(f"file changed while being read: {path.name}")
        return b"".join(chunks), after
    except BaseException:
        if stream is not None:
            stream.close()
        if fd >= 0:
            os.close(fd)
        raise


def _verify_file_digest(path: Path, *, expected_bytes: int, expected_sha256: str) -> None:
    _require_plain_file(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    stream = None
    digest = hashlib.sha256()
    length = 0
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise MissionBackupError("backup database is not a regular file")
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            while chunk := stream.read(_CHUNK_BYTES):
                length += len(chunk)
                digest.update(chunk)
            after = os.fstat(stream.fileno())
    except BaseException:
        if stream is not None:
            stream.close()
        if fd >= 0:
            os.close(fd)
        raise
    if _fingerprint(before) != _fingerprint(after):
        raise MissionBackupError("backup database changed while being read")
    if length != expected_bytes or digest.hexdigest() != expected_sha256:
        raise MissionBackupError("backup database length or SHA-256 does not match")


def _hash_file(path: Path) -> tuple[int, str]:
    _require_plain_file(path)
    digest = hashlib.sha256()
    length = 0
    with path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_BYTES):
            length += len(chunk)
            digest.update(chunk)
    return length, digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if path.name == MANIFEST_NAME and len(encoded) > MAX_MANIFEST_BYTES:
        raise MissionBackupError("backup manifest exceeds its 16 MiB format limit")
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def _new_destination(destination: str | Path) -> Path:
    candidate = Path(destination).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    if candidate.name in {"", ".", ".."}:
        raise MissionBackupError("destination must name a new directory")
    parent = candidate.parent.resolve(strict=True)
    if not parent.is_dir():
        raise MissionBackupError("destination parent must already be a directory")
    target = parent / candidate.name
    if os.path.lexists(target):
        raise MissionBackupError("destination already exists; refusing replacement")
    return target


def _existing_directory(path: str | Path) -> Path:
    candidate = Path(path).expanduser()
    try:
        _require_plain_directory(candidate)
        return candidate.resolve(strict=True)
    except OSError as exc:
        raise MissionBackupError("backup directory is missing or unsafe") from exc


def _store_roots(store: Any) -> tuple[Path, ...]:
    roots = []
    if str(store.database_path) != ":memory:":
        roots.append(Path(store.database_path).parent.resolve(strict=True))
    roots.append(Path(store.evidence_root).resolve(strict=True))
    return tuple(roots)


def _reject_overlap(target: Path, roots: tuple[Path, ...]) -> None:
    for root in roots:
        try:
            target.relative_to(root)
            raise MissionBackupError("backup/restore directory must be separate from source data")
        except ValueError:
            pass
        try:
            root.relative_to(target)
            raise MissionBackupError("backup/restore directory must not contain source data")
        except ValueError:
            pass


def _new_stage(target: Path, purpose: str) -> Path:
    try:
        return Path(tempfile.mkdtemp(prefix=f".{target.name}.{purpose}-", dir=target.parent))
    except OSError as exc:
        raise MissionBackupError(f"cannot create staging directory: {exc}") from exc


def _publish_no_replace(stage: Path, target: Path) -> None:
    source_bytes = os.fsencode(stage)
    target_bytes = os.fsencode(target)
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        rename_exclusive = getattr(libc, "renamex_np", None)
        if rename_exclusive is None:
            raise MissionBackupError("atomic no-replace directory publication is unavailable")
        rename_exclusive.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(source_bytes, target_bytes, 0x00000004)  # RENAME_EXCL
    elif sys.platform.startswith("linux"):
        rename_exclusive = getattr(libc, "renameat2", None)
        if rename_exclusive is None:
            raise MissionBackupError("atomic no-replace directory publication is unavailable")
        rename_exclusive.argtypes = (
            ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
        )
        rename_exclusive.restype = ctypes.c_int
        result = rename_exclusive(-100, source_bytes, -100, target_bytes, 1)  # AT_FDCWD, RENAME_NOREPLACE
    else:
        raise MissionBackupError("atomic no-replace directory publication is unsupported on this platform")
    if result != 0:
        error = ctypes.get_errno()
        if error == errno.EEXIST:
            raise MissionBackupError("destination already exists; refusing replacement")
        raise OSError(error, os.strerror(error), str(target))


def _require_plain_directory(path: Path) -> None:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise MissionBackupError(f"not a plain directory: {path}")


def _require_plain_file(path: Path) -> None:
    info = path.lstat()
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise MissionBackupError(f"not a plain file: {path}")


def _safe_evidence_id(value: Any) -> str:
    if not isinstance(value, str) or not _EVIDENCE_ID.fullmatch(value):
        raise MissionBackupError("evidence ID cannot be represented as a safe media filename")
    return value


def _integer(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise MissionBackupError(f"invalid {name}")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise MissionBackupError(f"invalid {name}")
    return value


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise MissionBackupError(f"backup manifest contains duplicate key {key!r}")
        result[key] = value
    return result


def _database_boolean(value: Any, name: str) -> bool:
    if isinstance(value, bool) or value not in (0, 1):
        raise MissionBackupError(f"invalid {name}")
    return bool(value)


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise MissionBackupError(f"invalid {name}")
    return value


def _fingerprint(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        int(value.st_dev), int(value.st_ino), int(value.st_size),
        int(value.st_mtime_ns), int(value.st_ctime_ns),
    )


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
