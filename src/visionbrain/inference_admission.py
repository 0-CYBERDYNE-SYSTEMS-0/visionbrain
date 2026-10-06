"""Host-wide inference admission for the inference host (the M2 Mini).

Why this exists
---------------
On the M2, several processes can race heavyweight model loads: the bridge
brain process (``python -m vb_bridge.server --brain ...``), VisionBrain
local live sessions, and CLI/batch jobs. There is no host-wide admission,
so two processes can load competing models into RAM/GPU at once and
thrash. This module provides that admission: at most ONE process on the
host may hold the lock, so heavyweight loads are serialized and contention
is explicit ("busy") rather than silent thrashing.

Mechanism
---------
A filesystem lock (``fcntl.flock``, exclusive) on a lock file:

- default path ``~/.visionbrain/inference.lock`` (parent dir created 0700)
- override with the ``VB_INFERENCE_LOCK`` env var

flock semantics give, for free:

- auto-release on process exit/crash (legitimate recovery after real
  completion or death — no stale-lock problem, no PID-file games)
- same-process reentrancy is NOT supported: a second acquire from the
  same process conflicts (separate open file descriptions) and returns
  None. Hold ONE handle for the whole session.

After acquiring, we truncate-write our pid/reason/timestamp into the file
so ``describe_holder()`` can answer "who holds it?" for busy diagnostics.
Release truncates the file. All filesystem failures (unwritable path,
unflockable file) raise ``AdmissionError`` — fail loudly, never silently
grant.

Intended usage
--------------
Hold the admission across WEIGHT LOADING *and* the live inference session
that uses the weights: acquire before loading SAM/Falcon/LFM/VLM weights,
release only when the session ends or the engine is disarmed. Short
ASK/REPORT calls may take the admission too, but prefer shared scheduling
upstream so chatty short calls do not starve streaming sessions — see
``VB_SPEC.md`` (repo root) for the upstream VisionBrain contract.

Pause/cancel: a pause or cancellation request does not prove native work has
ended. Keep the handle until native work has drained and its owning session
ends; do not release merely because work became idle or paused.
"""

from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Optional, Union

ENV_LOCK_PATH = "VB_INFERENCE_LOCK"
DEFAULT_LOCK_PATH = "~/.visionbrain/inference.lock"

# Poll cadence when try_acquire is given a timeout > 0.
_POLL_INTERVAL_S = 0.05

_MAX_REASON_LEN = 200


class AdmissionError(RuntimeError):
    """Raised when admission cannot be attempted (filesystem failure).

    Acquisition errors must fail loudly; callers should never interpret an
    AdmissionError as "lock granted".
    """


class AdmissionHandle:
    """A held admission grant. Context manager; release is idempotent."""

    def __init__(
        self,
        admission: "InferenceAdmission",
        fd: int,
        reason: str,
        pid: int,
        acquired_at: float,
    ) -> None:
        self.admission = admission
        self.reason = reason
        self.pid = pid
        self.acquired_at = acquired_at
        self._fd = fd
        self._released = False

    @property
    def is_mine(self) -> bool:
        return not self._released and self._fd >= 0

    def release(self) -> None:
        """Release the admission. Idempotent; safe to call twice."""
        if self._released:
            return
        self._released = True
        # Truncate first so busy diagnostics show an empty file the moment
        # the lock is handed back. Best-effort: the unlock below is what
        # actually frees the lock, and process exit would free it anyway.
        try:
            os.ftruncate(self._fd, 0)
        except OSError:
            pass
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            os.close(self._fd)
            self._fd = -1

    def __enter__(self) -> "AdmissionHandle":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class InferenceAdmission:
    """Host-wide, single-holder admission via an exclusive flock."""

    def __init__(self, lock_path: Union[str, Path, None] = None) -> None:
        if lock_path is not None:
            raw: Union[str, Path] = lock_path
        else:
            env = os.environ.get(ENV_LOCK_PATH)
            raw = env if env else DEFAULT_LOCK_PATH
        self.path = Path(raw).expanduser()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        except OSError as exc:
            raise AdmissionError(
                f"cannot create admission lock directory {self.path.parent}: {exc}"
            ) from exc

    def try_acquire(
        self, reason: str, timeout: float = 0.0
    ) -> Optional[AdmissionHandle]:
        """Try to take the host-wide admission for up to ``timeout`` seconds.

        timeout=0 (default) is a single non-blocking attempt. Returns an
        :class:`AdmissionHandle` on success, or ``None`` when the host is
        busy (callers surface busy; they do NOT poll in a tight loop — see
        VB_SPEC.md). Filesystem failures raise :class:`AdmissionError`.
        """
        fd = self._open_lock_file()
        deadline = time.monotonic() + max(0.0, float(timeout))
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return None
                time.sleep(min(_POLL_INTERVAL_S, max(0.0, deadline - time.monotonic())))
                continue
            except OSError as exc:
                os.close(fd)
                raise AdmissionError(f"cannot flock {self.path}: {exc}") from exc
            break
        acquired_at = time.time()
        try:
            self._write_holder(fd, reason=reason, pid=os.getpid(), acquired_at=acquired_at)
        except AdmissionError:
            # We hold the flock but could not record the holder. Release the
            # lock before propagating — otherwise this process keeps blocking
            # every other acquirer with no handle to release it.
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)
            raise
        return AdmissionHandle(
            self, fd=fd, reason=reason, pid=os.getpid(), acquired_at=acquired_at
        )

    def describe_holder(self) -> str:
        """Best-effort human-readable holder info from the lock file.

        Tolerates missing, empty, or unreadable files (returns a note
        instead of raising) — busy diagnostics must never crash a caller.
        """
        try:
            with open(self.path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
        except FileNotFoundError:
            return "<no holder info: lock file absent>"
        except OSError as exc:
            return f"<no holder info: {exc}>"
        return text if text else "<no holder info: lock file empty>"

    # -- internals ---------------------------------------------------------

    def _open_lock_file(self) -> int:
        try:
            return os.open(str(self.path), os.O_RDWR | os.O_CREAT, 0o600)
        except OSError as exc:
            raise AdmissionError(f"cannot open lock file {self.path}: {exc}") from exc

    def _write_holder(self, fd: int, reason: str, pid: int, acquired_at: float) -> None:
        clean = " ".join(str(reason).split())[:_MAX_REASON_LEN] or "unspecified"
        payload = json.dumps(
            {"pid": pid, "reason": clean, "acquired_at": acquired_at},
        ).encode("utf-8")
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, payload)
        except OSError as exc:  # we hold the lock but cannot record it
            raise AdmissionError(
                f"acquired {self.path} but could not write holder record: {exc}"
            ) from exc
