"""Service hardening primitives: shared-token auth and a job-slot queue.

Pure asyncio/stdlib — no fastapi or MLX imports — so the module is
importable anywhere (tests, CLI, field bridges), mirroring the
detection_core philosophy.

Environment variables:
    VB_TOKEN      shared access token; when set, /api/* (minus /api/healthz)
                  requires it via the X-Auth-Token header or ?token= query
                  parameter. Unset/empty disables auth entirely (default).
    VB_MAX_JOBS   maximum concurrent heavy subprocess jobs, clamped to 1..4
                  (default 1).
"""

from __future__ import annotations

import asyncio
import collections
import hmac
import os
from typing import Deque, Optional, Tuple

MAX_JOBS_DEFAULT = 1
MAX_JOBS_MIN = 1
MAX_JOBS_MAX = 4


def token_enabled() -> bool:
    """Return True when VB_TOKEN is set and non-empty (read at call time)."""
    return bool(os.environ.get("VB_TOKEN"))


def check_token(provided: Optional[str]) -> bool:
    """Constant-time compare of *provided* against the VB_TOKEN env var.

    Returns False when the token feature is not enabled or *provided* is
    None/empty; otherwise True only on an exact match.
    """
    expected = os.environ.get("VB_TOKEN") or ""
    if not expected or not provided:
        return False
    # hmac.compare_digest raises TypeError on non-ASCII str, so compare bytes.
    return hmac.compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def max_jobs() -> int:
    """Parse VB_MAX_JOBS, clamped to 1..4; invalid or unset gives the default."""
    raw = os.environ.get("VB_MAX_JOBS")
    if not raw:
        return MAX_JOBS_DEFAULT
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return MAX_JOBS_DEFAULT
    return max(MAX_JOBS_MIN, min(MAX_JOBS_MAX, value))


class JobQueue:
    """asyncio FIFO slot limiter for memory-hungry subprocess jobs.

    Waiters are kept in a collections.deque of futures; a release grants the
    freed slot to the head-of-line waiter, so no busy waiting and no third
    party dependencies. A waiter whose task is cancelled while queued is
    skipped cleanly and never consumes a slot.
    """

    def __init__(self, max_concurrent: int) -> None:
        self._capacity = max(1, int(max_concurrent))
        self._active: set[str] = set()
        self._waiters: Deque[Tuple[str, "asyncio.Future[None]"]] = collections.deque()

    @property
    def queued_count(self) -> int:
        """Number of jobs currently waiting for a slot (excludes cancelled)."""
        return sum(1 for _key, fut in self._waiters if not fut.done())

    def wait_position(self, key: str) -> int:
        """Live 1-based line spot for *key*; 0 when running or not present.

        Unlike the submit-time position acquire() returns, this stays
        truthful as waiters ahead of *key* are granted or cancelled.
        """
        if key in self._active:
            return 0
        for spot, (waiter, fut) in enumerate(self._waiters, start=1):
            if waiter == key and not fut.done():
                return spot
        return 0

    async def acquire(self, key: str) -> int:
        """Wait for a free slot; return this job's submit-time position.

        0 means the job started immediately, 1 means first in line, and so
        on. FIFO order is preserved.
        """
        if len(self._active) < self._capacity:
            self._active.add(key)
            return 0
        fut: "asyncio.Future[None]" = asyncio.get_running_loop().create_future()
        entry = (key, fut)
        self._waiters.append(entry)
        position = len(self._waiters)
        try:
            await fut
        except asyncio.CancelledError:
            try:
                self._waiters.remove(entry)
            except ValueError:
                pass  # already granted and popped by _grant()
            if key in self._active:
                # Granted in a race with cancellation — hand the slot back.
                self.release(key)
            raise
        return position

    def release(self, key: str) -> None:
        """Free the slot held by *key* and grant it to the next waiter."""
        self._active.discard(key)
        self._grant()

    def _grant(self) -> None:
        """Hand free slots to head-of-line waiters, skipping cancelled ones."""
        while len(self._active) < self._capacity and self._waiters:
            key, fut = self._waiters.popleft()
            if fut.done():
                continue  # cancelled while queued — skip cleanly
            self._active.add(key)
            fut.set_result(None)
