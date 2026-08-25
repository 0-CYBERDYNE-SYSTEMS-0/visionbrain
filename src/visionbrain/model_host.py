"""Shared MLX checkpoint residency host.

Refcounted cache so callers that can name the same weights hold ONE copy.
Born in the field bridge, where the LFM grounding engine and the ask/report
VLM could both point at the identical checkpoint but each kept a private
module-level cache — two resident copies of one model on a 16GB machine.

ModelHost is payload-agnostic: ``acquire()`` takes a key and a zero-argument
loader and returns whatever the loader returns unchanged. Its job is only
"is this key already loaded" and "how many holders still want it".

acquire()/release() must stay paired per logical hold, not per inference:
acquire once when a caller starts needing a checkpoint resident; a later
release() drops one reference and the entry is only actually freed (and MLX's
cache cleared) once every holder has released.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Hashable

log = logging.getLogger("visionbrain.model_host")


class _Entry:
    __slots__ = ("payload", "refcount")

    def __init__(self, payload: Any) -> None:
        self.payload = payload
        self.refcount = 0


class ModelHost:
    """Refcounted, thread-safe checkpoint residency cache."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[Hashable, _Entry] = {}

    def acquire(self, key: Hashable, loader: Callable[[], Any]) -> Any:
        """Return key's payload, loading it on first request. Bumps refcount."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                log.info("loading %s...", key)
                entry = _Entry(loader())
                self._entries[key] = entry
                log.info("%s loaded", key)
            entry.refcount += 1
            return entry.payload

    def release(self, key: Hashable) -> None:
        """Drop one reference; free and clear MLX's cache once it hits zero."""
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return
            entry.refcount -= 1
            if entry.refcount > 0:
                return
            del self._entries[key]
            log.info("evicted %s", key)
            _clear_mlx_cache()

    def resident(self) -> list[Hashable]:
        """Checkpoint keys currently loaded — for status/debugging."""
        with self._lock:
            return list(self._entries.keys())


def _clear_mlx_cache() -> None:
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass


HOST = ModelHost()
