"""Shared raw SAM 3.1 weights and serialized access for canonical consumers."""

from __future__ import annotations

import threading
from functools import wraps

from .loader import SAM31_HF_REPO
from .model_host import HOST

_KEY_PREFIX = "visionbrain.sam31.raw"
_SAM31_LOCK = threading.RLock()


def serialized_sam31(fn):
    """Serialize SAM 3.1 load, use, and release across inference modules."""

    @wraps(fn)
    def _serialized(*args, **kwargs):
        with _SAM31_LOCK:
            return fn(*args, **kwargs)

    return _serialized


def _model_key(model_id: str) -> tuple[str, str]:
    return _KEY_PREFIX, str(model_id)


def _load_sam31_weights(model_id: str):
    from mlx_vlm.utils import get_model_path, load_model

    return load_model(get_model_path(model_id))


@serialized_sam31
def acquire_sam31_model(model_id: str = SAM31_HF_REPO):
    """Acquire only the shared raw model; callers own processor/predictor state."""
    return HOST.acquire(
        _model_key(model_id), lambda: _load_sam31_weights(str(model_id))
    )


@serialized_sam31
def release_sam31_model(model_id: str = SAM31_HF_REPO) -> None:
    """Release one module-level raw-model hold after serialized use ends."""
    HOST.release(_model_key(model_id))
