"""Model registry and disk cache management.

VisionBrain does NOT download models — it uses what is already cached
on disk. This module reports cache status and resolves local paths.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# Disk locations
# ──────────────────────────────────────────────────────────────────────────────

HF_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
FALCON_HF_ID = "tiiuae--Falcon-Perception"
SAM31_HF_ID = "facebook--sam3.1"

# Path to Falcon-Perception git repo (read-only)
FALCON_REPO = Path.home() / "Falcon-Perception"


# ──────────────────────────────────────────────────────────────────────────────
# Model record
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ModelRecord:
    hf_id: str               # e.g. "tiiuae/Falcon-Perception"
    cache_dir: Path           # local cache path
    disk_gb: float           # size on disk
    is_cached: bool
    can_load: bool            # all runtime deps present
    note: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# Availability checks
# ──────────────────────────────────────────────────────────────────────────────

def _check_mlx() -> bool:
    """True if mlx and mlx_vlm are importable in the current Python env."""
    try:
        import mlx.core  # noqa: F401
        import mlx_vlm   # noqa: F401
        return True
    except ImportError:
        return False


def _cache_size(path: Path) -> float:
    """Return size in GB of a directory tree, or 0.0 if not present."""
    if not path.exists():
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for fn in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, fn))
            except OSError:
                pass
    return total / (1024 ** 3)


def falcon_perception_record() -> ModelRecord:
    """Status of Falcon Perception weights."""
    cached = HF_CACHE / f"models--{FALCON_HF_ID.replace('/', '--')}"
    size = _cache_size(cached)
    mlx_ok = _check_mlx()
    # Weights must be cached AND mlx must be available
    can_load = cached.exists() and size > 0.1 and mlx_ok
    return ModelRecord(
        hf_id="tiiuae/Falcon-Perception",
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=cached.exists() and size > 0.1,
        can_load=can_load,
        note="3B params, MLX, float16. Ready to use." if can_load
             else ("Weights not cached" if not cached.exists() else
                   "mlx/mlx_vlm not in Python path"),
    )


def sam31_record() -> ModelRecord:
    """Status of SAM 3.1 weights."""
    cached = HF_CACHE / f"models--{SAM31_HF_ID.replace('/', '--')}"
    size = _cache_size(cached)
    mlx_ok = _check_mlx()
    # Check for actual weight files (not just the refs/ placeholder)
    has_weights = cached.exists() and size > 0.5
    can_load = has_weights and mlx_ok
    return ModelRecord(
        hf_id="facebook/sam3.1",
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=has_weights,
        can_load=can_load,
        note="Meta SAM 3.1 with Object Multiplex for video tracking."
             if can_load else
             ("Run: huggingface-cli download facebook/sam3.1" if not has_weights else
              "mlx/mlx_vlm not in Python path"),
    )


def all_records() -> list[ModelRecord]:
    return [falcon_perception_record(), sam31_record()]


def print_status() -> None:
    """Print a human-readable model status table."""
    print("\n=== VisionBrain Model Status ===")
    for rec in all_records():
        status = "READY" if rec.can_load else ("CACHED" if rec.is_cached else "MISSING")
        print(f"  [{status}] {rec.hf_id}")
        print(f"         Cache: {rec.cache_dir}")
        print(f"         Size:  {rec.disk_gb} GB")
        print(f"         {rec.note}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# Falcon-Perception repo access
# ──────────────────────────────────────────────────────────────────────────────

def falcon_repo() -> Path:
    """Path to the Falcon-Perception git repo (read-only)."""
    if not FALCON_REPO.exists():
        raise RuntimeError(
            f"Falcon-Perception repo not found at {FALCON_REPO}. "
            "VisionBrain reads from the local git repo."
        )
    return FALCON_REPO


def fp_module_path() -> Path:
    """Path to falcon_perception Python package in the local git repo."""
    return falcon_repo() / "falcon_perception"


# ──────────────────────────────────────────────────────────────────────────────
# SAM 3.1 cache path (for mlx_vlm)
# ──────────────────────────────────────────────────────────────────────────────

def sam31_cache_path() -> Optional[Path]:
    """Path to cached SAM 3.1 weights, or None if not cached."""
    cached = HF_CACHE / f"models--{SAM31_HF_ID.replace('/', '--')}"
    if cached.exists() and _cache_size(cached) > 0.5:
        return cached
    return None


if __name__ == "__main__":
    print_status()
