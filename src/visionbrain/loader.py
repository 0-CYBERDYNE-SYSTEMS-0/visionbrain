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
SAM31_HF_ID = "mlx-community--sam3.1-bf16"
SAM31_HF_REPO = "mlx-community/sam3.1-bf16"
GEMMA4_HF_ID = "mlx-community--gemma-4-26b-a4b-it-4bit"
GEMMA4_HF_REPO = "mlx-community/gemma-4-26b-a4b-it-4bit"
FALCON_OCR_HF_ID = "tiiuae--Falcon-OCR"
FALCON_OCR_HF_REPO = "tiiuae/Falcon-OCR"
OLLAMA_GEMMA_MODEL = "gemma4:e2b"
OLLAMA_BASE_URL = "http://localhost:11434"

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

_mlx_error: str = ""


def _check_mlx() -> bool:
    """True if mlx and mlx_vlm can initialize in the current Python env."""
    global _mlx_error
    _mlx_error = ""
    try:
        import mlx.core  # noqa: F401
        import mlx_vlm   # noqa: F401
        return True
    except Exception as exc:
        _mlx_error = str(exc)
        return False


def _mlx_note() -> str:
    """Short explanation when MLX cannot initialize."""
    if _mlx_error:
        return f"mlx/mlx_vlm unavailable: {_mlx_error}"
    return "mlx/mlx_vlm not in Python path"


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
    # Weights, MLX, and the local Falcon-Perception checkout are all required.
    has_weights = cached.exists() and size > 0.1
    has_repo = FALCON_REPO.exists()
    can_load = has_weights and mlx_ok and has_repo
    return ModelRecord(
        hf_id="tiiuae/Falcon-Perception",
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=has_weights,
        can_load=can_load,
        note="3B params, MLX, float16. Ready to use." if can_load
             else ("Weights not cached" if not has_weights else
                   (f"Falcon-Perception repo not found at {FALCON_REPO}" if not has_repo else
                    _mlx_note())),
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
        hf_id="mlx-community/sam3.1-bf16",
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=has_weights,
        can_load=can_load,
        note="Meta SAM 3.1 with Object Multiplex for video tracking."
             if can_load else
             ("Run: huggingface-cli download mlx-community/sam3.1-bf16" if not has_weights else
              _mlx_note()),
    )


def falcon_ocr_record() -> ModelRecord:
    """Status of Falcon-OCR weights (registry-only; no MLX inference path yet)."""
    cached = HF_CACHE / f"models--{FALCON_OCR_HF_ID.replace('/', '--')}"
    size = _cache_size(cached)
    # Registry entry only: Falcon-OCR is served upstream via vLLM/CUDA and has
    # no MLX inference path in VisionBrain, so can_load stays False regardless
    # of MLX availability.
    has_weights = cached.exists() and size > 0.1
    return ModelRecord(
        hf_id=FALCON_OCR_HF_REPO,
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=has_weights,
        can_load=False,
        note="0.3B OCR companion (text, tables, formulas). Upstream serving is "
             "vLLM/CUDA; not wired to VisionBrain MLX inference yet."
             if has_weights else
             "Run: huggingface-cli download tiiuae/Falcon-OCR",
    )


def gemma4_record() -> ModelRecord:
    """Status of Gemma 4 26B weights."""
    cached = HF_CACHE / f"models--{GEMMA4_HF_ID.replace('/', '--')}"
    size = _cache_size(cached)
    mlx_ok = _check_mlx()
    has_weights = cached.exists() and size > 1.0
    can_load = has_weights and mlx_ok
    return ModelRecord(
        hf_id=GEMMA4_HF_REPO,
        cache_dir=cached,
        disk_gb=round(size, 2),
        is_cached=has_weights,
        can_load=can_load,
        note="Google Gemma 4 26B MoE (~3.8B active params), 4-bit quantized."
             if can_load else
             ("Run: huggingface-cli download mlx-community/gemma-4-26b-a4b-it-4bit" if not has_weights else
              _mlx_note()),
    )


def ollama_gemma_record() -> ModelRecord:
    """Status of Ollama gemma4:e2b (7.2 GB, localhost:11434)."""
    import urllib.request, json
    cached = True  # Ollama manages its own model storage
    has_weights = False
    size_gb = 7.2  # nominal size of gemma4:e2b
    try:
        req = urllib.request.Request(
            f"{OLLAMA_BASE_URL}/api/tags",
            headers={"Content-Type": "application/json"},
            method="GET",
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            names = [m.get("name", "") for m in data.get("models", [])]
            has_weights = OLLAMA_GEMMA_MODEL in names or "gemma4:latest" in names
    except Exception:
        pass
    return ModelRecord(
        hf_id=f"{OLLAMA_GEMMA_MODEL} (Ollama)",
        cache_dir=Path("ollama://models"),
        disk_gb=size_gb,
        is_cached=has_weights,
        can_load=has_weights,
        note=f"7.2 GB. Gemma 4 2B MoE via Ollama localhost:11434."
             if has_weights else "gemma4:e2b not loaded in Ollama.",
    )


def all_records() -> list[ModelRecord]:
    return [
        falcon_perception_record(),
        sam31_record(),
        ollama_gemma_record(),
        falcon_ocr_record(),
    ]


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
