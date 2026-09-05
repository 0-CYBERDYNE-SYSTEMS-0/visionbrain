"""Compatibility shims for the pinned mlx_vlm build.

mlx_vlm 0.4.4's Gemma 4 arch (``mlx_vlm/models/gemma4/language.py``) defines
``per_layer_model_projection`` as a local ``ScaledLinear`` — a plain
``nn.Module`` computing ``(x @ weight.T) * scalar``. Quantized checkpoints
(mlx-community/gemma-4-e2b-it-4bit) store that layer as a *quantized* linear
(``weight`` + ``scales`` + ``biases``), but the quantization predicate in
``mlx_vlm.utils.load_model`` only quantizes modules that expose a
``to_quantized()`` method, which ``ScaledLinear`` does not. ``load_weights``
then sees two parameters the model does not declare and the load dies in well
under a second with::

    ValueError: Received 2 parameters not in model:
    language_model.model.per_layer_model_projection.biases,
    language_model.model.per_layer_model_projection.scales.

``ensure_scaled_linear_quantization`` teaches the installed arch's
``ScaledLinear`` to quantize itself into a ``QuantizedScaledLinear`` (a
``nn.QuantizedLinear`` that re-applies the scalar after the quantized matmul).
The standard predicate then picks the layer up and loads exactly the stored
tensors, bit-for-bit as the checkpoint ships them. Idempotent, thread-safe,
and a permanent no-op once mlx_vlm ships equivalent support.

LiquidAI's LFM2.5-VL MLX checkpoints (450M and 3B alike) are internally
inconsistent: their ``config.json`` says ``projector_use_layernorm: false``
while the safetensors ship ``multi_modal_projector.layer_norm.{weight,bias}``.
mlx_vlm's arch therefore builds the projector with ``nn.Identity`` and
``load_weights`` rejects the two unexpected tensors (ValueError, again well
under a second). ``ensure_lfm_projector_layernorm`` wraps
``mlx_vlm.utils.load_config`` so a lfm2_vl config is corrected to
``projector_use_layernorm: true`` *only when the checkpoint's own weight
index proves the layernorm exists* — never guessed, so a genuinely
layernorm-free checkpoint is untouched.
"""

from __future__ import annotations

import functools
import json
import logging
import threading
from pathlib import Path

log = logging.getLogger("visionbrain.mlx_compat")

_lock = threading.Lock()
_applied = False
_kv_applied = False
_lfm_applied = False


def ensure_scaled_linear_quantization() -> None:
    """Add quantization support to mlx_vlm's gemma4 ``ScaledLinear`` (idempotent)."""
    global _applied
    with _lock:
        if _applied:
            return
        try:
            from mlx.nn import QuantizedLinear
            from mlx_vlm.models.gemma4.language import ScaledLinear
        except Exception:
            # mlx_vlm/arch not present in this environment; nothing to patch.
            _applied = True
            return

        if hasattr(ScaledLinear, "to_quantized"):
            _applied = True  # upstream already handles quantized ScaledLinear
            return

        class QuantizedScaledLinear(QuantizedLinear):
            """``QuantizedLinear`` that re-applies ``ScaledLinear``'s scalar."""

            def __call__(self, x):  # noqa: D102 - mirrors QuantizedLinear
                return super().__call__(x) * self.scalar

        def _to_quantized(
            self,
            group_size=None,
            bits=None,
            mode: str = "affine",
            quantize_input: bool = False,
        ):
            out_features, in_features = self.weight.shape
            ql = QuantizedScaledLinear(
                in_features,
                out_features,
                bias=False,
                group_size=group_size,
                bits=bits,
                mode=mode,
            )
            ql.scalar = self.scalar
            return ql

        ScaledLinear.to_quantized = _to_quantized
        _applied = True
        log.info(
            "mlx_compat: quantization support added to mlx_vlm gemma4 ScaledLinear"
        )


def ensure_lfm_projector_layernorm() -> None:
    """Fix lfm2_vl configs whose weights carry a projector layernorm they deny."""
    global _lfm_applied
    with _lock:
        if _lfm_applied:
            return
        try:
            import mlx_vlm.utils as vlm_utils
        except Exception:
            _lfm_applied = True
            return

        if getattr(vlm_utils.load_config, "_vb_lfm_patched", False):
            _lfm_applied = True
            return

        original = vlm_utils.load_config

        @functools.wraps(original)
        def _load_config_patched(model_path, **kwargs):
            config = original(model_path, **kwargs)
            try:
                if (
                    isinstance(config, dict)
                    and config.get("model_type") in ("lfm2_vl", "lfm2-vl")
                    and config.get("projector_use_layernorm") is not True
                ):
                    index = Path(model_path) / "model.safetensors.index.json"
                    if index.exists():
                        weight_map = json.loads(index.read_text()).get("weight_map", {})
                        if "multi_modal_projector.layer_norm.weight" in weight_map:
                            log.info(
                                "mlx_compat: checkpoint ships "
                                "multi_modal_projector.layer_norm; forcing "
                                "projector_use_layernorm=true"
                            )
                            config["projector_use_layernorm"] = True
            except Exception:  # never break a load over the correction
                log.exception("mlx_compat: lfm2_vl layernorm correction failed")
            return config

        _load_config_patched._vb_lfm_patched = True
        vlm_utils.load_config = _load_config_patched
        _lfm_applied = True
        log.info("mlx_compat: lfm2_vl projector layernorm config guard installed")


def apply_all() -> None:
    """Apply every shim; safe to call before any mlx_vlm checkpoint load."""
    ensure_scaled_linear_quantization()
    ensure_lfm_projector_layernorm()
    ensure_gemma4_kv_shared_attention()


def ensure_gemma4_kv_shared_attention() -> None:
    """Drop the dead k/v projections mlx_vlm's gemma4 builds for KV-shared layers.

    Gemma 4 reuses one layer's K/V for the trailing ``num_kv_shared_layers``
    (this checkpoint: layers 15-34 reuse layer 14). The HF checkpoint correctly
    ships no ``self_attn.k_norm/k_proj/v_proj`` for those layers — the forward
    pass never touches them when a cache is present — but the arch's
    ``Attention.__init__`` still instantiates them, so ``load_weights`` dies
    with "Missing 60 parameters" (this error was previously masked by the
    ScaledLinear extras error, which load_weights reports first). The arch's
    own convention for absent modules is ``self.module = None``; do the same.
    """
    global _kv_applied
    with _lock:
        if _kv_applied:
            return
        try:
            from mlx_vlm.models.gemma4.language import Attention
        except Exception:
            _kv_applied = True
            return

        if getattr(Attention, "_vb_kv_shared_patched", False):
            _kv_applied = True
            return

        original_init = Attention.__init__

        def _patched_init(self, config, layer_idx: int = 0, *args, **kwargs):
            original_init(self, config, layer_idx, *args, **kwargs)
            if getattr(self, "is_kv_shared_layer", False):
                self.k_norm = None
                self.k_proj = None
                self.v_proj = None

        Attention.__init__ = _patched_init
        Attention._vb_kv_shared_patched = True
        _kv_applied = True
        log.info(
            "mlx_compat: gemma4 KV-shared layers no longer allocate dead k/v modules"
        )
