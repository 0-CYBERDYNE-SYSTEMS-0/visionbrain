"""Raw SAM 3.1 residency and runtime isolation without native MLX."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from visionbrain import live_tracking, sam3_inference, sam3_model_host
from visionbrain.model_host import ModelHost


@pytest.fixture
def sam_consumers(monkeypatch):
    sam3_inference._clear_sam_model_cache()
    live_tracking._clear_loaded()

    host = ModelHost()
    monkeypatch.setattr(sam3_model_host, "HOST", host)
    raw_model = object()
    load_calls = []
    clear_calls = []

    class FakeProcessor:
        def __init__(self, path):
            self.path = path
            self.image_size = 1008

        @classmethod
        def from_pretrained(cls, path):
            return cls(path)

    class FakePredictor:
        def __init__(self, model, processor, score_threshold):
            self.model = model
            self.processor = processor
            self.score_threshold = score_threshold
            self.text_cache = {}

    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.__path__ = []
    models = types.ModuleType("mlx_vlm.models")
    models.__path__ = []
    sam3_package = types.ModuleType("mlx_vlm.models.sam3_1")
    sam3_package.__path__ = []
    utils = types.ModuleType("mlx_vlm.utils")
    utils.get_model_path = lambda model_id: f"/fake/{model_id}"
    utils.load_model = lambda path: load_calls.append(path) or raw_model
    processing = types.ModuleType("mlx_vlm.models.sam3_1.processing_sam3_1")
    processing.Sam31Processor = FakeProcessor
    generate = types.ModuleType("mlx_vlm.models.sam3_1.generate")
    generate.Sam3Predictor = FakePredictor
    mlx_vlm.models = models
    mlx_vlm.utils = utils
    models.sam3_1 = sam3_package
    sam3_package.processing_sam3_1 = processing
    sam3_package.generate = generate

    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    core = types.ModuleType("mlx.core")
    core.clear_cache = lambda: clear_calls.append(True)
    for name, module in {
        "mlx_vlm": mlx_vlm,
        "mlx_vlm.models": models,
        "mlx_vlm.models.sam3_1": sam3_package,
        "mlx_vlm.utils": utils,
        "mlx_vlm.models.sam3_1.processing_sam3_1": processing,
        "mlx_vlm.models.sam3_1.generate": generate,
        "mlx": mlx,
        "mlx.core": core,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(sam3_inference, "sam31_available", lambda: True)
    yield types.SimpleNamespace(
        host=host,
        raw_model=raw_model,
        load_calls=load_calls,
        clear_calls=clear_calls,
    )

    sam3_inference._clear_sam_model_cache()
    live_tracking._clear_loaded()


def test_raw_weights_are_shared_but_processors_and_predictors_are_isolated(sam_consumers):
    model_id = live_tracking.DEFAULT_MODEL
    sam_model, sam_processor, sam_predictor = sam3_inference._ensure_sam31(
        model_id, threshold=0.2, resolution=504
    )
    track_model, track_processor, track_predictor = live_tracking._ensure_loaded(
        model_id, resolution=504, threshold=0.7
    )

    assert sam_model is track_model is sam_consumers.raw_model
    assert len(sam_consumers.load_calls) == 1
    assert sam_processor is not track_processor
    assert sam_predictor is not track_predictor
    assert sam_predictor.text_cache is not track_predictor.text_cache
    assert sam_processor.image_size == track_processor.image_size == 504
    assert sam_predictor.score_threshold == 0.2
    assert track_predictor.score_threshold == 0.7
    key = sam3_model_host._model_key(model_id)
    assert sam_consumers.host._entries[key].refcount == 2

    sam3_inference._ensure_sam31(model_id, threshold=0.4, resolution=1008)
    assert sam_processor.image_size == 1008
    assert sam_predictor.score_threshold == 0.4
    assert track_processor.image_size == 504
    assert track_predictor.score_threshold == 0.7

    first_tracker = live_tracking.LiveSamTracker(
        model=model_id,
        resolution=504,
        threshold=0.2,
        detect_every=2,
        backbone_every=3,
        backbone_fn=lambda *args: None,
        detect_fn=lambda *args: None,
        preprocess_fn=lambda *args: None,
        tracker=object(),
    )
    second_tracker = live_tracking.LiveSamTracker(
        model=model_id,
        resolution=1008,
        threshold=0.7,
        detect_every=4,
        backbone_every=5,
        backbone_fn=lambda *args: None,
        detect_fn=lambda *args: None,
        preprocess_fn=lambda *args: None,
        tracker=object(),
    )
    first_tracker._backbone_cache = object()
    first_tracker._encoder_cache["prompt"] = object()
    assert second_tracker._backbone_cache is None
    assert second_tracker._encoder_cache == {}
    assert first_tracker.detect_every != second_tracker.detect_every
    assert first_tracker.backbone_every != second_tracker.backbone_every


def test_each_consumer_releases_one_ref_and_final_release_clears_residency(sam_consumers):
    model_id = live_tracking.DEFAULT_MODEL
    sam3_inference._ensure_sam31(model_id, threshold=0.2, resolution=504)
    live_tracking._ensure_loaded(model_id, resolution=504, threshold=0.7)
    key = sam3_model_host._model_key(model_id)

    sam3_inference._clear_sam_model_cache()
    assert sam_consumers.host._entries[key].refcount == 1
    assert key in sam_consumers.host.resident()
    assert sam_consumers.clear_calls == []

    live_tracking._clear_loaded()
    assert key not in sam_consumers.host.resident()
    assert sam_consumers.clear_calls == [True]

    sam3_inference._ensure_sam31(model_id, threshold=0.3, resolution=504)
    assert len(sam_consumers.load_calls) == 2


def test_shared_sam_lock_serializes_release_until_use_ends(sam_consumers):
    model_id = live_tracking.DEFAULT_MODEL
    sam3_model_host.acquire_sam31_model(model_id)
    key = sam3_model_host._model_key(model_id)
    entered = threading.Event()
    release = threading.Event()
    release_complete = threading.Event()

    @sam3_model_host.serialized_sam31
    def first_use():
        entered.set()
        assert release.wait(1.0)

    def release_model():
        sam3_model_host.release_sam31_model(model_id)
        release_complete.set()

    first = threading.Thread(target=first_use)
    second = threading.Thread(target=release_model)
    first.start()
    assert entered.wait(1.0)
    second.start()
    try:
        assert not release_complete.wait(0.05)
        assert key in sam_consumers.host.resident()
    finally:
        release.set()
    first.join(timeout=1.0)
    second.join(timeout=1.0)
    assert release_complete.is_set()
    assert key not in sam_consumers.host.resident()
    assert sam_consumers.clear_calls == [True]
