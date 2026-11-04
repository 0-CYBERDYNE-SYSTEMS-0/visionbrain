"""Threshold provider seam for mission SAM detection; no model is loaded."""

import sys
import types
from types import SimpleNamespace

import pytest

from visionbrain import mission_models
from visionbrain.mission_contracts import ToolContext, ToolRequest
from visionbrain.mission_models import VisionBrainPlanner, VisionBrainTools, build_local_tools


class _Lock:
    def __init__(self):
        self.held = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *exc):
        self.held = False
        return False


def _request():
    return ToolRequest(
        tool="detect_objects",
        arguments={"targets": ["cooler"]},
        mission_id="mission-1",
        cycle_id="cycle-1",
        execution_generation=1,
        reasoning_model="gemma",
        input_evidence_id="evidence-1",
        input_sha256="0" * 64,
        frame_id=1,
        source_binding=None,
    )


def _context():
    return ToolContext(b"jpeg", 32, 24, {}, (), None)


@pytest.fixture
def sam(monkeypatch):
    calls = []
    lock = _Lock()

    def detect_multi(image, prompts, **kwargs):
        calls.append({"prompts": list(prompts), "kwargs": dict(kwargs), "lock_held": lock.held})
        return []

    module = types.ModuleType("visionbrain.sam3_inference")
    module.detect_multi = detect_multi
    monkeypatch.setitem(sys.modules, "visionbrain.sam3_inference", module)
    monkeypatch.setattr(mission_models, "_sam31_available", lambda: True)
    monkeypatch.setattr(
        mission_models,
        "_open_original_jpeg",
        lambda jpeg, width, height: SimpleNamespace(width=width, height=height),
    )
    return SimpleNamespace(calls=calls, lock=lock)


def test_absent_provider_keeps_the_existing_detection_call(sam):
    tools = VisionBrainTools(sam.lock, planner=VisionBrainPlanner(sam.lock))
    result = tools.execute(_request(), _context())
    assert result.status == "empty"
    assert "threshold" not in sam.calls[0]["kwargs"]
    assert sam.calls[0]["lock_held"] is True


def test_provider_is_read_before_the_gpu_lock_and_its_value_is_passed_unchanged(sam):
    lock_states_at_read = []

    def provider():
        lock_states_at_read.append(sam.lock.held)
        return 0.9

    tools = VisionBrainTools(sam.lock, planner=VisionBrainPlanner(sam.lock), threshold_provider=provider)
    tools.execute(_request(), _context())
    assert lock_states_at_read == [False]
    assert sam.calls[0]["kwargs"]["threshold"] == 0.9


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0.0, -0.1, 1.5, True, "0.3", None])
def test_invalid_provider_values_fail_closed_without_running_detection(sam, value):
    tools = VisionBrainTools(sam.lock, planner=VisionBrainPlanner(sam.lock), threshold_provider=lambda: value)
    result = tools.execute(_request(), _context())
    assert result.status == "failed"
    assert result.error_code == "threshold_invalid"
    assert sam.calls == []


def test_provider_exceptions_are_reported_as_native_failures(sam):
    def broken():
        raise RuntimeError("threshold source unavailable")

    tools = VisionBrainTools(sam.lock, planner=VisionBrainPlanner(sam.lock), threshold_provider=broken)
    result = tools.execute(_request(), _context())
    assert result.status == "failed"
    assert result.error_code == "native_inference_failed"
    assert sam.calls == []


def test_local_tools_forward_the_provider_to_the_dispatcher(sam):
    provider = lambda: 0.4
    tools = build_local_tools(VisionBrainPlanner(sam.lock), gpu_lock=sam.lock, threshold_provider=provider)
    assert tools.threshold_provider is provider
    tools.execute(_request(), _context())
    assert sam.calls[0]["kwargs"]["threshold"] == 0.4
