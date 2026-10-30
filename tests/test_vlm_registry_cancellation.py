"""Canonical VLM cancellation hooks with fake MLX modules only."""

from __future__ import annotations

import sys
import threading
import types

import pytest

from visionbrain import vlm_registry


class FakeTokenizer:
    def __init__(self):
        self.stopping_criteria = lambda token: token == "eos"


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()


def install_fake_generation(monkeypatch, generate):
    mlx_vlm = types.ModuleType("mlx_vlm")
    mlx_vlm.__path__ = []
    generate_module = types.ModuleType("mlx_vlm.generate")
    generate_module.generate = generate
    prompt_module = types.ModuleType("mlx_vlm.prompt_utils")
    prompt_module.apply_chat_template = lambda processor, config, text, num_images: text
    mlx = types.ModuleType("mlx")
    mlx.__path__ = []
    core = types.ModuleType("mlx.core")
    core.synchronize = lambda: None
    monkeypatch.setitem(sys.modules, "mlx_vlm", mlx_vlm)
    monkeypatch.setitem(sys.modules, "mlx_vlm.generate", generate_module)
    monkeypatch.setitem(sys.modules, "mlx_vlm.prompt_utils", prompt_module)
    monkeypatch.setitem(sys.modules, "mlx", mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", core)


def test_generate_cancellation_composes_eos_restores_and_keeps_next_call_usable(monkeypatch):
    processor = FakeProcessor()
    eos = processor.tokenizer.stopping_criteria
    event = threading.Event()
    seen_criteria = []
    sync_calls = []

    def generate(_model, _processor, _prompt, **kwargs):
        criteria = kwargs["stopping_criteria"]
        seen_criteria.append(criteria)
        processor.tokenizer.stopping_criteria = criteria
        assert criteria("ordinary") is False
        assert criteria("eos") is True  # EOS still stops without cancellation.
        event.set()
        assert processor.tokenizer.stopping_criteria("cancel") is True
        # Cancellation stops at this token boundary.
        return types.SimpleNamespace(text="partial")

    install_fake_generation(monkeypatch, generate)
    monkeypatch.setattr(vlm_registry, "_ensure_loaded", lambda: (object(), processor, object()))
    monkeypatch.setattr(sys.modules["mlx.core"], "synchronize", lambda: sync_calls.append(True))

    with pytest.raises(InterruptedError, match="cancelled"):
        vlm_registry._generate("question", object(), 8, event)

    assert processor.tokenizer.stopping_criteria is eos
    assert sync_calls == [True]

    # A later uncancelled call uses mlx_vlm's normal EOS path and sees the
    # restored tokenizer state; injected legacy hooks are exercised in bridge tests.
    def normal_generate(_model, _processor, _prompt, **kwargs):
        assert "stopping_criteria" not in kwargs
        assert processor.tokenizer.stopping_criteria is eos
        return types.SimpleNamespace(text="  complete  ")

    install_fake_generation(monkeypatch, normal_generate)
    assert vlm_registry._generate("question", object(), 8) == "complete"
    assert processor.tokenizer.stopping_criteria is eos
    assert len(seen_criteria) == 1


def test_cancelled_before_load_skips_model_and_generation(monkeypatch):
    event = threading.Event()
    event.set()
    loads = []
    monkeypatch.setattr(vlm_registry, "_ensure_loaded", lambda: loads.append(True))

    with pytest.raises(InterruptedError, match="cancelled"):
        vlm_registry._generate("question", object(), 8, event)

    assert loads == []


def test_ask_and_report_forward_cancellation_event(monkeypatch):
    calls = []

    def fake_generate(user_text, image, max_tokens, cancel_event=None):
        calls.append((user_text, image, max_tokens, cancel_event))
        return "done"

    monkeypatch.setattr(vlm_registry, "_generate", fake_generate)
    event = threading.Event()
    image = object()

    assert vlm_registry.ask("where?", image=image, cancel_event=event) == "done"
    assert vlm_registry.generate_report(
        "1x cow", image=image, cancel_event=event
    ) == "done"
    assert [call[3] for call in calls] == [event, event]


def test_generate_exception_restores_eos_after_synchronize(monkeypatch):
    processor = FakeProcessor()
    eos = processor.tokenizer.stopping_criteria
    event = threading.Event()
    calls = []

    def fail_generate(_model, _processor, _prompt, **kwargs):
        processor.tokenizer.stopping_criteria = kwargs["stopping_criteria"]
        assert kwargs["stopping_criteria"]("eos") is True
        raise RuntimeError("generation failed")

    install_fake_generation(monkeypatch, fail_generate)
    monkeypatch.setattr(vlm_registry, "_ensure_loaded", lambda: (object(), processor, object()))
    monkeypatch.setattr(sys.modules["mlx.core"], "synchronize", lambda: calls.append("sync"))

    with pytest.raises(RuntimeError, match="generation failed"):
        vlm_registry._generate("question", object(), 8, event)

    assert calls == ["sync"]
    assert processor.tokenizer.stopping_criteria is eos
