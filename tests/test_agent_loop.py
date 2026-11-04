"""Tests for visionbrain.agent_loop — response/tool-call contract.

Covers both tool-call transports (native OpenAI-style tool_calls and the
textual <tool>...</tool> protocol), legacy str-returning clients, malformed
native JSON arguments, and VLMClient.chat response mapping.

Run with:
    .venv/bin/python -m pytest tests/test_agent_loop.py -v

No network, no model weights, no real openai client: VLM backends are faked
(scripts of ChatResponse/str per turn) and the tool implementations used
inside agent_loop are monkeypatched on the ``visionbrain.agent_loop`` module.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from types import SimpleNamespace

# Ensure visionbrain is importable
VBRAIN = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(VBRAIN))

import pytest
from PIL import Image

from visionbrain import agent_loop
from visionbrain.agent_loop import (
    AGENT_TOOLS,
    ChatResponse,
    VLMClient,
    _normalize_chat_response,
    run_agent,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fakes
# ──────────────────────────────────────────────────────────────────────────────

def _make_image() -> Image.Image:
    """Small RGB image for the loop (no inference runs against it)."""
    return Image.new("RGB", (64, 64), color=(120, 130, 140))


def _fake_mask(mask_id: int = 1) -> dict:
    """Mask metadata dict in the shape run_ground_expression returns.

    Keys match what _mask_dict_to_result consumes: id, centroid_norm,
    bbox_norm, area_fraction, image_region, rle.
    """
    return {
        "id": mask_id,
        "area_fraction": 0.12,
        "centroid_norm": {"x": 0.5, "y": 0.5},
        "bbox_norm": {"x1": 0.4, "y1": 0.4, "x2": 0.6, "y2": 0.6},
        "image_region": "center",
        "rle": [64, 0],
    }


class ScriptedVLM:
    """Duck-typed client popping one scripted response per chat() call.

    Responses may be ChatResponse instances or plain strings (legacy).
    """

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[list[dict]] = []

    def chat(self, messages, tools=None):
        self.calls.append([dict(m) for m in messages])
        assert tools == AGENT_TOOLS, "run_agent must pass AGENT_TOOLS"
        if not self.responses:
            raise AssertionError("ScriptedVLM ran out of scripted responses")
        return self.responses.pop(0)


@pytest.fixture
def fake_tools(monkeypatch):
    """Patch the tool implementations inside agent_loop's namespace."""
    calls: dict[str, list] = {
        "ground_expression": [],
        "get_crop": [],
        "compute_relations": [],
    }

    def fake_ground_expression(image, expression, **kwargs):
        calls["ground_expression"].append(expression)
        return {1: _fake_mask(1)}

    def fake_render_som(image, masks):
        return image.copy()

    def fake_get_crop(image, mask, pad=0.05):
        calls["get_crop"].append(getattr(mask, "mask_id", None))
        return image.copy()

    def fake_compute_relations(masks, mask_ids):
        calls["compute_relations"].append(list(mask_ids))
        return {"pairs": []}

    monkeypatch.setattr(agent_loop, "run_ground_expression", fake_ground_expression)
    monkeypatch.setattr(agent_loop, "render_som", fake_render_som)
    monkeypatch.setattr(agent_loop, "get_crop", fake_get_crop)
    monkeypatch.setattr(agent_loop, "compute_relations", fake_compute_relations)
    return calls


def _history_texts(history: list[dict], role: str) -> list[str]:
    """Concatenated text parts of all history messages with the given role."""
    out: list[str] = []
    for msg in history:
        if msg.get("role") != role:
            continue
        content = msg.get("content")
        if isinstance(content, str):
            out.append(content)
        elif isinstance(content, list):
            out.extend(
                part.get("text", "")
                for part in content
                if isinstance(part, dict)
            )
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Native tool_calls path
# ──────────────────────────────────────────────────────────────────────────────

class TestNativeToolCallsPath:
    def test_native_tool_calls_drive_loop(self, fake_tools):
        """ground_expression + answer arrive as native tool_calls, no <tool> tag."""
        client = ScriptedVLM(
            ChatResponse(
                content="Let me segment the scene.",
                tool_calls=[
                    {"name": "ground_expression", "parameters": {"expression": "person"}}
                ],
            ),
            ChatResponse(
                tool_calls=[
                    {
                        "name": "answer",
                        "parameters": {
                            "response": "One person near the center",
                            "supporting_mask_ids": [1],
                        },
                    }
                ]
            ),
        )

        result = run_agent(_make_image(), "How many people?", client)

        assert result.answer == "One person near the center"
        assert result.supporting_mask_ids == [1]
        assert result.n_fp_calls == 1
        assert result.n_vlm_calls == 2
        assert result.final_image is not None
        assert fake_tools["ground_expression"] == ["person"]
        # No textual <tool> tag was ever needed.
        for msg in result.history:
            if msg.get("role") == "assistant":
                for part in msg["content"]:
                    assert "<tool>" not in part.get("text", "")

    def test_native_history_note_keeps_turn_interpretable(self, fake_tools):
        """The appended assistant message carries content plus a compact JSON note."""
        client = ScriptedVLM(
            ChatResponse(
                content="Let me segment the scene.",
                tool_calls=[
                    {"name": "ground_expression", "parameters": {"expression": "person"}}
                ],
            ),
            ChatResponse(
                tool_calls=[{"name": "answer", "parameters": {"response": "done"}}]
            ),
        )

        result = run_agent(_make_image(), "How many people?", client)

        assistant_texts = _history_texts(result.history, "assistant")
        assert any("Let me segment the scene." in t for t in assistant_texts)
        note = [t for t in assistant_texts if "requested_tool_call" in t]
        assert note, "native tool call must be noted in assistant history"
        assert '"name":"ground_expression"' in note[0].replace(" ", "")
        assert '"expression":"person"' in note[0].replace(" ", "")

    def test_native_takes_precedence_over_textual_tag(self, fake_tools):
        """If both are present, the FIRST native call wins over the <tool> tag."""
        client = ScriptedVLM(
            ChatResponse(
                content='<tool>{"name": "answer", "parameters": {"response": "wrong"}}</tool>',
                tool_calls=[
                    {"name": "ground_expression", "parameters": {"expression": "person"}}
                ],
            ),
            ChatResponse(
                tool_calls=[{"name": "answer", "parameters": {"response": "right"}}]
            ),
        )

        result = run_agent(_make_image(), "How many people?", client)

        assert result.answer == "right"
        assert fake_tools["ground_expression"] == ["person"]
        assert result.n_fp_calls == 1


# ──────────────────────────────────────────────────────────────────────────────
# Textual <tool> tag path
# ──────────────────────────────────────────────────────────────────────────────

class TestTextualToolPath:
    def test_textual_protocol_still_works(self, fake_tools):
        """Plain content strings with <tool> JSON blocks keep driving the loop."""
        client = ScriptedVLM(
            '<tool>{"name": "ground_expression", '
            '"parameters": {"expression": "truck"}}</tool>',
            'Looking at the masks… <tool>{"name": "answer", '
            '"parameters": {"response": "Two trucks", "supporting_mask_ids": [1]}}</tool>',
        )

        result = run_agent(_make_image(), "How many trucks?", client)

        assert result.answer == "Two trucks"
        assert result.supporting_mask_ids == [1]
        assert result.n_fp_calls == 1
        assert result.n_vlm_calls == 2
        assert fake_tools["ground_expression"] == ["truck"]

    def test_textual_path_with_chat_response_wrapper(self, fake_tools):
        """A ChatResponse with empty tool_calls falls back to the textual tag."""
        client = ScriptedVLM(
            ChatResponse(
                content='<tool>{"name": "answer", "parameters": {"response": "clear"}}</tool>'
            ),
        )

        result = run_agent(_make_image(), "Status?", client)

        assert result.answer == "clear"
        assert result.n_fp_calls == 0


# ──────────────────────────────────────────────────────────────────────────────
# Legacy str-returning VLMClient subclass
# ──────────────────────────────────────────────────────────────────────────────

class TestLegacyStrSubclass:
    def test_str_returning_subclass_still_works(self, fake_tools):
        """Subclasses overriding chat() to return str keep working."""

        class LegacyStrClient(VLMClient):
            def __init__(self):  # skip VLMClient.__init__ (no openai import)
                self.model = "legacy"

            def chat(self, messages, tools=None):
                if "ground_expression" not in "".join(_history_texts(messages, "user")):
                    return (
                        '<tool>{"name": "ground_expression", '
                        '"parameters": {"expression": "roof"}}</tool>'
                    )
                return (
                    '<tool>{"name": "answer", '
                    '"parameters": {"response": "roof intact", '
                    '"supporting_mask_ids": [1]}}</tool>'
                )

        result = run_agent(_make_image(), "Is the roof ok?", LegacyStrClient())

        assert result.answer == "roof intact"
        assert result.n_fp_calls == 1
        assert fake_tools["ground_expression"] == ["roof"]


# ──────────────────────────────────────────────────────────────────────────────
# Malformed native JSON arguments
# ──────────────────────────────────────────────────────────────────────────────

class TestMalformedNativeArguments:
    def test_malformed_args_surface_retryable_error(self, fake_tools):
        """Bad native JSON args → user-role error to the VLM, loop continues.

        Decision (documented in the module docstring): the loop appends a
        user-role message describing the parse failure and retries, instead
        of crashing or dispatching the tool with empty parameters.
        """
        bad_msg = SimpleNamespace(
            content="",
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="ground_expression", arguments="{not valid json"
                    )
                )
            ],
        )
        client = ScriptedVLM(
            agent_loop._chat_response_from_message(bad_msg),
            ChatResponse(
                tool_calls=[
                    {"name": "ground_expression", "parameters": {"expression": "person"}}
                ]
            ),
            ChatResponse(
                tool_calls=[
                    {"name": "answer", "parameters": {"response": "one person"}}
                ]
            ),
        )

        result = run_agent(_make_image(), "How many people?", client)

        assert result.answer == "one person"
        assert result.n_fp_calls == 1
        assert result.n_vlm_calls == 3
        user_texts = _history_texts(result.history, "user")
        retry = [t for t in user_texts if "not valid JSON" in t]
        assert retry, "loop must surface a user-role error message for bad args"
        assert "ground_expression" in retry[0]

    def test_bad_args_do_not_dispatch_tool(self, fake_tools):
        """The tool is not called with the fallback empty parameters."""
        bad_msg = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="compute_relations", arguments='["a", 1]'
                    )
                )
            ],
        )
        client = ScriptedVLM(
            agent_loop._chat_response_from_message(bad_msg),
            ChatResponse(
                tool_calls=[{"name": "answer", "parameters": {"response": "gave up"}}]
            ),
        )

        result = run_agent(_make_image(), "Compare?", client)

        assert result.answer == "gave up"
        assert fake_tools["compute_relations"] == []


# ──────────────────────────────────────────────────────────────────────────────
# No tool call at all
# ──────────────────────────────────────────────────────────────────────────────

class TestNoToolCall:
    @pytest.mark.parametrize(
        "response",
        [
            ChatResponse(content="Sorry, I cannot answer that."),
            "Sorry, I cannot answer that.",
        ],
        ids=["chat-response", "plain-str"],
    )
    def test_no_tool_call_raises_value_error(self, fake_tools, response):
        """Neither native calls nor a <tool> tag → today's ValueError."""
        client = ScriptedVLM(response)
        with pytest.raises(ValueError, match="Could not parse <tool> tag"):
            run_agent(_make_image(), "How many people?", client)


# ──────────────────────────────────────────────────────────────────────────────
# VLMClient.chat response mapping (fake openai module)
# ──────────────────────────────────────────────────────────────────────────────

class _FakeCompletions:
    def __init__(self, captured, message):
        self._captured = captured
        self._message = message

    def create(self, **params):
        self._captured["params"] = params
        return SimpleNamespace(choices=[SimpleNamespace(message=self._message)])


class _FakeOpenAI:
    def __init__(self, captured, message):
        self._captured = captured
        self._message = message

    def __call__(self, api_key=None, base_url=None):
        client = SimpleNamespace()
        client.chat = SimpleNamespace(
            completions=_FakeCompletions(self._captured, self._message)
        )
        return client


def _install_fake_openai(monkeypatch, message) -> dict:
    """Inject a fake ``openai`` module into sys.modules; return captured params."""
    captured: dict = {}
    fake = types.ModuleType("openai")
    fake.OpenAI = _FakeOpenAI(captured, message)
    monkeypatch.setitem(sys.modules, "openai", fake)
    return captured


class TestVLMClientChatMapping:
    def test_chat_maps_native_tool_calls(self, monkeypatch):
        message = SimpleNamespace(
            content="Checking the scene.",
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="ground_expression",
                        arguments='{"expression": "person"}',
                    )
                )
            ],
        )
        captured = _install_fake_openai(monkeypatch, message)
        client = VLMClient(api_key="sk-test", model="gpt-4o")

        resp = client.chat([{"role": "user", "content": "hi"}], tools=AGENT_TOOLS)

        assert isinstance(resp, ChatResponse)
        assert resp.content == "Checking the scene."
        assert resp.tool_calls == [
            {"name": "ground_expression", "parameters": {"expression": "person"}}
        ]
        # Native tools were forwarded to the backend untouched.
        assert captured["params"]["model"] == "gpt-4o"
        assert captured["params"]["tools"] == AGENT_TOOLS
        assert captured["params"]["messages"] == [{"role": "user", "content": "hi"}]

    def test_chat_content_only_yields_empty_tool_calls(self, monkeypatch):
        message = SimpleNamespace(content="Hello there.", tool_calls=None)
        _install_fake_openai(monkeypatch, message)
        client = VLMClient(api_key="sk-test")

        resp = client.chat([{"role": "user", "content": "hi"}])

        assert isinstance(resp, ChatResponse)
        assert resp.content == "Hello there."
        assert resp.tool_calls == []

    def test_chat_malformed_arguments_fall_back_to_empty_params(self, monkeypatch):
        message = SimpleNamespace(
            content=None,
            tool_calls=[
                SimpleNamespace(
                    function=SimpleNamespace(
                        name="ground_expression", arguments="{expression: person}"
                    )
                )
            ],
        )
        _install_fake_openai(monkeypatch, message)
        client = VLMClient(api_key="sk-test")

        resp = client.chat([{"role": "user", "content": "hi"}])

        assert resp.content == ""
        assert len(resp.tool_calls) == 1
        call = resp.tool_calls[0]
        assert call["name"] == "ground_expression"
        assert call["parameters"] == {}
        assert "arguments_error" in call
        assert call["arguments_raw"] == "{expression: person}"


# ──────────────────────────────────────────────────────────────────────────────
# _normalize_chat_response
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeChatResponse:
    def test_chat_response_passthrough(self):
        resp = ChatResponse(content="x", tool_calls=[{"name": "answer", "parameters": {}}])
        assert _normalize_chat_response(resp) is resp

    def test_str_becomes_chat_response(self):
        resp = _normalize_chat_response("<tool>{}</tool>")
        assert isinstance(resp, ChatResponse)
        assert resp.content == "<tool>{}</tool>"
        assert resp.tool_calls == []

    def test_default_content_is_empty(self):
        assert ChatResponse().content == ""
        assert ChatResponse().tool_calls == []

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="must return ChatResponse or str"):
            _normalize_chat_response(42)
