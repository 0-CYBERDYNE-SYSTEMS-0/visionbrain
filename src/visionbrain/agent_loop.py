"""VisionBrain Agent Loop — VLM-powered visual reasoning over Falcon Perception.

The agent alternates between:
  1. Sending the current message history to a VLM (GPT-4o or OpenAI-compatible).
  2. Resolving the requested tool call from the response — either native
     OpenAI-style ``message.tool_calls`` (returned by ``VLMClient.chat`` as a
     ``ChatResponse``) or a textual ``<tool>...</tool>`` JSON block in the
     response content. Native tool calls take precedence.
  3. Executing the tool (FP inference, crop extraction, or relation computation).
  4. Appending the tool result back to the message history.

The loop terminates when the VLM calls the `answer` tool.

Error contract:
  - A native tool call whose ``function.arguments`` is not valid JSON is
    reported back to the VLM as a user-role error message so it can retry on
    the next turn (the loop does not crash).
  - A response containing neither native tool calls nor a ``<tool>`` tag
    raises ``ValueError``.

Usage::

    from visionbrain.agent_loop import run_agent, VLMClient

    client = VLMClient(api_key="sk-...", model="gpt-4o")
    result = run_agent(image, "Which animal is closest to the water trough?",
                       client=client)
    print(result.answer)
    result.final_image.show()
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from .agent_tools import (
    compute_relations,
    masks_to_vlm_json,
    run_ground_expression,
)
from .viz import get_crop, render_som, render_detections


# ──────────────────────────────────────────────────────────────────────────────
# System prompt loader
# ──────────────────────────────────────────────────────────────────────────────

_SYS_PROMPT_CACHE: Optional[str] = None


def _load_system_prompt() -> str:
    """Load the agent system prompt from the bundled reference file (cached per process)."""
    global _SYS_PROMPT_CACHE
    if _SYS_PROMPT_CACHE is not None:
        return _SYS_PROMPT_CACHE

    ref_path = Path(__file__).parent / "references" / "system_prompt.txt"
    if ref_path.exists():
        _SYS_PROMPT_CACHE = ref_path.read_text(encoding="utf-8").strip()
    else:
        # Fallback minimal prompt
        _SYS_PROMPT_CACHE = (
            "You are a visual reasoning assistant for camera and drone imagery. "
            "You have access to a segmentation model (Falcon Perception) that can "
            "detect and segment objects. Use the tools below to answer the user's question. "
            "When you are done, call answer() with your response."
        )
    return _SYS_PROMPT_CACHE


# ──────────────────────────────────────────────────────────────────────────────
# VLM Client interface
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatResponse:
    """Normalized assistant turn returned by :meth:`VLMClient.chat`.

    Attributes:
        content: assistant text content ("" when absent or None).
        tool_calls: native tool calls in backend order, each a dict of the
            form ``{"name": str, "parameters": dict}``. When a call's JSON
            arguments could not be parsed, ``parameters`` falls back to ``{}``
            and the extra keys ``arguments_error`` (reason) and
            ``arguments_raw`` (original JSON string) are included so callers
            (see :func:`run_agent`) can surface a retryable error.
    """

    content: str = ""
    tool_calls: list[dict] = field(default_factory=list)


def _chat_response_from_message(message: Any) -> ChatResponse:
    """Map a native OpenAI-style response message to a :class:`ChatResponse`.

    ``message.tool_calls[*].function`` entries are converted to plain
    ``{"name", "parameters"}`` dicts. ``function.arguments`` is a JSON string
    in the OpenAI format; on ``json.JSONDecodeError`` the mapping keeps going
    with ``parameters={}`` plus ``arguments_raw``/``arguments_error`` markers
    instead of crashing. Dict-shaped tool calls (some OpenAI-compatible
    backends) are handled too.
    """
    content = getattr(message, "content", None) or ""
    calls: list[dict] = []
    for tc in getattr(message, "tool_calls", None) or []:
        function = getattr(tc, "function", None)
        if function is None and isinstance(tc, dict):
            function = tc.get("function")
        name = getattr(function, "name", None)
        raw_args = getattr(function, "arguments", None)
        if isinstance(function, dict):
            name = function.get("name", name)
            raw_args = function.get("arguments", raw_args)

        call: dict[str, Any] = {"name": name or "", "parameters": {}}
        if isinstance(raw_args, dict):
            call["parameters"] = raw_args
        elif isinstance(raw_args, str) and raw_args.strip():
            try:
                parsed = json.loads(raw_args)
            except json.JSONDecodeError as exc:
                call["arguments_error"] = str(exc)
                call["arguments_raw"] = raw_args
            else:
                if isinstance(parsed, dict):
                    call["parameters"] = parsed
                else:
                    call["arguments_error"] = "tool arguments must be a JSON object"
                    call["arguments_raw"] = raw_args
        calls.append(call)
    return ChatResponse(content=content, tool_calls=calls)


def _normalize_chat_response(result: Any) -> ChatResponse:
    """Normalize a ``chat()`` return value to a :class:`ChatResponse`.

    Accepts a ``ChatResponse`` as-is, or a plain ``str`` for legacy
    ``VLMClient`` subclasses that override ``chat`` to return only the
    assistant text. Anything else raises ``TypeError``.
    """
    if isinstance(result, ChatResponse):
        return result
    if isinstance(result, str):
        return ChatResponse(content=result)
    raise TypeError(
        "VLMClient.chat() must return ChatResponse or str, got "
        f"{type(result).__name__}"
    )


class VLMClient:
    """Minimal VLM client for the agent loop.

    Subclass or wrap to support any OpenAI-compatible API (GPT-4o, Claude,
    Gemini via proxy, local model, etc.).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: Optional[str] = None,
    ):
        import openai
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.model = model

    def chat(
        self,
        messages: list[dict],
        tools: Optional[list] = None,
    ) -> ChatResponse:
        """Send a multi-modal message list, return the normalized assistant turn.

        Native OpenAI-style ``message.tool_calls`` are mapped to
        ``ChatResponse.tool_calls`` as ``{"name", "parameters"}`` dicts (the
        ``function.arguments`` JSON string is decoded; malformed JSON falls
        back to ``{}`` with ``arguments_error``/``arguments_raw`` markers).
        ``content`` stays ``message.content or ""``.
        """
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
        resp = self._client.chat.completions.create(**params)
        return _chat_response_from_message(resp.choices[0].message)


# ──────────────────────────────────────────────────────────────────────────────
# Result container
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    answer: str
    supporting_mask_ids: list[int] = field(default_factory=list)
    final_image: Optional[Image.Image] = None
    history: list[dict] = field(default_factory=list)
    n_fp_calls: int = 0
    n_vlm_calls: int = 0


# ──────────────────────────────────────────────────────────────────────────────
# Tool-call parsing
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_RE = re.compile(r"<tool>(.*?)</tool>", re.DOTALL)


def _parse_tool_call(text: str) -> Optional[dict]:
    """Extract and parse JSON inside the first <tool>...</tool> block."""
    m = _TOOL_RE.search(text)
    if not m:
        return None
    raw = m.group(1).strip().replace("}}}", "}}")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Tool definitions for the VLM
# ──────────────────────────────────────────────────────────────────────────────

AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "ground_expression",
            "description": (
                "Segment objects in the image matching a natural-language expression. "
                "Returns colored masks with numbered labels. "
                "Use for: 'person', 'vehicle', 'injured animal', 'roof', 'sign', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Natural-language expression to segment. "
                            "Be specific: 'red pickup truck' vs 'truck', 'damaged roof' vs 'roof'."
                        ),
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_crop",
            "description": "Zoom into a specific mask by ID to see fine details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_id": {
                        "type": "integer",
                        "description": "Mask ID from the last ground_expression result.",
                    }
                },
                "required": ["mask_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute_relations",
            "description": (
                "Compute spatial relationships (IoU, left/right, above/below, "
                "size ratio, centroid distance) between selected masks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "mask_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "List of mask IDs to compare (2 or more).",
                    }
                },
                "required": ["mask_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Return the final answer to the user's question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "response": {"type": "string"},
                    "supporting_mask_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Mask IDs that support this answer (optional).",
                    },
                },
                "required": ["response"],
            },
        },
    },
]


# ──────────────────────────────────────────────────────────────────────────────
# Context management
# ──────────────────────────────────────────────────────────────────────────────

def _count_images(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        content = msg.get("content", [])
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") in ("image_url", "text"):
                total += 1
    return total


_IMAGE_PLACEHOLDER = "[previous annotated image omitted]"


def _has_image(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list):
        return False
    return any(
        isinstance(item, dict) and item.get("type") == "image_url" for item in content
    )


def _strip_image(msg: dict) -> dict:
    """Return a copy of *msg* with image parts replaced by a text placeholder."""
    content = msg.get("content")
    if not isinstance(content, list):
        return msg

    new_content: list[Any] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "image_url":
            new_content.append({"type": "text", "text": _IMAGE_PLACEHOLDER})
        else:
            new_content.append(item)

    stripped = dict(msg)
    stripped["content"] = new_content
    return stripped


def _prune_context(messages: list[dict], max_tail: int = 8) -> list[dict]:
    """Keep message history compact and bounded.

    Strategy:
      - Always keep messages[0] (system) and messages[1] (original user image).
      - Keep only the most recent ``max_tail`` messages after that. Slices are
        computed by index so kept messages can never overlap or duplicate.
      - In the kept messages, replace the base64 payload of every image-bearing
        message except the most recent one with a short text placeholder; the
        VLM only needs to look at the latest render. Text parts are preserved.
      - messages[1] keeps its image payload regardless: the latest render may be
        a get_crop of one region, and stripping the original would leave the VLM
        with no view of the full scene.
      - Messages with unexpected structure are kept verbatim.

    At most two image payloads survive, so context stays bounded across turns.
    """
    if len(messages) <= 2:
        return messages

    start = max(2, len(messages) - max_tail)
    kept = messages[:2] + messages[start:]

    last_image_idx = None
    for i in range(len(kept) - 1, -1, -1):
        if _has_image(kept[i]):
            last_image_idx = i
            break

    if last_image_idx is None:
        return kept

    return [
        msg if i == last_image_idx or i == 1 else _strip_image(msg)
        for i, msg in enumerate(kept)
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Core agent loop
# ──────────────────────────────────────────────────────────────────────────────

def run_agent(
    image: Image.Image,
    question: str,
    client: VLMClient,
    *,
    system_prompt: Optional[str] = None,
    max_generations: int = 10,
    verbose: bool = False,
) -> AgentResult:
    """Run the VisionBrain agent on *image* answering *question*.

    Tool-call resolution per turn: if the chat result carries native
    ``tool_calls`` the FIRST one is used; otherwise the textual
    ``<tool>...</tool>`` block in the content is parsed. A response with
    neither raises ``ValueError``. A native call with malformed JSON
    arguments is reported back to the VLM as a user-role error message and
    the loop continues (consuming a generation).

    Args:
        image: PIL Image to analyze
        question: user's question about the image
        client: VLMClient instance for LLM calls (``chat`` may return a
            ``ChatResponse`` or, for legacy subclasses, a plain ``str``)
        system_prompt: optional custom system prompt
        max_generations: max tool-call rounds before giving up
        verbose: print step-by-step progress

    Returns:
        AgentResult with answer, supporting masks, and annotated image
    """
    messages: list[dict] = []
    current_masks: dict[int, dict] = {}

    sys_prompt = system_prompt or _load_system_prompt()
    messages.append({"role": "system", "content": sys_prompt})

    # Build the first user message with the image
    # For OpenAI-compatible APIs, send as image_url
    import base64, io
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode()
    img_url = f"data:image/jpeg;base64,{img_b64}"

    messages.append({
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": img_url}},
            {"type": "text", "text": question},
        ],
    })

    n_fp_calls = 0
    n_vlm_calls = 0

    for step in range(max_generations):
        if verbose:
            print(f"\n[Agent turn {step + 1}]")

        # ── Call VLM ────────────────────────────────────────────────────────
        n_vlm_calls += 1
        t0 = __import__("time").perf_counter()
        chat_result = _normalize_chat_response(client.chat(messages, tools=AGENT_TOOLS))
        response_text = chat_result.content
        if verbose:
            print(f"  VLM response in {__import__('time').perf_counter()-t0:.2f}s")
            think = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            if think:
                print(f"  [think] {think.group(1).strip()[:200]}")

        # Keep history faithful: for native tool calls, append a compact JSON
        # note of the requested call so the turn stays interpretable after
        # _prune_context (the backend never sees native tool_call rows here).
        assistant_text = response_text
        if chat_result.tool_calls:
            call_note = json.dumps(
                {"requested_tool_call": chat_result.tool_calls[0]},
                separators=(",", ":"),
                default=str,
            )
            assistant_text = (
                f"{response_text}\n{call_note}" if response_text.strip() else call_note
            )
        messages.append({"role": "assistant", "content": [{"type": "text", "text": assistant_text}]})

        # ── Resolve tool call (native first, then textual <tool> tag) ───────
        tool_call = (
            chat_result.tool_calls[0]
            if chat_result.tool_calls
            else _parse_tool_call(response_text)
        )
        if tool_call is None:
            raise ValueError(
                f"Could not parse <tool> tag from VLM response at step {step + 1}.\n"
                f"Response: {response_text[:500]}"
            )

        # Native call with unparseable JSON arguments: surface a retryable
        # user-role error to the VLM instead of dispatching or crashing.
        if tool_call.get("arguments_error"):
            retry_text = (
                f"Error: arguments for tool '{tool_call.get('name', '')}' were not "
                f"valid JSON ({tool_call['arguments_error']}). "
                f"Raw arguments: {tool_call.get('arguments_raw', '')!r}. "
                "Re-issue the tool call with valid JSON arguments."
            )
            if verbose:
                print(f"  ! {retry_text}")
            messages.append({"role": "user", "content": [{"type": "text", "text": retry_text}]})
            messages = _prune_context(messages)
            continue

        tool_name = tool_call.get("name", "")
        params = tool_call.get("parameters", {})

        # ── Execute tool ────────────────────────────────────────────────────

        if tool_name == "ground_expression":
            expression = params.get("expression", "")
            if verbose:
                print(f"  → ground_expression({expression!r})")

            current_masks = run_ground_expression(
                image,
                expression,
                max_new_tokens=2048,
            )
            n_fp_calls += 1
            n_masks = len(current_masks)

            if verbose:
                print(f"     → {n_masks} mask(s) returned")

            if n_masks == 0:
                tool_result_content: list[dict] = [
                    {"type": "text", "text": (
                        f"ground_expression({expression!r}) returned 0 masks. "
                        "Try a more general expression."
                    )},
                ]
            else:
                som_image = render_som(image, _masks_from_dict(current_masks))
                meta_json = json.dumps(
                    {"n_masks": n_masks, "masks": masks_to_vlm_json(current_masks)},
                    indent=2,
                )
                # Re-encode the SoM image
                som_buf = io.BytesIO()
                som_image.save(som_buf, format="JPEG", quality=85)
                som_b64 = base64.b64encode(som_buf.getvalue()).decode()
                som_url = f"data:image/jpeg;base64,{som_b64}"

                tool_result_content = [
                    {"type": "image_url", "image_url": {"url": som_url}},
                    {"type": "text", "text": (
                        f"ground_expression returned {n_masks} mask(s). "
                        f"The Set-of-Marks image is shown above.\n\n"
                        f"Mask metadata:\n{meta_json}"
                    )},
                ]

            messages.append({"role": "user", "content": tool_result_content})

        elif tool_name == "get_crop":
            mask_id = int(params.get("mask_id", -1))
            if verbose:
                print(f"  → get_crop(mask_id={mask_id})")

            if mask_id not in current_masks:
                messages.append({
                    "role": "user",
                    "content": [{"type": "text", "text": (
                        f"get_crop failed: mask_id={mask_id} does not exist. "
                        f"Available IDs: {sorted(current_masks.keys())}"
                    )}],
                })
            else:
                crop_img = get_crop(image, _mask_dict_to_result(current_masks[mask_id]))
                crop_buf = io.BytesIO()
                crop_img.save(crop_buf, format="JPEG", quality=85)
                crop_b64 = base64.b64encode(crop_buf.getvalue()).decode()
                crop_url = f"data:image/jpeg;base64,{crop_b64}"
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": crop_url}},
                        {"type": "text", "text": f"Zoomed crop of mask {mask_id}."},
                    ],
                })

        elif tool_name == "compute_relations":
            mask_ids = params.get("mask_ids", [])
            if verbose:
                print(f"  → compute_relations(mask_ids={mask_ids})")

            relations = compute_relations(current_masks, mask_ids)
            messages.append({
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": f"compute_relations result:\n{json.dumps(relations, indent=2)}",
                }],
            })

        elif tool_name == "answer":
            response_text_final = params.get("response", "")
            selected_ids = [int(i) for i in params.get("supporting_mask_ids", [])]

            if verbose:
                print(f"\n{'─' * 60}")
                print(f"  Answer: {response_text_final}")
                print(f"  Supporting masks: {selected_ids}")
                print(f"  FP calls: {n_fp_calls}  |  VLM calls: {n_vlm_calls}")
                print(f"{'─' * 60}\n")

            final_image = (
                render_som(image, _masks_from_dict(
                    {k: v for k, v in current_masks.items() if k in selected_ids}
                )) if selected_ids and current_masks else image.copy()
            )

            return AgentResult(
                answer=response_text_final,
                supporting_mask_ids=selected_ids,
                final_image=final_image,
                history=messages,
                n_fp_calls=n_fp_calls,
                n_vlm_calls=n_vlm_calls,
            )

        else:
            raise ValueError(
                f"Unknown tool '{tool_name}' at step {step + 1}. "
                "Expected: ground_expression, get_crop, compute_relations, answer."
            )

        # ── Context pruning ──────────────────────────────────────────────────
        messages = _prune_context(messages)

    raise RuntimeError(
        f"Agent exceeded max_generations={max_generations} without calling 'answer'."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _mask_dict_to_result(d: dict) -> "MaskResult":
    """Convert a mask dict from agent_tools to a MaskResult for viz.py."""
    from .fp_inference import MaskResult as MR
    return MR(
        mask_id=d["id"],
        centroid_x=d["centroid_norm"]["x"],
        centroid_y=d["centroid_norm"]["y"],
        bbox_x1=d["bbox_norm"]["x1"],
        bbox_y1=d["bbox_norm"]["y1"],
        bbox_x2=d["bbox_norm"]["x2"],
        bbox_y2=d["bbox_norm"]["y2"],
        area_fraction=d["area_fraction"],
        image_region=d["image_region"],
        rle=d["rle"],
    )


def _masks_from_dict(d: dict[int, dict]) -> list:
    return [_mask_dict_to_result(v) for v in d.values()]
