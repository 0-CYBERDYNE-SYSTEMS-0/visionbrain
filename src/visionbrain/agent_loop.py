"""VisionBrain Agent Loop — VLM-powered visual reasoning over Falcon Perception.

The agent alternates between:
  1. Sending the current message history to a VLM (GPT-4o or OpenAI-compatible).
  2. Parsing the <tool> call from the VLM response.
  3. Executing the tool (FP inference, crop extraction, or relation computation).
  4. Appending the tool result back to the message history.

The loop terminates when the VLM calls the `answer` tool.

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

def _load_system_prompt() -> str:
    """Load the agent system prompt from the bundled reference file."""
    ref_path = Path(__file__).parent / "references" / "system_prompt.txt"
    if ref_path.exists():
        return ref_path.read_text(encoding="utf-8").strip()
    # Fallback minimal prompt
    return (
        "You are a visual reasoning assistant for agricultural images. "
        "You have access to a segmentation model (Falcon Perception) that can "
        "detect and segment objects. Use the tools below to answer the user's question. "
        "When you are done, call answer() with your response."
    )


# ──────────────────────────────────────────────────────────────────────────────
# VLM Client interface
# ──────────────────────────────────────────────────────────────────────────────

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
    ) -> str:
        """Send a multi-modal message list, return the assistant's text response."""
        params: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if tools:
            params["tools"] = tools
        resp = self._client.chat.completions.create(**params)
        return resp.choices[0].message.content or ""


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
                "Use for: 'cow', 'sheep', 'injured animal', 'fence post', 'crop row', etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": (
                            "Natural-language expression to segment. "
                            "Be specific: 'lame cow' vs 'cow', 'wheat row' vs 'crop'."
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


def _prune_context(messages: list[dict], max_tail: int = 4) -> list[dict]:
    """Keep message history compact.

    Strategy:
      - Always keep messages[0] (system) and messages[1] (original user image).
      - Keep the last ground_expression assistant + user pair.
      - Keep all subsequent get_crop / compute_relations messages.
      - Discard everything in between.
    """
    if len(messages) <= 2:
        return messages

    system = messages[:1]
    original_user = messages[1:2]

    # Find the last assistant message that contains a tool call
    last_tool_idx = None
    for i in range(len(messages) - 1, 1, -1):
        if messages[i].get("role") == "assistant":
            content = messages[i].get("content", "")
            if isinstance(content, str) and "<tool>" in content:
                last_tool_idx = i
                break

    if last_tool_idx is None:
        return system + original_user + messages[-max_tail:]

    # Keep: system, original_user, messages[2:last_tool_idx+1], last max_tail
    kept = system + original_user + messages[2:last_tool_idx + 1] + messages[-max_tail:]
    return kept


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

    Args:
        image: PIL Image to analyze
        question: user's question about the image
        client: VLMClient instance for LLM calls
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
        response_text = client.chat(messages, tools=AGENT_TOOLS)
        if verbose:
            print(f"  VLM response in {__import__('time').perf_counter()-t0:.2f}s")
            think = re.search(r"<think>(.*?)</think>", response_text, re.DOTALL)
            if think:
                print(f"  [think] {think.group(1).strip()[:200]}")

        messages.append({"role": "assistant", "content": [{"type": "text", "text": response_text}]})

        # ── Parse tool call ─────────────────────────────────────────────────
        tool_call = _parse_tool_call(response_text)
        if tool_call is None:
            raise ValueError(
                f"Could not parse <tool> tag from VLM response at step {step + 1}.\n"
                f"Response: {response_text[:500]}"
            )

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
