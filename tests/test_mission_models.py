"""Model-free tests for the canonical adaptive-mission adapters."""

from __future__ import annotations

import json
from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

from visionbrain import mission_models as models
from visionbrain.mission_contracts import (
    Decision,
    GeometryItem,
    PlannerContext,
    ToolCallRecord,
    ToolContext,
    ToolRequest,
)


class RecordingLock:
    def __init__(self) -> None:
        self.depth = 0

    @property
    def held(self) -> bool:
        return self.depth > 0

    def __enter__(self):
        self.depth += 1
        return self

    def __exit__(self, *_exc):
        self.depth -= 1


def jpeg_image(size=(100, 80), color=(210, 30, 20)) -> bytes:
    image = Image.new("RGB", size, color)
    stream = BytesIO()
    image.save(stream, format="JPEG", quality=96)
    return stream.getvalue()


def make_context(
    *,
    allowed=("detect_objects", "segment_objects", "inspect_crop", "read_text", "request_closeup", "finish"),
    results=(),
    mode="inspect",
    image=None,
) -> PlannerContext:
    image = image or jpeg_image()
    with Image.open(BytesIO(image)) as decoded:
        width, height = decoded.size
    return PlannerContext(
        mission_id="mission-1",
        revision=4,
        cycle_id="cycle-1",
        execution_generation=2,
        profile_id="visual_inspection",
        profile_version=1,
        mode=mode,
        expertise="facility maintenance",
        goal="Find visible items that need closer inspection",
        reasoning_model="gemma",
        input_evidence_id="evidence-input",
        input_sha256="a" * 64,
        image_jpeg=image,
        image_width=width,
        image_height=height,
        source_binding=None,
        frame_id=None,
        input_transform=None,
        allowed_tools=tuple({"name": name} for name in allowed),
        tool_results=tuple(results),
        findings=(),
        generations_remaining=5,
        tool_calls_remaining=8,
    )


def make_request(tool: str, arguments: dict, *, reasoning_model="gemma") -> ToolRequest:
    return ToolRequest(
        tool=tool,
        arguments=arguments,
        mission_id="mission-1",
        cycle_id="cycle-1",
        execution_generation=2,
        reasoning_model=reasoning_model,
        input_evidence_id="evidence-input",
        input_sha256="a" * 64,
        frame_id=None,
        source_binding=None,
    )


def make_tool_context(image: bytes, item: GeometryItem) -> ToolContext:
    with Image.open(BytesIO(image)) as decoded:
        width, height = decoded.size
    return ToolContext(
        image_jpeg=image,
        image_width=width,
        image_height=height,
        grounded_items={item.item_id: item},
        prior_results=(),
    )


def test_module_import_and_adapter_construction_do_not_load_mlx(monkeypatch):
    planner = models.LocalMissionPlanner(gpu_lock=RecordingLock())
    assert planner._held_checkpoint is None
    assert planner.gpu_lock is not None


def test_local_checkpoint_readiness_requires_complete_cached_snapshot(tmp_path, monkeypatch):
    from visionbrain import loader, vlm_registry

    cache = tmp_path / "hub"
    monkeypatch.setattr(loader, "HF_CACHE", cache)
    repo = cache / f"models--{vlm_registry.MODELS['gemma'].replace('/', '--')}"
    model_dir = repo / "snapshots" / "revision"
    model_dir.mkdir(parents=True)
    (repo / "refs").mkdir()
    (repo / "refs" / "main").write_text("revision", encoding="utf-8")
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "processor_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert not models._checkpoint_is_cached(vlm_registry.MODELS["gemma"])

    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"a": "model-00001.safetensors", "b": "model-00002.safetensors"}}),
        encoding="utf-8",
    )
    (model_dir / "model-00001.safetensors").write_bytes(b"weights")
    assert not models._checkpoint_is_cached(vlm_registry.MODELS["gemma"])
    (model_dir / "model-00002.safetensors").write_bytes(b"weights")
    assert models._checkpoint_is_cached(vlm_registry.MODELS["gemma"])
    assert models._cached_checkpoint_path(vlm_registry.MODELS["gemma"]) == model_dir

    monkeypatch.setattr(models, "_mlx_vlm_available", lambda: True)
    planner = models.LocalMissionPlanner(gpu_lock=RecordingLock())
    assert planner.available("gemma")
    assert not planner.available("missing")


def test_lfm_projector_compat_drops_only_the_config_disabled_norm():
    from visionbrain.mlx_compat import _patch_lfm_projector_init

    class Norm:
        def parameters(self):
            return ("weight", "bias")

    class Identity:
        def parameters(self):
            return ()

    class Projector:
        def __init__(self, config):
            self.layer_norm = Norm()

    _patch_lfm_projector_init(Projector, Identity)

    without_norm = Projector(SimpleNamespace(projector_use_layernorm=False))
    with_norm = Projector(SimpleNamespace(projector_use_layernorm=True))

    assert isinstance(without_norm.layer_norm, Identity)
    assert without_norm.layer_norm.parameters() == ()
    assert isinstance(with_norm.layer_norm, Norm)
    assert with_norm.layer_norm.parameters() == ("weight", "bias")


def test_decision_parser_accepts_one_bounded_action_and_normalizes_targets():
    context = make_context(allowed=("detect_objects", "finish"))
    raw = json.dumps({
        "schema_version": 1,
        "tool": "detect_objects",
        "arguments": {"targets": [" pipe ", "pipe", "valve"]},
        "reason": "Locate visible infrastructure.",
    })

    decision = models.parse_decision(raw, context)

    assert decision == Decision(
        schema_version=1,
        tool="detect_objects",
        arguments={"targets": ["pipe", "valve"]},
        reason="Locate visible infrastructure.",
    )


@pytest.mark.parametrize("tool,targets", [
    ("detect_objects", ["pipe"]),
    ("segment_objects", ["pipe", "valve"]),
])
def test_parser_repairs_only_unambiguous_nested_planner_reason(tool, targets):
    raw = json.dumps({
        "schema_version": 1,
        "tool": tool,
        "arguments": {"targets": targets, "reason": "Find these visible objects."},
    })

    decision = models.parse_decision(raw, make_context(allowed=(tool,)))

    assert decision.arguments == {"targets": targets}
    assert decision.reason == "Find these visible objects."


def test_parser_hoists_finish_payload_placed_inside_arguments():
    raw = json.dumps({
        "schema_version": 1,
        "tool": "finish",
        "arguments": {"findings": [], "watch": {"targets": ["cooler"], "task": "detect"}},
        "reason": "Observations cover the objective.",
    })

    decision = models.parse_decision(raw, make_context(mode="watch", results=(COOLER_DETECTION,)))

    assert decision.arguments == {}
    assert decision.watch.targets == ("cooler",)


COOLER_DETECTION = ToolCallRecord(
    tool_result_id="result-detect",
    tool="detect_objects",
    status="ok",
    input_evidence_id="evidence-input",
    items=(GeometryItem("item-1", "cooler", 0.9, (0.1, 0.2, 0.5, 0.7)),),
)


@pytest.mark.parametrize("target", ["item-1", "result-detect", "evidence-input", "table"])
def test_parser_rejects_watch_target_that_is_not_an_observed_label(target):
    raw = json.dumps({
        "schema_version": 1,
        "tool": "finish",
        "arguments": {"findings": [], "watch": {"targets": [target], "task": "detect"}},
        "reason": "Observations cover the objective.",
    })

    with pytest.raises(models.InvalidPlannerOutput, match="observed label"):
        models.parse_decision(raw, make_context(mode="watch", results=(COOLER_DETECTION,)))


def test_parser_accepts_observed_label_watch_target_in_any_case():
    raw = json.dumps({
        "schema_version": 1,
        "tool": "finish",
        "arguments": {"findings": [], "watch": {"targets": ["Cooler"], "task": "detect"}},
        "reason": "Observations cover the objective.",
    })

    decision = models.parse_decision(raw, make_context(mode="watch", results=(COOLER_DETECTION,)))

    assert decision.watch.targets == ("Cooler",)


def test_decoding_schema_limits_watch_targets_to_observed_labels():
    schema = models.build_decoding_schema(make_context(mode="watch", results=(COOLER_DETECTION,)))
    text = json.dumps(schema)
    finish = next(b for b in schema["anyOf"] if b["properties"]["tool"]["const"] == "finish")
    watch = finish["properties"]["arguments"]["properties"]["watch"]["anyOf"][0]

    assert '"oneOf"' not in text
    assert watch["properties"]["targets"]["items"] == {"enum": ["cooler"]}
    assert watch["properties"]["targets"]["minItems"] == 1


def test_decoding_schema_without_observations_allows_one_target_and_no_watch():
    schema = models.build_decoding_schema(make_context(mode="watch"))
    branches = {b["properties"]["tool"]["const"]: b for b in schema["anyOf"]}

    assert branches["detect_objects"]["properties"]["arguments"]["properties"]["targets"]["maxItems"] == 1
    assert branches["finish"]["properties"]["arguments"]["properties"]["watch"] == {"type": "null"}


def test_watch_finish_prompt_lists_observed_labels_and_forbids_ids():
    prompt = models.build_mission_prompt(make_context(mode="watch", results=(COOLER_DETECTION,)))

    assert 'Observed labels for watch targets: ["cooler"]' in prompt
    assert "never an item_id" in prompt


@pytest.mark.parametrize("key,value", [
    ("watch", {"targets": ["cooler"], "task": "detect"}),
    ("findings", []),
])
def test_parser_rejects_finish_payload_supplied_both_inside_and_outside_arguments(key, value):
    raw = json.dumps({
        "schema_version": 1,
        "tool": "finish",
        "arguments": {key: value},
        "reason": "Observations cover the objective.",
        key: value,
    })

    with pytest.raises(models.InvalidPlannerOutput):
        models.parse_decision(raw, make_context(mode="watch"))


def test_parser_rejects_unknown_keys_inside_finish_arguments():
    raw = json.dumps({
        "schema_version": 1,
        "tool": "finish",
        "arguments": {"task": "detect"},
        "reason": "Observations cover the objective.",
    })

    with pytest.raises(models.InvalidPlannerOutput):
        models.parse_decision(raw, make_context(mode="watch"))


def test_parser_accepts_null_finish_slot_on_non_finish_action():
    raw = json.dumps({
        "schema_version": 1,
        "tool": "detect_objects",
        "arguments": {"targets": ["oven", "refrigerator"], "reason": "Find visible appliances."},
        "finish": None,
    })

    decision = models.parse_decision(raw, make_context(allowed=("detect_objects",)))

    assert decision.tool == "detect_objects"
    assert decision.arguments == {"targets": ["oven", "refrigerator"]}
    assert decision.reason == "Find visible appliances."


LIVE_UNCLOSED = (
    '{"tool": "detect_objects", "arguments": {"targets": ["refrigerator", "cabinet", "shelves"], '
    '"reason": "Identify storage units to verify inventory count for appliances and supplies."}, "finish": null'
)


@pytest.mark.parametrize("raw", [
    LIVE_UNCLOSED,
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x"',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"],"reason":"x"}',
])
def test_parser_closes_up_to_two_missing_trailing_brackets(raw):
    decision = models.parse_decision(raw, make_context(allowed=("detect_objects",)))

    assert decision.tool == "detect_objects"
    assert decision.reason


@pytest.mark.parametrize("raw", [
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"unterminated',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x",',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x"} junk',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x"}{"tool":"finish"',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x"]',
])
def test_parser_rejects_unsafe_bracket_repairs(raw):
    with pytest.raises(models.InvalidPlannerOutput):
        models.parse_decision(raw, make_context(allowed=("detect_objects",)))


def test_parser_accepts_live_lfm3b_action_without_schema_version():
    raw = (
        '{"tool":"detect_objects","arguments":{"targets":["electrical panel","obstructed area"],'
        '"reason":"Locate the panel and assess the obstructed area."},"finish":null}'
    )

    decision = models.parse_decision(raw, make_context(allowed=("detect_objects",)))

    assert decision.schema_version == 1
    assert decision.tool == "detect_objects"
    assert decision.arguments == {"targets": ["electrical panel", "obstructed area"]}
    assert decision.reason == "Locate the panel and assess the obstructed area."


def test_parser_rejects_extra_root_field_when_schema_version_is_absent():
    raw = (
        '{"tool":"detect_objects","arguments":{"targets":["electrical panel","obstructed area"],'
        '"reason":"Locate the panel and assess the obstructed area."},"finish":null,"confidence":0.9}'
    )

    with pytest.raises(models.InvalidPlannerOutput, match="unknown fields"):
        models.parse_decision(raw, make_context(allowed=("detect_objects",)))


@pytest.mark.parametrize("schema_version", [2, True, None])
def test_parser_rejects_explicit_invalid_schema_version(schema_version):
    raw = json.dumps({
        "schema_version": schema_version,
        "tool": "detect_objects",
        "arguments": {"targets": ["pipe"]},
        "reason": "Locate the visible pipe.",
    })

    with pytest.raises(models.InvalidPlannerOutput, match="schema version"):
        models.parse_decision(raw, make_context(allowed=("detect_objects",)))


def test_closeup_parser_keeps_detail_reason_separate_from_action_rationale():
    decision = models.parse_decision(
        json.dumps({
            "schema_version": 1,
            "tool": "request_closeup",
            "arguments": {
                "description": "visible label",
                "reason": "The printed characters are too small to read.",
            },
            "reason": "A closer image is needed to assess the visible label.",
        }),
        make_context(allowed=("request_closeup",)),
    )

    assert decision.arguments["reason"] == "The printed characters are too small to read."
    assert decision.reason == "A closer image is needed to assess the visible label."


@pytest.mark.parametrize("raw", [
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x","extra":1}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x","finish":true}',
    '{"schema_version":1,"tool":"finish","arguments":{},"reason":"x","finish":null}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"],"finish":null},"reason":"x"}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x"} {"tool":"finish"}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"],"url":"http://bad"},"reason":"x"}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x","tool":"finish"}',
    '{"schema_version":1,"tool":"segment_objects","arguments":{"targets":["a","b","c","d","e"]},"reason":"x"}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x","score":NaN}',
    '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"],"reason":"nested"},"reason":"top-level"}',
    '{"schema_version":1,"tool":"request_closeup","arguments":{"description":"visible label","reason":"detail"}}',
])
def test_decision_parser_rejects_invalid_or_ambiguous_output(raw):
    with pytest.raises(models.InvalidPlannerOutput):
        models.parse_decision(raw, make_context())


def test_parser_rejects_unavailable_tools_and_invented_crop_items():
    with pytest.raises(models.InvalidPlannerOutput):
        models.parse_decision(
            '{"schema_version":1,"tool":"read_text","arguments":{"item_id":"fake"},"reason":"Read it."}',
            make_context(allowed=("detect_objects", "finish")),
        )
    with pytest.raises(models.InvalidPlannerOutput, match="not grounded"):
        models.parse_decision(
            '{"schema_version":1,"tool":"inspect_crop","arguments":{"item_id":"fake","question":"Read the label."},"reason":"Inspect it."}',
            make_context(),
        )


def test_finish_references_must_exist_in_the_current_cycle():
    item = GeometryItem("item-1", "pipe", 0.9, (0.1, 0.2, 0.5, 0.7))
    detection = ToolCallRecord(
        tool_result_id="result-detect",
        tool="detect_objects",
        status="ok",
        input_evidence_id="evidence-input",
        items=(item,),
        evidence_ids=("evidence-detect",),
    )
    ocr = ToolCallRecord(
        tool_result_id="result-ocr",
        tool="read_text",
        status="ok",
        input_evidence_id="evidence-input",
        evidence_ids=("evidence-ocr",),
        text="VALVE 21",
    )
    context = make_context(results=(detection, ocr), allowed=("finish",))
    valid = {
        "schema_version": 1,
        "tool": "finish",
        "arguments": {},
        "reason": "The grounded review is complete.",
        "findings": [
            {
                "claim": "A pipe is visible.",
                "claim_type": "localized_object",
                "evidence_refs": ["evidence-detect"],
                "item_refs": [{"tool_result_id": "result-detect", "item_id": "item-1"}],
            },
            {
                "claim": "The marking reads VALVE 21.",
                "claim_type": "text_read",
                "evidence_refs": ["evidence-ocr"],
                "text_refs": ["result-ocr"],
            },
        ],
        "watch": None,
    }

    decision = models.parse_decision(json.dumps(valid), context)

    assert decision.findings[0].item_refs == (("result-detect", "item-1"),)
    assert decision.findings[1].text_refs == ("result-ocr",)

    valid["findings"][0]["item_refs"][0]["item_id"] = "invented-item"
    with pytest.raises(models.InvalidPlannerOutput, match="invented item"):
        models.parse_decision(json.dumps(valid), context)


def test_prompt_is_direct_and_omits_unneeded_identity_and_schema_payload():
    prompt = models.build_mission_prompt(make_context(allowed=("detect_objects", "finish")))

    assert '"schema_version":1,"tool":"detect_objects"' in prompt
    assert models.PLANNER_PROMPT_VERSION == "mission.v1-local-json-8"
    assert "finish with no findings and no watch" in prompt
    assert "request a close-up" in prompt
    assert "exactly one specific, visible physical object" in prompt
    assert "On the first action target exactly one highest-priority object" in prompt
    assert "Later actions may expand only when observations justify it" in prompt
    assert "Do not target a job, role, broad scene category, or unrelated inventory" in prompt
    assert "Explain in reason how this action serves the objective" in prompt
    assert "decision_schema" not in prompt
    assert "evidence-input" not in prompt
    assert "a" * 64 not in prompt
    assert "segment_objects:" not in prompt
    assert '"arguments":{"description":"visible region","reason":"The printed characters are too small to read."}' in models.build_mission_prompt(make_context(allowed=("request_closeup",)))


def test_prompt_only_includes_bounded_grounded_observations():
    items = tuple(
        GeometryItem(f"item-{i}", "pipe", 0.8, (0.1, 0.1, 0.3, 0.3))
        for i in range(models.MAX_TOOL_ITEMS + 2)
    )
    results = tuple(
        ToolCallRecord(
            tool_result_id=f"result-{i}",
            tool="detect_objects",
            status="ok",
            input_evidence_id="evidence-input",
            items=items,
        )
        for i in range(models.MAX_PROMPT_TOOL_RESULTS + 1)
    )
    data = models._bounded_context_data(make_context(results=results))

    assert len(data["observations"]) == models.MAX_PROMPT_TOOL_RESULTS
    assert len(data["observations"][0]["items"]) == models.MAX_PROMPT_ITEMS_PER_RESULT
    assert "input_evidence_id" not in data["observations"][0]


def test_planner_receives_original_pixels_and_reports_raw_validity(monkeypatch):
    image = jpeg_image(size=(24, 16), color=(180, 20, 40))
    context = make_context(allowed=("detect_objects",), image=image)
    lock = RecordingLock()
    planner = models.LocalMissionPlanner(gpu_lock=lock)
    received = {}

    def fake_generate(_planner, prompt, attached_image, **kwargs):
        received["prompt"] = prompt
        received["image"] = attached_image.copy()
        received["kwargs"] = kwargs
        return (
            '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["red valve"]},"reason":"Locate the visible valve."}',
            "checkpoint-gemma",
        )

    monkeypatch.setattr(models, "_generate_with_registry_checkpoint", fake_generate)
    response = planner.plan_with_response(context)

    assert response.valid is True
    assert response.decision is not None and response.decision.tool == "detect_objects"
    assert response.raw_text.startswith('{"schema_version"')
    assert response.checkpoint == "checkpoint-gemma"
    assert received["image"].size == (24, 16)
    assert received["image"].getpixel((10, 8))[0] > received["image"].getpixel((10, 8))[2]
    assert "targets" in received["prompt"]
    assert received["kwargs"]["max_tokens"] == 512


def test_planner_keeps_raw_invalid_response_for_qualification(monkeypatch):
    context = make_context(allowed=("detect_objects",))
    planner = models.LocalMissionPlanner(gpu_lock=RecordingLock())
    raw = '{"schema_version":1,"tool":"detect_objects","arguments":{"targets":["pipe"]},"reason":"x","tool_calls":[1]}'
    monkeypatch.setattr(models, "_generate_with_registry_checkpoint", lambda *_a, **_k: (raw, "cp"))

    response = planner.plan_with_response(context)

    assert response.valid is False
    assert response.decision is None
    assert response.raw_text == raw
    with pytest.raises(models.InvalidPlannerOutput) as caught:
        planner.plan(context)
    assert caught.value.raw_text == raw


def test_sam_tool_uses_shared_lock_and_returns_normalized_mask_geometry(monkeypatch):
    import visionbrain.sam3_inference as sam

    lock = RecordingLock()
    monkeypatch.setattr(models, "_sam31_available", lambda: True)

    class Detection:
        label = "pipe"
        score = 0.8
        bbox_xyxy = (10, 20, 50, 60)
        mask = [[0] * 100 for _ in range(80)]

    for y in range(25, 50):
        for x in range(15, 45):
            Detection.mask[y][x] = 1

    def detect_multi(_image, _targets, *, resolution, task):
        assert lock.held
        assert resolution == 1008
        assert task == "segment"
        return [Detection()]

    monkeypatch.setattr(sam, "detect_multi", detect_multi)
    item = GeometryItem("grounded", "pipe", 0.9, (0.1, 0.2, 0.5, 0.7))
    tools = models.LocalMissionTools(planner=SimpleNamespace(gpu_lock=lock), gpu_lock=lock)
    result = tools.execute(
        make_request("segment_objects", {"targets": ["pipe"]}),
        make_tool_context(jpeg_image(), item),
    )

    assert result.status == "ok", result.error_code
    assert result.items[0].box == pytest.approx((0.1, 0.25, 0.5, 0.75))
    assert result.items[0].polygon is not None
    assert result.metadata["mask_status"] == "complete"


def test_sam_tool_forwards_configured_resolution(monkeypatch):
    import visionbrain.sam3_inference as sam

    lock = RecordingLock()
    monkeypatch.setattr(models, "_sam31_available", lambda: True)
    received = {}

    def detect_multi(_image, targets, *, resolution, task):
        assert lock.held
        received.update(targets=targets, resolution=resolution, task=task)
        return []

    monkeypatch.setattr(sam, "detect_multi", detect_multi)
    tools = models.LocalMissionTools(
        planner=SimpleNamespace(gpu_lock=lock),
        gpu_lock=lock,
        sam_resolution=504,
    )
    result = tools.execute(
        make_request("detect_objects", {"targets": ["electrical panel"]}),
        make_tool_context(jpeg_image(), GeometryItem("grounded", "panel", 0.9, (0.1, 0.1, 0.4, 0.4))),
    )

    assert result.status == "empty"
    assert received == {"targets": ["electrical panel"], "resolution": 504, "task": "detect"}


def test_inspect_crop_uses_exact_grounded_item_and_preserves_original_provenance():
    lock = RecordingLock()
    item = GeometryItem("item-1", "valve", 0.9, (0.2, 0.25, 0.4, 0.5))
    original = jpeg_image()

    class CropPlanner:
        gpu_lock = lock

        def available(self, _key):
            return True

        def inspect_crop(self, question, image, *, model_key):
            with lock:
                assert lock.held
            assert question == "Is the connection visibly corroded?"
            assert model_key == "gemma"
            self.crop_size = image.size
            return "The connection surface is visible; corrosion is uncertain.", "gemma-checkpoint"

    planner = CropPlanner()
    tools = models.LocalMissionTools(planner=planner, gpu_lock=lock)
    result = tools.execute(
        make_request("inspect_crop", {"item_id": "item-1", "question": "Is the connection visibly corroded?"}),
        make_tool_context(original, item),
    )

    assert planner.crop_size == (24, 24)
    assert result.status == "ok", result.error_code
    assert result.artifacts[0].parent_evidence_id == "evidence-input"
    assert result.artifacts[0].kind == "crop"
    assert result.artifacts[0].crop_box == pytest.approx((0.18, 0.225, 0.42, 0.525))
    assert result.metadata["source_item_id"] == "item-1"
    with Image.open(BytesIO(result.artifacts[0].jpeg_bytes)) as crop:
        assert crop.size == planner.crop_size


def test_read_text_runs_canonical_ocr_on_grounded_crop_and_maps_geometry(monkeypatch):
    import visionbrain.fp_inference as fp

    lock = RecordingLock()
    item = GeometryItem("item-1", "label", 0.9, (0.2, 0.25, 0.4, 0.5))
    tools = models.LocalMissionTools(planner=SimpleNamespace(gpu_lock=lock), gpu_lock=lock)
    monkeypatch.setattr(models, "_falcon_ocr_available", lambda: True)

    def ocr(image, question):
        assert lock.held
        assert image.size == (24, 24)
        assert question == "Read the equipment label."
        return ([SimpleNamespace(cx=0.5, cy=0.5, w=0.5, h=0.5)], "PUMP 21", SimpleNamespace(generation_ms=25.0))

    monkeypatch.setattr(fp, "ocr", ocr)
    result = tools.execute(
        make_request("read_text", {"item_id": "item-1", "question": "Read the equipment label."}),
        make_tool_context(jpeg_image(), item),
    )

    assert result.status == "ok", result.error_code
    assert result.text == "PUMP 21"
    assert result.items[0].label == "text region"
    assert result.items[0].box == pytest.approx((0.24, 0.3, 0.36, 0.45))
    assert result.items[0].score == 0.0
    assert result.artifacts[0].parent_evidence_id == "evidence-input"
    assert result.metadata["engine"] == "falcon_ocr"


def test_read_text_reports_unsupported_without_a_ready_ocr_adapter(monkeypatch):
    lock = RecordingLock()
    tools = models.LocalMissionTools(planner=SimpleNamespace(gpu_lock=lock), gpu_lock=lock)
    monkeypatch.setattr(models, "_falcon_ocr_available", lambda: False)
    item = GeometryItem("item-1", "label", 0.9, (0.2, 0.25, 0.4, 0.5))

    result = tools.execute(
        make_request("read_text", {"item_id": "item-1"}),
        make_tool_context(jpeg_image(), item),
    )

    assert result.status == "unsupported"
    assert result.error_code == "ocr_unavailable"
