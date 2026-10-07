"""Deterministic tests for vision candidate validation and the Jev decision contract.

No test here touches a model, the network, or a real API key.
"""

import asyncio
import json
import socket
import urllib.error
import urllib.request

import pytest

from visionbrain import mission_autotarget as autotarget
from visionbrain.mission_autotarget import (
    AutotargetError,
    JevChoice,
    JevDecisionClient,
    VisionCandidate,
    build_choice_body,
    build_choice_request,
    build_decision_record,
    candidate_response_schema,
    parse_choice_payload,
    parse_vision_proposal,
    parse_vision_response,
)


def _candidate_payload(**overrides):
    item = {
        "target_label": "cooler",
        "objective": "Inspect the cooler's placement and access clearance",
        "reason": "The cooler partly blocks the access path",
        "visual_evidence": "A cooler lid is visible beside the door",
        "uncertainty": "Contents cannot be seen",
    }
    item.update(overrides)
    return item


def _response(candidates=None, *, usable=True, uncertainty=""):
    return json.dumps(
        {
            "scene_usable": usable,
            "scene_uncertainty": uncertainty,
            "candidates": [_candidate_payload()] if candidates is None else candidates,
        }
    )


def _jev_payload(choice="c1", probabilities=None, confidence=0.7):
    return {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {
            "selection": {
                "type": "choice",
                "choice": choice,
                "probabilities": probabilities or {"c1": 0.8, "none": 0.2},
                "confidence": confidence,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }


def test_schema_generation_caps_are_stricter_than_parser_limits():
    schema = candidate_response_schema()
    assert schema["type"] == "object"
    assert len(schema["anyOf"]) == 2
    branches = {branch["properties"]["scene_usable"]["const"]: branch for branch in schema["anyOf"]}
    usable, unusable = branches[True], branches[False]
    assert usable["properties"]["scene_usable"] == {"const": True}
    assert unusable["properties"]["scene_usable"] == {"const": False}
    assert unusable["properties"]["candidates"] == {"type": "array", "maxItems": 0}
    candidate = usable["properties"]["candidates"]["items"]["properties"]
    assert usable["properties"]["candidates"]["maxItems"] == 4
    assert candidate["target_label"] == {"type": "string", "minLength": 1, "maxLength": 40}
    assert candidate["objective"] == {"type": "string", "minLength": 1, "maxLength": 56}
    assert candidate["reason"] == {"type": "string", "minLength": 1, "maxLength": 56}
    assert candidate["visual_evidence"] == {"type": "string", "minLength": 1, "maxLength": 56}
    assert candidate["uncertainty"] == {"type": "string", "minLength": 0, "maxLength": 56}
    for branch in (usable, unusable):
        assert branch["properties"]["scene_uncertainty"]["minLength"] == 0
        assert branch["additionalProperties"] is False


def test_valid_proposal_assigns_cycle_local_candidate_ids():
    proposal = parse_vision_proposal(_response([_candidate_payload(), _candidate_payload(target_label="bin")]))
    assert proposal.scene_usable is True
    assert [item.candidate_id for item in proposal.candidates] == ["c1", "c2"]
    assert proposal.candidates[0].target_label == "cooler"


def test_vision_adapter_uses_actual_frame_context_and_candidate_schema(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace

    from PIL import Image
    from visionbrain import mission_models

    jpeg = BytesIO()
    Image.new("RGB", (32, 24), (20, 90, 170)).save(jpeg, format="JPEG")
    seen = []

    def generate(adapter, prompt, image, **kwargs):
        seen.append((prompt, image.size, kwargs))
        return _response(), "cached-checkpoint"

    monkeypatch.setattr(mission_models, "_generate_with_registry_checkpoint", generate)
    context = SimpleNamespace(
        reasoning_model="gemma", expertise="Storage inspection",
        image_jpeg=jpeg.getvalue(), image_width=32, image_height=24,
    )
    response = mission_models.VisionBrainPlanner().propose_candidates(context)
    assert response.proposal.candidates[0].target_label == "cooler"
    assert response.checkpoint == "cached-checkpoint"
    prompt, size, kwargs = seen[0]
    assert "Storage inspection" in prompt
    assert size == (32, 24)
    assert kwargs["model_key"] == "gemma"
    assert kwargs["json_schema"] == candidate_response_schema()
    assert kwargs["max_tokens"] == 512


def test_empty_candidate_list_is_a_valid_usable_scene():
    proposal = parse_vision_proposal(_response([]))
    assert proposal.scene_usable is True
    assert proposal.candidates == ()


@pytest.mark.parametrize(
    "raw",
    [
        _response(usable=False, candidates=[_candidate_payload()]),
        _response([_candidate_payload() for _ in range(5)]),
        _response([_candidate_payload(target_label="")]),
        _response([_candidate_payload(target_label="x" * 65)]),
        _response([_candidate_payload(), _candidate_payload(target_label="Cooler")]),
        _response([_candidate_payload(objective="x" * 301)]),
        _response([_candidate_payload(reason="   ")]),
        _response([{**_candidate_payload(), "extra": "field"}]),
        _response([{k: v for k, v in _candidate_payload().items() if k != "reason"}]),
        '{"scene_usable": true, "scene_uncertainty": "", "candidates": [',
        '{"scene_usable": true, "scene_usable": false, "scene_uncertainty": "", "candidates": []}',
        json.dumps({"scene_usable": "yes", "scene_uncertainty": "", "candidates": []}),
        json.dumps({"scene_usable": True, "candidates": [], "extra": 1}),
        "[]",
    ],
)
def test_invalid_vision_output_is_rejected_not_completed(raw):
    with pytest.raises(AutotargetError) as failure:
        parse_vision_proposal(raw)
    assert failure.value.code == "vision_invalid_output"


def test_invalid_vision_output_is_an_error_not_an_empty_proposal():
    response = parse_vision_response("not json", model_key="gemma", checkpoint="checkpoint")
    assert response.proposal is None
    assert response.error_code == "vision_invalid_output"
    assert response.raw_text == "not json"


def test_valid_choice_payload_returns_validated_choice():
    choice = parse_choice_payload(_jev_payload(), frozenset({"c1", "none"}))
    assert isinstance(choice, JevChoice)
    assert choice.choice == "c1"
    assert choice.model == "typesafe/jev-1.13-20260917"
    assert dict(choice.probabilities) == {"c1": 0.8, "none": 0.2}


@pytest.mark.parametrize(
    "payload",
    [
        {**_jev_payload(), "model": "jev-latest"},
        {"model": "typesafe/jev-1.13-20260917", "answers": {}},
        {"model": "typesafe/jev-1.13-20260917", "answers": {"selection": {**_jev_payload()["answers"]["selection"], "type": "noul"}}},
        _jev_payload(choice="c9"),
        _jev_payload(probabilities={"c1": 0.8}),
        _jev_payload(probabilities={"c1": 0.5, "none": 0.1}),
        _jev_payload(choice="c1", probabilities={"c1": 0.2, "none": 0.8}),
        _jev_payload(confidence=1.5),
        _jev_payload(confidence=True),
        _jev_payload(probabilities={"c1": float("nan"), "none": 0.2}),
    ],
)
def test_invalid_choice_payload_is_rejected(payload):
    with pytest.raises(AutotargetError) as failure:
        parse_choice_payload(payload, frozenset({"c1", "none"}))
    assert failure.value.code == "jev_invalid_response"


def test_choice_request_criteria_are_the_only_legal_answers_and_stay_structured():
    candidates = (
        VisionCandidate("c1", "cooler", "Inspect the cooler", "Blocks access", "Lid visible", ""),
        VisionCandidate("c2", "bin", "Check the bin lid", "Lid is open", "Open lid", "Contents unknown"),
    )
    state, criteria = build_choice_request(
        expertise="Storage inspection",
        scene_uncertainty="Door edge is dim",
        candidates=candidates,
        prior_label=None,
    )
    assert set(criteria) == {"c1", "c2", "none"}
    assert criteria["c1"]["target_label"] == "cooler"
    assert isinstance(criteria["c1"], dict)
    assert state["operator_context"] == "Storage inspection"
    assert len(state["candidates"]) == 2
    body = build_choice_body(state, criteria)
    assert body["model"] == "typesafe/jev-1.13"
    assert body["questions"]["selection"]["type"] == "choice"
    assert body["questions"]["selection"]["criteria"] == criteria


def test_client_reports_unconfigured_without_key_and_never_touches_network(monkeypatch):
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: pytest.fail("network must not be used"))
    assert JevDecisionClient(environ={}).configured() is False
    assert JevDecisionClient(environ={"OPENROUTER_API_KEY": "   "}).configured() is False
    assert JevDecisionClient(environ={"OPENROUTER_API_KEY": "key"}).configured() is True
    with pytest.raises(AutotargetError) as failure:
        asyncio.run(
            JevDecisionClient(environ={}).decide(
                state={}, criteria={"c1": {}, "none": "x"}, timeout_seconds=1.0
            )
        )
    assert failure.value.code == "jev_unconfigured"


def test_client_caps_timeout_at_three_seconds_even_when_more_time_is_offered(monkeypatch):
    seen = []

    def fake_post(self, body, key, timeout_seconds):
        seen.append(timeout_seconds)
        return _jev_payload()

    monkeypatch.setattr(JevDecisionClient, "_post", fake_post)
    client = JevDecisionClient(environ={"OPENROUTER_API_KEY": "key"})
    asyncio.run(client.decide(state={}, criteria={"c1": {}, "none": "x"}, timeout_seconds=10.0))
    asyncio.run(client.decide(state={}, criteria={"c1": {}, "none": "x"}, timeout_seconds=1.2))
    assert seen == [3.0, 1.2]


def test_client_rejects_a_spent_budget_before_sending(monkeypatch):
    monkeypatch.setattr(JevDecisionClient, "_post", lambda *a, **k: pytest.fail("no request expected"))
    client = JevDecisionClient(environ={"OPENROUTER_API_KEY": "key"})
    with pytest.raises(AutotargetError) as failure:
        asyncio.run(client.decide(state={}, criteria={"c1": {}, "none": "x"}, timeout_seconds=0.0))
    assert failure.value.code == "jev_timeout"


class _FakeResponse:
    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, limit):
        return self.body[:limit]


@pytest.mark.parametrize(
    ("status", "code"),
    [(401, "jev_auth_failed"), (403, "jev_auth_failed"), (422, "jev_rejected"), (429, "jev_rate_limited"), (529, "jev_overloaded"), (500, "jev_failed")],
)
def test_http_failures_map_to_explicit_codes_without_echoing_the_key(monkeypatch, status, code):
    def fail(request, timeout):
        raise urllib.error.HTTPError(request.full_url, status, "failed", {}, None)

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    client = JevDecisionClient(environ={"OPENROUTER_API_KEY": "k-secret"})
    with pytest.raises(AutotargetError) as failure:
        client._post(b"{}", "k-secret", 1.0)
    assert failure.value.code == code
    assert "k-secret" not in str(failure.value)


@pytest.mark.parametrize(
    ("raised", "code"),
    [
        (urllib.error.URLError(socket.timeout("slow")), "jev_timeout"),
        (socket.timeout("slow"), "jev_timeout"),
        (urllib.error.URLError("refused"), "jev_unreachable"),
        (OSError("reset"), "jev_unreachable"),
    ],
)
def test_transport_failures_are_typed(monkeypatch, raised, code):
    def fail(request, timeout):
        raise raised

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    with pytest.raises(AutotargetError) as failure:
        JevDecisionClient(environ={"OPENROUTER_API_KEY": "key"})._post(b"{}", "key", 1.0)
    assert failure.value.code == code


def test_malformed_and_oversized_service_bodies_are_invalid_responses(monkeypatch):
    client = JevDecisionClient(environ={"OPENROUTER_API_KEY": "key"})
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _FakeResponse(b"not json"))
    with pytest.raises(AutotargetError) as malformed:
        client._post(b"{}", "key", 1.0)
    assert malformed.value.code == "jev_invalid_response"

    oversized = b"{" + b" " * (autotarget.MAX_JEV_RESPONSE_BYTES + 10) + b"}"
    monkeypatch.setattr(urllib.request, "urlopen", lambda request, timeout: _FakeResponse(oversized))
    with pytest.raises(AutotargetError) as too_large:
        client._post(b"{}", "key", 1.0)
    assert too_large.value.code == "jev_invalid_response"


def test_decision_record_is_bounded_and_marks_none_without_an_objective():
    candidates = tuple(
        VisionCandidate(f"c{index}", f"item {index}", "o" * 1000, "r" * 1000, "v" * 1000, "u" * 1000)
        for index in range(1, 5)
    )
    record = build_decision_record(
        cycle_id="cycle-1",
        execution_generation=3,
        brief_version=2,
        goal_mode="inferred",
        outcome="none",
        error_code=None,
        candidates=candidates,
        choice=JevChoice("none", {"c1": 0.1, "c2": 0.1, "c3": 0.1, "c4": 0.1, "none": 0.6}, 0.61234, "typesafe/jev-1.13-20260917"),
        grounding=None,
        evidence_id="evidence-1",
        frame_id=7,
        source_id="scout-1",
        source_epoch="41",
        applied_configuration_revision=None,
        timings_ms={"vision_ms": 1234.56},
    )
    assert len(record["candidates"]) == 4
    assert len(record["candidates"][0]["objective"]) == 300
    assert len(record["candidates"][0]["visual_evidence"]) == 160
    assert record["selected_candidate_id"] == "none"
    assert record["selected_label"] is None
    assert record["objective_provenance"] is None
    assert record["decision_confidence"] == 0.6123
    assert record["timings_ms"] == {"vision_ms": 1234.6}


def test_decision_record_names_the_selected_vision_candidate():
    candidates = (VisionCandidate("c1", "cooler", "Inspect the cooler", "Blocks access", "Lid visible", ""),)
    record = build_decision_record(
        cycle_id="cycle-2",
        execution_generation=1,
        brief_version=1,
        goal_mode="inferred",
        outcome="watching",
        error_code=None,
        candidates=candidates,
        choice=JevChoice("c1", {"c1": 0.9, "none": 0.1}, 0.8, "typesafe/jev-1.13-20260917"),
        grounding={"tool": "detect_objects", "status": "ok", "matched_count": 1},
        evidence_id="evidence-2",
        frame_id=8,
        source_id="scout-1",
        source_epoch="41",
        applied_configuration_revision=5,
        timings_ms={},
    )
    assert record["selected_candidate_id"] == "c1"
    assert record["selected_label"] == "cooler"
    assert record["objective_provenance"] == "vision_candidate"
    assert record["grounding"]["matched_count"] == 1
    assert record["applied_configuration_revision"] == 5
