"""Finding schema-2 text-citation contract tests."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from visionbrain.mission_records import (
    Finding,
    RecordValidationError,
    deserialize_record,
    record_from_dict,
    record_to_dict,
    serialize_record,
    validate_record_bundle,
)


FIXTURE_ENV = "VISIONBRAIN_MISSION_RECORDS_V2_FIXTURE"
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "visionBrain-bridge"
    / "contracts"
    / "mission-records"
    / "v2"
    / "record_golden.json"
)
V1_FIXTURE_SHA256 = "89adcca972bbcd78f6a6bce94d79d907575652cc2a2ee26b6e8163b9222efa45"


def _fixture_path() -> Path:
    """Return the single bridge-owned V2 delta fixture."""
    path = Path(os.environ.get(FIXTURE_ENV, DEFAULT_FIXTURE))
    if not path.is_file():
        pytest.fail(
            f"Finding schema-2 fixture is required at {path}. Set {FIXTURE_ENV} "
            "to the paired bridge contracts/mission-records/v2/record_golden.json path."
        )
    return path


@pytest.fixture(scope="module")
def golden_fixture() -> dict[str, Any]:
    """Load shared vectors without a language-specific copy."""
    with _fixture_path().open(encoding="utf-8") as stream:
        return json.load(stream)


def _decode_bundle(case: dict[str, Any]):
    return [record_from_dict(value) for value in case["records"]]


def _assert_rejected(expected_reason: str, callback: Any) -> None:
    with pytest.raises(RecordValidationError) as caught:
        callback()
    assert caught.value.reason == expected_reason


def test_v2_delta_fixture_has_four_scenarios_and_preserves_v1_fixture_bytes(
    golden_fixture: dict[str, Any],
) -> None:
    """Keep V2 additive and leave the released V1 fixture frozen."""
    assert len(golden_fixture["scenarios"]) == 4
    v1_fixture = DEFAULT_FIXTURE.parents[1] / "v1" / "record_golden.json"
    assert hashlib.sha256(v1_fixture.read_bytes()).hexdigest() == V1_FIXTURE_SHA256


def test_shared_v2_bundle_scenarios_decode_validate_and_round_trip(
    golden_fixture: dict[str, Any],
) -> None:
    """Apply every shared cross-consumer vector and bounded variant."""
    for scenario in golden_fixture["scenarios"]:
        cases = scenario.get("variants", [scenario])
        for case in cases:
            records = _decode_bundle(case)
            if case["valid"]:
                validate_record_bundle(records)
                for record, raw in zip(records, case["records"], strict=True):
                    decoded = deserialize_record(serialize_record(record))
                    assert record_to_dict(decoded) == raw
            else:
                _assert_rejected(
                    case["expected_reason"], lambda records=records: validate_record_bundle(records)
                )


def test_finding_schema_two_requires_bounded_unique_text_refs(
    golden_fixture: dict[str, Any],
) -> None:
    """Enforce the V2-only required field and the existing four-reference limit."""
    valid_case = golden_fixture["scenarios"][0]
    raw = next(item for item in valid_case["records"] if item["record_type"] == "finding")

    missing = copy.deepcopy(raw)
    del missing["text_refs"]
    _assert_rejected("missing_field", lambda: record_from_dict(missing))

    ordinary = copy.deepcopy(raw)
    ordinary["claim_type"] = "localized_object"
    _assert_rejected("unexpected_text_reference", lambda: record_from_dict(ordinary))

    empty = copy.deepcopy(raw)
    empty["text_refs"] = []
    _assert_rejected("missing_text_reference", lambda: record_from_dict(empty))

    at_limit = copy.deepcopy(raw)
    at_limit["text_refs"] = [f"tr-{index}" for index in range(4)]
    record_from_dict(at_limit)

    duplicate = copy.deepcopy(raw)
    duplicate["text_refs"] = [raw["text_refs"][0], raw["text_refs"][0]]
    _assert_rejected("duplicate_id", lambda: record_from_dict(duplicate))

    too_many = copy.deepcopy(raw)
    too_many["text_refs"] = [f"tr-{index}" for index in range(5)]
    _assert_rejected("too_many_text_references", lambda: record_from_dict(too_many))

    max_length = copy.deepcopy(raw)
    max_length["text_refs"] = ["r" * 128]
    record_from_dict(max_length)

    too_long = copy.deepcopy(raw)
    too_long["text_refs"] = ["r" * 129]
    _assert_rejected("invalid_text", lambda: record_from_dict(too_long))


def test_v1_findings_keep_their_old_shape_and_reject_text_refs(
    golden_fixture: dict[str, Any],
) -> None:
    """V1 Findings remain citation-free and schema 2 stays Finding-only."""
    valid_case = golden_fixture["scenarios"][0]
    raw_finding = next(item for item in valid_case["records"] if item["record_type"] == "finding")
    v1_finding = copy.deepcopy(raw_finding)
    v1_finding["record_schema_version"] = 1
    v1_finding.pop("text_refs")
    decoded = record_from_dict(v1_finding)
    assert isinstance(decoded, Finding)
    assert "text_refs" not in record_to_dict(decoded)
    assert "text_refs" not in json.loads(serialize_record(decoded))
    _assert_rejected("invalid_sequence", lambda: record_to_dict(replace(decoded, text_refs=[])))

    v1_with_refs = copy.deepcopy(v1_finding)
    v1_with_refs["text_refs"] = raw_finding["text_refs"]
    _assert_rejected("unknown_field", lambda: record_from_dict(v1_with_refs))

    observation = copy.deepcopy(next(item for item in valid_case["records"] if item["record_type"] == "observation"))
    observation["record_schema_version"] = 2
    _assert_rejected("unsupported_record_version", lambda: record_from_dict(observation))
