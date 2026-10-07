"""Reference tests for the bridge-owned released-record golden fixture."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from visionbrain.mission_records import (
    EvidenceRecord,
    Finding,
    Observation,
    RecordValidationError,
    deserialize_record,
    record_from_dict,
    record_to_dict,
    serialize_record,
    validate_record_bundle,
)


FIXTURE_ENV = "VISIONBRAIN_MISSION_RECORDS_FIXTURE"
DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "visionBrain-bridge"
    / "contracts"
    / "mission-records"
    / "v1"
    / "record_golden.json"
)


def _fixture_path() -> Path:
    """Return the one bridge-owned fixture path or fail with remediation."""
    path = Path(os.environ.get(FIXTURE_ENV, DEFAULT_FIXTURE))
    if not path.is_file():
        pytest.fail(
            f"Released-record fixture is required at {path}. Set {FIXTURE_ENV} "
            "to the paired bridge contracts/mission-records/v1/record_golden.json path."
        )
    return path


@pytest.fixture(scope="module")
def golden_fixture() -> dict[str, Any]:
    """Load the exact bridge-owned fixture without a copied fallback."""
    with _fixture_path().open(encoding="utf-8") as stream:
        return json.load(stream)


def _apply_mutation(record: dict[str, Any], mutation: dict[str, Any] | None) -> dict[str, Any]:
    """Apply the fixture's explicit test-only mutation to a copied record."""
    mutated = copy.deepcopy(record)
    if mutation is None:
        return mutated
    target: Any = mutated
    for key in mutation["path"][:-1]:
        target = target[key]
    value = mutation["value"]
    target[mutation["path"][-1]] = float("nan") if value == "NaN" else value
    return mutated


def _assert_rejected(expected_reason: str, callback: Any) -> None:
    """Assert a stable validation reason without coupling to a display path."""
    with pytest.raises(RecordValidationError) as caught:
        callback()
    assert caught.value.reason == expected_reason


def test_fixture_metadata_declares_unimplemented_authority(golden_fixture: dict[str, Any]) -> None:
    """Keep review, packet, and business authority explicitly outside this codec."""
    assert golden_fixture["fixture_version"] == 1
    assert golden_fixture["record_schema_version"] == 1
    assert golden_fixture["dynamic_checks"] == {
        "checks": [
            "review and packet state transitions",
            "mission revision conflict handling",
            "durable persistence and restart recovery",
            "timestamp clock mapping and capture provenance verification",
            "approval, export, or business action authority",
        ],
        "implemented": False,
        "status": "not_implemented",
    }


@pytest.mark.parametrize("index", range(30))
def test_record_cases_decode_strictly_and_round_trip(
    golden_fixture: dict[str, Any], index: int
) -> None:
    """Exercise each standalone valid/invalid record vector from the shared fixture."""
    case = golden_fixture["record_cases"][index]
    record = _apply_mutation(case["record"], case.get("mutation"))
    if case["valid"]:
        decoded = deserialize_record(json.dumps(record, allow_nan=False))
        assert record_to_dict(decoded) == record
        assert record_to_dict(deserialize_record(serialize_record(decoded))) == record
    else:
        _assert_rejected(case["expected_reason"], lambda: record_from_dict(record))


@pytest.mark.parametrize("index", range(23))
def test_bundle_cases_preserve_lineage_and_capability_boundaries(
    golden_fixture: dict[str, Any], index: int
) -> None:
    """Validate cross-record references, source epochs, held frames, and bounds."""
    case = golden_fixture["bundle_cases"][index]
    records = [record_from_dict(record) for record in case["records"]]
    if case["valid"]:
        validate_record_bundle(records)
    else:
        _assert_rejected(
            case["expected_reason"], lambda: validate_record_bundle(records)
        )


def test_finding_evidence_lineage_is_checked_without_item_refs(
    golden_fixture: dict[str, Any]
) -> None:
    """A finding cannot point at another observation's evidence when item_refs is empty."""
    bundle = golden_fixture["bundle_cases"][0]
    records = [record_from_dict(record) for record in bundle["records"]]
    finding_index = next(i for i, record in enumerate(records) if isinstance(record, Finding))
    records[finding_index] = replace(
        records[finding_index], source_binding=None, observation_ids=("obs-old",), item_refs=()
    )
    _assert_rejected("finding_evidence_lineage_mismatch", lambda: validate_record_bundle(records))


def test_each_finding_evidence_ref_is_checked_independent_of_set_order(
    golden_fixture: dict[str, Any]
) -> None:
    """A consistent final set entry cannot mask another evidence/observation mismatch."""
    bundle = golden_fixture["bundle_cases"][0]
    records = [record_from_dict(record) for record in bundle["records"]]
    evidence = {
        record.evidence.evidence_id: record
        for record in records
        if isinstance(record, EvidenceRecord)
    }
    observations = {
        record.observation_id: record
        for record in records
        if isinstance(record, Observation)
    }
    finding_index = next(i for i, record in enumerate(records) if isinstance(record, Finding))
    finding = records[finding_index]
    extra_evidence_refs = ("ev-old",)
    evidence_refs = set(extra_evidence_refs) | {finding.evidence_id}
    last_evidence_id = list(evidence_refs)[-1]
    matching_observation_id = evidence[last_evidence_id].source_observation_id
    matching_observation = observations[matching_observation_id]
    matching_tool = matching_observation.tool_results[0]
    records[finding_index] = replace(
        finding,
        source_binding=None,
        evidence_refs=extra_evidence_refs,
        observation_ids=(matching_observation_id,),
        item_refs=((matching_tool.tool_result_id, matching_tool.items[0].item_id),),
    )
    _assert_rejected("finding_evidence_lineage_mismatch", lambda: validate_record_bundle(records))


def test_strict_json_decode_rejects_duplicate_keys() -> None:
    """Reject duplicate JSON keys before a record can be interpreted."""
    _assert_rejected(
        "duplicate_json_key",
        lambda: deserialize_record('{"record_type":"observation","record_type":"finding"}'),
    )


def test_strict_json_decode_rejects_nonfinite_constants() -> None:
    """Reject non-finite JSON constants before record decoding."""
    _assert_rejected("invalid_number", lambda: deserialize_record("NaN"))
