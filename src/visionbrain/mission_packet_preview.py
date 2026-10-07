"""Internal deterministic packet previews over persisted mission metadata.

Previews preserve recorded evidence availability and observation state. They do
not inspect evidence files and are not fresh media verification.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .mission_record_projection import project_mission_records
from .mission_records import (
    EvidenceRecord,
    Finding,
    InspectionPacket,
    Observation,
    Record,
    record_to_dict,
    validate_record_bundle,
)

_PREVIEW_FORMAT = "visionbrain.packet-preview.v1"


@dataclass(frozen=True)
class PacketPreview:
    """One revision-bound draft packet and its internal canonical comparison form."""

    packet: InspectionPacket
    records: tuple[Record, ...]
    canonical_json: str
    content_sha256: str


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def build_packet_preview(rows: Mapping[str, Any]) -> PacketPreview:
    """Build a validated deterministic draft packet from one metadata row read.

    The supplied mapping is the complete result of
    ``MissionStore.read_mission_record_rows``. No evidence path is opened or
    checked; persisted availability is carried into the preview as recorded.
    """
    snapshot = rows["snapshot"]
    mission_id = snapshot["mission_id"]
    mission_revision = snapshot["revision"]
    if (
        not isinstance(mission_id, str)
        or not mission_id
        or isinstance(mission_revision, bool)
        or not isinstance(mission_revision, int)
        or mission_revision < 1
    ):
        raise ValueError("mission metadata has no valid identity and revision")

    projected = project_mission_records(
        snapshot,
        rows["evidence_rows"],
        rows["tool_rows"],
    )
    record_data = [record_to_dict(record) for record in projected]
    identity_payload = {
        "format": _PREVIEW_FORMAT,
        "mission_id": mission_id,
        "mission_revision": mission_revision,
        "records": record_data,
    }
    identity_digest = hashlib.sha256(_canonical_json(identity_payload).encode("utf-8")).hexdigest()

    findings = tuple(sorted(
        record.finding_id for record in projected if isinstance(record, Finding)
    ))
    observations = tuple(sorted(
        record.observation_id for record in projected if isinstance(record, Observation)
    ))
    evidence = tuple(sorted(
        record.evidence.evidence_id for record in projected if isinstance(record, EvidenceRecord)
    ))
    packet = InspectionPacket(
        packet_id=f"packet-{identity_digest[:32]}",
        mission_id=mission_id,
        mission_revision=mission_revision,
        state="draft",
        finding_ids=findings,
        observation_ids=observations,
        evidence_ids=evidence,
    )
    bundle = (*projected, packet)
    validate_record_bundle(bundle)

    canonical_json = _canonical_json({
        "format": _PREVIEW_FORMAT,
        "records": [record_to_dict(record) for record in bundle],
    })
    content_sha256 = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return PacketPreview(packet, bundle, canonical_json, content_sha256)
