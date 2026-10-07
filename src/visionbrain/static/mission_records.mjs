/* Read-only browser/Node codec for released mission record schema version 1. */

export const RECORD_SCHEMA_VERSION = 1;
const MAX_EVIDENCE_BYTES = 2 * 1024 * 1024;
const MAX_POLYGON_POINTS = 64;
const MAX_I63 = 9_223_372_036_854_775_807;

export class RecordValidationError extends Error {
  constructor(reason, path = "") {
    super(path ? `${reason}: ${path}` : reason);
    this.name = "RecordValidationError";
    this.reason = reason;
    this.path = path;
  }
}

const fields = {
  observation: ["observation_id", "mission_id", "status", "outcome", "source_id", "source_epoch", "frame_id", "observed_frame_id", "width", "height", "evidence_ids", "items", "tool_results", "detector_tool_revision", "prompt_config_revision", "capture_time_ms", "capture_time_provenance", "capture_time_quality", "received_at_ms", "client_session_id", "inference_session_id", "archive_session_id", "error_code", "record_schema_version"],
  evidence: ["mission_id", "evidence", "byte_length", "availability", "availability_reason", "source_observation_id", "source_id", "source_epoch", "frame_id", "capture_time_ms", "capture_time_provenance", "capture_time_quality", "created_at_ms", "client_session_id", "inference_session_id", "archive_session_id", "closeup_request_id", "brief_version", "brief_sha256", "model_provenance", "record_schema_version"],
  review: ["finding_id", "state", "actor", "reviewed_at_ms", "mission_revision", "note", "record_schema_version"],
  finding: ["finding_id", "mission_id", "claim", "claim_type", "visual_state", "reason", "evidence_id", "evidence_refs", "observation_ids", "item_refs", "items", "localization", "source_binding", "frame_id", "brief_version", "brief_sha256", "model_provenance", "review", "record_schema_version"],
  inspection_packet: ["packet_id", "mission_id", "mission_revision", "state", "finding_ids", "observation_ids", "evidence_ids", "created_at_ms", "exported_at_ms", "supersedes_packet_id", "record_schema_version"],
  runtime_capability: ["runtime_id", "supported_record_versions", "supported_modes", "supported_tools", "max_evidence_bytes", "max_polygon_points", "record_schema_version"],
};
const geometryFields = ["item_id", "label", "score", "box", "polygon", "source"];
const transformFields = ["origin_width", "origin_height", "rotation_degrees", "resized", "scale_x", "scale_y", "flipped", "width", "height"];
const evidenceRefFields = ["evidence_id", "sha256", "width", "height", "kind", "parent_evidence_id", "crop_box", "input_transform"];
const toolFields = ["tool_result_id", "tool", "status", "input_evidence_id", "items", "evidence_ids", "text", "error_code", "brief_version", "brief_sha256", "model_provenance", "unavailable_evidence_ids", "source_binding", "frame_id", "input_sha256", "evidence_sha256", "created_at_ms"];
const localizationFields = ["status", "basis", "evidence_id", "statements"];
const statementFields = ["claim", "label", "count", "positions"];
const bindingFields = ["source_id", "source_epoch"];

function fail(reason, path = "") {
  throw new RecordValidationError(reason, path);
}

function object(value, path) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) fail("invalid_object", path);
  const prototype = Object.getPrototypeOf(value);
  if (prototype !== Object.prototype && prototype !== null) fail("invalid_object", path);
  return value;
}

function exact(value, names, path) {
  object(value, path);
  const allowed = new Set(names);
  if (Object.keys(value).some((key) => !allowed.has(key))) fail("unknown_field", path);
  if (names.some((key) => !Object.hasOwn(value, key))) fail("missing_field", path);
  return value;
}

function text(value, path, {optional = false, max = 512} = {}) {
  if (optional && value === null) return;
  if (typeof value !== "string" || value.trim().length === 0 || value.length > max) fail("invalid_text", path);
}

function int(value, path, {optional = false, min = 0, max = null} = {}) {
  if (optional && value === null) return;
  if (typeof value !== "number" || !Number.isInteger(value) || value < min) fail("invalid_integer", path);
  if (!Number.isSafeInteger(value)) fail("integer_precision_unsupported", path);
  if (max !== null && value > max) fail("out_of_bounds", path);
}

function number(value, path, min = 0, max = 1) {
  if (typeof value !== "number" || !Number.isFinite(value)) fail("invalid_number", path);
  if (value < min || value > max) fail("out_of_bounds", path);
}

function choice(value, choices, reason, path) {
  if (typeof value !== "string" || !choices.includes(value)) fail(reason, path);
}

function sha256(value, path, optional = false) {
  if (optional && value === null) return;
  if (typeof value !== "string" || !/^[0-9a-f]{64}$/.test(value)) fail("invalid_sha256", path);
}

function list(value, path) {
  if (!Array.isArray(value)) fail("invalid_sequence", path);
  return value;
}

function strings(value, path) {
  const seen = new Set();
  list(value, path).forEach((item, index) => {
    text(item, `${path}[${index}]`);
    if (seen.has(item)) fail("duplicate_id", `${path}[${index}]`);
    seen.add(item);
  });
}

function ints(value, path, min) {
  const seen = new Set();
  list(value, path).forEach((item, index) => {
    int(item, `${path}[${index}]`, {min});
    if (seen.has(item)) fail("duplicate_value", `${path}[${index}]`);
    seen.add(item);
  });
}

function scalarMapping(value, path) {
  object(value, path);
  for (const [key, item] of Object.entries(value)) {
    text(key, `${path}.key`, {max: 128});
    if (item !== null && !["string", "number", "boolean"].includes(typeof item)) fail("invalid_mapping_value", `${path}.${key}`);
    if (typeof item === "number" && !Number.isFinite(item)) fail("invalid_number", `${path}.${key}`);
  }
}

function modelValues(value, path) {
  list(value, path).forEach((pair, index) => {
    if (!Array.isArray(pair) || pair.length !== 2) fail("invalid_model_provenance", `${path}[${index}]`);
    const [key, item] = pair;
    text(key, `${path}[${index}].key`, {max: 128});
    if (value.slice(0, index).some(([prior]) => prior === key)) fail("invalid_model_provenance", `${path}[${index}]`);
    if (item !== null && !["string", "number", "boolean"].includes(typeof item)) fail("invalid_model_provenance", `${path}[${index}]`);
    if (typeof item === "number" && !Number.isFinite(item)) fail("invalid_model_provenance", `${path}[${index}]`);
  });
}

function captureTime(record, path) {
  int(record.capture_time_ms, `${path}.capture_time_ms`, {optional: true, max: MAX_I63});
  choice(record.capture_time_quality, ["unknown", "unverified", "verified"], "invalid_capture_time_quality", `${path}.capture_time_quality`);
  if (record.capture_time_ms === null) {
    if (record.capture_time_provenance !== null || record.capture_time_quality !== "unknown") fail("capture_time_provenance_mismatch", path);
    return;
  }
  text(record.capture_time_provenance, `${path}.capture_time_provenance`, {max: 128});
  if (record.capture_time_quality === "unknown") fail("capture_time_provenance_mismatch", path);
}

function validateBinding(value, path) {
  exact(value, bindingFields, path);
  text(value.source_id, `${path}.source_id`, {max: 128});
  text(value.source_epoch, `${path}.source_epoch`, {max: 128});
}

function validateTransform(value, path) {
  exact(value, transformFields, path);
  for (const name of ["origin_width", "origin_height", "width", "height"]) int(value[name], `${path}.${name}`, {optional: true, min: 1});
  if (![0, 90, 180, 270].includes(value.rotation_degrees) || typeof value.resized !== "boolean" || typeof value.flipped !== "boolean") fail("invalid_transform", path);
  for (const name of ["scale_x", "scale_y"]) if (value[name] !== null) number(value[name], `${path}.${name}`, 1e-10, Infinity);
}

function validateBox(value, path) {
  if (!Array.isArray(value) || value.length !== 4) fail("invalid_geometry", path);
  value.forEach((item, index) => number(item, `${path}[${index}]`));
  if (value[2] <= value[0] || value[3] <= value[1]) fail("degenerate_geometry", path);
}

function validateGeometry(value, path) {
  exact(value, geometryFields, path);
  text(value.item_id, `${path}.item_id`, {max: 128});
  text(value.label, `${path}.label`, {max: 128});
  text(value.source, `${path}.source`, {max: 128});
  number(value.score, `${path}.score`);
  validateBox(value.box, `${path}.box`);
  if (value.polygon !== null) {
    if (!Array.isArray(value.polygon) || value.polygon.length < 3 || value.polygon.length > MAX_POLYGON_POINTS) fail("invalid_polygon", `${path}.polygon`);
    value.polygon.forEach((point, index) => {
      if (!Array.isArray(point) || point.length !== 2) fail("invalid_polygon", `${path}.polygon[${index}]`);
      point.forEach((coordinate, coordinateIndex) => number(coordinate, `${path}.polygon[${index}][${coordinateIndex}]`));
    });
    const area = value.polygon.reduce((total, point, index) => {
      const next = value.polygon[(index + 1) % value.polygon.length];
      return total + point[0] * next[1] - next[0] * point[1];
    }, 0);
    if (area === 0) fail("degenerate_geometry", `${path}.polygon`);
  }
}

function validateEvidenceRef(value) {
  exact(value, evidenceRefFields, "evidence");
  text(value.evidence_id, "evidence.evidence_id", {max: 128});
  sha256(value.sha256, "evidence.sha256");
  int(value.width, "evidence.width", {min: 1});
  int(value.height, "evidence.height", {min: 1});
  text(value.kind, "evidence.kind", {max: 64});
  text(value.parent_evidence_id, "evidence.parent_evidence_id", {optional: true, max: 128});
  if (value.crop_box !== null) validateBox(value.crop_box, "evidence.crop_box");
  if (value.input_transform !== null) validateTransform(value.input_transform, "evidence.input_transform");
}

function validateTool(value, path) {
  exact(value, toolFields, path);
  text(value.tool_result_id, `${path}.tool_result_id`, {max: 128});
  text(value.tool, `${path}.tool`, {max: 128});
  choice(value.status, ["ok", "empty", "unsupported", "failed", "timeout"], "invalid_tool_status", `${path}.status`);
  text(value.input_evidence_id, `${path}.input_evidence_id`, {max: 128});
  list(value.items, `${path}.items`).forEach((item, index) => validateGeometry(item, `${path}.items[${index}]`));
  strings(value.evidence_ids, `${path}.evidence_ids`);
  strings(value.unavailable_evidence_ids, `${path}.unavailable_evidence_ids`);
  if (typeof value.text !== "string" || value.text.length > 4_000) fail("invalid_text", `${path}.text`);
  if (value.status === "empty" && (value.items.length || value.evidence_ids.length || value.text.trim())) fail("tool_status_output_mismatch", `${path}.status`);
  text(value.error_code, `${path}.error_code`, {optional: true, max: 128});
  int(value.brief_version, `${path}.brief_version`, {min: 1});
  sha256(value.brief_sha256, `${path}.brief_sha256`, true);
  scalarMapping(value.model_provenance, `${path}.model_provenance`);
  if (value.source_binding !== null) {
    const binding = object(value.source_binding, `${path}.source_binding`);
    if (Object.keys(binding).length !== 2 || !Object.hasOwn(binding, "source_id") || !Object.hasOwn(binding, "source_epoch")) fail("invalid_source_binding", `${path}.source_binding`);
    text(binding.source_id, `${path}.source_binding.source_id`, {max: 128});
    text(binding.source_epoch, `${path}.source_binding.source_epoch`, {max: 128});
  }
  int(value.frame_id, `${path}.frame_id`, {optional: true});
  sha256(value.input_sha256, `${path}.input_sha256`, true);
  scalarMapping(value.evidence_sha256, `${path}.evidence_sha256`);
  for (const [id, digest] of Object.entries(value.evidence_sha256)) {
    text(id, `${path}.evidence_sha256.key`, {max: 128});
    sha256(digest, `${path}.evidence_sha256.${id}`);
  }
  int(value.created_at_ms, `${path}.created_at_ms`, {optional: true, max: MAX_I63});
}

function validateLocalization(value) {
  exact(value, localizationFields, "localization");
  choice(value.status, ["supported"], "invalid_localization_state", "localization.status");
  text(value.basis, "localization.basis", {max: 128});
  text(value.evidence_id, "localization.evidence_id", {max: 128});
  list(value.statements, "localization.statements").forEach((statement, index) => {
    const path = `localization.statements[${index}]`;
    exact(statement, statementFields, path);
    text(statement.claim, `${path}.claim`, {max: 500});
    text(statement.label, `${path}.label`, {max: 128});
    int(statement.count, `${path}.count`, {min: 1});
    list(statement.positions, `${path}.positions`);
    if (statement.positions.length !== statement.count) fail("localization_count_mismatch", `${path}.positions`);
    statement.positions.forEach((position, positionIndex) => {
      if (!Array.isArray(position) || position.length !== 2) fail("invalid_geometry", `${path}.positions[${positionIndex}]`);
      position.forEach((coordinate, coordinateIndex) => number(coordinate, `${path}.positions[${positionIndex}][${coordinateIndex}]`));
    });
  });
}

function validateReview(value) {
  validateRecord(value);
  if (value.record_type !== "review") fail("invalid_nested_record", "review");
}

export function validateRecord(record) {
  object(record, "record");
  const kind = record.record_type;
  if (!Object.hasOwn(fields, kind)) fail("unknown_record_type", "record_type");
  exact(record, [...fields[kind], "record_type"], "record");
  int(record.record_schema_version, "record_schema_version", {min: 1});
  if (record.record_schema_version !== RECORD_SCHEMA_VERSION) fail("unsupported_record_version", "record_schema_version");

  if (kind === "observation") {
    text(record.observation_id, "observation_id", {max: 128});
    text(record.mission_id, "mission_id", {max: 128});
    choice(record.status, ["observed", "held", "stale", "failed", "unavailable", "capture-time-unknown"], "invalid_observation_state", "status");
    choice(record.outcome, ["nonempty", "empty", "failed", "not_run", "unavailable"], "invalid_observation_outcome", "outcome");
    text(record.source_id, "source_id", {optional: true, max: 128});
    text(record.source_epoch, "source_epoch", {optional: true, max: 128});
    if ((record.source_id === null) !== (record.source_epoch === null)) fail("incomplete_source_binding", "source_epoch");
    for (const name of ["frame_id", "observed_frame_id"]) int(record[name], name, {optional: true});
    int(record.received_at_ms, "received_at_ms", {optional: true, max: MAX_I63});
    if (record.status === "held" && record.observed_frame_id === null) fail("held_observation_missing_observed_frame", "observed_frame_id");
    if (record.status === "failed" && (record.outcome !== "failed" || !record.error_code)) fail("failed_observation_missing_error", "error_code");
    if (record.outcome === "failed" && record.status !== "failed") fail("observation_status_outcome_mismatch", "outcome");
    if (record.status === "unavailable" && !["unavailable", "not_run"].includes(record.outcome)) fail("observation_status_outcome_mismatch", "outcome");
    if (["unavailable", "not_run"].includes(record.outcome) && record.status !== "unavailable") fail("observation_status_outcome_mismatch", "outcome");
    captureTime(record, "observation");
    if (record.status === "capture-time-unknown" && record.capture_time_quality === "verified") fail("capture_time_state_mismatch", "capture_time_quality");
    text(record.error_code, "error_code", {optional: true, max: 128});
    strings(record.evidence_ids, "evidence_ids");
    list(record.items, "items").forEach((item, index) => validateGeometry(item, `items[${index}]`));
    const toolIds = new Set();
    list(record.tool_results, "tool_results").forEach((result, index) => {
      validateTool(result, `tool_results[${index}]`);
      if (toolIds.has(result.tool_result_id)) fail("duplicate_id", `tool_results[${index}].tool_result_id`);
      toolIds.add(result.tool_result_id);
    });
    if (record.outcome === "empty" && (record.items.length || record.tool_results.some((result) => result.status === "ok" && (result.items.length || result.evidence_ids.length || result.text.trim())))) fail("observation_empty_has_output", "outcome");
    text(record.detector_tool_revision, "detector_tool_revision", {optional: true, max: 128});
    text(record.prompt_config_revision, "prompt_config_revision", {optional: true, max: 128});
    for (const name of ["client_session_id", "inference_session_id", "archive_session_id"]) text(record[name], name, {optional: true, max: 128});
    if ((record.width === null) !== (record.height === null)) fail("incomplete_dimensions", "width");
    int(record.width, "width", {optional: true, min: 1});
    int(record.height, "height", {optional: true, min: 1});
  } else if (kind === "evidence") {
    text(record.mission_id, "mission_id", {max: 128});
    validateEvidenceRef(record.evidence);
    int(record.byte_length, "byte_length", {min: 1});
    choice(record.availability, ["available", "missing", "corrupt", "rolled_off"], "invalid_evidence_availability", "availability");
    if (record.availability === "available") {
      if (record.availability_reason !== null) fail("availability_reason_mismatch", "availability_reason");
    } else text(record.availability_reason, "availability_reason", {max: 128});
    text(record.source_observation_id, "source_observation_id", {optional: true, max: 128});
    text(record.source_id, "source_id", {optional: true, max: 128});
    text(record.source_epoch, "source_epoch", {optional: true, max: 128});
    if ((record.source_id === null) !== (record.source_epoch === null)) fail("incomplete_source_binding", "source_epoch");
    int(record.frame_id, "frame_id", {optional: true});
    int(record.created_at_ms, "created_at_ms", {optional: true, max: MAX_I63});
    int(record.brief_version, "brief_version", {optional: true, min: 1});
    captureTime(record, "evidence");
    for (const name of ["client_session_id", "inference_session_id", "archive_session_id", "closeup_request_id"]) text(record[name], name, {optional: true, max: 128});
    sha256(record.brief_sha256, "brief_sha256", true);
    modelValues(record.model_provenance, "model_provenance");
  } else if (kind === "review") {
    text(record.finding_id, "finding_id", {max: 128});
    choice(record.state, ["pending", "accepted", "corrected", "rejected"], "invalid_review_state", "state");
    if (record.state === "pending") {
      if ([record.actor, record.reviewed_at_ms, record.mission_revision].some((item) => item !== null)) fail("pending_review_has_attribution", "state");
    } else {
      text(record.actor, "actor", {max: 128});
      int(record.reviewed_at_ms, "reviewed_at_ms", {min: 1});
      int(record.mission_revision, "mission_revision", {min: 1});
    }
    text(record.note, "note", {optional: true, max: 1000});
  } else if (kind === "finding") {
    text(record.finding_id, "finding_id", {max: 128});
    text(record.mission_id, "mission_id", {max: 128});
    text(record.claim, "claim", {max: 500});
    choice(record.claim_type, ["localized_object", "text_read", "visual_hypothesis"], "invalid_claim_type", "claim_type");
    choice(record.visual_state, ["candidate", "supported", "unresolved"], "invalid_visual_state", "visual_state");
    text(record.reason, "reason", {max: 256});
    text(record.evidence_id, "evidence_id", {max: 128});
    strings(record.evidence_refs, "evidence_refs");
    strings(record.observation_ids, "observation_ids");
    list(record.item_refs, "item_refs").forEach((pair, index) => {
      if (!Array.isArray(pair) || pair.length !== 2) fail("invalid_item_reference", `item_refs[${index}]`);
      text(pair[0], `item_refs[${index}].tool_result_id`, {max: 128});
      text(pair[1], `item_refs[${index}].item_id`, {max: 128});
    });
    list(record.items, "items").forEach((item, index) => validateGeometry(item, `items[${index}]`));
    if (record.source_binding !== null) validateBinding(record.source_binding, "source_binding");
    int(record.frame_id, "frame_id", {optional: true});
    int(record.brief_version, "brief_version", {optional: true, min: 1});
    sha256(record.brief_sha256, "brief_sha256", true);
    modelValues(record.model_provenance, "model_provenance");
    if (record.review !== null) {
      validateReview(record.review);
      if (record.review.finding_id !== record.finding_id) fail("review_finding_mismatch", "review.finding_id");
    }
    if (record.localization !== null) validateLocalization(record.localization);
  } else if (kind === "inspection_packet") {
    text(record.packet_id, "packet_id", {max: 128});
    text(record.mission_id, "mission_id", {max: 128});
    int(record.mission_revision, "mission_revision", {min: 1});
    choice(record.state, ["draft", "approved", "exported", "superseded"], "invalid_packet_state", "state");
    for (const name of ["finding_ids", "observation_ids", "evidence_ids"]) strings(record[name], name);
    int(record.created_at_ms, "created_at_ms", {optional: true});
    int(record.exported_at_ms, "exported_at_ms", {optional: true});
    text(record.supersedes_packet_id, "supersedes_packet_id", {optional: true, max: 128});
  } else {
    text(record.runtime_id, "runtime_id", {max: 128});
    ints(record.supported_record_versions, "supported_record_versions", 1);
    strings(record.supported_modes, "supported_modes");
    strings(record.supported_tools, "supported_tools");
    int(record.max_evidence_bytes, "max_evidence_bytes", {min: 1});
    int(record.max_polygon_points, "max_polygon_points", {min: 3});
    if (record.max_evidence_bytes > MAX_EVIDENCE_BYTES || record.max_polygon_points > MAX_POLYGON_POINTS) fail("unsupported_capability_limit", "runtime_capability");
  }
  return record;
}

function geometryFor(record) {
  if (record.record_type === "observation") return [...record.items, ...record.tool_results.flatMap((result) => result.items)];
  if (record.record_type === "finding") return record.items;
  return [];
}

function capabilityShape(record) {
  return JSON.stringify([record.supported_record_versions, record.supported_modes, record.supported_tools, record.max_evidence_bytes, record.max_polygon_points]);
}

function sameCapability(left, right) {
  return left.runtime_id === right.runtime_id
    && left.record_schema_version === right.record_schema_version
    && capabilityShape(left) === capabilityShape(right);
}

function mapIds(records, type, idField) {
  return new Map(records.filter((item) => item.record_type === type).map((item) => [item[idField], item]));
}

function sameMission(records, missionId) {
  return records.every((item) => item.mission_id === missionId);
}

/** Parse JSON text and validate one record. Duplicate object keys follow JSON.parse semantics. */
export function parseRecord(payload) {
  if (typeof payload !== "string") fail("invalid_json");
  let record;
  try {
    record = JSON.parse(payload);
  } catch {
    fail("invalid_json");
  }
  return validateRecord(record);
}

function sorted(value) {
  if (Array.isArray(value)) return value.map(sorted);
  if (value !== null && typeof value === "object") {
    return Object.fromEntries(Object.keys(value).sort().map((key) => [key, sorted(value[key])]));
  }
  return value;
}

/** Serialize a validated record with lexically sorted object keys. */
export function serializeRecord(record) {
  validateRecord(record);
  return JSON.stringify(sorted(record));
}

/** Validate references and source lineage within one bundle; performs no transitions. */
export function validateBundle(records, capability = null) {
  list(records, "records").forEach(validateRecord);
  const ids = new Map();
  const identifier = {
    observation: "observation_id", evidence: "evidence_id", finding: "finding_id",
    inspection_packet: "packet_id", review: "finding_id", runtime_capability: "runtime_id",
  };
  for (const record of records) {
    const value = record.record_type === "evidence" ? record.evidence.evidence_id : record[identifier[record.record_type]];
    const key = `${record.record_type}:${value}`;
    if (ids.has(key)) fail("duplicate_id", value);
    ids.set(key, record);
  }

  const observations = mapIds(records, "observation", "observation_id");
  const evidence = new Map(records.filter((item) => item.record_type === "evidence").map((item) => [item.evidence.evidence_id, item]));
  const findings = mapIds(records, "finding", "finding_id");
  const packets = mapIds(records, "inspection_packet", "packet_id");
  const capabilities = records.filter((item) => item.record_type === "runtime_capability");
  const reviews = records.filter((item) => item.record_type === "review");
  if (new Set(capabilities.map(capabilityShape)).size > 1) fail("conflicting_capabilities", "runtime_capability");
  if (capability !== null) {
    validateRecord(capability);
    if (capability.record_type !== "runtime_capability") fail("capability_mismatch", "runtime_capability");
  }
  if (capability !== null && capabilities.length && !capabilities.some((item) => sameCapability(item, capability))) fail("capability_mismatch", "runtime_capability");
  const effective = capability ?? capabilities[0] ?? null;
  if (effective !== null) {
    for (const record of records) if (!effective.supported_record_versions.includes(record.record_schema_version)) fail("capability_mismatch", "record_schema_version");
    for (const record of records) for (const item of geometryFor(record)) if (item.polygon !== null && item.polygon.length > effective.max_polygon_points) fail("capability_mismatch", "max_polygon_points");
    for (const item of evidence.values()) if (item.byte_length > effective.max_evidence_bytes) fail("capability_mismatch", "max_evidence_bytes");
  }

  const toolOutputs = new Map();
  for (const observation of observations.values()) {
    for (const evidenceId of observation.evidence_ids) {
      if (!evidence.has(evidenceId)) fail("missing_reference", `observation.evidence_ids:${evidenceId}`);
      if (evidence.get(evidenceId).mission_id !== observation.mission_id) fail("observation_evidence_mission_mismatch", `observation:${observation.observation_id}`);
    }
    for (const result of observation.tool_results) {
      if (toolOutputs.has(result.tool_result_id)) fail("duplicate_id", `tool_result:${result.tool_result_id}`);
      toolOutputs.set(result.tool_result_id, [observation, result]);
      const referenced = new Set([result.input_evidence_id, ...result.evidence_ids, ...result.unavailable_evidence_ids]);
      if ([...referenced].some((id) => !evidence.has(id))) fail("missing_reference", `observation.tool_results:${result.tool_result_id}`);
      if ([...referenced].some((id) => evidence.get(id).mission_id !== observation.mission_id)) fail("observation_evidence_mission_mismatch", `observation:${observation.observation_id}`);
      if (result.input_sha256 !== null && result.input_sha256 !== evidence.get(result.input_evidence_id).evidence.sha256) fail("tool_input_hash_mismatch", `tool_result:${result.tool_result_id}`);
      if (Object.keys(result.evidence_sha256).some((id) => !result.evidence_ids.includes(id))) fail("tool_evidence_reference_mismatch", `tool_result:${result.tool_result_id}`);
      for (const [id, digest] of Object.entries(result.evidence_sha256)) if (digest !== evidence.get(id).evidence.sha256) fail("tool_evidence_hash_mismatch", `tool_result:${result.tool_result_id}`);
      if (result.source_binding !== null && observation.source_id !== null && (result.source_binding.source_id !== observation.source_id || result.source_binding.source_epoch !== observation.source_epoch)) fail("tool_observation_lineage_mismatch", `tool_result:${result.tool_result_id}`);
      const expectedFrame = observation.status === "held" ? observation.observed_frame_id : observation.frame_id;
      if (result.frame_id !== null && expectedFrame !== null && result.frame_id !== expectedFrame) fail("tool_observation_lineage_mismatch", `tool_result:${result.tool_result_id}`);
    }
  }

  for (const [evidenceId, item] of evidence) {
    if (item.source_observation_id !== null) {
      const observation = observations.get(item.source_observation_id);
      if (!observation) fail("missing_reference", `evidence.source_observation_id:${item.source_observation_id}`);
      if (item.mission_id !== observation.mission_id) fail("evidence_observation_mission_mismatch", `evidence:${evidenceId}`);
      const expectedFrame = observation.status === "held" ? observation.observed_frame_id : observation.frame_id;
      if ((item.source_id !== null && item.source_id !== observation.source_id) || (item.source_epoch !== null && item.source_epoch !== observation.source_epoch) || (observation.status === "held" && item.frame_id !== expectedFrame) || (observation.status !== "held" && item.frame_id !== null && item.frame_id !== expectedFrame)) fail("evidence_observation_lineage_mismatch", `evidence:${evidenceId}`);
    }
    const parentId = item.evidence.parent_evidence_id;
    if (parentId !== null && !evidence.has(parentId)) fail("missing_reference", `evidence.parent_evidence_id:${parentId}`);
    if (parentId !== null && evidence.get(parentId).mission_id !== item.mission_id) fail("parent_evidence_mission_mismatch", `evidence:${evidenceId}`);
  }

  for (const finding of findings.values()) {
    const evidenceRefs = new Set([finding.evidence_id, ...finding.evidence_refs]);
    if ([...evidenceRefs].some((id) => !evidence.has(id))) fail("missing_reference", "finding.evidence_refs");
    if (finding.observation_ids.some((id) => !observations.has(id))) fail("missing_reference", "finding.observation_ids");
    if ([...evidenceRefs].some((id) => evidence.get(id).mission_id !== finding.mission_id)) fail("finding_evidence_mission_mismatch", `finding:${finding.finding_id}`);
    if (finding.observation_ids.some((id) => observations.get(id).mission_id !== finding.mission_id)) fail("finding_observation_mission_mismatch", `finding:${finding.finding_id}`);
    if (finding.source_binding !== null) for (const id of finding.observation_ids) {
      const observation = observations.get(id);
      if (observation.source_id !== null && (finding.source_binding.source_id !== observation.source_id || finding.source_binding.source_epoch !== observation.source_epoch)) fail("finding_observation_lineage_mismatch", `finding:${finding.finding_id}`);
    }
    for (const id of evidenceRefs) {
      const evidenceRecord = evidence.get(id);
      if (finding.source_binding !== null && evidenceRecord.source_id !== null && (finding.source_binding.source_id !== evidenceRecord.source_id || finding.source_binding.source_epoch !== evidenceRecord.source_epoch)) fail("finding_evidence_lineage_mismatch", `finding:${finding.finding_id}`);
      if (evidenceRecord.source_observation_id !== null && finding.observation_ids.length && !finding.observation_ids.includes(evidenceRecord.source_observation_id)) fail("finding_evidence_lineage_mismatch", `finding:${finding.finding_id}`);
    }
    for (const [toolId, itemId] of finding.item_refs) {
      const entry = toolOutputs.get(toolId);
      if (!entry || !entry[1].items.some((item) => item.item_id === itemId)) fail("missing_tool_item_reference", `finding:${finding.finding_id}`);
      const observation = entry[0];
      if (observation.mission_id !== finding.mission_id) fail("finding_tool_mission_mismatch", `finding:${finding.finding_id}`);
      if (finding.observation_ids.length && !finding.observation_ids.includes(observation.observation_id)) fail("finding_tool_observation_mismatch", `finding:${finding.finding_id}`);
      if (finding.source_binding !== null && observation.source_id !== null && (finding.source_binding.source_id !== observation.source_id || finding.source_binding.source_epoch !== observation.source_epoch)) fail("finding_tool_lineage_mismatch", `finding:${finding.finding_id}`);
    }
    if (finding.review !== null && finding.review.finding_id !== finding.finding_id) fail("review_finding_mismatch", "finding.review");
    if (finding.localization !== null && !evidenceRefs.has(finding.localization.evidence_id)) fail("finding_localization_evidence_mismatch", `finding:${finding.finding_id}`);
  }
  for (const review of reviews) if (!findings.has(review.finding_id)) fail("missing_reference", `review.finding_id:${review.finding_id}`);
  for (const packet of packets.values()) {
    const packetFindings = packet.finding_ids.map((id) => findings.get(id));
    const packetObservations = packet.observation_ids.map((id) => observations.get(id));
    if ([...packetFindings, ...packetObservations].some((item) => item === undefined)) fail("missing_reference", `packet:${packet.packet_id}`);
    if (!sameMission([...packetFindings, ...packetObservations], packet.mission_id)) fail("packet_mission_mismatch", `packet:${packet.packet_id}`);
    if (packet.evidence_ids.some((id) => !evidence.has(id))) fail("missing_reference", `packet.evidence_ids:${packet.packet_id}`);
    if (packet.evidence_ids.some((id) => evidence.get(id).mission_id !== packet.mission_id)) fail("packet_mission_mismatch", `packet:${packet.packet_id}`);
    if (packet.supersedes_packet_id !== null) {
      if (!packets.has(packet.supersedes_packet_id)) fail("missing_reference", `packet.supersedes_packet_id:${packet.packet_id}`);
      if (packets.get(packet.supersedes_packet_id).mission_id !== packet.mission_id) fail("packet_supersedes_mission_mismatch", `packet:${packet.packet_id}`);
    }
  }
  return records;
}
