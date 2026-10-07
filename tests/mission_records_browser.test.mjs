import assert from "node:assert/strict";
import {existsSync, readFileSync} from "node:fs";
import {dirname, resolve} from "node:path";
import {fileURLToPath} from "node:url";
import test from "node:test";

import {
  RecordValidationError,
  parseRecord,
  serializeRecord,
  validateBundle,
  validateRecord,
} from "../src/visionbrain/static/mission_records.mjs";

const testDirectory = dirname(fileURLToPath(import.meta.url));
const fixturePath = resolve(
  process.env.VISIONBRAIN_MISSION_RECORDS_FIXTURE ??
    resolve(testDirectory, "../../visionBrain-bridge/contracts/mission-records/v1/record_golden.json"),
);
if (!existsSync(fixturePath)) {
  throw new Error(
    `Released-record fixture is required at ${fixturePath}. Set VISIONBRAIN_MISSION_RECORDS_FIXTURE to the paired bridge contracts/mission-records/v1/record_golden.json path.`,
  );
}
const fixture = JSON.parse(readFileSync(fixturePath, "utf8"));

function applyMutation(record, mutation) {
  const copy = structuredClone(record);
  if (mutation === undefined) return copy;
  let target = copy;
  for (const key of mutation.path.slice(0, -1)) target = target[key];
  target[mutation.path.at(-1)] = mutation.value === "NaN" ? Number.NaN : mutation.value;
  return copy;
}

function assertReason(expected, callback) {
  assert.throws(callback, (error) =>
    error instanceof RecordValidationError && error.reason === expected,
  );
}

test("fixture metadata and unimplemented authority remain explicit", () => {
  assert.equal(fixture.fixture_version, 1);
  assert.equal(fixture.record_schema_version, 1);
  assert.equal(fixture.record_cases.length, 30);
  assert.equal(fixture.bundle_cases.length, 23);
  assert.deepEqual(fixture.dynamic_checks, {
    checks: [
      "review and packet state transitions",
      "mission revision conflict handling",
      "durable persistence and restart recovery",
      "timestamp clock mapping and capture provenance verification",
      "approval, export, or business action authority",
    ],
    implemented: false,
    status: "not_implemented",
  });
});

for (const vector of fixture.record_cases) {
  test(`record fixture: ${vector.name}`, () => {
    const record = applyMutation(vector.record, vector.mutation);
    if (!vector.valid) {
      assertReason(vector.expected_reason, () => validateRecord(record));
      return;
    }
    assert.strictEqual(validateRecord(record), record);
    const serialized = serializeRecord(record);
    assert.deepEqual(JSON.parse(serialized), record);
    assert.deepEqual(parseRecord(serialized), record);
  });
}

for (const vector of fixture.bundle_cases) {
  test(`bundle fixture: ${vector.name}`, () => {
    const records = structuredClone(vector.records);
    if (!vector.valid) {
      assertReason(vector.expected_reason, () => validateBundle(records));
      return;
    }
    assert.strictEqual(validateBundle(records), records);
    const roundTripped = records.map((record) => parseRecord(serializeRecord(record)));
    assert.deepEqual(roundTripped, records);
    assert.strictEqual(validateBundle(roundTripped), roundTripped);
  });
}

test("invalid JSON is rejected without coercing inputs", () => {
  assertReason("invalid_json", () => parseRecord("{"));
  const record = structuredClone(fixture.record_cases[0].record);
  record.capture_time_ms = "123";
  assertReason("invalid_integer", () => validateRecord(record));
});

test("unsafe raw revisions fail closed and the safe integer boundary round-trips", () => {
  const packet = structuredClone(fixture.record_cases.find((item) => item.name === "packet-state-draft").record);
  const unsafeJson = JSON.stringify(packet).replace(
    /(\"mission_revision\":)\d+/,
    (_match, prefix) => `${prefix}9007199254740993`,
  );
  assert.match(unsafeJson, /"mission_revision":9007199254740993/);
  assertReason("integer_precision_unsupported", () => parseRecord(unsafeJson));

  const boundary = {...packet, mission_revision: Number.MAX_SAFE_INTEGER};
  const decoded = parseRecord(JSON.stringify(boundary));
  assert.equal(decoded.mission_revision, Number.MAX_SAFE_INTEGER);
  assert.equal(JSON.parse(serializeRecord(decoded)).mission_revision, Number.MAX_SAFE_INTEGER);
});

test("external capability runtime identity must match a bundle capability", () => {
  const bundle = structuredClone(fixture.bundle_cases.find((item) => item.valid).records);
  const external = structuredClone(bundle.find((record) => record.record_type === "runtime_capability"));
  external.runtime_id = "different-runtime";
  assertReason("capability_mismatch", () => validateBundle(bundle, external));
});

test("external capability equality ignores object key order", () => {
  const bundle = structuredClone(fixture.bundle_cases.find((item) => item.valid).records);
  const capability = bundle.find((record) => record.record_type === "runtime_capability");
  const reordered = Object.fromEntries(Object.entries(capability).reverse());
  assert.strictEqual(validateBundle(bundle, reordered), bundle);
});

test("external capability must be a runtime capability record", () => {
  const bundle = structuredClone(fixture.bundle_cases.find((item) => item.valid).records);
  const finding = structuredClone(bundle.find((record) => record.record_type === "finding"));
  assertReason("capability_mismatch", () => validateBundle(bundle, finding));
});
