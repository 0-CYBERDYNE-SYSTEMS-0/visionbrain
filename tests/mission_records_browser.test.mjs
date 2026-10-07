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
const v2FixturePath = resolve(
  process.env.VISIONBRAIN_MISSION_RECORDS_V2_FIXTURE ??
    resolve(testDirectory, "../../visionBrain-bridge/contracts/mission-records/v2/record_golden.json"),
);
if (!existsSync(v2FixturePath)) {
  throw new Error(
    `Finding schema-2 fixture is required at ${v2FixturePath}. Set VISIONBRAIN_MISSION_RECORDS_V2_FIXTURE to the paired bridge contracts/mission-records/v2/record_golden.json path.`,
  );
}
const v2Fixture = JSON.parse(readFileSync(v2FixturePath, "utf8"));

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

test("V2 delta fixture metadata remains additive and bounded", () => {
  assert.equal(v2Fixture.fixture_version, 1);
  assert.equal(v2Fixture.record_schema_version, 2);
  assert.deepEqual(v2Fixture.supported_record_versions, [1, 2]);
  assert.equal(v2Fixture.scenarios.length, 4);
  assert.equal(v2Fixture.dynamic_checks.implemented, false);
  assert.equal(v2Fixture.dynamic_checks.status, "not_implemented");
  assert.equal(v2Fixture.scenarios[3].v1_client_rejection.expected_reason, "unsupported_record_version");
});

for (const scenario of v2Fixture.scenarios) {
  for (const vector of scenario.variants ?? [scenario]) {
    test(`V2 bundle fixture: ${vector.name ?? scenario.name}`, () => {
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
}

test("Finding schema 2 requires unique bounded text references", () => {
  const record = structuredClone(v2Fixture.scenarios[0].records.find((item) => item.record_type === "finding"));
  assert.strictEqual(validateRecord(record), record);

  const missing = structuredClone(record);
  delete missing.text_refs;
  assertReason("missing_field", () => validateRecord(missing));

  const empty = {...record, text_refs: []};
  assertReason("missing_text_reference", () => validateRecord(empty));

  const atLimit = {...record, text_refs: ["tr-0", "tr-1", "tr-2", "tr-3"]};
  assert.strictEqual(validateRecord(atLimit), atLimit);

  const tooMany = {...record, text_refs: ["tr-0", "tr-1", "tr-2", "tr-3", "tr-4"]};
  assertReason("too_many_text_references", () => validateRecord(tooMany));

  const duplicate = {...record, text_refs: [record.text_refs[0], record.text_refs[0]]};
  assertReason("duplicate_id", () => validateRecord(duplicate));

  const maxLength = {...record, text_refs: ["r".repeat(128)]};
  assert.strictEqual(validateRecord(maxLength), maxLength);

  const tooLong = {...record, text_refs: ["r".repeat(129)]};
  assertReason("invalid_text", () => validateRecord(tooLong));

  const wrongClaimType = {...record, claim_type: "localized_object"};
  assertReason("unexpected_text_reference", () => validateRecord(wrongClaimType));
});

test("schema 2 remains Finding-only and V1 Findings reject the added field", () => {
  const observation = structuredClone(v2Fixture.scenarios[0].records.find((item) => item.record_type === "observation"));
  assertReason("unsupported_record_version", () => validateRecord({...observation, record_schema_version: 2}));

  const finding = structuredClone(fixture.record_cases.find((item) => item.valid && item.record.record_type === "finding").record);
  assertReason("unknown_field", () => validateRecord({...finding, text_refs: []}));
});

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
