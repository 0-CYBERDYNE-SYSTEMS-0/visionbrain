import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {webcrypto} from 'node:crypto';
import test from 'node:test';

import {createGroundControlMissionInspector, groundControlMissionLimits, jpegDimensions} from '../src/visionbrain/static/groundcontrol_missions.mjs';
import {validateBundle} from '../src/visionbrain/static/mission_records.mjs';

class FakeSocket {
  readyState = 1;
  listeners = new Map();
  sent = [];

  addEventListener(type, callback) {
    const callbacks = this.listeners.get(type) || [];
    callbacks.push(callback);
    this.listeners.set(type, callbacks);
  }

  send(value) { this.sent.push(JSON.parse(value)); }

  emit(type, value) {
    for (const callback of this.listeners.get(type) || []) callback(value);
  }

  reply(command, result, {ok = true, revision = null, error = undefined} = {}) {
    this.emit('message', {data: JSON.stringify({
      type: 'mission_reply', schema_version: 1, request_id: command.request_id,
      mission_id: command.mission_id, revision, ok, result, ...(error ? {error} : {}),
    })});
  }
}

function createClient(socket, options = {}) {
  let id = 0;
  const updates = [];
  const client = createGroundControlMissionInspector({
    onChange: (state) => updates.push(state),
    cryptoApi: webcrypto,
    idFactory: () => `req-${++id}`,
    ...options,
  });
  client.bindSocket(socket);
  return {client, updates};
}

async function loadMission(client, socket, missionId, revision, extra = {}) {
  const loading = client.selectMission(missionId);
  const get = socket.sent.at(-1);
  assert.equal(get.command, 'get');
  socket.reply(get, {snapshot: {
    mission_id: missionId, revision, last_sequence: 0, state: 'completed',
    source_binding: {source_id: 'camera-a', source_epoch: '9'},
    findings: [], evidence: [], ...extra,
  }}, {revision});
  await Promise.resolve();
  const events = socket.sent.at(-1);
  assert.equal(events.command, 'events_since');
  socket.reply(events, {events: [], last_sequence: 0}, {revision});
  await loading;
}

async function flushQueue() {
  await new Promise((resolve) => setImmediate(resolve));
}

function evidenceSnapshot(sha256, bytes, overrides = {}) {
  return {
    findings: [{finding_id: 'finding-1', claim: 'inspect roof', status: 'unresolved',
      evidence_id: 'primary-1', evidence_refs: ['support-1', 'primary-1'], review: null}],
    evidence: [{evidence_id: 'primary-1', available: true, sha256, width: 1, height: 2, bytes,
      source_id: 'camera-a', source_epoch: '9', frame_id: 44},
      {evidence_id: 'support-1', available: true, sha256, width: 1, height: 2, bytes,
        source_id: null, source_epoch: null, frame_id: null}],
    ...overrides,
  };
}

function tinyJpeg(length = 100) {
  const bytes = new Uint8Array(length);
  bytes.set([0xff, 0xd8, 0xff, 0xc0, 0x00, 0x0b, 0x08, 0x00, 0x02, 0x00, 0x01, 0x01, 0x01, 0x11, 0x00]);
  bytes.set([0xff, 0xd9], length - 2);
  return bytes;
}

function sha256(bytes) {
  return webcrypto.subtle.digest('SHA-256', bytes).then((digest) =>
    [...new Uint8Array(digest)].map((value) => value.toString(16).padStart(2, '0')).join(''),
  );
}

function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(',')}]`;
  if (value && typeof value === 'object') {
    return `{${Object.keys(value).sort().map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  }
  return JSON.stringify(value);
}

async function packetPreviewResult(missionId, revision, overrides = {}) {
  const packet = {
    record_type: 'inspection_packet', record_schema_version: 1, packet_id: 'packet-1',
    mission_id: missionId, mission_revision: revision, state: 'draft', finding_ids: [],
    observation_ids: [], evidence_ids: [], created_at_ms: null, exported_at_ms: null, supersedes_packet_id: null,
  };
  const bundle = [packet];
  const canonical = canonicalJson({format: 'visionbrain.packet-preview.v1', records: bundle});
  return {
    envelope_version: 1, kind: 'inspection_packet_preview',
    verification: {basis: 'persisted_metadata_only', fresh_media_verified: false},
    mission_id: missionId, mission_revision: revision, packet, bundle,
    canonical_json: canonical, content_sha256: await sha256(new TextEncoder().encode(canonical)),
    ...overrides,
  };
}

// Exact Python json.dumps(..., ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False) output.
const PYTHON_FLOAT_PREVIEW_JSON = '{"format":"visionbrain.packet-preview.v1","records":[{"archive_session_id":null,"availability":"available","availability_reason":null,"brief_sha256":null,"brief_version":null,"byte_length":1,"capture_time_ms":null,"capture_time_provenance":null,"capture_time_quality":"unknown","client_session_id":null,"closeup_request_id":null,"created_at_ms":null,"evidence":{"crop_box":null,"evidence_id":"evidence-floats","height":2,"input_transform":null,"kind":"original","parent_evidence_id":null,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","width":1},"frame_id":null,"inference_session_id":null,"mission_id":"mission-preview-floats","model_provenance":[],"record_schema_version":1,"record_type":"evidence","source_epoch":null,"source_id":null,"source_observation_id":null},{"brief_sha256":null,"brief_version":null,"claim":"Localized leaf café","claim_type":"localized_object","evidence_id":"evidence-floats","evidence_refs":["evidence-floats"],"finding_id":"finding-floats","frame_id":null,"item_refs":[],"items":[{"box":[0.0,-0.0,0.5,1.0],"item_id":"item-floats","label":"leaf","polygon":null,"score":1e-05,"source":"visionbrain"}],"localization":null,"mission_id":"mission-preview-floats","model_provenance":[["small_float",1e-07]],"observation_ids":[],"reason":"bounded geometry sample","record_schema_version":1,"record_type":"finding","review":null,"source_binding":null,"visual_state":"supported"},{"created_at_ms":null,"evidence_ids":["evidence-floats"],"exported_at_ms":null,"finding_ids":["finding-floats"],"mission_id":"mission-preview-floats","mission_revision":9,"observation_ids":[],"packet_id":"packet-1","record_schema_version":1,"record_type":"inspection_packet","state":"draft","supersedes_packet_id":null}]}';

test('Ground Control inspector sends only canonical read commands on the injected socket', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  const listed = client.refreshMissions();
  const command = socket.sent[0];
  assert.deepEqual(command, {
    type: 'mission_command', schema_version: 1, request_id: 'req-1', mission_id: null,
    command: 'list', args: {limit: groundControlMissionLimits.LIST_LIMIT},
  });
  socket.reply(command, {missions: [{mission_id: 'm-1', revision: 3, state: 'running', updated_at_ms: 12}]});
  await listed;
  assert.deepEqual([...new Set(socket.sent.map((item) => item.command))], ['list']);
  assert.equal(client.state.missions[0].mission_id, 'm-1');
});

test('missing mission capability or reply timeout is surfaced without a second transport', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket, {requestTimeoutMs: 5});
  await client.refreshMissions();
  assert.match(client.state.status, /mission_reply_timeout/);
  assert.equal(socket.sent.length, 1);
  assert.equal(socket.sent[0].command, 'list');
});

test('late get reply from an older mission selection cannot replace the current snapshot', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  const oldLoad = client.selectMission('mission-old');
  const oldGet = socket.sent.at(-1);
  const currentLoad = client.selectMission('mission-current');
  const currentGet = socket.sent.at(-1);
  socket.reply(currentGet, {snapshot: {mission_id: 'mission-current', revision: 8, last_sequence: 0, findings: [], evidence: []}}, {revision: 8});
  await Promise.resolve();
  const catchup = socket.sent.at(-1);
  socket.reply(catchup, {events: [], last_sequence: 0}, {revision: 8});
  await currentLoad;
  socket.reply(oldGet, {snapshot: {mission_id: 'mission-old', revision: 2, last_sequence: 0, findings: [], evidence: []}}, {revision: 2});
  await oldLoad;
  assert.equal(client.state.selectedMissionId, 'mission-current');
  assert.equal(client.state.snapshot.mission_id, 'mission-current');
  assert.equal(client.state.snapshot.revision, 8);
});

test('primary evidence assembles sequential bounded chunks and verifies exact hash and dimensions', async () => {
  const bytes = tinyJpeg(groundControlMissionLimits.EVIDENCE_CHUNK_BYTES + 8);
  const digest = await sha256(bytes);
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-1', 4, evidenceSnapshot(digest, bytes.length));
  client.setFinding('finding-1');
  assert.deepEqual(client.state.evidenceIds, ['primary-1', 'support-1']);
  const loading = client.selectEvidence('primary-1');
  await flushQueue();
  const first = socket.sent.at(-1);
  assert.equal(first.command, 'get_evidence');
  assert.deepEqual(first.args, {evidence_id: 'primary-1', offset: 0, length: groundControlMissionLimits.EVIDENCE_CHUNK_BYTES});
  assert.equal(socket.sent.filter((item) => item.command === 'get_evidence').length, 1);
  socket.reply(first, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes.slice(0, groundControlMissionLimits.EVIDENCE_CHUNK_BYTES)).toString('base64'), width: 1, height: 2}, {revision: 4});
  await Promise.resolve();
  await Promise.resolve();
  const second = socket.sent.at(-1);
  assert.equal(second.command, 'get_evidence');
  assert.equal(second.args.offset, groundControlMissionLimits.EVIDENCE_CHUNK_BYTES);
  assert.equal(second.args.length, 8);
  socket.reply(second, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length,
    offset: second.args.offset, jpeg_b64: Buffer.from(bytes.slice(second.args.offset)).toString('base64'), width: 1, height: 2}, {revision: 4});
  const media = await loading;
  assert.equal(media.width, 1);
  assert.equal(media.height, 2);
  assert.equal(media.lineage, 'Current source binding · camera-a · epoch 9');
  assert.deepEqual(media.bytes, bytes);
  assert.match(client.state.evidenceStatus, /^Verified · 1×2/);
});

test('hash mismatch and stale evidence completion never publish media', async () => {
  const bytes = tinyJpeg(64);
  const digest = await sha256(bytes);
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-2', 5, evidenceSnapshot(digest, bytes.length));
  client.setFinding('finding-1');
  const loading = client.selectEvidence('primary-1');
  await flushQueue();
  const fetch = socket.sent.at(-1);
  const corrupt = bytes.slice();
  corrupt[20] ^= 0x01;
  socket.reply(fetch, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(corrupt).toString('base64'), width: 1, height: 2}, {revision: 5});
  await loading;
  assert.equal(client.state.media, null);
  assert.match(client.state.evidenceStatus, /evidence_hash_mismatch/);

  const staleLoad = client.selectEvidence('primary-1');
  await flushQueue();
  const staleFetch = socket.sent.at(-1);
  client.setFinding('finding-1');
  const currentLoad = client.selectEvidence('support-1');
  socket.reply(staleFetch, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes).toString('base64'), width: 1, height: 2}, {revision: 5});
  await staleLoad;
  await flushQueue();
  const currentFetch = socket.sent.at(-1);
  assert.equal(currentFetch.args.evidence_id, 'support-1');
  socket.reply(currentFetch, {evidence_id: 'support-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes).toString('base64'), width: 1, height: 2}, {revision: 5});
  const currentMedia = await currentLoad;
  assert.equal(currentMedia.evidenceId, 'support-1');
  assert.equal(client.state.selectedEvidenceId, 'support-1');
  assert.equal(client.state.media.evidenceId, 'support-1');
});

test('decoded JPEG dimensions must match canonical and chunk metadata before media is inspected', async () => {
  const bytes = tinyJpeg(64);
  const digest = await sha256(bytes);
  const snapshot = evidenceSnapshot(digest, bytes.length);
  snapshot.evidence[0].width = 2;
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-dimensions', 6, snapshot);
  client.setFinding('finding-1');
  const loading = client.selectEvidence('primary-1');
  await flushQueue();
  const fetch = socket.sent.at(-1);
  socket.reply(fetch, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes).toString('base64'), width: 2, height: 2}, {revision: 6});
  await loading;
  assert.equal(client.state.media, null);
  assert.match(client.state.evidenceStatus, /evidence_dimensions_mismatch/);
});

test('missing media reasons and unknown provenance remain explicit', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-3', 2, {
    findings: [{finding_id: 'finding-1', evidence_id: 'primary-1', evidence_refs: ['support-1']}],
    evidence: [
      {evidence_id: 'primary-1', available: false, availability_reason: 'corrupt'},
      {evidence_id: 'support-1', available: true, width: 1, height: 2},
    ],
  });
  client.setFinding('finding-1');
  await client.selectEvidence('primary-1');
  assert.match(client.state.evidenceStatus, /Unavailable · corrupt/);
  await client.selectEvidence('support-1');
  assert.match(client.state.evidenceStatus, /Historical · source provenance unknown/);
});

test('unknown availability never triggers an evidence request', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-availability', 2, {
    findings: [{finding_id: 'finding-1', evidence_id: 'primary-1', evidence_refs: []}],
    evidence: [{evidence_id: 'primary-1', sha256: 'a'.repeat(64), width: 1, height: 2, bytes: 24}],
  });
  client.setFinding('finding-1');
  await client.selectEvidence('primary-1');
  assert.match(client.state.evidenceStatus, /availability unknown/);
  assert.equal(socket.sent.some((item) => item.command === 'get_evidence'), false);
});

test('evidence not linked by the selected finding is rejected before fetch', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-linkage', 3, {
    findings: [{finding_id: 'finding-1', evidence_id: 'primary-1', evidence_refs: ['support-1']}],
    evidence: [
      {evidence_id: 'primary-1', available: true, sha256: 'a'.repeat(64), width: 1, height: 2, bytes: 24},
      {evidence_id: 'support-1', available: true, sha256: 'b'.repeat(64), width: 1, height: 2, bytes: 24},
      {evidence_id: 'unrelated-1', available: true, sha256: 'c'.repeat(64), width: 1, height: 2, bytes: 24},
    ],
  });
  client.setFinding('finding-1');
  await client.selectEvidence('unrelated-1');
  assert.equal(client.state.selectedEvidenceId, null);
  assert.match(client.state.evidenceStatus, /not linked to the selected finding/);
  assert.equal(socket.sent.some((item) => item.command === 'get_evidence'), false);
});

test('rapid evidence selections keep one active transfer and coalesce to the latest selection', async () => {
  const bytes = tinyJpeg(64);
  const digest = await sha256(bytes);
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-rapid', 7, evidenceSnapshot(digest, bytes.length));
  client.setFinding('finding-1');
  const firstLoad = client.selectEvidence('primary-1');
  await flushQueue();
  const first = socket.sent.at(-1);
  const superseded = client.selectEvidence('support-1');
  const latest = client.selectEvidence('primary-1');
  assert.equal(socket.sent.filter((item) => item.command === 'get_evidence').length, 1);
  assert.match(client.state.evidenceStatus, /latest selection/);
  socket.reply(first, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes).toString('base64'), width: 1, height: 2}, {revision: 7});
  await firstLoad;
  assert.equal(await superseded, null);
  await flushQueue();
  const finalFetch = socket.sent.at(-1);
  assert.equal(finalFetch.command, 'get_evidence');
  assert.equal(finalFetch.args.evidence_id, 'primary-1');
  assert.equal(socket.sent.filter((item) => item.command === 'get_evidence').length, 2);
  socket.reply(finalFetch, {evidence_id: 'primary-1', sha256: digest, total_bytes: bytes.length, offset: 0,
    jpeg_b64: Buffer.from(bytes).toString('base64'), width: 1, height: 2}, {revision: 7});
  assert.equal((await latest).evidenceId, 'primary-1');
  assert.equal(client.state.media.evidenceId, 'primary-1');
});

test('event catch-up updates revision and clears media selection', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-4', 3, evidenceSnapshot('a'.repeat(64), 40));
  client.setFinding('finding-1');
  const evidenceLoading = client.selectEvidence('primary-1');
  await flushQueue();
  const evidenceFetch = socket.sent.at(-1);
  socket.emit('message', {data: JSON.stringify({type: 'mission_event', mission_id: 'mission-4', sequence: 1})});
  const eventRequest = socket.sent.at(-1);
  assert.equal(eventRequest.command, 'events_since');
  assert.deepEqual(eventRequest.args, {cursor: 0, limit: 100});
  socket.reply(eventRequest, {events: [{mission_id: 'mission-4', sequence: 1, revision: 4,
    data: {snapshot: {mission_id: 'mission-4', revision: 4, last_sequence: 1, findings: [], evidence: []}}}], last_sequence: 1}, {revision: 4});
  await flushQueue();
  socket.reply(evidenceFetch, {evidence_id: 'primary-1', sha256: 'a'.repeat(64), total_bytes: 40, offset: 0,
    jpeg_b64: Buffer.from(tinyJpeg(40)).toString('base64'), width: 1, height: 2}, {revision: 3});
  await evidenceLoading;
  assert.equal(client.state.snapshot.revision, 4);
  assert.equal(client.state.media, null);
  assert.equal(client.state.selectedFindingId, null);
});

test('canonical JPEG dimension parser rejects missing and malformed dimensions', () => {
  assert.deepEqual(jpegDimensions(tinyJpeg(24)), {width: 1, height: 2});
  assert.throws(() => jpegDimensions(new Uint8Array([0xff, 0xd8, 0xff, 0xd9])), /jpeg_dimensions_missing/);
  assert.throws(() => jpegDimensions(new Uint8Array([0xff, 0xd8, 0xff, 0xc0, 0x00, 0x01])), /invalid_jpeg_segment/);
});

test('static inspector is hub-only, memory-token labeled, and exposes no mutation controls', async () => {
  const html = await readFile(new URL('../src/visionbrain/static/index.html', import.meta.url), 'utf8');
  const helper = await readFile(new URL('../src/visionbrain/static/groundcontrol_missions.mjs', import.meta.url), 'utf8');
  assert.match(html, /<label for="gc-mi-token">Mission access token<\/label>/);
  assert.match(html, /Token is kept only in this page\. Reconnect after changing it\./);
  assert.match(html, /Server requires an access token:/);
  assert.doesNotMatch(html, /(?:<label for="gc-mi-token">VB_MISSION_TOKEN|separate from VB_TOKEN|Enter VB_MISSION_TOKEN|access token \(VB_TOKEN\))/);
  assert.match(html, /hello\.capabilities = \['mission\.v1'\]/);
  assert.match(html, /review\.actor/);
  assert.match(html, /if \(LIVE\.mode === 'hub'\)/);
  assert.match(html, /groundControlMissionInspector\?\.bindSocket\(ws\)/);
  const inlineScript = html.match(/<script>([\s\S]*?)<\/script>/)?.[1];
  assert.ok(inlineScript);
  assert.doesNotThrow(() => new Function(inlineScript));
  const moduleScript = html.match(/<script type="module">([\s\S]*?)<\/script>/)?.[1];
  assert.ok(moduleScript);
  assert.doesNotThrow(() => new Function(moduleScript.replace(/^\s*import .*;\s*/, '')));
  assert.doesNotMatch(helper, /localStorage/);
  for (const mutation of ['review_finding', 'attach_evidence', 'pause', 'resume', 'cancel', 'create']) {
    assert.equal(helper.includes(`'${mutation}'`), false, `${mutation} must not be emitted`);
  }
  assert.match(html, /type="module"/);
  assert.match(html, /Metadata-only: availability and stale status reflect the last persisted record state; media is not freshly checked\./);
  assert.match(helper, /'preview_packet'/);
});

test('packet preview binds to the selected mission revision and verifies the canonical bundle', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  await loadMission(client, socket, 'mission-preview', 9, {state: 'paused', cycle_id: null, watch_lease: null});
  const loading = client.previewSelectedPacket();
  const request = socket.sent.at(-1);
  assert.equal(request.command, 'preview_packet');
  assert.equal(request.mission_id, 'mission-preview');
  assert.equal(request.expected_revision, 9);
  assert.deepEqual(request.args, {});
  const result = await packetPreviewResult('mission-preview', 9);
  socket.reply(request, result, {revision: 9});
  const preview = await loading;
  assert.equal(preview.missionRevision, 9);
  assert.equal(preview.packet.state, 'draft');
  assert.equal(preview.contentSha256, result.content_sha256);
  assert.match(client.state.packetPreviewStatus, /^Verified record hash/);
  assert.deepEqual([...new Set(socket.sent.map((item) => item.command))].sort(), ['events_since', 'get', 'preview_packet']);
});

test('packet preview accepts Python canonical finite-float spellings', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  const missionId = 'mission-preview-floats';
  await loadMission(client, socket, missionId, 9, {state: 'completed', cycle_id: null, watch_lease: null});
  const loading = client.previewSelectedPacket();
  const request = socket.sent.at(-1);
  const packet = {
    record_type: 'inspection_packet', record_schema_version: 1, packet_id: 'packet-1',
    mission_id: missionId, mission_revision: 9, state: 'draft', finding_ids: ['finding-floats'],
    observation_ids: [], evidence_ids: ['evidence-floats'], created_at_ms: null, exported_at_ms: null, supersedes_packet_id: null,
  };
  const evidence = {
    record_type: 'evidence', record_schema_version: 1, mission_id: missionId,
    evidence: {evidence_id: 'evidence-floats', sha256: 'a'.repeat(64), width: 1, height: 2,
      kind: 'original', parent_evidence_id: null, crop_box: null, input_transform: null},
    byte_length: 1, availability: 'available', availability_reason: null,
    source_observation_id: null, source_id: null, source_epoch: null, frame_id: null,
    capture_time_ms: null, capture_time_provenance: null, capture_time_quality: 'unknown', created_at_ms: null,
    client_session_id: null, inference_session_id: null, archive_session_id: null, closeup_request_id: null,
    brief_version: null, brief_sha256: null, model_provenance: [],
  };
  const finding = {
    record_type: 'finding', record_schema_version: 1, finding_id: 'finding-floats', mission_id: missionId,
    claim: 'Localized leaf café', claim_type: 'localized_object', visual_state: 'supported',
    reason: 'bounded geometry sample', evidence_id: 'evidence-floats', evidence_refs: ['evidence-floats'],
    observation_ids: [], item_refs: [], items: [{
      item_id: 'item-floats', label: 'leaf', score: 1e-05, box: [0.0, -0.0, 0.5, 1.0],
      polygon: null, source: 'visionbrain',
    }],
    localization: null, source_binding: null, frame_id: null, brief_version: null, brief_sha256: null,
    model_provenance: [['small_float', 1e-07]], review: null,
  };
  const result = await packetPreviewResult(missionId, 9, {
    packet,
    bundle: [evidence, finding, packet],
    canonical_json: PYTHON_FLOAT_PREVIEW_JSON,
  });
  result.content_sha256 = await sha256(new TextEncoder().encode(result.canonical_json));
  socket.reply(request, result, {revision: 9});
  const preview = await loading;
  assert.equal(preview.packet.packet_id, 'packet-1');
  assert.equal(client.state.packetPreviewStatus, `Verified record hash · ${result.content_sha256} · persisted metadata only.`);

  const invalidLoading = client.previewSelectedPacket();
  const invalidRequest = socket.sent.at(-1);
  const noncanonical = {...result, canonical_json: result.canonical_json.replace('"score":1e-05', '"score":0.00001')};
  noncanonical.content_sha256 = await sha256(new TextEncoder().encode(noncanonical.canonical_json));
  socket.reply(invalidRequest, noncanonical, {revision: 9});
  assert.equal(await invalidLoading, null);
  assert.equal(client.state.packetPreview, null);
  assert.match(client.state.packetPreviewStatus, /Unavailable · packet_preview_projection_mismatch/);

  const unsafeLoading = client.previewSelectedPacket();
  const unsafeRequest = socket.sent.at(-1);
  const unsafeBundle = structuredClone(result.bundle);
  unsafeBundle[1].model_provenance = [['large_integer', Number.MAX_SAFE_INTEGER + 1]];
  assert.doesNotThrow(() => validateBundle(unsafeBundle));
  const unsafeInteger = {
    ...result,
    bundle: unsafeBundle,
    packet: unsafeBundle.at(-1),
    canonical_json: result.canonical_json.replace(
      '"model_provenance":[["small_float",1e-07]]',
      '"model_provenance":[["large_integer",9007199254740993]]',
    ),
  };
  unsafeInteger.content_sha256 = await sha256(new TextEncoder().encode(unsafeInteger.canonical_json));
  socket.reply(unsafeRequest, unsafeInteger, {revision: 9});
  assert.equal(await unsafeLoading, null);
  assert.equal(client.state.packetPreview, null);
  assert.match(client.state.packetPreviewStatus, /Unavailable · packet_preview_projection_mismatch/);
});

test('packet preview rejects hash, projection, identity, and stale revision mismatches', async () => {
  for (const mismatch of ['hash', 'projection', 'identity', 'stale']) {
    const socket = new FakeSocket();
    const {client} = createClient(socket);
    await loadMission(client, socket, `mission-preview-${mismatch}`, 9, {state: 'completed'});
    const loading = client.previewSelectedPacket();
    const request = socket.sent.at(-1);
    const result = await packetPreviewResult(`mission-preview-${mismatch}`, 9);
    let replyRevision = 9;
    if (mismatch === 'hash') result.content_sha256 = '0'.repeat(64);
    if (mismatch === 'projection') result.bundle[0].packet_id = 'packet-tampered';
    if (mismatch === 'identity') result.mission_id = 'mission-other';
    if (mismatch === 'stale') replyRevision = 10;
    socket.reply(request, result, {revision: replyRevision});
    assert.equal(await loading, null, mismatch);
    assert.equal(client.state.packetPreview, null, mismatch);
    assert.match(client.state.packetPreviewStatus, /Unavailable ·/, mismatch);
  }
});

test('packet preview rejects a structurally invalid record bundle with a valid projection and hash', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  const missionId = 'mission-preview-invalid-bundle';
  await loadMission(client, socket, missionId, 9, {state: 'completed', cycle_id: null, watch_lease: null});
  const loading = client.previewSelectedPacket();
  const request = socket.sent.at(-1);
  const result = await packetPreviewResult(missionId, 9);
  result.packet.unexpected_field = true;
  result.canonical_json = canonicalJson({format: 'visionbrain.packet-preview.v1', records: result.bundle});
  result.content_sha256 = await sha256(new TextEncoder().encode(result.canonical_json));
  socket.reply(request, result, {revision: 9});
  assert.equal(await loading, null);
  assert.equal(client.state.packetPreview, null);
  assert.match(client.state.packetPreviewStatus, /Unavailable · packet_preview_invalid_bundle/);
});

test('packet preview rejects a negative-zero integer token with a valid bundle and matching digest', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket);
  const missionId = 'mission-preview-negative-zero';
  await loadMission(client, socket, missionId, 9, {state: 'completed', cycle_id: null, watch_lease: null});
  const loading = client.previewSelectedPacket();
  const request = socket.sent.at(-1);
  const result = await packetPreviewResult(missionId, 9);
  result.packet.created_at_ms = 0;
  assert.doesNotThrow(() => validateBundle(result.bundle));
  result.canonical_json = canonicalJson({format: 'visionbrain.packet-preview.v1', records: result.bundle});
  assert.match(result.canonical_json, /"created_at_ms":0/);
  result.canonical_json = result.canonical_json.replace('"created_at_ms":0', '"created_at_ms":-0');
  result.content_sha256 = await sha256(new TextEncoder().encode(result.canonical_json));
  socket.reply(request, result, {revision: 9});
  assert.equal(await loading, null);
  assert.equal(client.state.packetPreview, null);
  assert.match(client.state.packetPreviewStatus, /Unavailable · packet_preview_projection_mismatch/);
});

test('packet preview rejects noncanonical JSON even when its digest matches', async () => {
  for (const format of ['whitespace', 'escaped_non_ascii', 'duplicate_key', 'unsorted_keys']) {
    const socket = new FakeSocket();
    const {client} = createClient(socket);
    const missionId = format === 'escaped_non_ascii' ? 'mission-café' : 'mission-preview-whitespace';
    await loadMission(client, socket, missionId, 9, {state: 'completed'});
    const loading = client.previewSelectedPacket();
    const request = socket.sent.at(-1);
    const result = await packetPreviewResult(missionId, 9);
    if (format === 'whitespace') {
      result.canonical_json = JSON.stringify(JSON.parse(result.canonical_json), null, 2);
    } else if (format === 'escaped_non_ascii') {
      result.canonical_json = result.canonical_json.replaceAll('é', '\\u00e9');
    } else if (format === 'duplicate_key') {
      result.canonical_json = result.canonical_json.replace(
        '"format":"visionbrain.packet-preview.v1",',
        '"format":"visionbrain.packet-preview.v1","format":"visionbrain.packet-preview.v1",',
      );
    } else {
      const canonical = JSON.parse(result.canonical_json);
      result.canonical_json = `{"records":${JSON.stringify(canonical.records)},"format":${JSON.stringify(canonical.format)}}`;
    }
    result.content_sha256 = await sha256(new TextEncoder().encode(result.canonical_json));
    socket.reply(request, result, {revision: 9});
    assert.equal(await loading, null, format);
    assert.equal(client.state.packetPreview, null, format);
    assert.match(client.state.packetPreviewStatus, /Unavailable ·/, format);
  }
});

test('packet preview reports unavailable without WebCrypto and sends no request', async () => {
  const socket = new FakeSocket();
  const {client} = createClient(socket, {cryptoApi: null});
  await loadMission(client, socket, 'mission-no-crypto', 4, {state: 'completed', cycle_id: null, watch_lease: null});
  const before = socket.sent.length;
  assert.equal(await client.previewSelectedPacket(), null);
  assert.equal(socket.sent.length, before);
  assert.match(client.state.packetPreviewStatus, /cannot verify the preview SHA-256/);
  assert.equal(client.state.packetPreview, null);
});

test('packet preview refuses active and non-ready mission states', async () => {
  for (const snapshot of [
    {state: 'running', cycle_id: null, watch_lease: null},
    {state: 'paused', cycle_id: 'cycle-1', watch_lease: null},
    {state: 'completed', cycle_id: null, watch_lease: {lease_id: 'active'}},
  ]) {
    const socket = new FakeSocket();
    const {client} = createClient(socket);
    await loadMission(client, socket, 'mission-not-ready', 5, {...snapshot, cycle_id: snapshot.cycle_id ?? null,
      watch_lease: snapshot.watch_lease ?? null});
    const before = socket.sent.length;
    assert.equal(await client.previewSelectedPacket(), null);
    assert.equal(socket.sent.length, before);
    assert.match(client.state.packetPreviewStatus, /completed or paused mission with no active cycle/);
  }
});
