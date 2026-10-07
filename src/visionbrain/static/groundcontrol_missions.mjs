import {validateBundle} from './mission_records.mjs';

const MAX_EVIDENCE_BYTES = 2 * 1024 * 1024;
const EVIDENCE_CHUNK_BYTES = 49152;
const LIST_LIMIT = 50;
const MAX_PACKET_PREVIEW_BYTES = 256 * 1024;

export function jpegDimensions(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 4 || bytes[0] !== 0xff || bytes[1] !== 0xd8) {
    throw new Error('invalid_jpeg');
  }
  const sof = new Set([0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7, 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf]);
  let offset = 2;
  while (offset < bytes.length) {
    if (bytes[offset++] !== 0xff) throw new Error('invalid_jpeg_marker');
    while (offset < bytes.length && bytes[offset] === 0xff) offset++;
    if (offset >= bytes.length) break;
    const marker = bytes[offset++];
    if (marker === 0xd9 || marker === 0xda) break;
    if (marker === 0x01 || (marker >= 0xd0 && marker <= 0xd7)) continue;
    if (offset + 2 > bytes.length) throw new Error('invalid_jpeg_segment');
    const length = (bytes[offset] << 8) | bytes[offset + 1];
    if (length < 2 || offset + length > bytes.length) throw new Error('invalid_jpeg_segment');
    if (sof.has(marker)) {
      if (length < 7) throw new Error('invalid_jpeg_dimensions');
      const height = (bytes[offset + 3] << 8) | bytes[offset + 4];
      const width = (bytes[offset + 5] << 8) | bytes[offset + 6];
      if (!width || !height) throw new Error('invalid_jpeg_dimensions');
      return {width, height};
    }
    offset += length;
  }
  throw new Error('jpeg_dimensions_missing');
}

function decodeBase64(value) {
  if (typeof value !== 'string' || !value) throw new Error('invalid_evidence_chunk');
  const decoded = atob(value);
  const bytes = new Uint8Array(decoded.length);
  for (let i = 0; i < decoded.length; i++) bytes[i] = decoded.charCodeAt(i);
  return bytes;
}

function sameRevision(value, expected) {
  return Number.isSafeInteger(value) && value === expected;
}

function sameJson(left, right) {
  if (left === right) return true;
  if (left === null || right === null || typeof left !== 'object' || typeof right !== 'object') return false;
  if (Array.isArray(left) || Array.isArray(right)) {
    return Array.isArray(left) && Array.isArray(right) && left.length === right.length &&
      left.every((value, index) => sameJson(value, right[index]));
  }
  const leftKeys = Object.keys(left).sort();
  const rightKeys = Object.keys(right).sort();
  return leftKeys.length === rightKeys.length && leftKeys.every((key, index) =>
    key === rightKeys[index] && sameJson(left[key], right[key]));
}

function pythonStringOrder(left, right) {
  const leftPoints = Array.from(left, (value) => value.codePointAt(0));
  const rightPoints = Array.from(right, (value) => value.codePointAt(0));
  for (let index = 0; index < Math.min(leftPoints.length, rightPoints.length); index++) {
    if (leftPoints[index] !== rightPoints[index]) return leftPoints[index] - rightPoints[index];
  }
  return leftPoints.length - rightPoints.length;
}

function hasUnpairedSurrogate(value) {
  for (let index = 0; index < value.length; index++) {
    const code = value.charCodeAt(index);
    if (code >= 0xd800 && code <= 0xdbff) {
      const next = value.charCodeAt(index + 1);
      if (!(next >= 0xdc00 && next <= 0xdfff)) return true;
      index++;
    } else if (code >= 0xdc00 && code <= 0xdfff) {
      return true;
    }
  }
  return false;
}

function pythonFloatToken(value) {
  if (!Number.isFinite(value)) return null;
  if (Object.is(value, -0)) return '-0.0';
  if (value === 0) return '0.0';
  const sign = value < 0 ? '-' : '';
  const absolute = Math.abs(value);
  let shortest = absolute.toString();
  let exponent = 0;
  const exponentIndex = shortest.indexOf('e');
  if (exponentIndex >= 0) {
    exponent = Number(shortest.slice(exponentIndex + 1));
    shortest = shortest.slice(0, exponentIndex);
  }
  const decimalIndex = shortest.indexOf('.');
  let decimalPosition = (decimalIndex < 0 ? shortest.length : decimalIndex) + exponent;
  let digits = shortest.replace('.', '');
  while (digits.length > 1 && digits[0] === '0') {
    digits = digits.slice(1);
    decimalPosition--;
  }
  while (digits.length > 1 && digits.endsWith('0')) digits = digits.slice(0, -1);

  if (absolute < 1e-4 || absolute >= 1e16) {
    const power = decimalPosition - 1;
    const exponentSign = power >= 0 ? '+' : '-';
    const exponentDigits = Math.abs(power).toString().padStart(2, '0');
    const significand = digits[0] + (digits.length > 1 ? `.${digits.slice(1)}` : '');
    return `${sign}${significand}e${exponentSign}${exponentDigits}`;
  }
  if (decimalPosition <= 0) return `${sign}0.${'0'.repeat(-decimalPosition)}${digits}`;
  if (decimalPosition >= digits.length) return `${sign}${digits}${'0'.repeat(decimalPosition - digits.length)}.0`;
  return `${sign}${digits.slice(0, decimalPosition)}.${digits.slice(decimalPosition)}`;
}

function isCanonicalPythonJson(text) {
  let offset = 0;

  function stringToken() {
    const start = offset;
    if (text[offset] !== '"') return null;
    offset++;
    while (offset < text.length) {
      const code = text.charCodeAt(offset);
      if (code === 0x22) {
        offset++;
        const token = text.slice(start, offset);
        let value;
        try { value = JSON.parse(token); } catch { return null; }
        if (typeof value !== 'string' || hasUnpairedSurrogate(value) || JSON.stringify(value) !== token) return null;
        return value;
      }
      if (code === 0x5c) offset += 2;
      else if (code < 0x20) return null;
      else offset++;
    }
    return null;
  }

  function value() {
    const current = text[offset];
    if (current === '"') return stringToken() !== null;
    if (current === '{') {
      offset++;
      if (text[offset] === '}') { offset++; return true; }
      const keys = new Set();
      let previous = null;
      while (offset < text.length) {
        const key = stringToken();
        if (key === null || keys.has(key) || (previous !== null && pythonStringOrder(previous, key) >= 0)) return false;
        keys.add(key);
        previous = key;
        if (text[offset++] !== ':') return false;
        if (!value()) return false;
        if (text[offset] === '}') { offset++; return true; }
        if (text[offset++] !== ',') return false;
      }
      return false;
    }
    if (current === '[') {
      offset++;
      if (text[offset] === ']') { offset++; return true; }
      while (offset < text.length) {
        if (!value()) return false;
        if (text[offset] === ']') { offset++; return true; }
        if (text[offset++] !== ',') return false;
      }
      return false;
    }
    for (const literal of ['true', 'false', 'null']) {
      if (text.startsWith(literal, offset)) {
        offset += literal.length;
        return true;
      }
    }
    const number = /-?(?:0|[1-9]\d*)(?:\.\d+)?(?:e[+-]?\d+)?/y;
    number.lastIndex = offset;
    const match = number.exec(text);
    if (!match) return false;
    const token = match[0];
    if (token === '-0') return false;
    const numeric = Number(token);
    if (!Number.isFinite(numeric)) return false;
    if (!token.includes('.') && !token.includes('e') && !Number.isSafeInteger(numeric)) return false;
    if ((token.includes('.') || token.includes('e')) && pythonFloatToken(numeric) !== token) return false;
    offset = number.lastIndex;
    return true;
  }

  try {
    return value() && offset === text.length;
  } catch {
    return false;
  }
}

export function createGroundControlMissionInspector({onChange = () => {}, cryptoApi = globalThis.crypto, idFactory, requestTimeoutMs = 12000} = {}) {
  let socket = null;
  let socketGeneration = 0;
  let selectionGeneration = 0;
  let packetPreviewGeneration = 0;
  let requestSequence = 0;
  let pending = new Map();
  let activeEvidenceTransfer = null;
  let latestEvidenceSelection = null;
  let eventSync = null;
  let beforeUpdatedAt = null;
  const makeId = idFactory || (() => globalThis.crypto?.randomUUID?.() || `gc-${Date.now()}-${++requestSequence}`);
  let state = {
    connected: false,
    status: 'Connect to the field hub with a mission credential.',
    descriptor: null,
    missions: [],
    moreMissions: false,
    snapshot: null,
    selectedMissionId: null,
    selectedFindingId: null,
    selectedEvidenceId: null,
    evidenceIds: [],
    evidenceLabels: {},
    evidenceStatus: '',
    media: null,
    packetPreview: null,
    packetPreviewStatus: '',
    eventCursor: 0,
  };

  function publish(patch) {
    state = {...state, ...patch};
    onChange(state);
  }

  function rejectPending(reason) {
    for (const item of pending.values()) {
      clearTimeout(item.timer);
      item.reject(new Error(reason));
    }
    pending = new Map();
  }

  function clearQueuedEvidence() {
    const queued = latestEvidenceSelection;
    latestEvidenceSelection = null;
    queued?.resolve(null);
  }

  function clearSelection(status, keepMissionList = true) {
    selectionGeneration++;
    clearQueuedEvidence();
    eventSync = null;
    publish({
      ...(keepMissionList ? {} : {missions: [], moreMissions: false}),
      snapshot: null,
      selectedMissionId: null,
      selectedFindingId: null,
      selectedEvidenceId: null,
      evidenceIds: [],
      evidenceLabels: {},
      evidenceStatus: '',
      media: null,
      packetPreview: null,
      packetPreviewStatus: '',
      eventCursor: 0,
      status,
    });
  }

  function bindSocket(nextSocket) {
    if (socket === nextSocket) return;
    socketGeneration++;
    selectionGeneration++;
    clearQueuedEvidence();
    rejectPending('disconnected');
    socket = nextSocket || null;
    beforeUpdatedAt = null;
    eventSync = null;
    const connected = socket?.readyState === 1;
    publish({
      connected,
      descriptor: null,
      snapshot: null,
      selectedMissionId: null,
      selectedFindingId: null,
      selectedEvidenceId: null,
      evidenceIds: [],
      evidenceLabels: {},
      evidenceStatus: '',
      media: null,
      packetPreview: null,
      packetPreviewStatus: '',
      eventCursor: 0,
      missions: [],
      moreMissions: false,
      status: connected ? 'Connected · refresh mission list.' : 'Connecting to field hub…',
    });
    if (!socket) return;
    const bound = socket;
    const generation = socketGeneration;
    socket.addEventListener('open', () => {
      if (socket !== bound || generation !== socketGeneration) return;
      publish({connected: true, status: 'Connected · refresh mission list.'});
    });
    socket.addEventListener('close', () => {
      if (socket !== bound || generation !== socketGeneration) return;
      selectionGeneration++;
      clearQueuedEvidence();
      rejectPending('disconnected');
      publish({
        connected: false,
        snapshot: null,
        selectedMissionId: null,
        selectedFindingId: null,
        selectedEvidenceId: null,
        evidenceIds: [],
        evidenceLabels: {},
        evidenceStatus: '',
        media: null,
        packetPreview: null,
        packetPreviewStatus: '',
        eventCursor: 0,
        status: 'Disconnected · cached mission and media cleared.',
      });
    });
    socket.addEventListener('message', (event) => {
      if (socket !== bound || generation !== socketGeneration || typeof event.data !== 'string') return;
      let message;
      try { message = JSON.parse(event.data); } catch { return; }
      if (message.type === 'mission_capabilities') {
        publish({descriptor: message.descriptor || null});
      } else if (message.type === 'mission_reply') {
        const request = pending.get(message.request_id);
        if (!request) return;
        pending.delete(message.request_id);
        clearTimeout(request.timer);
        if (message.mission_id !== request.missionId) {
          request.reject(new Error('reply_identity_mismatch'));
        } else {
          request.resolve(message);
        }
      } else if (message.type === 'mission_event' && message.mission_id === state.selectedMissionId) {
        catchUpEvents(selectionGeneration).catch(() => {});
      }
    });
  }

  function request(command, missionId = null, args = {}, expectedRevision = null) {
    if (!socket || socket.readyState !== 1) return Promise.reject(new Error('disconnected'));
    if (!['capabilities', 'list', 'get', 'events_since', 'get_evidence', 'preview_packet'].includes(command)) {
      return Promise.reject(new Error('read_only_command_rejected'));
    }
    if (command === 'preview_packet' &&
        (!missionId || !Number.isSafeInteger(expectedRevision) || expectedRevision < 0 || Object.keys(args).length !== 0)) {
      return Promise.reject(new Error('invalid_preview_request'));
    }
    const requestId = makeId();
    const envelope = {
      type: 'mission_command',
      schema_version: 1,
      request_id: requestId,
      mission_id: missionId,
      command,
      args,
    };
    if (expectedRevision !== null) envelope.expected_revision = expectedRevision;
    const socketAtSend = socket;
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        if (!pending.has(requestId)) return;
        pending.delete(requestId);
        reject(new Error('mission_reply_timeout'));
      }, requestTimeoutMs);
      pending.set(requestId, {missionId, resolve, reject, timer});
      try {
        socketAtSend.send(JSON.stringify(envelope));
      } catch (error) {
        pending.delete(requestId);
        clearTimeout(timer);
        reject(error);
      }
    });
  }

  function replyResult(reply) {
    if (!reply.ok) {
      const code = reply.error?.code || 'read_failed';
      const message = reply.error?.message || '';
      throw new Error(message ? `${code}: ${message}` : code);
    }
    return reply.result || {};
  }

  async function refreshMissions({more = false} = {}) {
    const before = more ? beforeUpdatedAt : null;
    const args = {limit: LIST_LIMIT};
    if (before !== null) args.before_updated_at_ms = before;
    try {
      const result = replyResult(await request('list', null, args));
      const rows = Array.isArray(result.missions) ? result.missions : [];
      const existing = more ? state.missions : [];
      const known = new Set(existing.map((item) => item.mission_id));
      const missions = [...existing, ...rows.filter((item) => item?.mission_id && !known.has(item.mission_id))];
      const oldest = rows.reduce((value, item) => Number.isSafeInteger(item?.updated_at_ms)
        ? Math.min(value, item.updated_at_ms) : value, Number.MAX_SAFE_INTEGER);
      beforeUpdatedAt = oldest === Number.MAX_SAFE_INTEGER ? null : oldest;
      publish({
        missions,
        moreMissions: rows.length === LIST_LIMIT && beforeUpdatedAt !== null,
        status: `Loaded ${missions.length} mission${missions.length === 1 ? '' : 's'} · list is server-authoritative.`,
      });
      return missions;
    } catch (error) {
      publish({status: `Mission list unavailable · ${error.message}`});
      return [];
    }
  }

  async function selectMission(missionId) {
    if (typeof missionId !== 'string' || !missionId) return;
    const generation = ++selectionGeneration;
    clearQueuedEvidence();
    eventSync = null;
    publish({
      snapshot: null,
      selectedMissionId: missionId,
      selectedFindingId: null,
      selectedEvidenceId: null,
      evidenceIds: [],
      evidenceLabels: {},
      evidenceStatus: '',
      media: null,
      packetPreview: null,
      packetPreviewStatus: '',
      eventCursor: 0,
      status: `Loading mission ${missionId}…`,
    });
    try {
      const reply = await request('get', missionId, {});
      if (generation !== selectionGeneration) return;
      const snapshot = replyResult(reply).snapshot;
      if (!snapshot || snapshot.mission_id !== missionId || !sameRevision(reply.revision, snapshot.revision)) {
        throw new Error('snapshot_identity_or_revision_mismatch');
      }
      publish({snapshot, eventCursor: Number.isSafeInteger(snapshot.last_sequence) ? snapshot.last_sequence : 0,
        status: `Mission loaded · revision ${snapshot.revision}.`});
      await catchUpEvents(generation);
    } catch (error) {
      if (generation === selectionGeneration) {
        clearSelection(`Mission unavailable · ${error.message}`);
        publish({selectedMissionId: missionId});
      }
    }
  }

  function evidenceFor(snapshot, evidenceId) {
    return snapshot?.evidence?.find((item) => item?.evidence_id === evidenceId) || null;
  }

  function evidenceLineage(snapshot, evidence) {
    const source = evidence?.source_id;
    const epoch = evidence?.source_epoch;
    const binding = snapshot?.source_binding;
    if (typeof source !== 'string' || !source || (typeof epoch !== 'string' && !Number.isSafeInteger(epoch))) {
      return 'Historical · source provenance unknown';
    }
    if (binding && source === binding.source_id && String(epoch) === String(binding.source_epoch)) {
      return `Current source binding · ${source} · epoch ${epoch}`;
    }
    return `Historical · ${source} · epoch ${epoch}`;
  }

  function setFinding(findingId) {
    const snapshot = state.snapshot;
    const finding = snapshot?.findings?.find((item) => item?.finding_id === findingId);
    if (!finding) return;
    selectionGeneration++;
    clearQueuedEvidence();
    const primary = typeof finding.evidence_id === 'string' ? finding.evidence_id : null;
    const refs = Array.isArray(finding.evidence_refs) ? finding.evidence_refs.filter((id) => typeof id === 'string') : [];
    const evidenceIds = [...new Set([...(primary ? [primary] : []), ...refs])];
    const labels = Object.fromEntries(evidenceIds.map((id) => [id, id === primary ? 'Primary' : 'Supporting reference']));
    publish({selectedFindingId: findingId, evidenceIds, evidenceLabels: labels, selectedEvidenceId: null,
      evidenceStatus: evidenceIds.length ? 'Select evidence to inspect.' : 'No evidence references are recorded.', media: null});
  }

  function evidenceSelectionCurrent(selection) {
    return selection.generation === selectionGeneration && state.selectedMissionId === selection.missionId &&
      state.selectedFindingId === selection.findingId && state.selectedEvidenceId === selection.evidenceId &&
      state.snapshot?.revision === selection.revision;
  }

  async function transferEvidence(selection) {
    const {generation, snapshot, missionId, findingId, evidenceId, evidence, revision, lineage} = selection;
    try {
      if (!evidenceSelectionCurrent(selection)) return null;
      const chunks = [];
      let offset = 0;
      let totalBytes = null;
      let metadata = null;
      while (totalBytes === null || offset < totalBytes) {
        if (!evidenceSelectionCurrent(selection)) return null;
        const length = Math.min(EVIDENCE_CHUNK_BYTES, totalBytes === null ? EVIDENCE_CHUNK_BYTES : totalBytes - offset);
        const reply = await request('get_evidence', missionId, {evidence_id: evidenceId, offset, length});
        if (!evidenceSelectionCurrent(selection)) return null;
        if (!sameRevision(reply.revision, revision)) throw new Error('stale_mission_revision');
        const result = replyResult(reply);
        if (result.evidence_id !== evidenceId || result.offset !== offset) throw new Error('evidence_identity_mismatch');
        if (!Number.isSafeInteger(result.total_bytes) || result.total_bytes <= 0 || result.total_bytes > MAX_EVIDENCE_BYTES) {
          throw new Error('evidence_size_out_of_bounds');
        }
        if (totalBytes === null) {
          totalBytes = result.total_bytes;
          metadata = result;
          if (Number.isSafeInteger(evidence.bytes) && evidence.bytes !== totalBytes) throw new Error('evidence_size_mismatch');
          if (String(result.sha256).toLowerCase() !== evidence.sha256.toLowerCase() ||
              result.width !== evidence.width || result.height !== evidence.height) {
            throw new Error('evidence_metadata_mismatch');
          }
        } else if (result.total_bytes !== totalBytes || String(result.sha256).toLowerCase() !== String(metadata.sha256).toLowerCase() ||
                   result.width !== metadata.width || result.height !== metadata.height) {
          throw new Error('evidence_chunk_metadata_mismatch');
        }
        const bytes = decodeBase64(result.jpeg_b64);
        if (!bytes.length || bytes.length > length || offset + bytes.length > totalBytes) throw new Error('evidence_chunk_size_invalid');
        chunks.push(bytes);
        offset += bytes.length;
        if (bytes.length < length && offset < totalBytes) throw new Error('evidence_chunk_truncated');
      }
      const all = new Uint8Array(totalBytes);
      let cursor = 0;
      for (const chunk of chunks) { all.set(chunk, cursor); cursor += chunk.length; }
      if (cursor !== totalBytes) throw new Error('evidence_size_mismatch');
      if (!cryptoApi?.subtle) throw new Error('sha256_unavailable');
      const digest = new Uint8Array(await cryptoApi.subtle.digest('SHA-256', all));
      const actualHash = [...digest].map((byte) => byte.toString(16).padStart(2, '0')).join('');
      if (actualHash !== evidence.sha256.toLowerCase() || actualHash !== String(metadata.sha256).toLowerCase()) {
        throw new Error('evidence_hash_mismatch');
      }
      const dimensions = jpegDimensions(all);
      if (dimensions.width !== evidence.width || dimensions.height !== evidence.height ||
          dimensions.width !== metadata.width || dimensions.height !== metadata.height) {
        throw new Error('evidence_dimensions_mismatch');
      }
      if (!evidenceSelectionCurrent(selection)) return null;
      const media = {missionId, findingId, evidenceId, revision, sha256: actualHash,
        width: dimensions.width, height: dimensions.height, bytes: all, lineage};
      publish({media, evidenceStatus: `Verified · ${dimensions.width}×${dimensions.height} · ${actualHash.slice(0, 12)}… · ${lineage}`});
      return media;
    } catch (error) {
      if (generation === selectionGeneration && state.snapshot === snapshot && evidenceSelectionCurrent(selection)) {
        publish({media: null, evidenceStatus: `Unavailable · ${error.message} · ${lineage}`});
      }
      return null;
    }
  }

  function startEvidenceTransfer(selection) {
    const transfer = {selection};
    activeEvidenceTransfer = transfer;
    const task = transferEvidence(selection).finally(() => {
      if (activeEvidenceTransfer === transfer) activeEvidenceTransfer = null;
      const queued = latestEvidenceSelection;
      latestEvidenceSelection = null;
      if (!queued) return;
      if (!evidenceSelectionCurrent(queued.selection)) {
        queued.resolve(null);
        return;
      }
      startEvidenceTransfer(queued.selection).then(queued.resolve, () => queued.resolve(null));
    });
    transfer.promise = task;
    return task;
  }

  function selectEvidence(evidenceId) {
    const generation = ++selectionGeneration;
    clearQueuedEvidence();
    const snapshot = state.snapshot;
    const missionId = state.selectedMissionId;
    const findingId = state.selectedFindingId;
    const revision = snapshot?.revision;
    if (!snapshot || !missionId || !findingId || !evidenceId) {
      publish({selectedEvidenceId: null, media: null, evidenceStatus: ''});
      return Promise.resolve(null);
    }
    if (!state.evidenceIds.includes(evidenceId)) {
      publish({selectedEvidenceId: null, media: null,
        evidenceStatus: 'Unavailable · evidence is not linked to the selected finding.'});
      return Promise.resolve(null);
    }
    const evidence = evidenceFor(snapshot, evidenceId);
    if (!evidence) {
      publish({selectedEvidenceId: evidenceId, media: null,
        evidenceStatus: 'Unavailable · evidence record is missing from this snapshot (reason unknown).'});
      return Promise.resolve(null);
    }
    const lineage = evidenceLineage(snapshot, evidence);
    publish({selectedEvidenceId: evidenceId, media: null, evidenceStatus: ''});
    if (evidence.available === false) {
      publish({evidenceStatus: `Unavailable · ${evidence.availability_reason || 'reason unknown'} · ${lineage}`});
      return Promise.resolve(null);
    }
    if (evidence.available !== true) {
      publish({evidenceStatus: `Unavailable · availability unknown · ${lineage}`});
      return Promise.resolve(null);
    }
    if (typeof evidence.sha256 !== 'string' || !/^[0-9a-f]{64}$/i.test(evidence.sha256) ||
        !Number.isSafeInteger(evidence.width) || evidence.width <= 0 ||
        !Number.isSafeInteger(evidence.height) || evidence.height <= 0) {
      publish({evidenceStatus: `Unavailable · verified hash or dimensions are missing · ${lineage}`});
      return Promise.resolve(null);
    }
    const selection = {generation, snapshot, missionId, findingId, evidenceId, evidence, revision, lineage};
    const label = state.evidenceLabels[evidenceId] || 'evidence';
    if (activeEvidenceTransfer) {
      let resolve;
      const promise = new Promise((done) => { resolve = done; });
      latestEvidenceSelection = {selection, resolve};
      publish({evidenceStatus: `Waiting to inspect latest selection · current chunk will finish first · ${lineage}`});
      return promise;
    }
    publish({evidenceStatus: `Fetching ${label} · ${lineage}`});
    return startEvidenceTransfer(selection);
  }

  async function previewSelectedPacket() {
    const previewGeneration = ++packetPreviewGeneration;
    const snapshot = state.snapshot;
    const missionId = state.selectedMissionId;
    const revision = snapshot?.revision;
    if (!snapshot || !missionId || !Number.isSafeInteger(revision) || revision < 0 ||
        !['completed', 'paused'].includes(snapshot.state) || snapshot.cycle_id !== null || snapshot.watch_lease !== null) {
      publish({packetPreview: null, packetPreviewStatus: 'Unavailable · select a completed or paused mission with no active cycle.'});
      return null;
    }
    if (!cryptoApi?.subtle || typeof TextEncoder === 'undefined') {
      publish({packetPreview: null, packetPreviewStatus: 'Unavailable · this browser cannot verify the preview SHA-256.'});
      return null;
    }
    const generation = selectionGeneration;
    publish({packetPreview: null, packetPreviewStatus: `Building metadata-only preview · mission revision ${revision}…`});
    try {
      const reply = await request('preview_packet', missionId, {}, revision);
      if (generation !== selectionGeneration || state.selectedMissionId !== missionId ||
          state.snapshot?.revision !== revision) {
        throw new Error('stale_mission_revision');
      }
      if (reply.type !== 'mission_reply' || reply.mission_id !== missionId || !sameRevision(reply.revision, revision)) {
        throw new Error('preview_reply_identity_or_revision_mismatch');
      }
      const result = replyResult(reply);
      if (result.envelope_version !== 1 || result.kind !== 'inspection_packet_preview' ||
          result.mission_id !== missionId || !sameRevision(result.mission_revision, revision) ||
          result.verification?.basis !== 'persisted_metadata_only' ||
          result.verification?.fresh_media_verified !== false ||
          typeof result.canonical_json !== 'string' ||
          new TextEncoder().encode(result.canonical_json).length > MAX_PACKET_PREVIEW_BYTES ||
          typeof result.content_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(result.content_sha256) ||
          !Array.isArray(result.bundle) || !result.packet || typeof result.packet !== 'object') {
        throw new Error('invalid_packet_preview_envelope');
      }
      const digest = new Uint8Array(await cryptoApi.subtle.digest('SHA-256', new TextEncoder().encode(result.canonical_json)));
      const actualHash = [...digest].map((byte) => byte.toString(16).padStart(2, '0')).join('');
      if (actualHash !== result.content_sha256) throw new Error('packet_preview_hash_mismatch');
      const canonical = JSON.parse(result.canonical_json);
      if (!canonical || typeof canonical !== 'object' || Array.isArray(canonical) ||
          !isCanonicalPythonJson(result.canonical_json) ||
          Object.keys(canonical).sort().join(',') !== 'format,records' ||
          canonical.format !== 'visionbrain.packet-preview.v1' || !Array.isArray(canonical.records) ||
          !sameJson(canonical.records, result.bundle) || !sameJson(canonical.records.at(-1), result.packet) ||
          result.packet.record_type !== 'inspection_packet' || result.packet.state !== 'draft' ||
          result.packet.mission_id !== missionId || !sameRevision(result.packet.mission_revision, revision)) {
        throw new Error('packet_preview_projection_mismatch');
      }
      try {
        validateBundle(result.bundle);
      } catch {
        throw new Error('packet_preview_invalid_bundle');
      }
      if (generation !== selectionGeneration || previewGeneration !== packetPreviewGeneration ||
          state.selectedMissionId !== missionId || state.snapshot?.revision !== revision) {
        throw new Error('stale_mission_revision');
      }
      const preview = {
        missionId,
        missionRevision: revision,
        packet: result.packet,
        bundle: result.bundle,
        contentSha256: actualHash,
      };
      publish({packetPreview: preview, packetPreviewStatus: `Verified record hash · ${actualHash} · persisted metadata only.`});
      return preview;
    } catch (error) {
      if (generation === selectionGeneration && previewGeneration === packetPreviewGeneration &&
          state.selectedMissionId === missionId) {
        publish({packetPreview: null, packetPreviewStatus: `Unavailable · ${error.message}. Refresh mission before retrying.`});
        if (error.message.includes('revision_conflict')) selectMission(missionId);
      }
      return null;
    }
  }

  async function refreshSnapshot(generation) {
    const missionId = state.selectedMissionId;
    const reply = await request('get', missionId, {});
    if (generation !== selectionGeneration) return;
    const snapshot = replyResult(reply).snapshot;
    if (!snapshot || snapshot.mission_id !== missionId || !sameRevision(reply.revision, snapshot.revision)) {
      throw new Error('snapshot_identity_or_revision_mismatch');
    }
    selectionGeneration++;
    clearQueuedEvidence();
    publish({snapshot, eventCursor: Number.isSafeInteger(snapshot.last_sequence) ? snapshot.last_sequence : 0,
      selectedFindingId: null, selectedEvidenceId: null, evidenceIds: [], evidenceLabels: {}, media: null,
      packetPreview: null, packetPreviewStatus: '',
      evidenceStatus: '', status: `Mission refreshed · revision ${snapshot.revision}. Select a finding again.`});
  }

  async function catchUpEvents(generation) {
    if (eventSync) return eventSync;
    const missionId = state.selectedMissionId;
    if (!missionId || !state.snapshot) return;
    const sync = (async () => {
      const result = replyResult(await request('events_since', missionId, {cursor: state.eventCursor, limit: 100}));
      if (generation !== selectionGeneration) return;
      if (result.resync_required) return refreshSnapshot(generation);
      const events = Array.isArray(result.events) ? result.events : [];
      let cursor = state.eventCursor;
      let snapshot = state.snapshot;
      for (const event of events) {
        if (event?.mission_id !== missionId || !Number.isSafeInteger(event.sequence) || event.sequence !== cursor + 1) {
          return refreshSnapshot(generation);
        }
        cursor = event.sequence;
        const next = event.data?.snapshot;
        if (next?.mission_id === missionId && Number.isSafeInteger(next.revision) && next.revision >= snapshot.revision) snapshot = next;
      }
      if (Number.isSafeInteger(result.last_sequence) && result.last_sequence > cursor) return refreshSnapshot(generation);
      if (snapshot.revision !== state.snapshot.revision) {
        selectionGeneration++;
        clearQueuedEvidence();
        publish({snapshot, eventCursor: cursor, selectedFindingId: null, selectedEvidenceId: null,
          evidenceIds: [], evidenceLabels: {}, evidenceStatus: '', media: null,
          packetPreview: null, packetPreviewStatus: '',
          status: `Mission updated · revision ${snapshot.revision}. Select a finding again.`});
      } else {
        publish({snapshot, eventCursor: cursor});
      }
    })().catch((error) => {
      if (generation === selectionGeneration) publish({status: `Event catch-up unavailable · ${error.message}`});
    });
    eventSync = sync;
    try {
      return await sync;
    } finally {
      if (eventSync === sync) eventSync = null;
    }
  }

  return {
    get state() { return state; },
    bindSocket,
    refreshMissions,
    loadMoreMissions: () => refreshMissions({more: true}),
    selectMission,
    setFinding,
    selectEvidence,
    previewSelectedPacket,
    refreshSelectedMission: () => state.selectedMissionId ? selectMission(state.selectedMissionId) : Promise.resolve(),
  };
}

export const groundControlMissionLimits = Object.freeze({MAX_EVIDENCE_BYTES, EVIDENCE_CHUNK_BYTES, LIST_LIMIT});
