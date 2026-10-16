#!/usr/bin/env node
/** Dependency-free structural and cross-record checks. No network or policy execution. */
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
const isObject = value => value !== null && typeof value === 'object' && !Array.isArray(value);
const idPattern = /^[a-z0-9][a-z0-9._-]{0,95}$/;
const sha256Pattern = /^[a-f0-9]{64}$/;
const commitPattern = /^[a-f0-9]{40}$/;
const stages = new Set(['calibration', 'development', 'reserved', 'video']);
const evidenceKinds = new Set(['receipt', 'source', 'model', 'trajectory', 'video', 'review', 'reproduction', 'other']);

export function canonicalJson(value) {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new TypeError('Canonical JSON refuses nonfinite numbers');
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) if (!own(value, i)) throw new TypeError('Canonical JSON refuses sparse arrays');
    return `[${value.map(canonicalJson).join(',')}]`;
  }
  if (isObject(value)) return `{${Object.keys(value).sort().map(key => `${JSON.stringify(key)}:${canonicalJson(value[key])}`).join(',')}}`;
  throw new TypeError('Canonical JSON only accepts JSON values');
}

export function fingerprintContract(requirements) {
  return createHash('sha256').update(canonicalJson(requirements)).digest('hex');
}

/** Returns nanoseconds since epoch, preserving microsecond provenance ordering. */
function utcInstant(value) {
  if (typeof value !== 'string') return null;
  const match = /^(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})(?:\.(\d{1,9}))?(?:Z|\+00:00)$/.exec(value);
  if (!match) return null;
  const normalized = `${match[1]}T${match[2]}Z`;
  const date = new Date(normalized);
  if (!Number.isFinite(date.valueOf()) || date.toISOString().slice(0, 19) !== normalized.slice(0, 19)) return null;
  return BigInt(date.valueOf()) * 1_000_000n + BigInt((match[3] || '').padEnd(9, '0'));
}

function publicHttp(value) {
  if (typeof value !== 'string' || value.length > 4096 || /[\s\\\u0000-\u001f\u007f]/.test(value)) return false;
  try {
    const url = new URL(value);
    if (!['https:', 'http:'].includes(url.protocol) || url.username || url.password) return false;
    const host = url.hostname.toLowerCase();
    if (!host || host === 'localhost' || host.endsWith('.localhost') || host.endsWith('.local') || host.endsWith('.internal')) return false;
    if (host.startsWith('[')) {
      const ip = host.slice(1, -1);
      if (ip === '::' || ip === '::1' || /^(?:fc|fd|fe8|fe9|fea|feb)/i.test(ip) || ip.startsWith('::ffff:')) return false;
    }
    if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) {
      const [a, b] = host.split('.').map(Number);
      if (a === 0 || a === 10 || a === 127 || a >= 224 || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31) || (a === 192 && b === 168) || (a === 100 && b >= 64 && b <= 127)) return false;
    }
    return true;
  } catch { return false; }
}

export function validateRegistry(registry, { base = null } = {}) {
  const errors = [];
  const fail = (path, message) => errors.push(`${path}: ${message}`);
  const object = (value, path, allowed, required = []) => {
    if (!isObject(value)) { fail(path, 'must be an object'); return false; }
    for (const key of Object.keys(value)) if (!allowed.includes(key)) fail(`${path}.${key}`, 'unknown field');
    for (const key of required) if (!own(value, key)) fail(`${path}.${key}`, 'is required');
    return true;
  };
  const string = (value, path, max = 2000) => {
    if (typeof value !== 'string' || !value.trim() || value.length > max || /[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]/.test(value)) fail(path, `must be a nonempty string of at most ${max} characters`);
  };
  const id = (value, path) => { if (typeof value !== 'string' || !idPattern.test(value)) fail(path, 'must be a stable lowercase ID'); };
  const boolean = (value, path) => { if (typeof value !== 'boolean') fail(path, 'must be boolean'); };
  const number = (value, path, { min = 0, max = Infinity, integer = false, positive = false } = {}) => {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (positive && value <= 0) || (integer && !Number.isSafeInteger(value))) fail(path, `must be a finite ${integer ? 'safe integer' : 'number'} in the permitted range`);
  };
  const hash = (value, path) => { if (typeof value !== 'string' || !sha256Pattern.test(value)) fail(path, 'must be a full lowercase SHA256'); };
  const url = (value, path) => { if (!publicHttp(value)) fail(path, 'must be an absolute public HTTP(S) URL without credentials'); };
  const date = (value, path, nullable = false) => {
    if (nullable && value === null) return;
    if (utcInstant(value) === null) fail(path, 'must be a real UTC ISO-8601 timestamp (Z or +00:00)');
  };
  const ordered = (a, b, path) => {
    const aa = utcInstant(a), bb = utcInstant(b);
    if (aa !== null && bb !== null && aa > bb) fail(path, 'timestamp order is reversed');
  };
  const array = (value, path, min = 0) => {
    if (!Array.isArray(value) || value.length < min) { fail(path, `must be an array with at least ${min} item(s)`); return false; }
    return true;
  };
  const strings = (value, path, min = 0) => { if (array(value, path, min)) value.forEach((v, i) => string(v, `${path}[${i}]`)); };
  const identity = (value, path, reviewer = false) => {
    if (!object(value, path, reviewer ? ['id', 'name', 'url', 'independent'] : ['id', 'name', 'url'], reviewer ? ['id', 'name', 'url', 'independent'] : ['id', 'name', 'url'])) return;
    id(value.id, `${path}.id`); string(value.name, `${path}.name`, 150); url(value.url, `${path}.url`);
    if (reviewer) boolean(value.independent, `${path}.independent`);
  };
  const seedRange = (value, path) => {
    const hasStart = own(value, 'seedStart'), hasEnd = own(value, 'seedEnd');
    if (hasStart !== hasEnd) fail(path, 'seedStart and seedEnd must appear together');
    if (hasStart) {
      number(value.seedStart, `${path}.seedStart`, { integer: true }); number(value.seedEnd, `${path}.seedEnd`, { integer: true });
      if (value.seedEnd - value.seedStart + 1 !== value.required) fail(path, 'seed range must contain exactly required distinct episodes');
    }
  };
  const source = (value, path) => {
    if (!object(value, path, ['url', 'commit', 'sha256'], ['url', 'commit'])) return;
    url(value.url, `${path}.url`);
    if (typeof value.commit !== 'string' || !commitPattern.test(value.commit)) fail(`${path}.commit`, 'must be a full lowercase forty-character Git commit');
    if (own(value, 'sha256')) hash(value.sha256, `${path}.sha256`);
  };
  const evidence = (value, path, registeredAt) => {
    const found = new Map();
    if (!array(value, path, 1)) return found;
    value.forEach((entry, index) => {
      const p = `${path}[${index}]`;
      if (!object(entry, p, ['id', 'kind', 'url', 'sha256', 'publishedAt', 'description'], ['id', 'kind', 'url', 'sha256', 'publishedAt'])) return;
      id(entry.id, `${p}.id`);
      if (found.has(entry.id)) fail(`${p}.id`, 'duplicate evidence ID');
      found.set(entry.id, entry);
      if (!evidenceKinds.has(entry.kind)) fail(`${p}.kind`, 'unknown evidence kind');
      url(entry.url, `${p}.url`); hash(entry.sha256, `${p}.sha256`); date(entry.publishedAt, `${p}.publishedAt`, true);
      ordered(entry.publishedAt, registeredAt, `${p}.publishedAt`);
      if (own(entry, 'description')) string(entry.description, `${p}.description`);
    });
    return found;
  };
  const refs = (value, path, entries, kind) => {
    if (!array(value, path, 1)) return;
    const seen = new Set();
    value.forEach((reference, index) => {
      id(reference, `${path}[${index}]`);
      if (!entries.has(reference)) fail(`${path}[${index}]`, 'references missing evidence');
      if (seen.has(reference)) fail(`${path}[${index}]`, 'duplicate evidence reference');
      seen.add(reference);
    });
    if (kind && !value.some(reference => entries.get(reference)?.kind === kind)) fail(path, `must include ${kind} evidence`);
  };

  if (!object(registry, '$', ['schemaVersion', 'protocols', 'records'], ['schemaVersion', 'protocols', 'records'])) return errors;
  if (registry.schemaVersion !== 1) fail('$.schemaVersion', 'must be 1');
  const protocols = new Map();
  if (array(registry.protocols, '$.protocols', 1)) registry.protocols.forEach((protocol, index) => {
    const p = `$.protocols[${index}]`;
    if (!object(protocol, p, ['id', 'title', 'linksMin', 'linksMax', 'source', 'fingerprint', 'requirements'], ['id', 'title', 'linksMin', 'linksMax', 'source', 'fingerprint', 'requirements'])) return;
    id(protocol.id, `${p}.id`); string(protocol.title, `${p}.title`, 200);
    if (protocols.has(protocol.id)) fail(`${p}.id`, 'duplicate protocol ID');
    protocols.set(protocol.id, protocol);
    number(protocol.linksMin, `${p}.linksMin`, { integer: true, positive: true }); number(protocol.linksMax, `${p}.linksMax`, { integer: true, positive: true });
    if (protocol.linksMax < protocol.linksMin) fail(p, 'linksMax must be at least linksMin');
    source(protocol.source, `${p}.source`); hash(protocol.fingerprint, `${p}.fingerprint`);
    const req = protocol.requirements;
    const reqFields = ['durationSeconds', 'commandsPerEpisode', 'actionHz', 'uprightAngleRad', 'uprightConsecutiveSteps', 'railLimitMeters', 'forceLimitNewtons', 'calibrationTolerance', 'integration', 'mechanics', 'reset', 'acceptance', 'cohorts'];
    if (!object(req, `${p}.requirements`, reqFields, reqFields)) return;
    for (const field of ['durationSeconds', 'actionHz', 'uprightAngleRad', 'railLimitMeters', 'forceLimitNewtons']) number(req[field], `${p}.requirements.${field}`, { positive: true });
    for (const field of ['commandsPerEpisode', 'uprightConsecutiveSteps']) number(req[field], `${p}.requirements.${field}`, { integer: true, positive: true });
    number(req.calibrationTolerance, `${p}.requirements.calibrationTolerance`);
    if (req.uprightConsecutiveSteps > req.commandsPerEpisode) fail(`${p}.requirements`, 'upright hold exceeds full episode');
    if (Math.abs(req.durationSeconds * req.actionHz - req.commandsPerEpisode) > 1e-9) fail(`${p}.requirements`, 'duration × actionHz must equal commandsPerEpisode');
    if (object(req.integration, `${p}.requirements.integration`, ['engine', 'engineVersion', 'precision', 'integrator', 'timestepSeconds', 'substeps'], ['engine', 'engineVersion', 'precision', 'integrator', 'timestepSeconds', 'substeps'])) {
      for (const field of ['engine', 'engineVersion', 'precision', 'integrator']) string(req.integration[field], `${p}.requirements.integration.${field}`, 100);
      number(req.integration.timestepSeconds, `${p}.requirements.integration.timestepSeconds`, { positive: true }); number(req.integration.substeps, `${p}.requirements.integration.substeps`, { integer: true, positive: true });
      if (Math.abs(req.integration.timestepSeconds * req.integration.substeps * req.actionHz - 1) > 1e-9) fail(`${p}.requirements.integration`, 'integration timestep and substeps must match actionHz');
    }
    const mechanicsFields = ['cartMassKg', 'totalLinkMassKg', 'totalChainLengthMeters', 'capsuleRadiusMeters', 'armature', 'totalHingeDamping', 'cartDamping', 'frictionLoss'];
    if (object(req.mechanics, `${p}.requirements.mechanics`, mechanicsFields, mechanicsFields)) for (const field of mechanicsFields) number(req.mechanics[field], `${p}.requirements.mechanics.${field}`, { positive: ['cartMassKg', 'totalLinkMassKg', 'totalChainLengthMeters', 'capsuleRadiusMeters'].includes(field) });
    if (object(req.reset, `${p}.requirements.reset`, ['position', 'noiseStd', 'cartPositionNoiseStd', 'rngOrder'], ['position', 'noiseStd', 'cartPositionNoiseStd', 'rngOrder'])) {
      string(req.reset.position, `${p}.requirements.reset.position`); string(req.reset.rngOrder, `${p}.requirements.reset.rngOrder`);
      number(req.reset.noiseStd, `${p}.requirements.reset.noiseStd`); number(req.reset.cartPositionNoiseStd, `${p}.requirements.reset.cartPositionNoiseStd`);
    }
    const acceptanceFields = ['angleConvention', 'angleComparison', 'samples', 'fullEpisodeRequired', 'stateOverwriteAfterInit', 'liveRetriesAllowed', 'allFiniteRequired'];
    if (object(req.acceptance, `${p}.requirements.acceptance`, acceptanceFields, acceptanceFields)) {
      for (const field of acceptanceFields.slice(0, 3)) string(req.acceptance[field], `${p}.requirements.acceptance.${field}`);
      for (const field of acceptanceFields.slice(3)) boolean(req.acceptance[field], `${p}.requirements.acceptance.${field}`);
    }
    const gateIds = new Set();
    const ranges = [];
    if (array(req.cohorts, `${p}.requirements.cohorts`, 1)) req.cohorts.forEach((cohort, i) => {
      const cp = `${p}.requirements.cohorts[${i}]`;
      if (!object(cohort, cp, ['id', 'stage', 'noise', 'required', 'seedStart', 'seedEnd'], ['id', 'stage', 'noise', 'required'])) return;
      id(cohort.id, `${cp}.id`);
      if (gateIds.has(cohort.id)) fail(`${cp}.id`, 'duplicate cohort ID'); gateIds.add(cohort.id);
      if (!stages.has(cohort.stage)) fail(`${cp}.stage`, 'unknown cohort stage');
      if (!['noisy', 'exact'].includes(cohort.noise)) fail(`${cp}.noise`, 'must be noisy or exact');
      number(cohort.required, `${cp}.required`, { integer: true, positive: true }); seedRange(cohort, cp);
      if (own(cohort, 'seedStart')) {
        if (ranges.some(range => cohort.seedStart <= range[1] && cohort.seedEnd >= range[0])) fail(cp, 'protocol cohort seed ranges must be disjoint');
        ranges.push([cohort.seedStart, cohort.seedEnd]);
      }
    });
    try { if (protocol.fingerprint !== fingerprintContract(req)) fail(`${p}.fingerprint`, 'does not match canonical requirements SHA256'); }
    catch { fail(`${p}.requirements`, 'cannot be canonicalized'); }
  });

  const records = new Map();
  const disposition = new Set();
  if (array(registry.records, '$.records')) registry.records.forEach((record, index) => {
    const p = `$.records[${index}]`;
    if (!isObject(record)) { fail(p, 'must be an object'); return; }
    id(record.id, `${p}.id`);
    if (records.has(record.id)) fail(`${p}.id`, 'duplicate record ID');
    if (['supersession', 'retraction'].includes(record.type)) {
      const fields = ['id', 'type', 'targetId', 'reason', 'timestamps', 'evidence', 'authority'];
      if (record.type === 'supersession') fields.push('replacementId');
      object(record, p, fields, fields);
      id(record.targetId, `${p}.targetId`); string(record.reason, `${p}.reason`);
      identity(record.authority, `${p}.authority`);
      if (object(record.timestamps, `${p}.timestamps`, ['registeredAt'], ['registeredAt'])) date(record.timestamps.registeredAt, `${p}.timestamps.registeredAt`);
      evidence(record.evidence, `${p}.evidence`, record.timestamps?.registeredAt);
      const target = records.get(record.targetId);
      if (!target || target.type !== 'result') fail(`${p}.targetId`, 'must refer to an earlier result record');
      if (disposition.has(record.targetId)) fail(`${p}.targetId`, 'result already has a supersession or retraction');
      disposition.add(record.targetId);
      ordered(target?.timestamps?.registeredAt, record.timestamps?.registeredAt, `${p}.timestamps.registeredAt`);
      if (record.type === 'supersession') {
        id(record.replacementId, `${p}.replacementId`);
        const replacement = records.get(record.replacementId);
        if (!replacement || replacement.type !== 'result') fail(`${p}.replacementId`, 'must refer to an earlier result record');
        if (record.replacementId === record.targetId) fail(`${p}.replacementId`, 'cannot supersede itself');
        if (disposition.has(record.replacementId)) fail(`${p}.replacementId`, 'replacement must still be active');
        if (target && replacement && (target.entrant?.id !== replacement.entrant?.id || target.protocolId !== replacement.protocolId || target.links !== replacement.links)) fail(p, 'replacement must concern the same entrant, protocol and link count');
        ordered(replacement?.timestamps?.registeredAt, record.timestamps?.registeredAt, `${p}.timestamps.registeredAt`);
      }
      records.set(record.id, record); return;
    }
    const fields = ['id', 'type', 'protocolId', 'protocolFingerprint', 'links', 'qualified', 'entrant', 'method', 'source', 'model', 'runtime', 'timestamps', 'evidence', 'cohorts', 'metrics', 'verification', 'notes'];
    object(record, p, fields, fields);
    if (record.type !== 'result') fail(`${p}.type`, 'must be result, supersession or retraction');
    records.set(record.id, record);
    id(record.protocolId, `${p}.protocolId`); hash(record.protocolFingerprint, `${p}.protocolFingerprint`);
    const protocol = protocols.get(record.protocolId), req = protocol?.requirements;
    if (!protocol) fail(`${p}.protocolId`, 'references missing protocol');
    else if (record.protocolFingerprint !== protocol.fingerprint) fail(`${p}.protocolFingerprint`, 'does not match declared protocol');
    number(record.links, `${p}.links`, { integer: true, positive: true }); boolean(record.qualified, `${p}.qualified`);
    if (protocol && (record.links < protocol.linksMin || record.links > protocol.linksMax)) fail(`${p}.links`, 'outside protocol link range');
    identity(record.entrant, `${p}.entrant`);
    if (object(record.method, `${p}.method`, ['name', 'summary', 'wallClockRealtime', 'usesObservedStateOnly', 'disclosures'], ['name', 'summary', 'wallClockRealtime', 'usesObservedStateOnly', 'disclosures'])) {
      string(record.method.name, `${p}.method.name`, 200); string(record.method.summary, `${p}.method.summary`);
      boolean(record.method.wallClockRealtime, `${p}.method.wallClockRealtime`); boolean(record.method.usesObservedStateOnly, `${p}.method.usesObservedStateOnly`); strings(record.method.disclosures, `${p}.method.disclosures`, 1);
    }
    source(record.source, `${p}.source`);
    if (object(record.model, `${p}.model`, ['url', 'sha256'], ['url', 'sha256'])) { url(record.model.url, `${p}.model.url`); hash(record.model.sha256, `${p}.model.sha256`); }
    const runtimeFields = ['engine', 'engineVersion', 'language', 'languageVersion', 'os', 'architecture', 'precision'];
    if (object(record.runtime, `${p}.runtime`, [...runtimeFields, 'dependencies'], runtimeFields)) {
      for (const field of runtimeFields) string(record.runtime[field], `${p}.runtime.${field}`, 100);
      if (own(record.runtime, 'dependencies')) {
        if (!isObject(record.runtime.dependencies)) fail(`${p}.runtime.dependencies`, 'must be a dependency/version object');
        else for (const [name, version] of Object.entries(record.runtime.dependencies)) { string(name, `${p}.runtime.dependencies key`, 100); string(version, `${p}.runtime.dependencies.${name}`, 100); }
      }
      if (req?.integration) for (const field of ['engine', 'engineVersion', 'precision']) if (record.runtime[field] !== req.integration[field]) fail(`${p}.runtime.${field}`, 'must match this protocol runtime');
    }
    if (object(record.timestamps, `${p}.timestamps`, ['runAt', 'frozenAt', 'firstPublicAt', 'registeredAt'], ['runAt', 'frozenAt', 'firstPublicAt', 'registeredAt'])) {
      for (const field of ['runAt', 'frozenAt', 'firstPublicAt']) date(record.timestamps[field], `${p}.timestamps.${field}`, true);
      date(record.timestamps.registeredAt, `${p}.timestamps.registeredAt`);
      ordered(record.timestamps.frozenAt, record.timestamps.runAt, `${p}.timestamps.runAt`);
      ordered(record.timestamps.runAt, record.timestamps.firstPublicAt, `${p}.timestamps.firstPublicAt`);
      for (const field of ['runAt', 'frozenAt', 'firstPublicAt']) ordered(record.timestamps[field], record.timestamps.registeredAt, `${p}.timestamps.registeredAt`);
      if (record.qualified && record.timestamps.frozenAt === null) fail(`${p}.timestamps.frozenAt`, 'qualified results require a recorded freeze');
    }
    const entries = evidence(record.evidence, `${p}.evidence`, record.timestamps?.registeredAt);
    const seenGates = new Set(), actualRanges = []; let episodeCount = 0;
    if (array(record.cohorts, `${p}.cohorts`, record.qualified ? 1 : 0)) record.cohorts.forEach((cohort, i) => {
      const cp = `${p}.cohorts[${i}]`;
      if (!object(cohort, cp, ['id', 'stage', 'noise', 'required', 'attempted', 'passed', 'seedStart', 'seedEnd', 'evidenceId'], ['id', 'stage', 'noise', 'required', 'attempted', 'passed', 'evidenceId'])) return;
      id(cohort.id, `${cp}.id`);
      if (seenGates.has(cohort.id)) fail(`${cp}.id`, 'duplicate cohort ID'); seenGates.add(cohort.id);
      const gate = Array.isArray(req?.cohorts) ? req.cohorts.find(item => isObject(item) && item.id === cohort.id) : null;
      if (!gate) fail(`${cp}.id`, 'not a declared protocol cohort');
      else {
        for (const field of ['stage', 'noise', 'required']) if (cohort[field] !== gate[field]) fail(`${cp}.${field}`, 'must match the protocol cohort');
        for (const field of ['seedStart', 'seedEnd']) if (own(gate, field) && cohort[field] !== gate[field]) fail(`${cp}.${field}`, 'must match the protocol cohort');
      }
      for (const field of ['required', 'attempted', 'passed']) number(cohort[field], `${cp}.${field}`, { integer: true, positive: field === 'required' });
      seedRange(cohort, cp);
      if (own(cohort, 'seedStart')) {
        if (actualRanges.some(range => cohort.seedStart <= range[1] && cohort.seedEnd >= range[0])) fail(cp, 'actual cohort seed ranges must be disjoint');
        actualRanges.push([cohort.seedStart, cohort.seedEnd]);
      }
      if (cohort.passed > cohort.attempted) fail(cp, 'passed cannot exceed attempted');
      if (cohort.attempted > cohort.required) fail(cp, 'attempts exceed the declared cohort; retries cannot be hidden');
      if (record.qualified && (cohort.passed !== cohort.required || cohort.attempted !== cohort.required)) fail(cp, 'qualified results must pass every cohort without replacement attempts');
      if (!entries.has(cohort.evidenceId)) fail(`${cp}.evidenceId`, 'references missing evidence');
      episodeCount += cohort.attempted;
    });
    if (record.qualified && Array.isArray(req?.cohorts)) for (const gate of req.cohorts) if (isObject(gate) && !seenGates.has(gate.id)) fail(`${p}.cohorts`, `missing required cohort ${gate.id}`);
    const metricFields = ['commandsPerEpisode', 'secondsPerEpisode', 'minUprightSeconds', 'maxCartMeters', 'maxNativeDefect', 'maxInitialGap', 'auditedEpisodes', 'auditedCommands'];
    if (object(record.metrics, `${p}.metrics`, metricFields, metricFields)) {
      for (const field of metricFields) number(record.metrics[field], `${p}.metrics.${field}`, { integer: ['commandsPerEpisode', 'auditedEpisodes', 'auditedCommands'].includes(field), positive: ['commandsPerEpisode', 'secondsPerEpisode'].includes(field) });
      if (record.metrics.auditedEpisodes > episodeCount) fail(`${p}.metrics.auditedEpisodes`, 'cannot exceed actual cohort attempts');
      if (record.metrics.auditedCommands > record.metrics.auditedEpisodes * record.metrics.commandsPerEpisode) fail(`${p}.metrics.auditedCommands`, 'exceeds audited episode command total');
      if (record.metrics.minUprightSeconds > record.metrics.secondsPerEpisode) fail(`${p}.metrics.minUprightSeconds`, 'cannot exceed episode duration');
      if (record.qualified && req) {
        if (record.metrics.commandsPerEpisode !== req.commandsPerEpisode || record.metrics.secondsPerEpisode !== req.durationSeconds) fail(`${p}.metrics`, 'qualified result must complete the whole declared episode');
        if (record.metrics.minUprightSeconds + 1e-12 < req.uprightConsecutiveSteps / req.actionHz) fail(`${p}.metrics.minUprightSeconds`, 'below the required upright hold');
        if (record.metrics.maxCartMeters > req.railLimitMeters) fail(`${p}.metrics.maxCartMeters`, 'outside the declared rail');
        if (record.metrics.maxNativeDefect > req.calibrationTolerance || record.metrics.maxInitialGap > req.calibrationTolerance) fail(`${p}.metrics`, 'native defect or initial gap exceeds calibration tolerance');
        const calibrationEpisodes = Array.isArray(req.cohorts)
          ? req.cohorts.filter(gate => isObject(gate) && gate.stage === 'calibration').reduce((sum, gate) => sum + gate.required, 0)
          : 0;
        // Published qualification requires full native calibration, not a cold
        // native re-audit of every development/reserved transition. Aggregate
        // coverage is a lower-bound check; reviewers inspect calibration proof.
        if (record.metrics.auditedEpisodes < calibrationEpisodes || record.metrics.auditedCommands < calibrationEpisodes * req.commandsPerEpisode) fail(`${p}.metrics`, 'qualified result requires complete native calibration audit coverage');
      }
    }
    if (object(record.verification, `${p}.verification`, ['level', 'reviewer', 'reviewedAt', 'evidenceIds', 'reproducedAt', 'reproductionEvidenceIds'], ['level'])) {
      const verification = record.verification;
      if (!['author-attested', 'evidence-reviewed', 'third-party-reproduced'].includes(verification.level)) fail(`${p}.verification.level`, 'unknown review level');
      if (verification.level === 'author-attested') {
        if (Object.keys(verification).length !== 1) fail(`${p}.verification`, 'author-attested records cannot carry implied external-review fields');
      } else {
        identity(verification.reviewer, `${p}.verification.reviewer`, true); date(verification.reviewedAt, `${p}.verification.reviewedAt`);
        ordered(record.timestamps?.frozenAt, verification.reviewedAt, `${p}.verification.reviewedAt`); ordered(verification.reviewedAt, record.timestamps?.registeredAt, `${p}.verification.reviewedAt`);
        refs(verification.evidenceIds, `${p}.verification.evidenceIds`, entries, 'review');
        if (verification.reviewer?.id === record.entrant?.id) fail(`${p}.verification.reviewer.id`, 'an entrant cannot award itself an external review level');
        if (verification.level === 'third-party-reproduced') {
          if (verification.reviewer?.independent !== true) fail(`${p}.verification.reviewer.independent`, 'third-party reproduction requires declared independence');
          date(verification.reproducedAt, `${p}.verification.reproducedAt`);
          ordered(record.timestamps?.frozenAt, verification.reproducedAt, `${p}.verification.reproducedAt`); ordered(verification.reproducedAt, verification.reviewedAt, `${p}.verification.reproducedAt`);
          refs(verification.reproductionEvidenceIds, `${p}.verification.reproductionEvidenceIds`, entries, 'reproduction');
        } else if (own(verification, 'reproducedAt') || own(verification, 'reproductionEvidenceIds')) fail(`${p}.verification`, 'reproduction fields require third-party-reproduced level');
      }
    }
    strings(record.notes, `${p}.notes`);
  });

  if (base !== null) {
    if (!isObject(base) || base.schemaVersion !== registry.schemaVersion || !Array.isArray(base.protocols) || !Array.isArray(base.records)) fail('$base', 'base registry has an incompatible structure');
    else for (const group of ['protocols', 'records']) {
      if (!Array.isArray(registry[group]) || registry[group].length < base[group].length) { fail(`$.${group}`, 'append-only history cannot remove entries'); continue; }
      base[group].forEach((entry, index) => {
        try { if (canonicalJson(entry) !== canonicalJson(registry[group][index])) fail(`$.${group}[${index}]`, 'append-only history cannot edit, replace or reorder existing entries'); }
        catch { fail(`$.${group}[${index}]`, 'invalid value in append-only comparison'); }
      });
    }
  }
  return errors;
}

async function main() {
  const args = process.argv.slice(2);
  let file = 'dist/data/registry.json', baseFile = null;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--base-file' && args[i + 1]) baseFile = args[++i];
    else if (!args[i].startsWith('-') && file === 'dist/data/registry.json') file = args[i];
    else throw new Error('Usage: node scripts/validate-registry.mjs [registry.json] [--base-file base.json]');
  }
  const registry = JSON.parse(await readFile(file, 'utf8'));
  const base = baseFile ? JSON.parse(await readFile(baseFile, 'utf8')) : null;
  const errors = validateRegistry(registry, { base });
  if (errors.length) { console.error(errors.join('\n')); process.exitCode = 1; }
  else console.log(`Registry valid: ${registry.protocols.length} protocol(s), ${registry.records.length} append-only record(s). Hashes and review claims still require human verification.`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main().catch(error => { console.error(error.message); process.exitCode = 1; });
