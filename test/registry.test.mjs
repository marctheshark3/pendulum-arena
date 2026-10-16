import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { canonicalJson, fingerprintContract, validateRegistry } from '../scripts/validate-registry.mjs';

const H = 'a'.repeat(64);
const C = 'b'.repeat(40);
const clone = value => structuredClone(value);

function fixture() {
  const requirements = {
    durationSeconds: 30, commandsPerEpisode: 1500, actionHz: 50,
    uprightAngleRad: 0.15, uprightConsecutiveSteps: 250, railLimitMeters: 3,
    forceLimitNewtons: 80, calibrationTolerance: 1e-8,
    integration: { engine: 'MuJoCo', engineVersion: '3.3.5', precision: 'binary64', integrator: 'RK4', timestepSeconds: 0.005, substeps: 4 },
    mechanics: { cartMassKg: 1, totalLinkMassKg: 1, totalChainLengthMeters: 3, capsuleRadiusMeters: 0.025, armature: 0.0005, totalHingeDamping: 0.015, cartDamping: 0.02, frictionLoss: 0 },
    reset: { position: 'relative [0, pi, 0, ...]', noiseStd: 0.05, cartPositionNoiseStd: 0, rngOrder: 'cart zero-scale draw; joint positions; all velocities; replacement cart velocity' },
    acceptance: { angleConvention: 'wrapped cumulative absolute', angleComparison: 'strictly-less-than', samples: 'post-command only', fullEpisodeRequired: true, stateOverwriteAfterInit: false, liveRetriesAllowed: false, allFiniteRequired: true },
    cohorts: [
      { id: 'calibration', stage: 'calibration', noise: 'exact', required: 1 },
      { id: 'dev20', stage: 'development', noise: 'noisy', required: 20 },
      { id: 'dev100', stage: 'development', noise: 'noisy', required: 100 },
      { id: 'reserved-noisy20', stage: 'reserved', noise: 'noisy', required: 20, seedStart: 215000, seedEnd: 215019 },
      { id: 'reserved-noisy100', stage: 'reserved', noise: 'noisy', required: 100, seedStart: 215100, seedEnd: 215199 },
      { id: 'reserved-exact20', stage: 'reserved', noise: 'exact', required: 20, seedStart: 215200, seedEnd: 215219 },
      { id: 'video', stage: 'video', noise: 'noisy', required: 1, seedStart: 215500, seedEnd: 215500 }
    ]
  };
  const fingerprint = fingerprintContract(requirements);
  return {
    schemaVersion: 1,
    protocols: [{ id: 'canonical15-v1', title: 'Published uniform chain, fifteen links', linksMin: 15, linksMax: 15, source: { url: 'https://github.com/corbensorenson/cartpole', commit: C }, fingerprint, requirements }],
    records: [{
      id: 'example15', type: 'result', protocolId: 'canonical15-v1', protocolFingerprint: fingerprint, links: 15, qualified: true,
      entrant: { id: 'entrant', name: 'Example entrant', url: 'https://example.org/team' },
      method: { name: 'Example method', summary: 'Native model planning', wallClockRealtime: false, usesObservedStateOnly: true, disclosures: ['Simulation only; wall-clock control deadlines not demonstrated'] },
      source: { url: `https://github.com/example/controller/tree/${C}`, commit: C, sha256: H },
      model: { url: 'https://example.org/model.xml', sha256: H },
      runtime: { engine: 'MuJoCo', engineVersion: '3.3.5', language: 'Python', languageVersion: '3.12.3', os: 'Linux', architecture: 'aarch64', precision: 'binary64', dependencies: { numpy: '2.5.2' } },
      timestamps: { runAt: null, frozenAt: '2026-10-16T17:54:52.208417Z', firstPublicAt: null, registeredAt: '2026-10-16T20:00:00Z' },
      evidence: [{ id: 'audit', kind: 'receipt', url: 'https://example.org/audit.json', sha256: H, publishedAt: null }],
      cohorts: requirements.cohorts.map(gate => ({ ...gate, attempted: gate.required, passed: gate.required, evidenceId: 'audit' })),
      metrics: { commandsPerEpisode: 1500, secondsPerEpisode: 30, minUprightSeconds: 6.38, maxCartMeters: 2.7244540435262334, maxNativeDefect: 0, maxInitialGap: 0, auditedEpisodes: 262, auditedCommands: 393000 },
      verification: { level: 'author-attested' }, notes: ['Same owner, separate scorer; no external reproduction claim']
    }]
  };
}

function rejects(mutator, message) {
  const value = fixture();
  mutator(value, value.records[0], value.protocols[0]);
  const errors = validateRegistry(value);
  assert.ok(errors.length, 'invalid registry unexpectedly passed');
  if (message) assert.match(errors.join('\n'), message);
}

function reviewed(value, level = 'evidence-reviewed') {
  const result = value.records[0];
  result.evidence.push({ id: 'review', kind: 'review', url: 'https://example.net/review.json', sha256: H, publishedAt: '2026-10-16T19:30:00Z' });
  result.verification = { level, reviewer: { id: 'reviewer', name: 'Named reviewer', url: 'https://example.net/reviewer', independent: false }, reviewedAt: '2026-10-16T19:30:00Z', evidenceIds: ['review'] };
  if (level === 'third-party-reproduced') {
    result.evidence.push({ id: 'rerun', kind: 'reproduction', url: 'https://example.net/rerun.json', sha256: H, publishedAt: '2026-10-16T19:00:00Z' });
    result.verification.reviewer.independent = true;
    result.verification.reproducedAt = '2026-10-16T19:00:00Z';
    result.verification.reproductionEvidenceIds = ['rerun'];
  }
  return value;
}

function retraction(targetId = 'example15') {
  return { id: 'withdraw-example15', type: 'retraction', targetId, reason: 'Published correction identifies invalid source identity', timestamps: { registeredAt: '2026-10-17T12:00:00Z' }, evidence: [{ id: 'correction', kind: 'review', url: 'https://example.org/correction.json', sha256: H, publishedAt: '2026-10-17T11:00:00Z' }], authority: { id: 'entrant', name: 'Example entrant', url: 'https://example.org/team' } };
}

test('complete exact/dev20/dev100/reserved141 claim is internally consistent without external-review inflation', () => {
  const value = fixture();
  assert.equal(value.records[0].cohorts.reduce((n, cohort) => n + cohort.attempted, 0), 262);
  assert.deepEqual(validateRegistry(value), []);
});

test('fingerprint is deterministic across key order and sensitive to arrays/mechanics', () => {
  assert.equal(canonicalJson({ b: [2, { z: 3, a: 4 }], a: 1 }), '{"a":1,"b":[2,{"a":4,"z":3}]}');
  assert.equal(fingerprintContract({ b: 2, a: 1 }), fingerprintContract({ a: 1, b: 2 }));
  assert.notEqual(fingerprintContract({ a: [1, 2] }), fingerprintContract({ a: [2, 1] }));
  for (const invalid of [NaN, Infinity, undefined, { a: undefined }, Array(1)]) assert.throws(() => canonicalJson(invalid));
  rejects((_v, _r, protocol) => { protocol.requirements.forceLimitNewtons = 10; }, /canonical requirements SHA256/);
});

test('a distinct link-count protocol is supported without silently comparing fingerprints', () => {
  const value = fixture();
  const protocol = value.protocols[0];
  protocol.id = 'another-five'; protocol.linksMin = 5; protocol.linksMax = 5;
  value.records[0].protocolId = protocol.id; value.records[0].links = 5;
  assert.deepEqual(validateRegistry(value), []);
  rejects((_v, record) => { record.links = 16; }, /outside protocol link range/);
  rejects((_v, record) => { record.protocolFingerprint = 'c'.repeat(64); }, /does not match declared protocol/);
});

test('every leaf gate is necessary, including dev20 and video', () => {
  for (const gateId of ['calibration', 'dev20', 'dev100', 'reserved-noisy20', 'reserved-noisy100', 'reserved-exact20', 'video']) {
    rejects((_v, record) => { record.cohorts = record.cohorts.filter(cohort => cohort.id !== gateId); }, new RegExp(`missing required cohort ${gateId}`));
  }
});

test('cohort arithmetic refuses hidden failures, replacement attempts and shifted reserved seeds', () => {
  rejects((_v, r) => { r.cohorts[1].passed = 19; }, /must pass every cohort/);
  rejects((_v, r) => { r.cohorts[1].attempted = 21; }, /retries cannot be hidden/);
  rejects((_v, r) => { r.cohorts[1].passed = 21; }, /passed cannot exceed attempted/);
  rejects((_v, r) => { r.cohorts[3].seedStart++; r.cohorts[3].seedEnd++; }, /must match the protocol cohort/);
  rejects((_v, r) => { r.cohorts[3].seedEnd--; }, /seed range must contain exactly/);
  rejects((_v, r) => { r.cohorts[3].evidenceId = 'missing'; }, /references missing evidence/);
});

test('entrant-selected development labels are permitted but must remain disjoint', () => {
  const value = fixture(), result = value.records[0];
  Object.assign(result.cohorts[0], { seedStart: 3156000, seedEnd: 3156000 });
  Object.assign(result.cohorts[1], { seedStart: 3156200, seedEnd: 3156219 });
  Object.assign(result.cohorts[2], { seedStart: 3156300, seedEnd: 3156399 });
  assert.deepEqual(validateRegistry(value), []);
  Object.assign(result.cohorts[2], { seedStart: 3156200, seedEnd: 3156299 });
  assert.match(validateRegistry(value).join('\n'), /actual cohort seed ranges must be disjoint/);
});

test('declared native audit coverage is bounded by actual episodes and excludes forecasts', () => {
  const value = fixture();
  value.records[0].metrics.auditedEpisodes = 261;
  value.records[0].metrics.auditedCommands = 391500;
  assert.deepEqual(validateRegistry(value), []);
  rejects((_v, r) => { r.metrics.auditedEpisodes = 263; }, /cannot exceed actual cohort attempts/);
  rejects((_v, r) => { r.metrics.auditedCommands = 400000; }, /exceeds audited episode command total/);
});

test('qualification requires complete native calibration but not cold audits of all cohorts', () => {
  const value = fixture();
  value.records[0].metrics.auditedEpisodes = 1;
  value.records[0].metrics.auditedCommands = 1500;
  assert.deepEqual(validateRegistry(value), []);
  value.records[0].metrics.auditedEpisodes = 0;
  value.records[0].metrics.auditedCommands = 0;
  assert.match(validateRegistry(value).join('\n'), /complete native calibration audit coverage/);
  value.records[0].metrics.auditedEpisodes = 1;
  value.records[0].metrics.auditedCommands = 450;
  assert.match(validateRegistry(value).join('\n'), /complete native calibration audit coverage/);
});

test('finite metrics, complete duration, hold, rail and calibration tolerances are enforced', () => {
  for (const bad of [NaN, Infinity, -Infinity, -1, '0']) rejects((_v, r) => { r.metrics.maxNativeDefect = bad; }, /finite/);
  rejects((_v, r) => { r.metrics.maxNativeDefect = 1.0001e-8; }, /calibration tolerance/);
  rejects((_v, r) => { r.metrics.maxInitialGap = 2e-8; }, /calibration tolerance/);
  rejects((_v, r) => { r.metrics.maxCartMeters = 3.0001; }, /outside the declared rail/);
  rejects((_v, r) => { r.metrics.minUprightSeconds = 4.999; }, /below the required upright hold/);
  rejects((_v, r) => { r.metrics.commandsPerEpisode = 1499; }, /whole declared episode/);
  rejects((_v, r) => { r.metrics.secondsPerEpisode = 29.98; }, /whole declared episode/);
  rejects((_v, r) => { r.metrics.minUprightSeconds = 31; }, /cannot exceed episode duration/);
});

test('partial/failing claims remain eligible for an explicitly unqualified record', () => {
  const value = fixture(), result = value.records[0];
  result.qualified = false;
  result.timestamps.frozenAt = null;
  result.cohorts = [{ ...result.cohorts[0], passed: 0 }];
  result.metrics = { commandsPerEpisode: 1500, secondsPerEpisode: 30, minUprightSeconds: 0, maxCartMeters: 3.1, maxNativeDefect: 0, maxInitialGap: 0, auditedEpisodes: 1, auditedCommands: 300 };
  assert.deepEqual(validateRegistry(value), []);
});

test('UTC dates reject impossible calendar days, offsets, invalid hours and date-only strings', () => {
  for (const bad of ['2026-02-30T12:00:00Z', '2026-10-16T24:00:00Z', '2026-10-16T12:00:60Z', '2026-10-16T12:00:00-04:00', '2026-10-16', 'not-a-date', 123, null]) {
    rejects((_v, r) => { r.timestamps.registeredAt = bad; }, /real UTC ISO-8601/);
  }
  const value = fixture();
  value.records[0].timestamps.frozenAt = '2026-10-16T17:54:52.208417+00:00';
  assert.deepEqual(validateRegistry(value), []);
});

test('microsecond freeze ordering and known public/registry order remain honest', () => {
  rejects((_v, r) => { r.timestamps.runAt = '2026-10-16T17:54:52.208416Z'; }, /timestamp order is reversed/);
  rejects((_v, r) => { r.timestamps.runAt = '2026-10-16T19:00:00Z'; r.timestamps.firstPublicAt = '2026-10-16T18:00:00Z'; }, /timestamp order is reversed/);
  rejects((_v, r) => { r.timestamps.firstPublicAt = '2026-10-17T00:00:00Z'; }, /timestamp order is reversed/);
  rejects((_v, r) => { r.timestamps.frozenAt = null; }, /require a recorded freeze/);
  rejects((_v, r) => { r.evidence[0].publishedAt = '2026-10-17T00:00:00Z'; }, /timestamp order is reversed/);
});

test('artifact/source identity cannot be shortened, uppercased or omitted', () => {
  rejects((_v, r) => { r.source.commit = C.slice(0, 7); }, /full lowercase forty-character/);
  for (const bad of ['f'.repeat(63), 'F'.repeat(64), 'sha256:' + H, null]) rejects((_v, r) => { r.evidence[0].sha256 = bad; }, /full lowercase SHA256/);
  rejects((_v, r) => { delete r.model.sha256; }, /is required/);
  const value = fixture(); delete value.records[0].source.sha256;
  assert.deepEqual(validateRegistry(value), []);
});

test('public links refuse script/data/file, credentials, local hosts and numeric-IP obfuscation', () => {
  for (const bad of ['javascript:alert(1)', 'data:text/plain,hello', 'file:///tmp/a', '/local/path', 'https://user:secret@example.org/a', 'http://localhost/a', 'http://test.local/a', 'http://127.1/a', 'http://0x7f000001/a', 'http://10.0.0.1/a', 'http://172.16.1.1/a', 'http://192.168.2.1/a', 'http://169.254.169.254/a', 'http://[::1]/a', 'http://[::ffff:127.0.0.1]/a', 'https://example.org/ a', 'https:\\example.org/a']) {
    rejects((_v, r) => { r.evidence[0].url = bad; }, /absolute public HTTP/);
  }
});

test('duplicate IDs and undeclared cohort overlap are refused', () => {
  rejects((v, r) => { v.records.push(clone(r)); }, /duplicate record ID/);
  rejects((v, _r, p) => { v.protocols.push(clone(p)); }, /duplicate protocol ID/);
  rejects((_v, r) => { r.evidence.push(clone(r.evidence[0])); }, /duplicate evidence ID/);
  rejects((_v, r) => { r.cohorts.push(clone(r.cohorts[0])); }, /duplicate cohort ID/);
  rejects((_v, _r, p) => { p.requirements.cohorts[4].seedStart = 215000; p.requirements.cohorts[4].seedEnd = 215099; p.fingerprint = fingerprintContract(p.requirements); }, /seed ranges must be disjoint/);
});

test('runtime deviations cannot claim the exact protocol fingerprint', () => {
  for (const [key, value] of [['engine', 'other'], ['engineVersion', '3.4.0'], ['precision', 'binary32']]) rejects((_v, r) => { r.runtime[key] = value; }, /must match this protocol runtime/);
  rejects((_v, _r, p) => { p.requirements.integration.substeps = 5; p.fingerprint = fingerprintContract(p.requirements); }, /must match actionHz/);
});

test('external review requires named reviewer, linked review proof and a disclosed relationship', () => {
  assert.deepEqual(validateRegistry(reviewed(fixture())), []);
  rejects((_v, r) => { r.verification.level = 'evidence-reviewed'; }, /reviewer/);
  const self = reviewed(fixture()); self.records[0].verification.reviewer.id = 'entrant';
  assert.match(validateRegistry(self).join('\n'), /cannot award itself/);
  const missing = reviewed(fixture()); missing.records[0].verification.evidenceIds = ['audit'];
  assert.match(validateRegistry(missing).join('\n'), /must include review evidence/);
  rejects((_v, r) => { r.verification.reviewer = { id: 'x' }; }, /implied external-review fields/);
});

test('third-party reproduction needs independent identity and distinct rerun evidence', () => {
  assert.deepEqual(validateRegistry(reviewed(fixture(), 'third-party-reproduced')), []);
  const value = reviewed(fixture(), 'third-party-reproduced');
  value.records[0].verification.reviewer.independent = false;
  assert.match(validateRegistry(value).join('\n'), /requires declared independence/);
  value.records[0].verification.reviewer.independent = true;
  value.records[0].verification.reproductionEvidenceIds = ['review'];
  assert.match(validateRegistry(value).join('\n'), /must include reproduction evidence/);
  const evidenceOnly = reviewed(fixture()); evidenceOnly.records[0].verification.reproducedAt = '2026-10-16T19:00:00Z';
  assert.match(validateRegistry(evidenceOnly).join('\n'), /require third-party-reproduced level/);
});

test('append-only comparison permits new events but prevents history edits, removal or reorder', () => {
  const base = fixture(), value = clone(base);
  value.records.push(retraction());
  assert.deepEqual(validateRegistry(value, { base }), []);
  for (const mutate of [v => { v.records[0].notes.push('silent rewrite'); }, v => { v.records.shift(); }, v => { v.records.reverse(); }, v => { v.protocols[0].title += ' changed'; }]) {
    const altered = clone(value); mutate(altered);
    assert.match(validateRegistry(altered, { base }).join('\n'), /append-only history/);
  }
});

test('supersession retains both results and refuses cross-entrant or cyclic disposition', () => {
  const base = fixture(), value = clone(base), replacement = clone(value.records[0]);
  replacement.id = 'example15-corrected'; replacement.notes.push('Corrected attribution'); replacement.timestamps.registeredAt = '2026-10-17T10:00:00Z';
  value.records.push(replacement);
  const event = retraction(); event.id = 'replace-example15'; event.type = 'supersession'; event.replacementId = replacement.id;
  value.records.push(event);
  assert.deepEqual(validateRegistry(value, { base }), []);
  replacement.entrant.id = 'different-entrant';
  assert.match(validateRegistry(value).join('\n'), /same entrant, protocol and link count/);
  replacement.entrant.id = 'entrant';
  value.records.push({ ...retraction(), id: 'withdraw-again' });
  assert.match(validateRegistry(value).join('\n'), /already has a supersession or retraction/);
});

test('disposition events cannot target a missing/future record or erase their evidence', () => {
  const value = fixture(); value.records.push(retraction('missing'));
  assert.match(validateRegistry(value).join('\n'), /earlier result record/);
  value.records[1] = retraction(); value.records[1].evidence = [];
  assert.match(validateRegistry(value).join('\n'), /at least 1 item/);
});

test('malformed nested JSON produces diagnostics rather than crashing the validator', () => {
  for (const value of [null, [], {}, { schemaVersion: 1, protocols: [null], records: [null] }]) assert.ok(validateRegistry(value).length);
  for (const mutate of [
    v => { v.protocols[0].requirements.cohorts = {}; },
    v => { v.protocols[0].requirements.cohorts = [null]; },
    v => { v.protocols[0].requirements = null; },
    v => { v.records[0].cohorts = null; },
    v => { v.records[0].cohorts = [null]; },
    v => { v.records[0].verification = null; },
    v => { v.records[0].evidence = [null]; },
    v => { v.records[0].timestamps = null; },
    v => { v.records[0].runtime = []; }
  ]) {
    const value = fixture(); mutate(value);
    assert.ok(validateRegistry(value).length);
  }
});

test('machine-readable schema is valid JSON with strict top-level and review/event contracts', async () => {
  const schema = JSON.parse(await readFile(new URL('../schemas/registry.schema.json', import.meta.url), 'utf8'));
  assert.equal(schema.$schema, 'https://json-schema.org/draft/2020-12/schema');
  assert.equal(schema.additionalProperties, false);
  assert.ok(schema.$defs.verification.allOf.length);
  assert.equal(schema.$defs.event.additionalProperties, false);
});
