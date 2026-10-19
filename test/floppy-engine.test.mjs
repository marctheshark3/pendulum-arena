import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import loadMujoco from '../dist/floppy/engine/vendor/mujoco.js';
import { createGame, deliveredForce, RUNTIME_VERSION, BROWSER_PROFILE } from '../dist/floppy/engine/game.js';

const base = new URL('../dist/floppy/engine/', import.meta.url);
const configuration = JSON.parse(await readFile(new URL('profile.json', base), 'utf8'));
const serves = JSON.parse(await readFile(new URL('serves.json', base), 'utf8'));
const modelXML = await readFile(new URL('model.xml', base), 'utf8');
const wasmBinary = await readFile(new URL('vendor/mujoco.wasm', base));
const loaderBytes = await readFile(new URL('vendor/mujoco.js', base));
const module = await loadMujoco({ wasmBinary });
const make = options => createGame({ configuration, serves, modelXML, module, ...options });

test('browser runtime is pinned by version and binary hash', () => {
  assert.equal(module.mj_versionString(), RUNTIME_VERSION);
  assert.equal(RUNTIME_VERSION, '3.8.0');
  assert.match(BROWSER_PROFILE, /human-mujoco-3\.8\.0-v1$/);
  assert.equal(createHash('sha256').update(wasmBinary).digest('hex'), 'a91233f2b4d77bdac310e2050a07e3b882da7fade32d94c80c960a9ec5f5ea53');
  assert.equal(createHash('sha256').update(loaderBytes).digest('hex'), '66777739ae30c38c7ead0221a79e05cbcbad31da3eabeea8fb3f4789c80f22cc');
});

test('force delivery rejects malformed input and preserves F32 clipping', () => {
  for (const invalid of [NaN, Infinity, -Infinity, true, '5', null, undefined]) assert.throws(() => deliveredForce(invalid));
  assert.equal(deliveredForce(123), 80);
  assert.equal(deliveredForce(-123), -80);
  assert.equal(deliveredForce(1 / 3), Math.fround(1 / 3));
});

test('First Light retains loaded passive geometry and original fifteen-brick score', async () => {
  const game = await make();
  try {
    const snapshot = game.snapshot();
    assert.equal(snapshot.board.bricks.length, 15);
    assert.equal(snapshot.board.id, 'first-light');
    assert.equal(snapshot.lives, 3);
    assert.equal(snapshot.phase, 'warmup');
    assert.equal(snapshot.life, 1);
    assert.deepEqual(snapshot.own.paddle_world, [0, -3]);
    assert.match(modelXML, /mass="0\.05"/);
    assert.match(modelXML, /gravcomp="1"/);
    assert.doesNotMatch(modelXML, /top_cart|top_hinge|joint="bottom_hinge"/);
    assert.equal(snapshot.profile.timestep, .0000625);
  } finally { game.dispose(); }
});

test('ball remains force-free between contacts and a command contains 320 RK4 steps', async () => {
  const game = await make();
  try {
    game.restoreValidationState({ qpos: [0, 0, 0, 0], qvel: [0, 0, 1, 2], phase: 'rally' });
    const snapshot = game.step(0);
    assert.equal(snapshot.steps, 320);
    assert.equal(snapshot.time, .02);
    assert.ok(Math.abs(snapshot.ball_world[0] - .02) < 1e-12);
    assert.ok(Math.abs(snapshot.ball_world[1] - .04) < 1e-12);
    assert.deepEqual(snapshot.ball_velocity_world, [1, 2]);
    assert.deepEqual(snapshot.qpos.slice(0, 2), [0, 0]);
  } finally { game.dispose(); }
});

test('a native brick collision damages once, then removes only after separation', async () => {
  const game = await make();
  try {
    game.restoreValidationState({ qpos: [0, 0, -1.44, 1.55], qvel: [0, 0, 0, 3.5], phase: 'rally' });
    let pendingSeen = false;
    const snapshot = game.step(0, frame => {
      if (frame.health.b00 === 0 && frame.active.b00) {
        pendingSeen = true;
        assert.ok(frame.pending.includes('b00'));
      }
    });
    assert.ok(pendingSeen, 'destroyed brick retains contact until separated');
    assert.equal(snapshot.score, 100);
    assert.equal(snapshot.health.b00, 0);
    assert.equal(snapshot.active.b00, false);
    assert.equal(snapshot.events.filter(event => event.kind === 'brick_damage').length, 1);
    const start = snapshot.events.find(event => event.kind === 'contact_begin');
    assert.equal(start.geom, 'brick_b00');
    assert.ok(start.motion);
    assert.ok(snapshot.ball_velocity_world[1] < 0);
  } finally { game.dispose(); }
});

test('clearing the last brick waits for native contact separation', async () => {
  const game = await make();
  try {
    const health = Object.fromEntries(configuration.board.bricks.map(brick => [brick.id, brick.id === 'b00' ? 1 : 0]));
    game.restoreValidationState({ qpos: [0, 0, -1.44, 1.55], qvel: [0, 0, 0, 3.5], phase: 'rally', health });
    const snapshot = game.step(0);
    assert.equal(snapshot.outcome, 'cleared');
    assert.equal(snapshot.phase, 'finished');
    assert.deepEqual(snapshot.pending, []);
    const kinds = snapshot.events.map(event => event.kind);
    assert.ok(kinds.indexOf('brick_removed') < kinds.indexOf('episode_end'));
  } finally { game.dispose(); }
});

test('rail fault loses an ordinary life and preserves damaged bricks after reset', async () => {
  const game = await make();
  try {
    const health = { ...game.snapshot().health, b00: 0 };
    game.restoreValidationState({ qpos: [2.001, 0, 0, 0], qvel: [0, 0, 0, 0], phase: 'rally', health });
    const fault = game.step(0);
    assert.equal(fault.lives, 2);
    assert.equal(fault.phase, 'new_life');
    assert.deepEqual(fault.events.find(event => event.kind === 'life_lost').reasons, ['rail']);
    const next = game.step(0);
    assert.equal(next.life, 2);
    assert.equal(next.phase, 'warmup');
    assert.equal(next.health.b00, 0);
    assert.equal(next.active.b00, false);
    assert.equal(next.qpos[0], 0);
  } finally { game.dispose(); }
});

test('timeout truncates the command and restart exactly restores an identical trajectory', async () => {
  const game = await make({ duration: .01 });
  try {
    const first = game.step(2);
    assert.equal(first.outcome, 'timeout');
    assert.equal(first.steps, 160);
    game.reset();
    const repeated = game.step(2);
    assert.deepEqual(repeated.qpos, first.qpos);
    assert.deepEqual(repeated.qvel, first.qvel);
    assert.equal(repeated.score, 0);
    assert.equal(repeated.lives, 3);
  } finally { game.dispose(); }
});

test('human posture assistance does not advance physics or read/retarget to the ball', async () => {
  const game = await make();
  try {
    const original = game.snapshot();
    const first = game.assistedForce(1);
    assert.ok(first < 0 && first >= -80);
    assert.deepEqual(game.snapshot().qpos, original.qpos);
    assert.equal(game.snapshot().time, 0);
    game.reset();
    game.restoreValidationState({ qpos: [0, 0, 1, 2], qvel: [0, 0, -3, 4], phase: 'rally' });
    assert.equal(game.assistedForce(1), first);
  } finally { game.dispose(); }
});
