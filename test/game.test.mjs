import test from "node:test";
import assert from "node:assert/strict";
import {
  TOY_PARAMETERS,
  createToyState,
  advanceToy,
  inspectToy,
  demonstrationInput,
} from "../dist/assets/arena-game.js";

test("a rod at rest directly beneath the cart stays at equilibrium", () => {
  let state = { x: 0, velocity: 0, angle: Math.PI, angularVelocity: 0, time: 0 };
  for (let tick = 0; tick < 1200; tick += 1) state = advanceToy(state, 0, 1 / 120);
  assert.ok(Math.abs(state.x) < 1e-10);
  assert.ok(Math.abs(Math.abs(state.angle) - Math.PI) < 1e-10);
  assert.ok(Math.abs(state.time - 10) < 1e-10);
});

test("a rightward push moves an upright cart right and initially tilts its pole left", () => {
  const state = { x: 0, velocity: 0, angle: 0, angularVelocity: 0, time: 0 };
  const pushed = advanceToy(state, 1, 0.05);
  assert.ok(pushed.x > 0);
  assert.ok(pushed.velocity > 0);
  assert.ok(pushed.angle < 0);
  assert.ok(pushed.angularVelocity < 0);
  assert.deepEqual(state, { x: 0, velocity: 0, angle: 0, angularVelocity: 0, time: 0 });
});

test("saturated input has the same physical result as a maximum push", () => {
  const state = createToyState();
  assert.deepEqual(advanceToy(state, 50, 0.04), advanceToy(state, 1, 0.04));
  assert.deepEqual(advanceToy(state, -50, 0.04), advanceToy(state, -1, 0.04));
});

test("the balance demo earns its five-second hold without hitting a rail", () => {
  let state = createToyState();
  let hold = 0;
  let maximumCart = 0;
  for (let tick = 0; tick < 1200; tick += 1) {
    state = advanceToy(state, demonstrationInput(state), 1 / 120);
    const result = inspectToy(state);
    assert.equal(result.finite, true);
    assert.equal(result.fallen, false);
    assert.equal(result.railHit, false);
    maximumCart = Math.max(maximumCart, Math.abs(state.x));
    hold = result.upright ? hold + 1 / 120 : 0;
  }
  assert.ok(hold >= TOY_PARAMETERS.goalSeconds);
  assert.ok(maximumCart < 0.5);
});

test("letting go cannot silently earn a successful balance run", () => {
  let state = createToyState();
  let fallen = false;
  for (let tick = 0; tick < 600 && !fallen; tick += 1) {
    state = advanceToy(state, 0, 1 / 120);
    fallen = inspectToy(state).fallen;
  }
  assert.equal(fallen, true);
  assert.ok(state.time < TOY_PARAMETERS.goalSeconds);
});

test("hanging-start motion stays finite through a minute of alternating pushes", () => {
  let state = createToyState("swingup");
  for (let tick = 0; tick < 7200; tick += 1) {
    const input = Math.sin(tick / 85);
    state = advanceToy(state, input, 1 / 120);
    assert.equal(inspectToy(state, "swingup").finite, true);
    assert.ok(Math.abs(state.angle) <= Math.PI);
  }
});

test("failure and goal checks reject nonfinite state and respect rail and angle limits", () => {
  assert.equal(inspectToy({ ...createToyState(), x: NaN }).finite, false);
  assert.equal(inspectToy({ ...createToyState(), time: Infinity }).finite, false);
  assert.equal(inspectToy({ ...createToyState(), angle: Math.PI * 2 }).upright, true);
  assert.equal(inspectToy({ ...createToyState(), angle: TOY_PARAMETERS.uprightAngle }).upright, false);
  assert.equal(inspectToy({ ...createToyState(), x: TOY_PARAMETERS.railLimit + 0.001 }).railHit, true);
  assert.equal(inspectToy(createToyState("swingup"), "swingup").fallen, false);
  assert.equal(inspectToy(createToyState("swingup")).fallen, true);
});

test("integration refuses invalid clocks, inputs, and uncontrolled jumps", () => {
  for (const seconds of [NaN, Infinity, -0.01, 0.2]) {
    assert.throws(() => advanceToy(createToyState(), 0, seconds), RangeError);
  }
  assert.throws(() => advanceToy(createToyState(), NaN, 1 / 120), RangeError);
  assert.throws(() => advanceToy({ ...createToyState(), angularVelocity: NaN }, 0, 1 / 120), RangeError);
});
