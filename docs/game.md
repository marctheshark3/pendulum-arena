# The browser challenge

`dist/assets/arena-game.js` is a dependency-free ES module. Import and call it after the container exists:

```js
import { mountArena } from "/assets/arena-game.js";
const toy = mountArena(document.querySelector("#play-arena"));
// Before removing or replacing the container:
toy.destroy();
```

The module owns the container's children and installs scoped `arena-game-*` styles. It uses a 2D canvas, `requestAnimationFrame`, pointer events, and a resize observer when available. It never starts moving before an explicit action. All motion stops when paused, hidden, blurred, or detached through `destroy()`. A stalled frame pauses the session rather than crediting unseen seconds. There are no idle animation loops or confetti, including for people who prefer reduced motion.

The default challenge begins near upright. A second, harder mode starts hanging. Use arrow keys, A/D, or the on-screen direction buttons to apply cart force. Keyboard controls are active only while focus is inside the toy. Space pauses when the canvas has focus. Buttons remain keyboard accessible; activating a direction button with a switch or keyboard supplies a short impulse. The canvas has an accessible description; start, pause, success, and failure messages also announce through a polite status region. The animation itself is a visual game and does not have a nonvisual equivalent.

The target is five consecutive simulated seconds within 0.2 radians of upright, while the cart remains within ±2.4 metres. A balance run ends when the pole falls past horizontal; hanging-start play allows complete rotations. Runs stop after 60 simulated seconds. The maximum completed streak is stored locally under `pendulum-arena-toy-v1`, separately for each mode. If storage is unavailable, play still works. Demo runs use a fixed linear feedback controller and never update a personal best. No score or state is sent to a server.

The physics is a planar single uniform rod on a cart, with finite force, simple drag, and RK4 integration at at most 1/240-second substeps. It is an educational browser toy, **not** MuJoCo, the canonical multibody benchmark, a submission evaluator, or evidence of a ranked result. The persistent UI label says so. Pole count, geometry, force, success threshold, reset, and scoring all differ from the ranked task. Keep this separation when changing the page.

The pure exports `createToyState`, `advanceToy`, `inspectToy`, `demonstrationInput`, and `TOY_PARAMETERS` are tested with Node's built-in test runner:

```sh
node --test test/game.test.mjs
```

These tests check physical response signs, a stationary hanging equilibrium, input saturation, a successful demo, uncontrolled failure, finite hanging-start motion, and refusal of nonfinite state or invalid time steps. They do not claim equivalence to the native benchmark.
