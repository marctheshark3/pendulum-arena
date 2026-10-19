# First Light browser engine

This public alpha is a distinct `floppy-first-light-human-mujoco-3.8.0-v1`
profile. It uses the official single-threaded `@mujoco/mujoco@3.8.0` WASM with
a documented static Embind patch to its JavaScript loader, and checks the
runtime's own reported version before compilation. It is not
a retained native MuJoCo 3.3.5 campaign or a ranked benchmark. Selected original
source, exported assets, dependency hashes and licenses are in
`provenance.json`; the original game concept belongs to Corben.

The loader's two JavaScript code-generation sites use the upstream Emscripten
4.0.10 `DYNAMIC_EXECUTION=0` static closures instead. This keeps the site's strict
JavaScript CSP without an `unsafe-eval` exception. The WASM is unchanged; see
`vendor/CSP-PATCH.md`, the reproducible patch and the Emscripten license.

`createGame()` asynchronously loads same-origin assets and compiles the model.
The result provides:

- `step(force, retain?)`: deliver one finite, clipped Float32 cart force and
  advance at most one 20ms command, using 320 RK4 native substeps. The optional
  callback receives a snapshot after each completed substep for bounded checks.
- `snapshot()`: copy current state, ordinary lives, brick health, pending contact
  removals, score and events. The board and profile are read-only configuration.
- `assistedForce(targetX)`: a human target drives fixed loaded-pendulum posture
  feedback with a limited reference rate. This helper never reads the ball.
- `reset()`: restart the same original fifteen-brick board, three ordinary lives
  and the same three NumPy-generated serves for seed 20261019.
- `dispose()`: release the model and data on the WASM heap.
- `restoreValidationState(state)`: a reproducibility hook for explicitly
  synthetic finite fixtures; the gameplay UI does not call it.

The exported model retains the loaded stem and paddle, passive hinge, uniform
mechanism gravity, ball gravity compensation and native compliant collisions.
Only declared serves assign ball position/velocity. Bricks retain collision
until native separation; damage happens once per contact episode, before fault
resolution. This alpha retains the original 100-point one-HP brick reward.

For Node checks, pass `module`, `configuration`, `serves` and `modelXML` to
`createGame`; the normal browser entry needs none. These injections allow
finite reproducibility without external services or a native installation.
