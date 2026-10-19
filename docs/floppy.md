# Floppy Brick Breaker: First Light alpha

Floppy game concept and human-versus-bot challenge idea by Corben;
developed together by Corben and Marc, with their agents.

The [browser game](https://arena.neuralyogi.com/floppy/) is one human-play board,
with a separately labelled retained autonomous recording. It is an unranked
alpha, with no leaderboard, bot upload or shared score submission.

## Human-play contract

`floppy-first-light-human-mujoco-3.8.0-v1` uses the selected original one-link
First Light model and contact rules revision 2, in MuJoCo 3.8.0 WebAssembly.
The passive hinge and massive paddle are driven only through cart force. The
ball's gravity is compensated; the support retains normal gravity.

First Light has 15 one-health bricks. Each removed brick awards 100 points,
regardless of row depth or color. Damage is registered once per contact episode;
a destroyed brick stays physically active until its contact separates. The
board clears after all final contacts separate. A life fault during unresolved
final damage does not award a win.

The alpha starts with three lives, a fixed seed `20261019` and a two-second
warmup before each serve. It ends after 120 **simulated** seconds without
awarding a timeout victory. This bounded one-board human mode is separate from
the ordinary autonomous campaign, which has no timeout win and carries lives
across three boards with one bonus life per clear.

Two input divisions are available:

- **Assisted director:** the person steers a cart target within ±1.65 m. A fixed
  posture controller reads the mechanism state and supplies cart force; it never
  reads or aims at the ball. The hinge remains passive.
- **Direct cart force:** holding left or right requests −20 N or +20 N. There
  is no balancing assistance. The motor's delivery contract clamps finite force
  to ±80 N and converts it to Float32, held for each 20 ms command.

Arrow keys or A/D and the touch buttons steer. Space pauses. Reset restores the
same board, all bricks, three lives and the same serve schedule. Leaving the
window, switching modes or a slow frame pauses play instead of silently running
the game behind the interface. The displayed simulation clock can differ from
elapsed wall time. Human play does not change the Corben-versus-Marc autonomous
competition: participants submit controllers and press Go, without live steering.

## Selected recording

`dist/floppy/replay.json` is a deliberate geometry/statistics export from an
earlier retained native MuJoCo 3.3.5 recording. The autonomous controller is
**Corben / Codex's N1 campaign planner v13**, SHA256
`7c965d2756a0534b31ce574dcaaf3ed201d59e62fefe361d4a2bb95c4dedd250`.

This complete First Light segment uses seed `20261022` and clears 15/15 bricks
in 45.0281875 simulated seconds: 1,500 points, 12 legal paddle hits and four
lives after the campaign's clear bonus. The later campaign was interrupted on
Prism. The replay therefore shows a **board clear**, not a full campaign win.

The display keeps every fifth original 50 Hz frame, plus all event-bearing
frames and the terminal: 482 of 2,252 original display frames. Any smooth screen
interpolation is a presentation effect. The earlier full cold comparison was
exact across 959,811 native states, including 720,451 states on this board;
exporting the recording adds no new experiment or reproduction sample.

| Identity | SHA256 / revision |
| --- | --- |
| Retained recording source freeze | `7b9c11451e6bbb343309084236b28cff18d75ae6` |
| Original compressed First Light frames | `bd97c1f0902e7a9a646582d213e2384bf24677f69a77aceb383a087535533be5` |
| Retained First Light receipt | `d188ca39c4e947e0e8c9921c159c77e5e865eb0739a29d6f36dee68e6c36108b` |
| Selected public replay JSON | `314b3359cd42331d950a8a62206b029b1b4ddf7e067b5f740a9fbd32d5657e96` |

## Physics source and validation scope

The selected model, board and rule source was reviewed at revision
`74bde0779fc9bc6b569ffd10e07b0db18de6ad85`. The public XML preserves the loaded
one-link masses, RK4 integrator, 62.5 µs timestep, contact parameters and passive
paddle. The three serve packets are a selected fixed NumPy PCG64 schedule;
there is no replacement browser random-number generator.

The browser vendors the official [`@mujoco/mujoco` 3.8.0 WASM](https://www.npmjs.com/package/@mujoco/mujoco/v/3.8.0)
with a documented static Embind patch to its JavaScript loader, and verifies
the reported runtime version. The two patched functions use the upstream
Emscripten 4.0.10 `DYNAMIC_EXECUTION=0` fallbacks, retaining strict JavaScript CSP
without an `unsafe-eval` exception; the WASM is unchanged. The original loader
hash and reproducible patch are retained in
[`CSP-PATCH.md`](../dist/floppy/engine/vendor/CSP-PATCH.md). MuJoCo is developed by
[Google DeepMind and its contributors](https://github.com/google-deepmind/mujoco).
Its Apache 2.0 license is retained at
[`dist/floppy/engine/vendor/LICENSE`](../dist/floppy/engine/vendor/LICENSE).

| Selected public asset | SHA256 |
| --- | --- |
| Model XML | `e6f17e245384c1edfe23952c52f6c677595ebae79828117914fdbf8c2da0a0b5` |
| Profile and First Light board | `486d6315dfcfaf720935659271fe1a36085dac01447aab4680b1dada4822043a` |
| Fixed serve schedule | `c0b3eed11e47044a70adfc755a4a432675001f2827c78c5c8a905ee3cc6fd3af` |
| MuJoCo loader with documented static Embind patch | `66777739ae30c38c7ead0221a79e05cbcbad31da3eabeea8fb3f4789c80f22cc` |
| Official MuJoCo WebAssembly | `a91233f2b4d77bdac310e2050a07e3b882da7fade32d94c80c960a9ec5f5ea53` |

Ten fixed short integration fixtures compare the port with native MuJoCo 3.3.5:
equilibrium, both force pulses, brick damage/separation, centered and offset
paddle contacts, illegal side contact, stem contact, a miss and final-brick clear.
All compared game-state/event identities pass. Maximum position and velocity
differences are `9.54e-18` and `6.66e-15` respectively. This is a finite local
port check. It does not establish full-campaign equivalence between runtimes,
unseen-serve reliability, phone performance or a ranked research result.

The [alpha validation record](../dist/floppy/validation.json) retains the
116-test result, fixture identity, enforced browser policy and one desktop timing
screen (500 commands; 2.9 ms p95 within a 20 ms command period). Keyboard input,
release, pause/reset, pagehide and the recording terminal were checked in the
browser. A 390×844 viewport has no horizontal overflow and 60 px steering
buttons. Physical phone and emulator performance remain unverified.

## Public export boundary

This release intentionally publishes selected joint model/rules, authored
browser implementation and whitelisted replay geometry/statistics with visible
origin and controller attribution. It does not publish the research workspace,
raw archives, correspondence, contacts, credentials or native installations.
Publication of this selected joint release does not assert a license for the
rest of the research workspace. The website code follows this repository's
license; separately bundled MuJoCo retains Apache 2.0 and its upstream notices.

The static game requires no account or native service. Play happens in the
browser; the game does not submit input, score or recording data to a server.
