# Recheck the canonical sixteen-link result

This is a curated public export of the qualified sixteen-link controller. Twenty-one Python files and the runtime manifest are byte-exact copies of the original controller identity. Original package/configuration/license documents accompany them. `frozen-manifest.json` identifies every copied byte and the original isolated snapshot commit `d06e1478834519f797d28fc57a1b8bad09c61d67`. Private Git history, the full 515-file source bundle, unrelated robot/legacy code, native runtime binaries and model JIT libraries are excluded.

`public_support.py`, `audit_saved.py` and the empty package initializer are NEW public adapters. They were authored after qualification and were outside the historical freeze. The original identity (`36076035…`) and freeze (`28d97cd6…`) remain unchanged. Historical receipts contain original absolute local paths as provenance, not download locations. Relocation resolves SHA256/size-bound rows through the downloaded manifest and creates a separate in-memory controller loading view; it never rewrites or rehashes a historical identity as the qualified result.

The task follows [Corben Sorenson's published release](https://github.com/corbensorenson/cartpole/tree/b84c6413974d5c0831cdf261ce15e58ba65cd3f5): sixteen passive uniform links totaling 3 m / 1 kg, a 1 kg cart, ±80 N force, ±3 m rail, four 5 ms MuJoCo RK4 steps per float32 command, 1,500 commands / 30 seconds. A pass requires any 250 consecutive post-command samples with every wrapped cumulative absolute angle strictly below 0.15 rad, plus the complete finite in-rail episode. No velocity or cart-target condition is added to the upright gate.

## Evidence layout

Extract `canonical16-evidence.tar.gz` and pass its `canonical16-evidence/` directory as `--proof-root`. It contains:

```text
metadata/{identity,freeze,qualification,verified-release,release-manifest,authority}.json
payload/{controller.npz,configuration.json,runtime-closure.json,model.xml}
calibration/{episodes/...,selected-prediction-3167109.npz,native-proof/...}
development20/{episodes/...,selected-prediction-*.npz,receipt.json}
development100/{episodes/...,selected-prediction-*.npz,receipt.json}
reserved/{noisy20,noisy100,exact20,video}/...
policy-proof/{receipt.json,<gate>/episode-<seed>/...}
claims/...  owners/...  reviews/...  render/...
negative-predecessor/...  files.json
```

All **262 actual episodes** are retained: one exact source calibration, 20 + 100 fresh disjoint noisy development episodes and 141 once-only held-out episodes. Each includes its raw live tape, all 1,500 command-journal entries and original receipt. Selected forecast arrays are retained for every episode; the held-out causal proof also contains each selected 651-state / 650-command forecast, all recomputed commands and proof journals.

Failed **unselected** shadow forecasts were originally retained as summaries and command counts, not complete state arrays. That retention limit is explicit; no missing arrays are synthesized for this export. Two development100 cases selected `+1e-6` m; all held-out cases selected zero. The older fixed seventeen-second policy remains 98/100, with its two failing actual tapes preserved separately. It is not relabeled as this qualified candidate.

## Portable hash audit — no physics installation needed

Python 3.11+ and the standard library suffice:

```sh
python3 benchmark/canonical16/audit_saved.py \
  --proof-root /path/to/canonical16-evidence --hashes-only
```

This verifies every payload byte against the pinned public manifest, all copied frozen source, original identities, exact complete cohort labels, 262 original episode bindings and cached scores, native source-calibration bindings and all 141 original command/suffix proof bindings. It does **not** rerun native physics, the controller, initialization or a benchmark label. Add `--video /path/to/canonical16.mp4` to hash-check the original scientific MP4. The soundtrack derivative is separate and deliberately fails that scientific-video hash check. Without `--video`, the report says the video binary was not checked. An optional `--out NEW.json` refuses an existing file.

The original exact source calibration checked 1,500 cold and 1,500 continuous native transitions on **one episode**, with zero initial-state gap and native dynamics defect. There is no claim of a separate all-262-episode native transition replay. The held-out causal policy proof independently reproduced 211,500 raw float64 commands and float32 deliveries plus 141 full selected forecast/live suffixes, including state 1500, byte for byte. These checks were separate implementations under the **same project owner**, not external third-party verification. The archive preserves that exact scope.

## Frozen controller loading API

The NEW `public_support.load_controller(proof_root, allow_runtime_mismatch=False)` first performs the full hash audit, then constructs the unchanged controller from its separately verified local loading view. It returns `(policy, disclosure)` and itself issues no controller command or native step. Run it in a fresh interpreter: pre-existing `rage_mill` imports are refused to prevent accidental package substitution. The returned policy receives only the current observed relative qpos/qvel state and consecutive tick; it receives no seed, cohort label or future actual state. Calling it at tick 850 performs causal native forecasts, so the default hash CLI never invokes this API.

The recorded runtime was Linux/aarch64, Python **3.12.3**, NumPy **2.5.2** and MuJoCo **3.3.5**, with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. `load_controller` compares the recorded runtime metadata by default and refuses a metadata mismatch. Its explicit `allow_runtime_mismatch=True` permits an exploratory loading view only, still requiring MuJoCo 3.3.5 and one thread. The loader does not verify installed native/runtime binary hashes; its disclosure reports `runtime_binary_hashes_verified: false`. Matching metadata does not establish an identical runtime. The historical runtime manifest retains exact file hashes as evidence, but its binaries are not redistributed. No portability result is claimed here.

The frozen `pyproject.toml` and `uv.lock` are historical identity documents, not instructions to install the unrelated original private project. For this controller, use an isolated Python 3.12.3 environment with `numpy==2.5.2` and `mujoco==3.3.5`; the portable hash check requires neither dependency. There is no public reserved-label activation command and no native-replay claim from the publication smoke check.

## Method, clocks and license

The live policy parks for 850 commands, then follows 650 unchanged feedback entries toward a 5 Hz bent upright orbit. Its bounded causal launch selector tries eleven registered cart-error offsets and chooses the first full in-rail forecast with the required hold. Full world-frame physics is preserved. All 141 held-out episodes earn 367 upright ticks / 7.34 seconds. Fifty Hz denotes **simulated action cadence**; the measured launch computation burst is about 236 ms median and 320 ms maximum wall time, so real-time 50 Hz hardware control is not established. Global priority, disturbance robustness and hardware transfer are not established.

The original source snapshot records author-generated Git time 2026-10-16T18:41:00Z. The later gate-freeze JSON has no explicit UTC field. Neither clock is an external timestamp authority; GitHub publication is a separate clock.

Copied project code carries its original MIT license in `frozen/LICENSE`. No unlicensed upstream controller source is copied here; its publication/configuration is credited as the benchmark definition. External packages retain their own licenses and are not vendored. The plain project-generated scientific video is separate from the Elton John soundtrack share copy; that third-party music is not covered by the code's MIT license and is excluded from both source and evidence archives.

From the public repository, `python3 scripts/package-canonical16.py --out NEW.tar.gz` creates a deterministic source archive containing only `benchmark/canonical16/`, excluding bytecode caches and Git history.
