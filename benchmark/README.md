# Recheck the canonical fifteen-link result

For reference generation, gain/value design, parking construction and a runnable
design reconstruction, see the [synthesis export](../synthesis/README.md).

This directory publishes a curated, runnable export of the fifteen-link controller and native saved-trajectory auditor. Eleven files in `frozen/` are **byte-exact copies** from the original frozen commit `886267f09ce67d7215c166c92e14a0adaca8e34f`. Their SHA256 values and the seven-file original `source_identity` are in [`frozen-manifest.json`](frozen-manifest.json). This export contains no private Git history or full repository bundle.

The blank `frozen/src/rage_mill/__init__.py` is a new package adapter. `public_support.py`, `run_episode.py`, and `audit_saved.py` are also new public adapters, authored after qualification. They are outside the original freeze. The adapters verify the frozen source and leave every original receipt unchanged. Original absolute paths remain historical metadata; relocation happens through `--proof-root`, never by rewriting or rehashing a freeze.

The task follows [Corben Sorenson's published cartpole configuration and release protocol](https://github.com/corbensorenson/cartpole/tree/b84c6413974d5c0831cdf261ce15e58ba65cd3f5): fifteen passive links, total length 3 m and mass 1 kg, a 1 kg cart, ±80 N cart force, ±3 m rail, 50 Hz commands, 30 s episodes, and at least 250 consecutive post-command samples with every wrapped cumulative absolute angle strictly below 0.15 rad. The precise copied configuration is [`frozen/configs/canonical15.json`](frozen/configs/canonical15.json).

The historical result contains one exact calibration, disjoint noisy development cohorts of 20 and 100, and frozen held-out cohorts of 20 noisy, 100 noisy, 20 exact, plus one video episode. All 262 actual episodes completed all 1,500 commands and passed. The original cold-transition audit checked 393,000 commands with zero native dynamics defect and zero initial-state gap. It was a separate implementation under the same project owner, **not external third-party verification**. Timestamps inside the original receipts are author-generated historical records; they are not a trusted external timestamp service.

## Downloaded proof layout

Download and extract the separate proof archive from this project's release. Pass the directory containing these paths as `--proof-root`:

```text
frozen-release02/
  freeze.json
  requirements-evaluation.txt
  candidate/{controller.npz,policy.json,model.xml,provenance.json,identity.json}
planned-candidate02-calibration02/{receipt.json,cases/...}
development120-01/{receipt.json,cases/...}
reserved141-01/{receipt.json,cases/...}
independent-final-audit01/receipt.json
heldout-video01/receipt.json
```

Keep every case's `episode.npz`, `receipt.json`, and retained `forecast-*.npz` files. The original private `source.bundle` is deliberately not distributed and is not required by these public adapters. The archive's own download hash should be checked against its release manifest before extraction.

## Check the evidence without installing physics dependencies

Python 3.11 or newer is sufficient for the hash check and CLI help:

```sh
python3 benchmark/audit_saved.py --proof-root /path/to/extracted-proof --hashes-only
```

This verifies the exact original freeze and final-audit hashes, exported source, candidate payload, runtime requirements, all cohort receipts, all 262 live recordings, every retained planning forecast, and the held-out video's receipt/source binding. It enforces the complete original cohort labels. It does **not** run native physics or rerun the controller. The MP4 is a separate download; optionally check its bytes with `--video /path/to/canonical15-published-benchmark.mp4`. Without that option, the report explicitly says the video binary was not checked.

## Recheck saved transitions in native MuJoCo

The qualified runtime was **Linux/aarch64**, Python **3.12.3**, NumPy **2.5.2**, SciPy **1.18.1**, native MuJoCo **3.3.5**, with `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`. The original freeze records the interpreter and shared-library hashes. Package versions alone do not establish an identical runtime. Portability to another architecture, numerical library, or runtime is not claimed.

For example, prepare an isolated environment using the original package inventory:

```sh
uv venv --python 3.12.3 .venv-canonical15
uv pip install --python .venv-canonical15/bin/python -r /path/to/extracted-proof/frozen-release02/requirements-evaluation.txt
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-canonical15/bin/python benchmark/audit_saved.py \
  --proof-root /path/to/extracted-proof --workers 4 --out saved-audit-local.json
```

The default refuses any mismatch with the full frozen runtime identity. On a different runtime, `--allow-runtime-mismatch` permits an explicitly exploratory check, while still requiring MuJoCo 3.3.5 and one BLAS/OMP thread. It records every differing identity field and does not make a portability or identical-runtime claim.

After verifying all file hashes, the adapter calls the unmodified original `check_case` function for every episode. It checks the reset and RNG order, float32 action delivery, strict angle scoring, native one-command dynamics, retained forecast order and hashes, and byte equality of selected forecasts with the live suffix. The native audit never invokes the policy and never creates replacement evaluation episodes. It writes a new optional report and cannot overwrite the historical qualification.

## Run one fresh diagnostic

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv-canonical15/bin/python benchmark/run_episode.py \
  --proof-root /path/to/extracted-proof --seed 3156500 --out diagnostic-3156500
```

This constructs a new controller from the frozen candidate and runs one native episode. The reset seed is used only to generate the initial plant state; the controller receives observed state and tick, with no seed or cohort label. Use `--exact` for an exact hanging start, or `--initial-state observed-state.json` for a diagnostic beginning with a JSON list of 32 finite qpos/qvel values. The output directory must be new. It contains the live recording, all attempted planning forecasts, and a clearly marked diagnostic receipt. These runs do **not** add qualification evidence or alter frozen reserved outcomes.

The method settles the live hanging system for 800 commands, then performs bounded deterministic route planning in separate native models from the observed launch state. It tests a frozen ordered list of at most 32 search profiles, retaining failed forecasts, and selects the first complete valid route. Every following live state must match its selected predicted state byte for byte. This is a simulation-control result; it does not establish real-time 50 Hz control, robustness to later disturbances, learned representation control, hardware transfer, or global priority.

## Packaging and license

From the public repository, `python3 scripts/package-canonical15.py` creates a deterministic `canonical15-source.tar.gz` under `.artifacts/release/`. Archive paths, metadata, ordering, and gzip timestamp are fixed; Python bytecode caches are excluded. This archive contains the complete `benchmark/` directory, not the separate proof payload.

The frozen code is distributed under its original [MIT license](frozen/LICENSE), included byte for byte. The copied `pyproject.toml` and `uv.lock` preserve original source identity; they are not a new installable distribution or a command to install the private original project. Use the public adapters and explicit runtime environment above.
