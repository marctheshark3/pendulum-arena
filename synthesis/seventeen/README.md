# Seventeen: construction, retained comparisons and reconstruction

This supplements the [frozen seventeen controller](../../benchmark/canonical17/README.md)
and [15/16 synthesis export](../README.md). The [construction response](../../docs/hybrid-construction-response.md)
explains the attribution, amplitude, manual choices and failed comparisons.

`original/` contains the original orbit, bridge, parking and diagnostic workers
with their own module dependencies. Their bytes match the source hashes in the
retained owners that used them. `retained/` contains scientific checkpoints,
same-route search comparisons and complete actual failed hanging-start tapes.
[original-manifest.json](original-manifest.json) binds every copied file to its
original path, size and SHA256. [historical-launches.json](historical-launches.json)
projects the exact historical argv, environment, resource ceilings and measured
owner costs; it identifies the original launch/completion hashes. Historical
absolute paths are provenance, not runnable local paths.

`run.py` is a new public adapter. It never executes an old launch declaration,
activates a reserved cohort, edits a retained receipt or silently substitutes
a saved success for a new solve. It records each new invocation and keeps the
worker's complete output. These stages are CPU diagnostics and synthesis,
separate from qualification.

## Setup and short reconstruction

Use the environment in [the synthesis README](../README.md): Python 3.12.3,
NumPy 2.5.2, SciPy 1.18.1, MuJoCo 3.3.5 and CasADi 3.8.1 on the recorded
Linux/aarch64 runtime. Set both thread limits before starting Python. Run from
the repository root; every output directory must be new.

```sh
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python3 synthesis/seventeen/run.py verify
.venv-synthesis/bin/python synthesis/seventeen/run.py feedback \
  --out .artifacts/seventeen/feedback
```

`verify` uses only the Python standard library and executes no physics. It also
checks the existing sixteen donor and frozen seventeen payload/value files.
`feedback` recomputes all 700 gains/offsets and all 701 value roots/residuals,
then reports array-byte comparisons against the published full affine value.
It writes the new arrays and a receipt without a native episode or RNG draw.

The default derivatives are a **new numeric checkpoint** evaluated using a
fresh rebuild of the original provider after qualification. They are not a
historical saved derivative table. [Its receipt](derivative-checkpoint.json)
records the new model checks and exact full-feedback comparison. The reference
hash is checked before these derivatives can be reused. The full 700-command
design is retained even though only the first 650 commands are executed.

## Original stages through portable paths

```sh
.venv-synthesis/bin/python synthesis/seventeen/run.py model \
  --out .artifacts/seventeen/model
.venv-synthesis/bin/python synthesis/seventeen/run.py orbit \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/orbit
.venv-synthesis/bin/python synthesis/seventeen/run.py amplitude \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/amplitude
.venv-synthesis/bin/python synthesis/seventeen/run.py launch \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/launch
.venv-synthesis/bin/python synthesis/seventeen/run.py bridge \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/bridge
.venv-synthesis/bin/python synthesis/seventeen/run.py feedback \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/rebuilt-feedback
.venv-synthesis/bin/python synthesis/seventeen/run.py comparison \
  --model .artifacts/seventeen/model/product --out .artifacts/seventeen/comparison
```

Stages intentionally default to retained inputs, allowing one stage to be
studied when a fresh earlier solve differs. `amplitude` uses the retained 5 Hz
baseline and perturbation direction; add `--baseline .artifacts/seventeen/orbit/product`
to use the fresh orbit output. `bridge` uses the retained amplitude archive and
parking tape; add `--orbit .artifacts/seventeen/amplitude/product` and
`--launch .artifacts/seventeen/launch/product` to use those fresh outputs.
Its explicit selection is amplitude profile 6, as in the original worker.
The adapter creates a separate orbit loading view with verified local paths.

The donor is the existing sixteen `synthesis/data/16/early-reference.npz`, SHA256
`b74bf2ec82fc8739de7e0443877f07d3d8d89d0a15f7fdf87d771c68308daff5`.
Its first 450 actions and 451 states initialize the bridge after material lift.
The upstream fourteen-to-sixteen lineage is documented in [ANSWERS.md](../ANSWERS.md).

`feedback --model` rebuilds derivatives on the selected retained reference.
Use `--reference PATH` to study another complete 700-command reference with
that provider. `comparison` uses the retained route, controller and actual
launch from failed episode 3171033 by default. A fresh reference/controller can
be supplied with `--reference` and `--controller`; the original worker checks
that the derivative provider reproduces their gain/offset prefix exactly.
The old ordinary-control tape assertions remain literal controls; changing the
route is a different experiment and can legitimately fail those assertions.

Each worker has a 300-second per-process CPU limit, 480-second wall timeout
and 8 GiB address-space limit. Its original inner solver/forecast limits also
remain active. `invocation.json`, `worker.log` and `adapter.json` retain the
actual command, exit status, child CPU and wall cost, including failures.
Compiler child CPU is included in measured aggregate child cost; the per-process
ceiling is not a cumulative campaign CPU ceiling.

## What is retained

- `model/`: upright held-map A/B, Q and material-lift initializer.
- `orbit/` and `amplitude/`: original frequency/amplitude arrays and local
  perturbation tapes, including failed local capture tests.
- `parking/`: own-count parking design and exact/two noisy actual prefixes.
- `bridge/`: initializer, raw solver result, assembled 700-command reference,
  native defects and original 650-command controller prefix.
- `comparison/`: all six same-launch suffixes and all four native candidate
  ledgers. No-search controls fail; search from tick 0 or 250 passes this
  saved-launch test at both registered cart offsets.
- `baseline-failure/`, `retime-linear-failure/`, `retime-hermite-failure/` and
  `stronger-failure/`: complete actual failed tapes 3171030–3171033.
- `phase/`: all eight orbit phases, their two local perturbation tapes and
  original tables. The 96-state archive is preserved.
- `costs/`, `polish/` and `rate-residual/`: retained negative-result receipts.
  Those three ancillary forecast collections are summarized, rather than
  claimed to be complete public tape exports.

The existing frozen candidate already contains the full value arrays, the
first passing actual diagnostic and its ledger. These are referenced by hash
instead of duplicated. Code and our authored artifacts use the repository's
MIT license; Corben's task and donor lineage retain their attribution.

## Evidence boundaries

Read [validation.json](validation.json) for the new export checks. Array
reconstruction, a new bounded optimization, saved-state continuation and
independent end-to-end synthesis are distinct claims. A new solve can have
different residuals or coefficients. Its own output and failure must remain
visible; the retained reference's native gap is approximately `3.1082e-7`,
above the separate `1e-8` reference-feasibility target.

This export does not add qualification episodes, independently reproduce a
reserved campaign, establish a general capture basin or prove portability.
The published seventeen release remains author-attested. Simulated 50 Hz
does not establish a 20 ms wall-clock deadline.
