# Response on the hybrid construction

Marc / Codex, reviewing the retained construction and evidence in October 2026.
This response covers attribution, failure mechanisms, next experiments and
seventeen's reconstruction. The accompanying [seventeen export](../synthesis/seventeen/README.md)
adds original synthesis workers, exact historical settings, selected inputs and
same-route negative/positive tapes. No new qualification is claimed here.

## What the selected constructions actually do

The broad proposal is sound: preserve the complete native execution law while
improving the construction of a reachable bridge and upright continuation.
The useful distinction is between a good orbit, a reachable entry and a policy
that completes the full episode. The retained failures separate those questions.

Fifteen starts from **Corben Sorenson's partial-fifteen actual tape**, reflected
and repaired. A successful native-value episode subsequently supplied the
700-command physical reference. It is not the same periodic-orbit construction
as sixteen/seventeen. Sixteen starts from **Corben's successful fourteen donor**
and material-coordinate lift, then connects to a separately constructed 5 Hz
upright orbit. Seventeen lifts the first 450 commands of our retained sixteen
early-bridge reference and rebuilds the mechanics, parking, orbit and bridge at
seventeen. [The earlier answers](../synthesis/ANSWERS.md) and
[donor inventory](../synthesis/upstream-inputs.json) retain the upstream credit.

Already implemented ingredients include early quintic blending with the
corresponding velocity derivative, delivered-F32 transition values with a
disclosed straight-through optimizer input Jacobian, native-defect affine QR,
physical-reference rebuilding at fifteen, and complete causal native launch
forecasts. Seventeen uses full affine native-value search from route tick zero;
sixteen's selected controller uses simple affine feedback with its offset
selector. Joint optimization of bridge, orbit phase and tail, followed by
feedback rebuilding, was **not** part of the selected seventeen construction.

## The 0.14 parameter and manual selections

The amplitude worker computes

```text
z = (exp(2*pi*i*6.25*0.02) I - A)^(-1) B
gamma = 0.14 / max(abs(cumsum(z[1:18])))
```

Here A/B describe seventeen's upright held-input map and normalized force.
Thus 0.14 is the desired maximum cumulative-angle amplitude of the **linear
initializer**. It is not every joint's amplitude or the refined orbit's measured
maximum. Gamma is approximately 0.878316547 normalized force, or 70.2653 N.
The refined saved orbit reaches approximately **0.1231573425 rad**, with maximum
cart magnitude **0.0384471993 m**. The strict 0.15-rad acceptance gate is unchanged.
See the [original amplitude worker](../synthesis/seventeen/original/scripts/probe_canonical17_orbit_amplitude.py).

The amplitude discriminator's original robust-local selection returned **none**.
The subsequent stronger-bridge experiment explicitly selected its profile 6
(6.25 Hz / 0.14) because it had the best observed fragile local hold, 167 ticks.
That is a disclosed manual experiment choice, not a successful automatic
local-basin selection. The [stronger wrapper](../synthesis/seventeen/original/scripts/synthesize_canonical17_strong_orbit.py)
records it without rewriting the failed discriminator.

The bridge starts from the actual exact-hanging state after 850 parking commands
with cart target -0.05 m. It blends from route tick 150 to phase zero at tick
450, solves a 450-command bridge, appends 250 orbit commands, designs all 700
feedback/value steps and executes the first 650. The original 96-state orbit
archive is retained; its actions repeat every eight commands, while its repeated
states differ slightly. It should not be silently reduced to eight states.

The original solve uses IPOPT/MUMPS, limited-memory Hessian, adaptive barrier,
150 iterations / 180 solver CPU seconds, `tol=constr_viol_tol=1e-9`, no NLP
scaling, no bound relaxation and no acceptable-iteration shortcut. Warm-control
and warm-state weights are 1e-4 / 1e-7. Normalized actions are bounded by 0.95,
cart by 2.8 m and other state coordinates by 80. Endpoints are assigned to the
declared initial/orbit states before all assembled native rows are checked.
The saved solve reaches the iteration limit; reference native gap remains
**3.1082206586e-7**. Actual controller success does not erase that approximation.
The [worker](../synthesis/seventeen/original/scripts/synthesize_canonical17.py)
and [historical invocations](../synthesis/seventeen/historical-launches.json)
expose the details, including original donor and launch hashes.

## The requested same-route search comparison exists

All six arms use the same saved actual postparking state and retained stronger
reference. They are suffix experiments, not six independent reset cohorts:

- Search disabled, zero offset: rail exit after **431 commands**, **88** upright ticks.
- Search disabled, +1e-6 m offset: rail exit after **578 commands**, **188** upright ticks.
- Search from tick 0, zero / +1e-6 m: both complete **650 commands**, **368** upright ticks;
  **8,005 / 8,036** distinct native candidate evaluations.
- Search from tick 250, zero / +1e-6 m: both complete **650 commands**, **368** upright ticks;
  **5,502 / 5,496** candidate evaluations.

The first passing registered arm, tick zero / zero offset, became the adopted
policy. These measurements support a native-search contribution on this route.
They do not establish a robustness advantage for starting at zero over 250.
[The original receipt](../synthesis/seventeen/retained/comparison/receipt.json),
all six tapes and the four full candidate ledgers are included. The original
worker verifies ordinary-control bytes and the preactivation prefixes.

## Failure modes and the next experiment

Correction headroom needs measured demand. The 70.27 N reference peak leaves
about 9.73 N to the physical limit at those commands, before affine correction
or tracking error. A nominal force reserve is not a control-authority proof.
The failed ordinary +1e-6 continuation shows large opposing feedback terms;
proximal/middle rate error grows before distal motion, with first saturation
immediately after the 188-tick hold. A distal-only correction was poorly motivated.

`A-BK` describes the unsaturated affine backbone. Seventeen also changes its
F32 proposals, brackets and selected candidate with the state. Entry sensitivity
must therefore be measured by perturbing the observed state and continuing the
**complete search policy**, including candidate switches and the full remaining
task. A lower one-step value is only a selection criterion. The launch forecast
must run that same policy and verify its final state, with independent mutable
clocks and predictor state. Full roots **and affine residuals**, delivered-byte
deduplication, first-minimum ties and original arithmetic matter.

The useful negative comparisons are already retained: 440-tick linear/Hermite
retiming improved reference residuals but earned no actual upright ticks;
R=100/1e4/1e6 screens, 64 finer cart offsets, native orbit polishing and a
bounded proximal-rate residual did not produce a passing continuation. Polishing
reduced cyclic gap roughly tenfold without fixing entry. Eight local phase tests
found fragile first-250 holds at phases 1/2 for 1e-15 perturbations, while every
phase failed the 1e-11 test. That component did not test repaired hanging-start
entries into each phase. These failures limit the tested families, not all future
phase or refinement methods.

I would first reconstruct the selected baseline, then test a small declared
phase/early-blend bracket with complete native policy admission. A concrete first
bracket is phases 0/1/2 crossed with blend starts 125/150, fixed bridge end 450,
including the original phase-zero/start-150 control. This is a proposed experiment,
not an executed search. Rebuild feedback for every proposal and retain the
passing incumbent. If phase/timing alone does not improve the declared outcome,
allow a bounded late-bridge/early-tail block to move jointly and repeat native
admission after each update. Give each family a fixed repair/candidate budget.
Gain norm and reference residual remain diagnostics; full task outcome comes first.

The public export makes the retained successful recipe inspectable. Its
[validation record](../synthesis/seventeen/validation.json) distinguishes new
coefficient reconstruction, stage regeneration and saved-launch reruns from
independent end-to-end synthesis and qualification.
