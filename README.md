# Pendulum Arena

One cart. Many pendulums. An open benchmark registry with evidence you can inspect,
publication timestamps, and a browser challenge you can play.

Live at **[arena.neuralyogi.com](https://arena.neuralyogi.com)**.

Built by Rage Industries / [NeuralYogi](https://neuralyogi.com).

## What belongs here

- **Registered campaigns:** exact protocol fingerprints, frozen controllers,
  model hashes, complete cohorts, runtime disclosures and linked proof.
- **Public history:** attributed milestones with primary sources, including
  Kache's six-link demonstration and Corben Sorenson's fourteen-link release.
- **A browser playground:** a small single-pole control toy, explicitly separate
  from native MuJoCo scoring. Toy scores never enter the registry.

This is a community evidence registry, not a priority authority. An author-attested
result remains author-attested until a documented review or independent rerun is
published. Imported milestones do not imply the author's submission or endorsement.

## Run locally

Requires Node 22+ and Python 3 for the static development server. The website has
no npm dependencies and no build step.

```sh
npm test
npm run check
npm run dev
```

Open `http://127.0.0.1:4315`. Application source and public assets live in `dist/`.

## Publish a benchmark

[Open a result submission](https://github.com/marctheshark3/pendulum-arena/issues/new?template=benchmark.yml)
or submit a data PR following [CONTRIBUTING.md](CONTRIBUTING.md). GitHub provides
the server-side issue/PR timestamp. The registry separately records controller
freeze, author-reported execution, first supported public evidence and registry
insertion. Unknown dates stay unknown.

Issues do not automatically create entries. CI validates structure, hashes,
cohort accounting and append-only history; maintainers assess the linked evidence.
The validator does not execute a controller or certify the claim.

Existing results are retained. Retractions and supersessions are appended with
reasons and sources. Source/credit corrections have a
[dedicated form](https://github.com/marctheshark3/pendulum-arena/issues/new?template=correction.yml).

## Eighteen links

The eighteen-link campaign passes **141/141 frozen held-out episodes**, every
one completing 1,500 commands and an **8.04 s strict upright hold**. Exact source
calibration and fresh disjoint development20+100 pass under one unchanged
classical policy. All 141 causal saved-policy proofs reproduce raw/float32
commands, selected native pairs, complete launch forecasts and final states
byte for byte. Cold and continuous native replay cover all **262 original
calibration/development/reserved episodes: 393,000 unique commands**, with zero
defect. These are same-owner checks; the entry remains author-attested.

The [eighteen source view](benchmark/canonical18/README.md),
[original synthesis and retained failures](synthesis/eighteen/README.md), and
[complete evidence release](https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical18-20261017)
retain the own38-state model, route/orbit/full700-step value tables and gains.
Original execution paths remain authoritative; public byte auditors make no
portable numerical identity or independent-reproduction claim. Seventeen’s
negative JEPA ranking study remains negative; no eighteen learned policy was
adopted. Fifty hertz denotes simulated cadence, with no hardware deadline claim.

The current hero is the actual held-out218500 recording in the standard
1280×1080 telemetry layout. [Scientific](dist/assets/canonical18.mp4) and
[silent telemetry](dist/assets/canonical18-telemetry.mp4) copies preserve all
1,500 post-command states at50fps/30s. Rendering performs no dynamics,
inference, reset or interpolation. The new `render_telemetry_video_v2.py`
retains the old renderer and accepts the native multiply/legacy divide duration
representations of an unchanged strict tick count; all other checks remain.

Marc’s original soundtrack wording, “super star by queen”, was clarified to
**Queen — “Don’t Stop Me Now”**, with YouTube accepted. His follow-up requests
one benchmark MP4 with the music mixed in. The [single-video view](https://arena.neuralyogi.com/queen-eighteen)
and [download](dist/assets/canonical18-telemetry-dont-stop-me-now-timed.mp4) use that
same combined file. The song window17.12..47.12s aligns the measured musical
energy entrance at video17.002s with the first active swing-up frame at17.000s
(cue ambiguity approximately±0.2s). The next strong accent falls at19.392s during
the larger swing. Audio pitch/tempo and every video packet/frame/time remain
unchanged; this is source-window alignment. The [official recording](https://www.youtube.com/watch?v=HgzGwKwLmgM)
is credited in the [soundtrack provenance](dist/assets/canonical18-telemetry-dont-stop-me-now-timed.provenance.json)
and [full credits](dist/assets/canonical18-timed-soundtrack-CREDITS.md). All1500video
packets, decoded frames and timings match the original silent telemetry;
[independent review](dist/assets/canonical18-timed-soundtrack-review.json) also verifies
finite, non-silent, unclipped stereo audio. The music remains outside the
website’s MIT license; no formal license grant is asserted. The
[original paired view](dist/history/canonical18-paired-20261017.html) and
[its original provenance](dist/assets/canonical18-queen-paired-view.provenance.json)
remain historical. Scientific/silent recordings and all historical source,
protocols, registry records, correction history and media remain available.

## Seventeen links

The [hybrid-construction response](docs/hybrid-construction-response.md) and
[seventeen synthesis supplement](synthesis/seventeen/README.md) share the original
orbit/bridge workers, exact settings, manual selections and retained search
comparisons, including failures.

Public evidence was first released on **16 October 2026 at 23:25:50 UTC**.
The registry preserves its original prepublication entry and appends a corrected
publication record with the [verified release metadata](dist/evidence/canonical17/publication-status.json).
Active records appear before superseded history. The qualification bytes are unchanged.

The seventeen-link campaign passes **141/141 frozen held-out episodes**, each
completing 1,500 commands and a **7.36 s upright hold**. Exact source calibration
and fresh disjoint development20+100 pass under the same policy. Separate saved
checks reproduce all 211,500 held-out raw commands, delivered float32 actions,
and complete selected forecasts; native cold and continuous replay have zero
defect. These are same-owner checks and the registry remains author-attested.

The [seventeen-link evidence release](https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical17-20261016)
retains all 262 actual calibration/development/reserved recordings. Native replay
coverage is source calibration plus 141 reserved episodes: **142 episodes /
213,000 unique commands**, checked cold and continuously. The 120 development
episodes are scored but are not added to native replay coverage. Original-path
runtime identities remain bound to that execution; the [public source view](benchmark/canonical17/README.md)
is for inspection, without a portable qualification claim.

The adopted controller is classical observed-state feedback with bounded native
one-step value selection. The bounded action-conditioned JEPA warm-start ranking
comparison failed its adoption gates and is retained as a negative result.
This campaign does not establish learned-control progress or hardware realtime.

The featured [telemetry replay](dist/assets/canonical17-telemetry-rising-tide.mp4)
shows every actual post-command state from held-out seed 217500, once, at 50 fps
for 30 seconds, using the [standard 1280×1080 layout](docs/video-standard.md).
The [silent presentation](dist/assets/canonical17-telemetry.mp4),
[scientific recording](dist/assets/canonical17.mp4),
[render provenance](dist/assets/canonical17-telemetry.provenance.json) and
[separate soundtrack provenance](dist/assets/canonical17-telemetry-rising-tide.provenance.json)
remain available. No dynamics or inference runs during rendering.

“Rising Tide” Kevin MacLeod (incompetech.com). Licensed under Creative Commons:
[By Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/).
[Official source](https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1900010).
The edit uses the first 30 seconds, with brief fades; video frames are unchanged.
See the [independent saved-media review](dist/evidence/canonical17/media-review.json),
[complete bounded-owner costs](dist/evidence/canonical17/owner-costs.json) and
[separately measured preparation costs](dist/evidence/canonical17/measured-preparation-costs.json).
Music is separate from the website’s MIT license. All historical registry rows,
protocols and media remain intact.

## Sixteen links

The sixteen-link Rage Mill campaign reports **141/141 frozen held-out passes**,
each with a **7.34 s upright hold**, after disjoint noisy development sets of
20 and 100. The exact native source calibration has zero dynamics defect and
initial-state gap. Separate saved-policy checks reproduce all 211,500 held-out
commands and the selected forecasts byte for byte. These are same-owner checks,
so the registry entry is author-attested.

The [sixteen-link evidence release](https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical16-20261016)
includes all 262 actual calibration, development and held-out recordings, their
selected predictions and the frozen benchmark-specific source. See
[the public inspection adapters](benchmark/canonical16/README.md) and
[the exact sixteen-link protocol](dist/data/protocol-canonical16.json).
Failed forecast summaries are retained; full unselected failed forecast state
tapes were not saved. This differs from fifteen’s forecast retention and is
disclosed in the result.

The sixteen-link replay uses our [standard video format](docs/video-standard.md),
matching fifteen: a 1280×1080 fixed scene, cart position and applied force,
wrapped absolute angle, upright streak, and Rage Industries / NeuralYogi footer.
The [new sixteen-link telemetry video](dist/assets/canonical16-telemetry-im-still-standing.mp4)
shows every recorded post-command state from seed 216500, once each, with
Elton John’s **“I’m Still Standing”** soundtrack. The chorus cue coincides with
the first frame of the final continuous upright hold.

The [silent telemetry render](dist/assets/canonical16-telemetry.mp4) and
[original qualification video](dist/assets/canonical16.mp4) are both available.
The new presentation uses the same recorded trajectory; its different layout
has its own [render provenance](dist/assets/canonical16-telemetry.provenance.json).
The [music provenance](dist/assets/canonical16-telemetry-im-still-standing.provenance.json)
binds the unchanged telemetry frames and reused soundtrack. Historical release
assets and registry records remain unchanged. Third-party music is outside the
website’s MIT license.

## First registered campaign

The fifteen-link Rage Mill campaign reports exact calibration, disjoint noisy
DEV20 and DEV100, then 141/141 frozen reserved episodes. A separate same-owner
native audit checked all 262 actual episodes / 393,000 commands, with zero
native defect and initial-state gap in the recorded runtime. This is an
**author-attested simulation result**. It does not establish global priority,
wall-clock realtime control, robustness to disturbances, or hardware transfer.

The [public evidence release](https://github.com/marctheshark3/pendulum-arena/releases/tag/canonical15-20261016)
contains the raw live trajectories, all retained planning forecasts, frozen
payload and byte-preserving benchmark source export. See [benchmark/README.md](benchmark/README.md)
for inspection and native replay. The original private repository's unrelated
history is not included in the export.

## Reproduce the controller design

The [fifteen/sixteen synthesis export](synthesis/README.md) includes original
reference-generation source, orbit and parking construction, selected intermediate
checkpoints, feedback/value regeneration and a fresh exact-hanging diagnostic.
Its [detailed answers](synthesis/ANSWERS.md) explain the five-hertz orbit,
700-to-650 feedback prefix, actual arithmetic and retained before/after evidence.
The [stage recipes](synthesis/RECIPES.md) distinguish regenerated designs,
optimizer feasibility and complete native episode success. This is an additive
design export; historical qualification source and evidence remain preserved.

## Credits and protocol

The task is derived from [Corben Sorenson's published cartpole work](https://github.com/corbensorenson/cartpole),
pinned in each protocol. [Attribution and primary publication clocks](docs/attribution.md)
explain the historical record. [Corben's fourteen-link import](docs/corben14-import.md)
records exactly what was inspected and what was not.

Read the [registry contract](docs/registry.md), [JSON schema](schemas/registry.schema.json)
and the native [fifteen-link](dist/data/protocol-canonical15.json) and
[sixteen-link](dist/data/protocol-canonical16.json) and
[seventeen-link](dist/data/protocol-canonical17.json) protocols.

## Hosting

The static site is published through Sites. `.openai/hosting.json` retains its
project identity and `dist/` output. Public GitHub source includes the deployed
static assets and separate synthesis tooling. No user credentials or submission
secrets are shipped to the browser. GitHub issues and PRs are the public review queue; there is no pretend
upload service or background score submission.

Website code is MIT. External source, media and evidence retain their own licenses.
