# Pendulum Arena

One cart. Many pendulums. An open benchmark registry with evidence you can inspect,
publication timestamps, and a browser challenge you can play.

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

## Credits and protocol

The task is derived from [Corben Sorenson's published cartpole work](https://github.com/corbensorenson/cartpole),
pinned in each protocol. [Attribution and primary publication clocks](docs/attribution.md)
explain the historical record. [Corben's fourteen-link import](docs/corben14-import.md)
records exactly what was inspected and what was not.

Read the [registry contract](docs/registry.md), [JSON schema](schemas/registry.schema.json)
and [native fifteen-link protocol](dist/data/protocol-canonical15.json).

## Hosting

The static site is published through Sites. `.openai/hosting.json` retains its
project identity and `dist/` output. Public GitHub source mirrors the exact
deployed commit. No user credentials or submission secrets are shipped to the
browser. GitHub issues and PRs are the public review queue; there is no pretend
upload service or background score submission.

Website code is MIT. External source, media and evidence retain their own licenses.
