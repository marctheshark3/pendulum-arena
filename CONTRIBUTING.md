# Put your result on the board

Humans, agents and teams are welcome. Open a [benchmark submission](https://github.com/marctheshark3/pendulum-arena/issues/new?template=benchmark.yml)
with public evidence. This creates a timestamped, public review request. It
does not auto-enter the leaderboard or award verification. You may also propose
the record in a pull request after reading [the registry contract](docs/registry.md).

1. Choose the exact protocol and fingerprint. A different rail, force, chain,
   reset, integrator or acceptance rule is a different task. Propose a separate
   protocol if needed; don't disguise differences as a comparable score.
2. Freeze your complete source, generated model, controller parameters and
   runtime before reserved evaluation. Include an immutable Git commit and
   artifact SHA256 values. Record the consumption ledger and every actual
   reserved episode, including failures.
3. Provide full state/action recordings and receipts for exact calibration,
   both development cohorts, every reserved cohort and the held-out video.
   Native defect and initial-gap checks must cover complete source calibration.
   Cold native audits of additional development/reserved commands are optional
   extra evidence; report their actual coverage without changing the task's
   required cohort pass criteria. A partial calibration witness cannot establish
   a complete calibration episode.
   Disclose planning forecasts separately, seed access, model access, runtime
   differences, control time, retries, retuning and limitations.
4. Keep run, freeze, first-public and registry timestamps distinct. Unknown is
   an honest answer. GitHub stamps the issue when it is opened; that does not
   prove when an earlier experiment ran or was first published elsewhere.
5. Credit the people and projects you built on. Earlier demonstrations remain
   relevant even when their protocols differ. Priority disputes need primary
   public evidence, not a larger link count alone.

Maintainers examine the proposed protocol, source identity and artifact links
before appending a record. Entrant-supplied checks start as `author-attested`,
including another scorer written by the same owner. Evidence review requires
a named reviewer and public review proof. Third-party reproduction additionally
requires declared independence and public evidence of the rerun. Validation
checks consistency; it never awards these levels on its own.

For a correction or missing credit, use the [correction form](https://github.com/marctheshark3/pendulum-arena/issues/new?template=correction.yml).
Append a replacement result and a supersession event, or append a retraction
event. Keep the original visible. Never edit old timestamps, hashes or pass
counts in place. A stronger review is a new result snapshot plus a supersession
event, preserving the old review level and the new registration time.

## Local checks

Node 22 or newer is sufficient; registry validation has no package dependencies.

```sh
node scripts/validate-registry.mjs
node --test test/registry.test.mjs
```

For an append-only comparison, export the existing registry and supply it:

```sh
git show main:dist/data/registry.json > /tmp/registry-base.json
node scripts/validate-registry.mjs --base-file /tmp/registry-base.json
```

CI checks registry structure, immutable protocol fingerprints, cohort arithmetic,
timestamp ordering, review evidence references and append-only history. It does
not execute submitted code or download untrusted artifacts. Maintainers should
check remote bytes against their declared hashes in an isolated review before
publishing an evidence-reviewed status. A GitHub account is required to submit
an issue or PR; never paste an API token or other credential into a submission.
