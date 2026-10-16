# Registry contract

`dist/data/registry.json` is the public append-only registry. `schemaVersion` is
currently `1`. A result describes a submitted claim, not a certificate issued by
the website. A passing JSON validator establishes internal consistency; it does
not run MuJoCo, inspect remote artifacts, or establish priority.

## Shape for the first protocol

This is the exact protocol requirements shape. Additional protocols may choose
other numeric values and cohort sizes. `fingerprint` is the lowercase SHA256 of
`canonicalJson(requirements)`, exported by `scripts/validate-registry.mjs`.
Canonicalization recursively sorts object keys, preserves array order, rejects
nonfinite numbers, and uses ordinary JSON string/number encoding without spaces.

```json
{
  "id": "canonical15-v1",
  "title": "Published uniform-chain task, fifteen links",
  "linksMin": 15,
  "linksMax": 15,
  "source": {
    "url": "https://github.com/corbensorenson/cartpole/tree/b84c6413974d5c0831cdf261ce15e58ba65cd3f5",
    "commit": "b84c6413974d5c0831cdf261ce15e58ba65cd3f5"
  },
  "fingerprint": "<fingerprintContract(requirements)>",
  "requirements": {
    "durationSeconds": 30,
    "commandsPerEpisode": 1500,
    "actionHz": 50,
    "uprightAngleRad": 0.15,
    "uprightConsecutiveSteps": 250,
    "railLimitMeters": 3,
    "forceLimitNewtons": 80,
    "calibrationTolerance": 1e-8,
    "integration": {
      "engine": "MuJoCo",
      "engineVersion": "3.3.5",
      "precision": "binary64",
      "integrator": "RK4",
      "timestepSeconds": 0.005,
      "substeps": 4
    },
    "mechanics": {
      "cartMassKg": 1,
      "totalLinkMassKg": 1,
      "totalChainLengthMeters": 3,
      "capsuleRadiusMeters": 0.025,
      "armature": 0.0005,
      "totalHingeDamping": 0.015,
      "cartDamping": 0.02,
      "frictionLoss": 0
    },
    "reset": {
      "position": "relative [0, pi, 0, ...]",
      "noiseStd": 0.05,
      "cartPositionNoiseStd": 0,
      "rngOrder": "cart position zero-scale draw; joint positions; all velocities; replacement cart velocity"
    },
    "acceptance": {
      "angleConvention": "wrapped cumulative absolute",
      "angleComparison": "strictly-less-than",
      "samples": "post-command only",
      "fullEpisodeRequired": true,
      "stateOverwriteAfterInit": false,
      "liveRetriesAllowed": false,
      "allFiniteRequired": true
    },
    "cohorts": [
      {"id":"calibration","stage":"calibration","noise":"exact","required":1},
      {"id":"dev20","stage":"development","noise":"noisy","required":20},
      {"id":"dev100","stage":"development","noise":"noisy","required":100},
      {"id":"reserved-noisy20","stage":"reserved","noise":"noisy","required":20,"seedStart":215000,"seedEnd":215019},
      {"id":"reserved-noisy100","stage":"reserved","noise":"noisy","required":100,"seedStart":215100,"seedEnd":215199},
      {"id":"reserved-exact20","stage":"reserved","noise":"exact","required":20,"seedStart":215200,"seedEnd":215219},
      {"id":"video","stage":"video","noise":"noisy","required":1,"seedStart":215500,"seedEnd":215500}
    ]
  }
}
```

A result has this structure (replace placeholders with genuine evidence):

```json
{
  "id": "rage-mill-canonical15-20261016",
  "type": "result",
  "protocolId": "canonical15-v1",
  "protocolFingerprint": "<same fingerprint>",
  "links": 15,
  "qualified": true,
  "entrant": {"id":"rage-industries","name":"Rage Industries","url":"https://neuralyogi.com"},
  "method": {
    "name": "Bounded native route planning",
    "summary": "Full-state native-model controller; frozen route search after hanging settle.",
    "wallClockRealtime": false,
    "usesObservedStateOnly": true,
    "disclosures": ["Simulation only", "No post-initialization disturbance claim"]
  },
  "source": {"url":"https://github.com/example/project/tree/<commit>","commit":"<40 lowercase hex>","sha256":"<archive SHA256>"},
  "model": {"url":"https://example.org/model.xml","sha256":"<exact generated XML SHA256>"},
  "runtime": {
    "engine":"MuJoCo","engineVersion":"3.3.5","language":"Python",
    "languageVersion":"3.12.3","os":"Linux","architecture":"aarch64","precision":"binary64",
    "dependencies":{"numpy":"2.5.2","scipy":"1.18.1"}
  },
  "timestamps": {"runAt":null,"frozenAt":"2026-10-16T17:54:52.208417Z","firstPublicAt":null,"registeredAt":"2026-10-16T18:40:00Z"},
  "evidence": [{"id":"audit","kind":"receipt","url":"https://example.org/receipt.json","sha256":"<SHA256>","publishedAt":null}],
  "cohorts": [{"id":"calibration","stage":"calibration","noise":"exact","required":1,"attempted":1,"passed":1,"evidenceId":"audit"}],
  "metrics": {"commandsPerEpisode":1500,"secondsPerEpisode":30,"minUprightSeconds":6.38,"maxCartMeters":2.7244540435262334,"maxNativeDefect":0,"maxInitialGap":0,"auditedEpisodes":262,"auditedCommands":393000},
  "verification": {"level":"author-attested"},
  "notes": ["Separate scorer by the same owner; no third-party reproduction claimed."]
}
```

The abbreviated cohort list above is intentionally incomplete; a qualified
record must contain every protocol cohort. Each cohort repeats the protocol's
stage, noise, required size and reserved seed range, records actual attempted
and passed counts, and binds an evidence ID. A result may set `qualified:false`
for a partial or failed campaign. Such a record is not a qualified leaderboard
entry. `auditedEpisodes` and `auditedCommands` describe the actual native audit
coverage, excluding planning forecasts. An episode with a partial audit is not
silently counted as a full episode of audited commands. `maxNativeDefect` and
`maxInitialGap` report finite maxima over that declared audited scope.

Published qualification requires complete source calibration with finite native
dynamics defect and initial-state gap no greater than the protocol's calibration
tolerance. It also requires the complete development and reserved cohort pass
proofs under the frozen release. It does **not** require a cold native re-audit
of every development and reserved transition. Audit coverage must include the
complete calibration cohort: for this protocol, at least one actual episode and
all 1,500 of its commands. A 450-command witness is partial evidence, not a
1,500-command calibration. The validator checks aggregate coverage bounds;
reviewers must inspect the linked calibration receipt to establish its scope.
Rage Mill additionally reports a cold native audit of all 262 actual episodes /
393,000 commands. That is extra evidence, not a new published qualification gate,
and same-owner checks remain author-attested.

Calibration and development gates may omit fixed seed ranges in the protocol:
entrants then declare their own fresh ranges in their result. Every declared
range must contain the required number of labels, and the result's ranges must
be disjoint. A protocol's fixed reserved ranges must match exactly.

## Evidence and clocks

Only absolute HTTP(S) URLs are accepted. Credentials, local file URLs and
JavaScript URLs are refused. SHA256 values must be full lowercase hex; source
commits are full forty-character lowercase Git SHA1s. The source archive SHA256
is optional when the source is pinned by commit. Exact model XML and all evidence
objects always require SHA256 values. Hashes make changes detectable; the
validator does not automatically fetch or authenticate the bytes behind URLs.

Timestamps must be real UTC ISO-8601 instants (`Z` or `+00:00`). `runAt` is the
last actual episode's completion time when known, `frozenAt` is the controller
freeze, `firstPublicAt` is the earliest demonstrated public publication, and
`registeredAt` is editorial insertion into this registry. Unknown historical
times are `null`. The site must not turn file modification time, an imported
claim, or today's registry date into proof of an earlier public result. These
are declared timestamps, not an external timestamp-authority service.

## Review levels

- `author-attested`: supplied by the entrant, including checks made with another
  implementation by the same owner. This is the default.
- `evidence-reviewed`: a named reviewer examined the linked immutable artifacts.
  Requires reviewer identity, review time, an explicit independence disclosure
  and linked review evidence. It does not assert reproduction.
- `third-party-reproduced`: a declared independent third party reran the exact
  protocol and published its own result evidence. Requires review and
  reproduction evidence; mere replay of a video is insufficient.

Schema validity never promotes a record. Maintainers review changes to these
fields; an entrant cannot self-award a higher level through an issue form.

## Append-only corrections

An existing record is never silently edited or removed. Append a
`type:"supersession"` event with `targetId`, `replacementId`, `reason`,
`timestamps:{registeredAt}`, `evidence`, and `authority:{id,name,url}`. Both
result IDs must already exist, must concern the same entrant and protocol, and
must not have another disposition. Append a `type:"retraction"` event with the
same fields except `replacementId` to withdraw a record. Events are public
annotations, not erasure; the UI must retain the original and show its status.
An event's evidence needs at least one artifact. A PR changing historical
records must fail the append-only comparison against its base branch.

PRs run `node scripts/validate-registry.mjs` and tests. The optional
`--base-file <registry.json>` argument requires all old protocols and records to
remain byte-for-byte equivalent under canonical JSON and only permits appended
entries. Issue forms create review requests, never automatic records.
