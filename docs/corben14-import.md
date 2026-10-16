# Corben's fourteen-link release: sourced historical import

Show Corben Sorenson's fourteen-link accomplishment prominently beside the
leaderboard as **author-published evidence**. This is a historical import,
without a claim that Corben submitted to this registry or that its maintainers
reproduced his controller. The machine-readable research record is
[`data/corben14-import.json`](../data/corben14-import.json).

The [GitHub release](https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002)
was published **2026-10-02T10:41:20Z**, according to the platform's
[`published_at` metadata](https://api.github.com/repos/corbensorenson/cartpole/releases/tags/fourteen-link-canonical-20261002).
That is a release publication time, not an inferred run time or proof of global
priority. Its creation time is separately 10:40:06Z.

## What the original receipts report

| Stage | Literal labels | Full thirty-second passes | Minimum upright hold |
| --- | --- | --- | --- |
| Exact source-candidate calibration | 20264100, explicit nominal state | 1/1 | 24.16 s |
| Noisy development20 | 20264200–20264219 | 20/20 | 8.16 s |
| Disjoint noisy development100 | 20264300–20264399 | 100/100 | 8.16 s |
| Reserved noisy20 | 214000–214019 | 20/20 | 8.16 s |
| Reserved noisy100 | 214100–214199 | 100/100 | 8.16 s |
| Reserved exact20 | 214200–214219 | 20/20 | 8.16 s |
| Reserved video | 214500 | 1/1 | 8.16 s |

All **141 reserved episodes** report 1,500 commands; maximum reserved cart
excursion is **1.8631687146015137 m**. These numbers come from the original
[release manifest](https://raw.githubusercontent.com/corbensorenson/cartpole/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/runs/swingup14_uniform/fourteen_link_swingup_manifest.json)
and the four linked evaluation receipts. This review parsed their episode rows
and checked receipt hashes. It did not execute their controller or simulator.

The public source-candidate calibration is the
[controller JSON](https://raw.githubusercontent.com/corbensorenson/cartpole/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/runs/swingup14_uniform/fourteen_link_controller.json).
It has a complete successful episode with trajectory integrity and a feasible
450-command reference whose reported native defect and initial gap are zero.
This calibration precedes the sixteen-second parked launch of the released
policy. Its 24.16-second hold must not be substituted for the reserved 8.16-second
result, and its reference diagnostics are not a cold audit of every episode.

Both development reports were found inside the public
[`synthesis_inputs.tar.gz` release asset](https://github.com/corbensorenson/cartpole/releases/download/fourteen-link-canonical-20261002/synthesis_inputs.tar.gz),
at the paths bound by the
[policy](https://raw.githubusercontent.com/corbensorenson/cartpole/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/runs/swingup14_uniform/fourteen_link_policy.json).
Their bytes match the policy's hashes; all 120 rows report complete successful
episodes. A read-only tar stream inspected these two reports and stopped early;
the entire 193 MB archive was not independently hashed here. Its recorded archive
digest comes from GitHub's asset metadata and the author's checksum file.

## Matching mechanics and provenance

The [pinned fourteen-link report](https://github.com/corbensorenson/cartpole/blob/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/docs/fourteen_link_swingup_paper.md)
states fourteen equal passive links totaling 3 m and 1 kg, a 1 kg cart, one
±80 N cart force, ±3 m rail, and thirty-second episodes. Commands run at 50 Hz
with float32 delivery; native binary64 MuJoCo RK4 uses four 5 ms substeps.
The goal requires 250 consecutive post-command samples with every wrapped
cumulative absolute angle strictly below 0.15 rad. Published damping,
armature, capsule geometry, noisy reset, and no-reset rules are part of the task.

The source metadata records MuJoCo 3.3.5, Python 3.12.14, NumPy 2.5.3 and
SciPy 1.18.1 on macOS 26.6.2 arm64. The policy is dated
**2026-10-02T10:26:56Z**; clean source snapshot metadata is dated
**2026-10-02T10:26:59Z**. These author-recorded timestamps precede the four
reserved receipts. They are not externally witnessed freeze times.

Frozen source commit: `3277b4173a9333819bf35acb848e384f52a227ec`.
The downloaded original Git bundle hashes to
`23335958906e74f3f8d9752a93f3d03ad726113b8e8c494845d27bd803f5ddef`.
The generated fourteen-link XML hash reported consistently by policy and
evaluation receipts is
`02fb1317dc1428860578eb9b548809b521063a6fe60c118e5b09b80d1dd678d2`.
All direct evidence URLs and hashes are included in the research JSON.

## Why this stays a historical milestone

The current strict registry requires a public artifact URL for the exact model
XML and explicit actual-episode cold-audit accounting. This review located the
XML hash but no standalone fourteen-link XML artifact, and no evidence supporting
a claim that all 262 episodes underwent independent cold native replay. Seven
stage totals add to 262 reported episodes; that is not 262 independently audited
episodes. No such numbers should be invented to satisfy the schema.

Keep the sourced **fourteen links, 141/141 reserved, 8.16 s upright** milestone
visible, with its publication time and original evidence. A future registry
import can preserve the fourteen-specific contract and disclose its exact
verification scope when the missing fields are supported.

Reviewed 2026-10-16. Only public artifact parsing and hash checks were performed.
