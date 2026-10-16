# Credit, evidence, and dates

Pendulum Arena builds on public work by Kache and Corben Sorenson. Credit is
part of the benchmark: link to the people, preserve their stated boundaries,
and keep each result attached to the task it actually attempted. Inclusion here
does not imply either person has submitted to, endorsed, or reviewed this site.

## Kache — @yacineMTB

Kache published a six-pendulum cart-pole demonstration on **June 9, 2026 at
00:50:41 UTC**. His post describes building an environment and training an AI
for the task. This is a known public demonstration and an inspiration for our
work. Its priority language remains the author's claim; this registry does not
certify a global first. [Original post](https://x.com/yacineMTB/status/2064148140899348779)

The post alone does not establish the exact mechanics, acceptance thresholds,
noise distribution, frozen source, or disjoint evaluation cohorts required by
our ranked protocol. Give it prominent historical credit without assigning a
comparable leaderboard score. Kache's own
[GitHub profile](https://github.com/yacineMTB) links his X identity and
[personal site](https://yacine.ca/); neither was used as numerical benchmark
evidence. This review found no additional primary source establishing a matched
canonical release for this demonstration.

## Corben Sorenson — @SorensonCorben

Corben's public cartpole project supplies the task contract used for our matched
fifteen-link campaign. His **fourteen-link GitHub release was published October
2, 2026 at 10:41:20 UTC**. The release reports 141 successful reserved episodes
and a complete thirty-second video. The accompanying report states 8.16 seconds
upright, native MuJoCo execution, and no reset after initialization. These are
author-published results with public evidence, rather than a claim that Pendulum
Arena independently reran the controller.
[Release](https://github.com/corbensorenson/cartpole/releases/tag/fourteen-link-canonical-20261002),
[pinned report](https://github.com/corbensorenson/cartpole/blob/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/docs/fourteen_link_swingup_paper.md)

The matched task uses a 3 m / 1 kg chain, 1 kg cart, ±80 N cart force,
±3 m rail, 50 Hz commands, and a five-second strict absolute-angle threshold
of 0.15 rad within a complete thirty-second episode. Mechanics, numerical
runtime, initialization, and validation gates also matter. A link count alone
does not identify the benchmark.
[Pinned fifteen-link configuration](https://github.com/corbensorenson/cartpole/blob/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/configs/swingup15_uniform.yaml)

His **October 15 fifteen-link update is explicitly partial**: a three-second
nominal hold followed by rail failure does not pass the five-second/full-episode
task. Preserve this as useful development history; do not promote it to a
successful fifteen-link entry. Its post was published at 15:24:26 UTC.
[Primary evidence note](https://github.com/corbensorenson/cartpole/blob/b84c6413974d5c0831cdf261ce15e58ba65cd3f5/docs/fifteen_link_partial_result.md),
[author's post](https://x.com/SorensonCorben/status/2110753674184040921)

The report pins the accepted fourteen source at
`3277b4173a9333819bf35acb848e384f52a227ec`; this review pins the surrounding public
repository at `b84c6413974d5c0831cdf261ce15e58ba65cd3f5`. Preserve both identities
when reproducing or attributing the work.

## How the clock works

- **Run time:** a submitter's reported time for executing an experiment. This
  remains a claim unless an identified witness or timestamp service attests it.
- **Publication time:** when a named public platform says a particular post or
  release was published. This proves a dated publication existed; it does not
  prove that no earlier result existed. The dates above come from X's official
  syndication metadata and GitHub's release `published_at`, not Git author dates.
- **Registry time:** when a submission or review entered this registry. Never
  backdate it to the experiment or imported publication date. The lineage
  dataset's `updatedAt` records this content review, not an experiment, result
  submission, or independent reproduction.

Unknown dates stay null. A source's `publishedAt` belongs to that source; a
timeline `date` identifies the specific publication described beside it. The
timeline is a selection of known primary evidence, not an exhaustive priority
history. Frozen hashes establish artifact identity, not historical priority.

Rage Mill's earlier results used a different plant and acceptance rule. Its
matched canonical fifteen campaign must be assessed using its own published
receipts. A separate audit implementation owned by the same team is not an
external reproduction. Simulation evidence makes no hardware claim.

Review performed October 16, 2026. No controller was rerun for this attribution
review, and no message was sent to either author.
