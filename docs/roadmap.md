# Roadmap

What is built is marked **done**; everything else is a plan and may change.

## 1. Finish round 2b and check it in Atlas

- Atlas tok/step on the matched and RigMark sets, plus one visit with a fixed
  verify width (`ATLAS_DFLASH_FIXED_WIDTH=7`) to separate acceptance from the
  width policy.
- Report greedy-prefix acceptance per category next to teacher-forced `eal`.

## 2. Data scaling

The selector study points at the body: the drafter misses tokens the target
is sure of, most of all in prose. The levers are more in-distribution tokens
and, later, capacity.

- **Public-only rounds.** Retrain from scratch on the published builders only
  (templates + structured coverage + new prose builders), so every number in
  a round can be reproduced from this repository.
- **Volume.** Rounds so far trained on 20-44M generated tokens. The next steps
  are about 100M and then about 200M generated tokens if the curves keep
  climbing. At 24,600 B per stored token (FP8) 200M tokens are about 5 TB of
  shards, more than two nodes hold at once, so capture and train in rounds.
- **Prose.** Prose has the lowest acceptance (template prose `eal` 2.2 vs code
  3.2 and structured 3.6). New prose builders (explanations, narrative,
  letters, summaries over synthetic documents) are the most useful data
  contribution.
- **Capture throughput.** Short rows are bound by the engine's per-request cost
  (~0.3 s); batching several short rows per request would raise it.

## 3. Appliance model: capture, train, delete

**done** (process, not code): a training round runs on a pair of nodes as a
self-contained appliance: capture shards to local disk, train, keep
checkpoints and receipts, delete the shards. Only text (prompts and
generations) and checkpoints are kept, because any row can be recaptured. This
is also the unit a volunteer site would run.

## 4. Low-communication training across sites (DiLoCo-style)

Goal: let several sites with GB10-class machines train one drafter without a
fast interconnect between them.

- Each site runs the appliance on its own shards: H inner steps of AdamW on
  its replica (the trainer already does replicated data-parallel training
  within a site).
- Every H steps, sites send the difference between their weights and the last
  global weights (the outer gradient) to a coordinator, which averages them and
  applies an outer optimizer (Nesterov momentum in DiLoCo), then broadcasts the
  new global weights. Streaming DiLoCo's variant syncs parameter fragments on
  a staggered schedule and overlaps communication with compute.
- A full-precision delta is 4.7 GB (1.17B fp32); bf16 or lower-precision
  deltas and fragment-wise sync reduce it. At H = 500 steps of ~3 s, one sync
  every ~25 minutes.
- **Not built:** the coordinator, the site client, delta compression, and
  handling sites that join, leave or lag. A local single-site prototype of the
  outer loop is the first step.

## 5. Validating contributions

Anyone can send wrong or malicious data or updates (poisoned captures, deltas
that degrade or backdoor the drafter). A drafter cannot change the target's
output under greedy verification, but a bad drafter can make serving slower,
and a backdoored one could behave differently on chosen inputs. Planned checks,
none built yet:

- **Data:** contributors share prompts and generations, not features; a
  trusted node recaptures them, or recaptures a random sample and compares
  bit-for-bit (the capture is deterministic without the prefix cache).
  Builders and their outputs are reviewed like code.
- **Updates:** every outer gradient is evaluated on a held-out set the
  contributor does not see before it is merged; robust aggregation (for
  example coordinate-wise median or trimmed mean instead of the mean) limits
  any one site's weight; outliers and sites whose updates repeatedly hurt the
  held-out proxy are excluded.
- **Provenance:** every checkpoint already carries a stamped lineage
  (`enntity_drafter.guard`); contributed updates would be recorded in it.
- **Release:** a release candidate is checked in Atlas for exactness and
  tok/step against the previous one before it is published.

## 6. Weights

No weights are published yet. Published weights would come from public-only
data rounds, with the data card, training command, provenance stamp and Atlas
receipts.
