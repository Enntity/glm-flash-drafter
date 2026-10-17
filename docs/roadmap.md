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

**done** (code and CPU tests; not deployed): the trainer's `--sync-mode diloco` (process group, shared directory,
or `external` rounds for volunteer sites), the coordinator (`coordinator/`), the validator and merger
(`validator/`) and the site client (`drafter-join`, `enntity_drafter/join.py`). Plain averaging (outer learning
rate 1, no momentum) is the default: in a two-site, three-sync A/B on the full model it matched or beat
synchronous training on held-out groups, while the OpenDiLoCo momentum setting fell well behind. Details:
[volunteer.md](volunteer.md).

Next:

- A hosted coordinator and validator, then a pilot with invited sites.
- A full-size run across two real sites (the recipe pipeline in `drafter-join` and `setup`'s hardware checks are
  untested on GB10 so far).
- int8 deltas (1.18 GB instead of 2.34 GB, 0.79% error on a real delta) and overlapped fragment syncs
  (Streaming DiLoCo).

## 5. Validating contributions

**done** for weight deltas: pins checked before any bytes move; the same delta checks on the site and on the
validator (`enntity_drafter/validation.py`); held-out evaluation of every delta on a private, rotating held-out
set with conservative thresholds; quarantine with automatic re-validation when the validator changes; spot-check
replay of recipe shards (hook built; real recipes need operator-side capture); per-delta provenance (site, mode,
recipe or tag, attestation) recorded with each merge. No data is uploaded, so there are no data contributions to
validate.

Planned: robust aggregation (coordinate-wise median or trimmed mean) once there are many sites; attributing
held-out regressions in a merge back to its deltas; a release candidate is checked in Atlas for exactness and
tok/step against the previous one before it is published.

## 6. Weights

No weights are published yet. Published weights would come from public-only
data rounds, with the data card, training command, provenance stamp and Atlas
receipts.
