# Results

All runs on two GB10 nodes (one GPU each, 200G RoCE). Atlas measurements use
the same engine image, profile and settings for every drafter (two-node
tensor/expert parallel GLM-5.3-Flash NVFP4, greedy, one request at a time,
copy drafts on as in production); arms differ only in `--draft-model`. Visits
ran in ABBA order (scratch, DFlash2, DFlash2, scratch), and both visits of an
arm gave identical tok/step. Rounds 1-2b are single training runs.

## Data per round

| round | data | rows | generated tokens | supervised tokens trained |
|---|---|---|---|---|
| 1 | template prompts (Enntity-authored) + Enntity-internal agentic and chat traffic | 7,547 | 20.2M | 22.6M |
| 2a | round 1 rows, source-balanced (template 35%, internal sources 65%) | same | same | +46.3M (continued from round 1) |
| 2b | round 1 rows + 23,942 structured-coverage rows (FP8), small sources capped at 2 epochs | 31,489 | 43.6M | +114.1M (continued from 2a) |

Template rows were 92% of round 1's anchors. The internal sources are not
published; see the README for what that means for reproduction.

## Atlas: accepted tokens per verify step

| category | DFlash2 (incoai) | round 1 | round 2a | 2a / DFlash2 |
|---|---|---|---|---|
| matched prose, 12 prompts x 256 forced tokens | 2.501 | 1.589 | 1.785 | 0.71 |
| matched code, 12 prompts x 256 forced tokens | 4.890 | 2.471 | 2.863 | 0.59 |
| RigMark code | 4.347 | 2.134 | 2.537 | 0.58 |
| RigMark prose | 2.044 | 1.404 | 1.573 | 0.77 |
| RigMark structured | 7.569 | 2.081 | 2.412 | 0.32 |

Decode tok/s (engine per-request rate, mean of 2 visits):

| category | DFlash2 | round 1 | round 2a |
|---|---|---|---|
| matched prose | 45.8 | 31.1 | 34.9 |
| matched code | 69.4 | 44.8 | 49.5 |
| RigMark code | 64.0 | 38.5 | 45.3 |
| RigMark prose | 39.5 | 27.2 | 31.0 |
| RigMark structured | 90.7 | 38.6 | 43.0 |

DFlash2's matched prose / code tok/step (2.501 / 4.890) reproduce its earlier
reference measurements (2.501 / 4.881) on the same harness. RigMark rows use
thinking on, low effort.

Engine per-position survival (share of verify steps accepting at least d
drafts, weighted by steps):

| category | DFlash2 | round 1 | round 2a |
|---|---|---|---|
| prose | .74 .46 .18 .07 .02 .01 .00 | .46 .13 .00 .00 .00 .00 .00 | .55 .21 .02 .00 .00 .00 .00 |
| code | .90 .78 .60 .51 .42 .33 .25 | .71 .43 .18 .08 .03 .01 .00 | .80 .55 .25 .14 .06 .02 .01 |
| RigMark structured | 1.0 .95 .95 .95 .93 .90 .90 | .78 .30 .00 .00 .00 .00 .00 | .83 .54 .04 .00 .00 .00 .00 |

The drop to near zero at depth 3 is mostly Atlas's confidence-based verify
width (calibrated on DFlash2's confidences) cutting short verifies, which the
engine then counts as rejected: the trainer's proxy for the same categories
shows a gradual decline (template prose .58 .28 .12). A fixed-width visit
(`ATLAS_DFLASH_FIXED_WIDTH=7`) is the check; not run yet.

**Exactness.** On every matched prompt (4 non-streamed with logprobs, 24
streamed), the text was identical with every drafter and in every visit.
Decode logprob hashes differ between drafters; they depend on the verify path
in the shipped engine and are not a gate. On held-out internal agentic rows,
39 of 40 matched (tool-call ids ignored); the one difference came from the
engine's forced end-of-thinking at a token budget, whose position follows
verify-step boundaries.

## Trainer proxy (held-out, teacher-forced)

`eal` = expected accepted length per block (1-8), `oracle` = the same if the
best of the unary top-16 were always picked. Held-out groups are disjoint
from training by prompt id.

### Round 1 (fresh init, 910 steps, first trainer)

Held-out internal agentic rows: `eal` 2.74, oracle 3.90 at 22.6M supervised tokens.

### Round 2a (from round 1, fast trainer, 3.3 h)

| step | supervised tokens (M) | eal | oracle | loss |
|---|---|---|---|---|
| 0 | 0.0 | 2.741 | 3.898 | 2.762 |
| 258 | 4.0 | 2.871 | 4.143 | 2.624 |
| 516 | 8.1 | 2.953 | 4.298 | 2.480 |
| 774 | 12.1 | 3.059 | 4.425 | 2.403 |
| 1032 | 16.1 | 3.125 | 4.507 | 2.356 |
| 1290 | 20.2 | 3.161 | 4.569 | 2.309 |
| 1548 | 24.2 | 3.189 | 4.619 | 2.402 |
| 1806 | 28.2 | 3.251 | 4.706 | 2.338 |
| 2064 | 32.2 | 3.295 | 4.751 | 2.327 |
| 2322 | 36.3 | 3.325 | 4.767 | 2.394 |
| 2580 | 40.3 | 3.351 | 4.755 | 2.366 |
| 2838 | 44.3 | 3.369 | 4.783 | 2.412 |
| 2967 | 46.3 | 3.347 | 4.784 | 2.465 |

The step-2,910 checkpoint was exported (the run was stopped at 3,070 once the
curve flattened). Conditional acceptance by depth at the end: .75 .69 .69 .73
.76 .79 .79. The small internal sources reached 4-7 epochs and were being
memorized (held-out loss stopped improving after ~step 1,300), which is why
round 2b caps epochs and adds new data.

### Round 2b (from round 2a; 4,620 steps, 114.1M supervised tokens)

| step | supervised tokens (M) | agentic eal | template eal | structured eal | structured oracle |
|---|---|---|---|---|---|
| 0 | 0.0 | 3.361 | 2.647 | 2.753 | 4.216 |
| 308 | 7.6 | 3.286 | 2.532 | 3.973 | 5.557 |
| 616 | 15.2 | 3.280 | 2.580 | 4.376 | 5.954 |
| 924 | 22.8 | 3.291 | 2.652 | 4.601 | 6.153 |
| 1232 | 30.4 | 3.326 | 2.719 | 4.724 | 6.252 |
| 1540 | 38.0 | 3.385 | 2.784 | 4.908 | 6.405 |
| 1848 | 45.6 | 3.401 | 2.821 | 5.015 | 6.467 |
| 2156 | 53.2 | 3.388 | 2.874 | 5.127 | 6.551 |
| 2464 | 60.8 | 3.431 | 2.945 | 5.207 | 6.623 |
| 4620 (end) | 114.1 | 3.507 | 3.141 | 5.551 | 6.882 |

At the end, template by category: code 3.40, prose 2.34, structured 3.82
(at step 0: 2.89, 2.05, 3.03). The template group's prompts were partly seen
by the parent rounds (about a quarter of their anchor positions), so its
step-0 value is optimistic; round 2b never trains on them.

Atlas tok/step (shipped adaptive verify width): matched prose 1.905, matched
code 3.574, RigMark structured 4.944 (0.76x, 0.73x and 0.65x of DFlash2);
RigMark structured decode rose from 43 to 71 tok/s. A fixed verify width of 7
raised tok/step 5-18% but lowered decode speed 3-34%, so the adaptive policy
is the right one for this drafter. Prose survival by draft depth is
.60 .27 .09 .03: the remaining prose gap is real, not a width artefact.

## Throughput

| | value |
|---|---|
| capture, rows 8-24K tokens | 2,480-2,560 sequence tok/s (two nodes, SP prefill) |
| capture, rows under 2K tokens | ~940 tok/s (fixed per-request cost) |
| FP8 storage | 24,600 B per stored token |
| training, one GB10, fast path | 53.9% MFU, 3,553 supervised tok/s |
| training, two GB10s, round 2b | 51% MFU, ~6,900 supervised tok/s |
| teacher text via API | ~600 tok/s at concurrency 16; about $0.5-1.7 per M output tokens incl. input, by host and length |
