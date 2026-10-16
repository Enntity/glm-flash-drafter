# Training

`scripts/train.py` (= `python -m enntity_drafter.train`) trains the drafter
from a fresh initialization on captured features.

## Model

`configs/full.json`: Speculators `DFlash2DraftModel` (pinned revision
`824125a`), 5 layers, hidden 4096, intermediate 12288, 32 query / 8 KV heads of
128, non-causal sliding window 2048, block 8 (anchor + 7 drafts), grouped
dynamic convolution (kernel 2, group 16), candidate selector rank 256 over the
unary top-16, mask token 154856, target layers 5, 14, 24, 33, 42.
1,171,080,448 trained parameters in 81 tensors (the Atlas key set, checked by
`tests/test_train.py`). `configs/tiny.json` is a 2-layer, hidden-256 version
for CPU tests.

The embedding, the target's final norm (applied to the captured final hidden
to build the teacher distribution) and the lm_head are read from the target
checkpoint through the license guard, frozen, and never exported: Atlas fills
them from the target at load.

### Initialization (`enntity_drafter.model.scratch_init`)

Deterministic per seed and identical on every rank (ranks compare a digest
before step 0):

| tensors | init |
|---|---|
| Linear weights | N(0, 0.02) |
| `o_proj`, `down_proj` | N(0, 0.02 / sqrt(2 L)) |
| RMSNorm gains | 1 |
| output `norm` | copy of the target's final-norm gain, so drafter hiddens reach the shared lm_head at the expected scale |
| grouped convolutions | identity |
| selector `predecessor_codebook`, `hidden_projection` | N(0, 0.02) |
| selector `successor_codebook` | 0: the initial selector order is the unary top-k order |

## Objectives (`--objective`)

- `upstream` (used for every published round): `DFlash2DraftModel.forward`'s
  loss, i.e. KL from the target distribution (`--unary-loss kl_div`) plus the
  K-way selector cross-entropy (`--selector-alpha`), weighted per block
  position by `fixed-exp-decay` (gamma 4) or `dpace`.
- `selector_inset`: as upstream, but the selector term only on slots whose
  target is already in the unary top-16 (upstream injects a missing target as
  the 16th candidate, a pick serving cannot make). Experimental.
- `selector_prefix`: each slot weighted by the detached probability that the
  served selector path survives the earlier slots; optional differentiable
  expected-accepted-length term (`--eal-weight`). Experimental.

A selector-only study on the round-2a weights found that with the body frozen
no selector objective gains more than about +0.02 accepted tokens per block on
prompt-disjoint rows, against a gap of about 1.4 to the top-16 oracle: the gap
is in the body's unary predictions (on positions where the target is at least
95% sure, the body ranks the target first only 40-51% of the time). Effort
goes to data and capacity, not to the selector objective.

## Data path

- `--capture-index` + `--capture-root`: Atlas captures read in place
  (`enntity_drafter.capture_io`). A sample is one row: up to `--anchors` (512)
  anchors drawn inside a random crop of `--crop` (4096) eligible positions, and
  the union of their windows `[a - 2048, a + 8)` read with `np.memmap`.
  Disjoint windows become separate documents in one packed sequence, with
  absolute `position_ids`. Anchors without the full window are skipped, never
  padded, except at a true sequence start. A test checks that the packed
  forward equals the full-row forward exactly.
- `--source-weights template=0.4,structured=0.6`: source-balanced coverage
  sampling. Each source cycles its own items; a deterministic smooth weighted
  round robin makes each source's share of anchors equal its weight; one
  source epoch uses every eligible position of every row as an anchor exactly
  once. `--source-max-epochs` caps small sources once the step plan is known.
- `--pack-rows`: several short row chunks share one micro-step (round 2b; the
  structured rows average 923-1,959 tokens).
- `--fp8-device`: FP8 rows reach the GPU as bytes + scales.
- `--eval-groups NAME=SRC[+SRC][:ROWS],...` and `--eval-breakdown category`:
  separate held-out groups. Held-out rows are chosen by prompt id hash
  (`scripts/heldout_split.py`), never by anchor within a row.
- `--features`: the older `enntity.drafter_rows/v1` row store
  (`enntity_drafter.data`), used by the synthetic CPU fixtures.

## Evaluation

The held-out proxy is the expected accepted length per block (`eal`, 1-8):
1 + the leading run of draft slots where the selector's greedy pick among the
unary top-16 equals the target's argmax, with teacher-forced predecessors, on
fixed anchors. Also logged: the top-16 oracle accepted length, recall@16,
selector accuracy, losses, and per-depth conditional acceptance
(`accept_by_depth`) and survival (`survive_by_depth`, `1 + sum = eal`).

The proxy tracks Atlas tok/step in direction but not in level: Atlas also
applies a confidence-based verify-width policy, and engine survival counts
verifies the policy did not attempt. Check in the engine (`scripts/atlas_check.py`).

## Parallelism and safety

The body is replicated on every rank (fp32 masters, gradients and AdamW
state fit one GB10). Gradients accumulate locally in fp32 over `--grad-accum`
micro-steps; only the last backward all-reduces them, in bf16 buckets launched
as backward produces them. FSDP2 sharding moved about 7 GB per rank per
micro-step and left the GPUs idle; replication moves about 2.3 GB per
optimizer step. Evals split rows across ranks and sum metric totals.

Resume is rank-local (params, optimizer, step, counters) and the data stream
is a pure function of `(seed, rank, item)`, so a resumed run replays the same
stream. `--init-from` starts a new run from one of this trainer's checkpoints
(stamped provenance and matching config required; fresh optimizer).

GB10 memory is shared by CPU and GPU; exhaustion can hang the host.
`--probe` runs a worst-case-size forward/backward before step 0 and counts the
not-yet-allocated optimizer state against `--min-free-gb`; `--min-free-gb`
stops every rank cleanly when host MemAvailable falls below it;
`--max-mem-fraction` caps the CUDA allocator; an OOM exits the rank at once.
`--time-budget-h` plans the step count from smoke-step timings
(`--smoke-steps`).

## Performance (`enntity_drafter.fastpath`, every piece opt-in)

Ceiling: 95 TFLOPS dense BF16 matmul per GB10. MFU = analytic matmul FLOPs of
forward + backward / time / 95 TFLOPS; "useful" counts attention only on the
(query, key) pairs the mask allows.

Where the first trainer's time went (one micro-step, 256 anchors, SDPA):
GEMMs 24% (already at ~94% of peak), attention 16%, weight-sized casts and
fp32 gradient accumulation 12%, activation-sized fp32 elementwise kernels 31%,
other 17%.

| flag | removes |
|---|---|
| `--attn flex_compiled` (+ `--compile-layers`) | dense masked attention: block-sparse flex attention with upstream's block mask, compiled inside each layer |
| `--attn packed_sdpa` | same, without compile: anchors grouped into chunks that gather only their reachable keys |
| `--param-dtype bf16` | per-micro autocast weight casts: bf16 compute weights, fp32 masters get each micro's gradient (bit-identical to autocast) |
| `--compile-layers` | unfused norm / conv / rotary / residual kernels: `torch.compile(dynamic=True)` per decoder layer; the fused loss kernels stay eager |
| `--adamw-impl fused` | foreach AdamW kernels |
| `--deferred-metrics` | ~35 host syncs per micro-step |
| `--train-metrics light` | metric-only full-vocabulary passes in training steps (loss and gradients bit-identical) |
| `--pin-loader` | synchronous pageable host-to-device copies |

One GB10, 512 anchors x 4 micro-steps, real rows:

| configuration | step (s) | MFU | MFU useful | supervised tok/s | GPU W |
|---|---|---|---|---|---|
| first trainer (`--attn sdpa`) | 6.68 | 23.8% | 21.0% | ~1,560 | 37.6 |
| packed SDPA, bf16, compile, fused AdamW, deferred, light | 3.76 | 41.5% | 36.4% | 2,722 | 42.7 |
| + flex attention compiled | 3.01 | 51.6% | 45.6% | 3,444 | 49.1 |
| + pinned loader (final) | 2.85 | **53.9%** | **47.3%** | **3,553** | 48.8 |

Final configuration:

```
--attn flex_compiled --param-dtype bf16 --compile-layers --adamw-impl fused \
  --deferred-metrics --train-metrics light --pin-loader
```

Equivalence: the packed and flex paths match the first trainer's SDPA path
within the noise between its own two attention kernels (eager vs SDPA) on the
full model (gradient relative difference 1.5% vs 1.7%). Recompiles happen in
the first ~8 steps only.

What is left is bandwidth-bound work proportional to the 1.17B parameters and
the [tokens, 154,880] logits: at 220 GB/s, AdamW, fp32 master-gradient
accumulation, clipping and copies take about 0.66 s of a 2.5 s step while
GEMMs run at 88% of peak. Next levers: fp32 gradient accumulation inside the
GEMM epilogue, larger accumulation, a fused logits head.

On two nodes (200G RoCE, NCCL ~20 GB/s), round 2a ran at 41% MFU because short
rows halved anchors per micro-step; round 2b's row packing brings it to 51%.
Round 2a also slowed from 2.95 to 3.9 s per step over 3 h at constant work
(not diagnosed); the trainer now logs SM clocks, throttle reasons, host CPU and
memory, allocator and loader state per log window to find it.

## Commands

Two nodes, one GPU each (run on both with `--node_rank` 0 and 1):

```
torchrun --nnodes=2 --nproc_per_node=1 --node_rank=$RANK --master_addr=$MASTER --master_port=29500 \
  scripts/train.py --config configs/full.json --teacher /models/GLM-5.3-Flash \
  --capture-index data/index/index-w2.json --capture-root data --output runs/r1 --device cuda \
  --anchors 512 --grad-accum 4 --crop 4096 --lr 6e-4 --warmup 100 --min-lr-ratio 0.1 \
  --attn flex_compiled --param-dtype bf16 --compile-layers --adamw-impl fused \
  --deferred-metrics --train-metrics light --pin-loader --fp8-device --pack-rows \
  --source-weights template=0.4,structured=0.6 --source-max-epochs 2 \
  --eval-groups template=template:96,structured=structured:96 --eval-breakdown category \
  --time-budget-h 6 --smoke-steps 20 --eval-points 30 --checkpoint-points 4 --keep-checkpoints 2 \
  --probe --min-free-gb 8 --peak-tflops 95 --export
```

Round 2a's continuation used `--init-from <round-1 checkpoint> --lr 3e-4
--warmup 50` (half the round-1 peak, fresh AdamW state). The teacher path
must be the GLM-5.3-Flash target directory (`model.safetensors.index.json`,
`config.json`); `--teacher` is checked for a drafter-shaped or
non-commercial-licensed directory and refused if so.
