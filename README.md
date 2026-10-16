# glm-flash-drafter

An open effort to train a DFlash2-style speculative-decoding drafter for
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) from scratch, with
code, data builders and results that anyone can use, check and extend. The
drafter targets GB10 / DGX Spark-class hardware served by the
[Atlas](https://github.com/Enntity/atlas) engine (as in
[SparkGLM](https://github.com/Enntity/sparkglm)).

Status: **research, early.** No weights are published yet. Everything here is
measured on Enntity's own two-node GB10 setup; numbers are from single runs
unless stated.

## Why

A drafter proposes several tokens per step and the target model verifies them
in one pass; the output is the target's own (greedy verification). How much
faster decoding gets depends on how many drafted tokens the target accepts.

The DFlash2 drafter commonly used with GLM-5.3-Flash on this hardware,
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
is licensed CC BY-NC-ND 4.0: no commercial use, no derivatives. The method and
the code are open (DFlash/DFlash2 code is MIT and Apache-2.0, see
[docs/prior-art.md](docs/prior-art.md)), and the target model is MIT. Training
new weights from a fresh initialization, on data we are allowed to use, gives
a drafter without that restriction. This repository contains no incoai
weights, configs, tokenizer files or text, and `enntity_drafter/guard.py`
refuses to load anything that looks like them (by path, provenance files,
safetensors metadata and known hashes).

## License

**MIT** ([LICENSE](LICENSE), [NOTICE](NOTICE)), matching the target model's own
license. The few functions re-implemented from
[vllm-project/speculators](https://github.com/vllm-project/speculators) keep its
Apache-2.0 terms ([LICENSES/Apache-2.0.txt](LICENSES/Apache-2.0.txt)); speculators
itself is installed as a dependency, not vendored. The target,
GLM-5.3-Flash, is MIT, which permits this use; no target file is redistributed
(the trainer and Atlas read the embedding and lm_head from your own copy).
Teacher text from hosted endpoints is also subject to each host's terms (see
[docs/pipeline.md](docs/pipeline.md)). The Atlas engine (AGPL-3.0-only) is only
used as a separate server process; no engine code is copied here.

## Current results

The drafter has the DFlash2 shape: 5 non-causal sliding-window layers
(hidden 4096, 32/8 heads, window 2048), grouped dynamic convolution, a
rank-256 candidate selector over the top-16 draft tokens, conditioned on the
target's hidden states at layers 5, 14, 24, 33 and 42, proposing 7 tokens per
step. 1.17B trained parameters; the embedding and lm_head are the target's,
frozen.

Atlas accepted tokens per verify step (higher is better; greedy, one request
at a time, same server and settings for every drafter; both visits per arm
gave identical tok/step):

| category | DFlash2 (incoai) | round 1 | round 2a |
|---|---|---|---|
| matched prose, 12 prompts x 256 tokens | 2.501 | 1.589 | 1.785 |
| matched code, 12 prompts x 256 tokens | 4.890 | 2.471 | 2.863 |
| RigMark code | 4.347 | 2.134 | 2.537 |
| RigMark prose | 2.044 | 1.404 | 1.573 |
| RigMark structured | 7.569 | 2.081 | 2.412 |

Output text was identical across all drafters on every matched prompt.
Round 2a reaches 0.58-0.77x of DFlash2's tok/step on these rows (0.32x on
structured output); it is not yet a replacement.

Round 2b (running as of 2026-10-16) adds 23,942 rows of structured-output data
(JSON, tool calls, configs, code edits, tables) and held-out groups for it. On
the trainer's teacher-forced proxy (expected accepted length per block, 1-8),
at step 2,464 of 4,620:

| held-out group | start (round 2a weights) | step 2,464 |
|---|---|---|
| structured (coverage prompts) | 2.75 | 5.21 |
| template (code, prose, structured, tools) | 2.65 | 2.95 |
| agentic (internal sessions) | 3.36 | 3.43 |

These are proxy numbers, not Atlas measurements; round 2b has not been checked
in the engine yet. Details, curves and caveats: [docs/results.md](docs/results.md).

## Pipeline

1. **Prompts** (`scripts/generate_prompts.py`): Enntity-authored synthetic
   templates and structured-output builders, deterministic, no third-party text.
2. **Teacher text** (`scripts/api_generate.py`): GLM-5.3-Flash completions from
   hosted endpoints (OpenRouter, `data_collection: deny`, Z.AI's endpoint
   excluded by its terms), thinking mode, temperature 1.0 / top_p 0.95 like
   serving.
3. **Hidden-state capture** (`scripts/prepare_items.py`, `scripts/capture.py`):
   each prompt + completion is prefilled once by the target in Atlas with the
   capture patch (branch
   [`exp/drafter-capture-20261014`](https://github.com/Enntity/atlas/tree/exp/drafter-capture-20261014)),
   which writes the features exactly as serving computes them for the drafter.
4. **FP8 shards** (`python -m enntity_drafter.fp8`): e4m3 with one scale per
   row per slot, 24,600 bytes per token, 2.6% relative L2, teacher top-1
   unchanged on 98.9% of positions.
5. **Training** (`scripts/train.py`): from-scratch init, upstream DFlash2
   objective (KL to the target distribution + selector cross-entropy),
   data-parallel across nodes, deterministic resume.
6. **Export** (`scripts/export.py` or `--export`): the 81-tensor BF16 layout
   Atlas loads with `--draft-model`; config generated from our own config.
7. **Atlas check** (`scripts/atlas_check.py`): exactness and tok/step per
   category against a running server.

More: [docs/pipeline.md](docs/pipeline.md), [docs/capture.md](docs/capture.md)
(format spec), [docs/training.md](docs/training.md).

## Hardware notes (GB10)

- Measured ceiling: 95 TFLOPS dense BF16 matmul per GB10 (SM clock capped
  near 2.2 GHz).
- The trainer's fast path (block-sparse flex attention, bf16 weights with fp32
  masters, compiled decoder layers, fused AdamW, deferred metrics, pinned
  loader) reaches **53.9% MFU** on one GB10 (47.3% counting only the attention
  pairs the mask allows), 2.3x the first trainer. Two nodes over 200G RoCE run
  at 51% (same formula) with row packing.
- CPU and GPU share 128 GB. Running out of memory can hang the host instead
  of raising an error. Never run the target and the trainer on the same node;
  run a memory watchdog that kills the job above a floor; use the trainer's
  `--probe` (worst-case forward/backward before step 0), `--min-free-gb` (clean
  stop on every rank) and `--max-mem-fraction`. Raise the open-file limit for
  memory-mapped data (`ulimit -n 524288`).

## How to help

Ready now:

- **Generate data.** Run the prompt builders and `api_generate.py` with your
  own key. Better, more diverse prompt builders (especially prose, where the
  drafter is weakest) are the most useful contribution today.
- **Run capture and training locally**, on one or two GB10-class machines with
  your own GLM-5.3-Flash copy. The CPU tests exercise the full trainer on a
  tiny model with no GPU.
- **Reproduce results.** The Atlas check and the trainer's held-out evals are
  scripted. Rounds 1-2b also used Enntity-internal agentic and chat traffic,
  which is not published, so those exact runs cannot be repeated from this
  repo alone; runs on the public builders can, and we will publish theirs.

Planned, not built:

- **Training across volunteer sites** with DiLoCo-style low-communication
  updates (each site trains on its own captured shards; sites exchange model
  deltas every few hundred steps). The sync coordinator does not exist yet.
- **Validation of contributions.** Any shared data or update can be wrong or
  malicious. Nothing contributed will be merged without checks such as
  recapturing a sample of rows on a trusted node, held-out evaluation of each
  update, and rejecting outliers. The design is open; see
  [docs/roadmap.md](docs/roadmap.md).

See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Quick start (CPU tests)

```
python -m venv .venv && . .venv/bin/activate
pip install torch numpy safetensors transformers pyyaml
bash scripts/install_speculators.sh
pip install -e .
OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

The suite trains tiny models with 1-2 CPU processes (gloo) and takes tens of
minutes; run one file with `-p test_train.py`.
