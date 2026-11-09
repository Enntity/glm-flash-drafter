# glm-flash-drafter

**A DFlash2 speculative-decoding drafter for
[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash), trained from
scratch on 2x DGX Spark.** Fresh random initialization, no third-party drafter
weights: all 1.17B trained parameters come from Enntity runs on two GB10 nodes. It is specific
to GLM-5.3-Flash (it reads the target's hidden states and shares its embedding
and lm_head) and has the DFlash2 shape, so the
[Atlas](https://github.com/Enntity/atlas) engine (as in
[SparkGLM](https://github.com/Enntity/sparkglm)) serves it as a drop-in DFlash2
drafter. Results are measured against the **reference drafter**: the best
public DFlash2 drafter for GLM-5.3-Flash,
[incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
This repository holds the code, data builders and results, for anyone to use,
check and extend.

Status: **research; outside contributors welcome.** The current version is
**v81** (2026-11-05): 87-95% of the reference drafter's decode speed on public prompts and
115-122% on held-out coding-agent turns. SparkGLM currently ships v80; the
inference weights are not public yet. Signed training globals are available to
registered compute sites. Everything here is measured on Enntity's own
two-node GB10 setup; numbers are from single runs unless stated.

## Why

A drafter proposes several tokens per step and the target model verifies them
in one pass; the output is the target's own (greedy verification). How much
faster decoding gets depends on how many drafted tokens the target accepts.

The reference drafter is licensed CC BY-NC-ND 4.0: no commercial use, no derivatives. The method and
the code are open (DFlash/DFlash2 code is MIT and Apache-2.0, see
[docs/prior-art.md](docs/prior-art.md)), and the target model is MIT. Training
new weights from a fresh initialization, on data we are allowed to use, gives
a drafter without that restriction. This repository contains none of the
reference drafter's weights, configs, tokenizer files or text, and `enntity_drafter/guard.py`
refuses to load anything that looks like them (by path, provenance files,
safetensors metadata and known hashes).

## License

**Apache-2.0** ([LICENSE](LICENSE), [NOTICE](NOTICE)), the same license as
[vllm-project/speculators](https://github.com/vllm-project/speculators), which the
trainer builds on, so the whole training stack has one set of terms and an
explicit patent grant. The target,
GLM-5.3-Flash, is MIT, which permits this use; no target file is redistributed
(the trainer and Atlas read the embedding and lm_head from your own copy).
Teacher text from hosted endpoints is also subject to each host's terms (see
[docs/pipeline.md](docs/pipeline.md)). The Atlas engine (AGPL-3.0-only) is only
used as a separate server process; no engine code is copied here.

## Current results (v81)

The drafter has the DFlash2 shape: 5 non-causal sliding-window layers
(hidden 4096, 32/8 heads, window 2048), grouped dynamic convolution, a
rank-256 candidate selector over the top-16 draft tokens, conditioned on the
target's hidden states at layers 5, 14, 24, 33 and 42, proposing 7 tokens per
step. 1.17B trained parameters; the embedding and lm_head are the target's,
frozen.

Atlas decode speed on a two-node GB10 server (NVFP4 target, greedy, one
request at a time, shipped adaptive verify width; every arm on the same server
and settings, differing only in `--draft-model`; 2026-11-05). Engine tok/s,
accepted tokens per verify step in brackets:

| category | v81 | v80 | v81 vs v80 | reference | v81 / reference |
|---|---|---|---|---|---|
| prose, 12 prompts x 256 tokens | 40.9 (2.12) | 40.1 (2.06) | +1.9% | 46.8 (2.50) | 87% |
| code, 12 prompts x 256 tokens | 63.0 (4.11) | 61.5 (3.99) | +2.4% | 69.2 (4.89) | 91% |
| structured output, 4 prompts | 79.6 (6.01) | 76.9 (5.74) | +3.4% | 87.3 (6.98) | 91% |
| RigMark prose | 36.0 (1.86) | 35.8 (1.83) | +0.6% | 39.0 (2.04) | 92% |
| RigMark code | 63.3 (4.27) | 61.8 (4.12) | +2.4% | 66.9 (4.73) | 95% |
| RigMark structured | 84.2 (6.73) | 79.8 (6.25) | +5.6% | 91.0 (7.64) | 93% |
| held-out coding-agent turns, set A (16) | 51.1 (3.42) | 52.0 (3.46) | -1.6% | 42.0 (2.70) | 122% |
| held-out coding-agent turns, set B (24) | 48.8 (3.17) | 47.6 (3.10) | +2.4% | 42.3 (2.68) | 115% |

Run-to-run noise is about ±2%. Output text was identical across all drafters
on every public prompt; on the held-out agentic turns 39 of 40 matched (the
one difference is the engine's forced end-of-thinking at a token budget, see
[docs/results.md](docs/results.md)). The reference drafter is still ahead on public prose,
code and structured output; v81 is ahead on agentic coding traffic like the
traffic it was trained on.

Versions: v79 (2026-10-31, fresh agentic sessions), v80 (2026-11-01,
served-path objective), v81 (2026-11-05, one epoch over 76K rows / 119M stored
tokens). Since round 2b (0.73-0.76x of the reference drafter's tok/step on matched prose and
code, 0.65x on RigMark structured), v81 is at 0.85x, 0.84x and 0.88x. Every
version descends from the same fresh initialization. Per-round data, curves
and caveats: [docs/results.md](docs/results.md).

## Pipeline

1. **Prompts** (`scripts/generate_prompts.py`): Enntity-authored synthetic
   templates and structured-output builders, deterministic, no third-party text.
2. **Teacher text** (`scripts/api_generate.py`): GLM-5.3-Flash completions from
   hosted endpoints (OpenRouter, `data_collection: deny`, Z.AI's endpoint
   excluded by its terms), thinking mode, temperature 1.0 / top_p 0.95 like
   serving. Later rounds add GLM-5.3-Flash coding-agent sessions on
   Enntity-authored tasks, where every answer and tool call is GLM's own.
3. **Hidden-state capture** (`scripts/prepare_items.py`, `scripts/capture.py`):
   each prompt + completion is prefilled once by the target in Atlas with the
   capture patch (branch
   [`exp/drafter-capture-20261014`](https://github.com/Enntity/atlas/tree/exp/drafter-capture-20261014)),
   which writes the features exactly as serving computes them for the drafter.
4. **FP8 shards** (`python -m enntity_drafter.fp8`): e4m3 with one scale per
   row per slot, 24,600 bytes per token, 2.6% relative L2, teacher top-1
   unchanged on 98.9% of positions.
5. **Training** (`scripts/train.py`): from-scratch init (round 1), then each
   round continues from the last global. Rounds 1-3 used the upstream DFlash2
   objective (KL to the target distribution + selector cross-entropy); v80 and
   v81 use `selector_prefix` (each slot weighted by the chance the served path
   reaches it, plus an expected-accepted-length term) with `nla` unary loss.
   Two-node plain averaging (DiLoCo, outer lr 1), deterministic resume.
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
  memory-mapped data to the host's hard limit, `ulimit -n $(ulimit -Hn)`, in the
  shell that trains (a fixed value above the hard limit fails; `drafter-join
  setup` warns when the soft limit is low).

## How to help

- **Run a volunteer training site** (live, invite-only pilot).
  [Request an invite](https://github.com/Enntity/glm-flash-drafter/issues/new?template=compute-site.yml);
  no invite is needed to contribute code or docs. Sites train the whole drafter on their own data for
  short rounds and upload only weight deltas; a validator checks each delta on a private held-out set and merges
  accepted ones by token-weighted plain averaging (DiLoCo with outer learning rate 1, which matched or beat
  synchronous training in a two-site A/B). Joining is by invite: with a link from the operator, follow
  [JOIN.md](JOIN.md) (written so a coding agent can execute it). The coordinator URL is the part of your invite
  link before `/join`; its public dashboard (sites, totals, held-out curve) is at that URL. Design, costs per
  round, trust model and what we promise contributors: [docs/volunteer.md](docs/volunteer.md); running the
  coordinator and validator: [docs/operators.md](docs/operators.md).
- **Train on your own data.** A site can bring its own prompts, chat logs or agent traffic: one command
  (`scripts/byo_prepare.py`) checks the teacher rules, generates missing GLM-5.3-Flash completions, captures,
  verifies and prints the `drafter-join` command; the data never leaves the machine. Guide:
  [docs/byo-data.md](docs/byo-data.md).
- **Generate data.** Run the prompt builders and `api_generate.py` with your own key. Better, more diverse prompt
  builders (especially prose, where the drafter is weakest) are the most useful code contribution today.
- **Run capture and training locally**, on one or two GB10-class machines with your own GLM-5.3-Flash copy. The
  CPU tests exercise the full trainer, the coordinator and the validator on a tiny model with no GPU.
- **Reproduce results.** The Atlas check and the trainer's held-out evals are scripted. Every round so far,
  v81 included, also used Enntity-internal agentic and chat traffic, which is not published, so those exact runs
  cannot be repeated from this repo alone; runs on the public builders can, and we will publish theirs.

No data is ever uploaded to the network: recipe sites generate Enntity shard recipes locally, and
bring-your-own-data sites keep their data. See [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request.

## Quick start (CPU tests)

```
python -m venv .venv && . .venv/bin/activate
pip install 'torch>=2.11,<2.13' numpy safetensors transformers pyyaml
bash scripts/install_speculators.sh
pip install -e .
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m unittest discover -s tests -p test_volunteer.py -v
```

This command checks the contributor client and merge protocol on tiny CPU models.
Run the test file relevant to your change; the full suite takes tens of minutes. The volunteer-training tests are
`tests/test_diloco.py`, `tests/test_coordinator.py` and `tests/test_volunteer.py`.
