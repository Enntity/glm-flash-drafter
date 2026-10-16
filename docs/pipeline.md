# Pipeline

```
prompt builders ──► teacher text (API) ──► capture items ──► Atlas prefill capture ──► FP8 shards
                                                                                          │
              Atlas check ◄── export (BF16, 81 tensors) ◄── training (from scratch) ◄──────┘
```

Every step writes plain files under a data directory of your choice (the
examples use `data/`, which git ignores). Paths below are examples.

## 0. Setup

- Python 3.10+, PyTorch with CUDA on the GPU nodes (CPU works for tests).
- `bash scripts/install_speculators.sh && pip install -e .[tokenize,validate]`
- Your own copy of the GLM-5.3-Flash target (for example
  `nvidia/GLM-5.3-Flash-NVFP4`, MIT): the trainer reads its embedding, final
  norm and lm_head; prompt rendering reads its `tokenizer.json` and
  `chat_template.jinja`.
- For capture: an Atlas build of the capture branch serving the target (docs/capture.md).

## 1. Prompts

```
python scripts/generate_prompts.py templates --out data/prompts/templates
python scripts/generate_prompts.py coverage  --out data/prompts/coverage1
python scripts/generate_prompts.py coverage  --out data/prompts/coverage2 --profile long --prefix cov2 \
    --exclude data/prompts/coverage1/train.jsonl
python scripts/heldout_split.py --prompts data/prompts/templates/train.jsonl \
    --prompts data/prompts/coverage1/train.jsonl --prompts data/prompts/coverage2/train.jsonl \
    --count-sources template --out data/heldout-ids.jsonl
```

- `templates`: 1,592 code, prose, structured and tool-call prompts (round 1's
  template rows, same ids and text). Prompts matching the Atlas check's
  benchmark prompts and their topics are excluded.
- `coverage`: structured-output prompts over 24 fictional domains: strict JSON
  with `response_format` schemas, JSON extraction, OpenAI-format tool calls
  (single, parallel, follow-ups, multi-turn), config files, code edits (diff,
  SEARCH/REPLACE, whole file), Markdown tables and short code in several
  languages. Each row carries a `check` for its validator. Round 2b used
  4,000 base + 8,000 long prompts (two samples each).
- Both are deterministic: rerunning gives byte-identical files.

## 2. Teacher text

```
export OPENROUTER_API_KEY=...
python scripts/api_generate.py --prompts data/prompts/templates/train.jsonl --out data/gen/templates \
    --sources template --minutes 60 --concurrency 64 --spend-cap 20 --samples 2
```

Terms: Z.ai's own terms of use bar using its service to develop competing
models, so its endpoint is excluded from routing; generation goes through
third-party hosts with `data_collection: deny`. Read each host's terms before a
large run.

Hosted GLM-5.3-Flash is not our quantized target and hosted sampling is not
deterministic, so the text is approximately on-policy. That is acceptable:
capture re-reads every row through the target, and the KL targets come from
the target's own distribution at every position. Match the reasoning effort
your serving traffic uses (the default effort produced 91% reasoning tokens
and truncated 44% of code answers at 16K tokens).

## 3. Capture items

```
python scripts/prepare_items.py --prompts data/prompts/templates/train.jsonl \
    --generations data/gen/templates/generations.jsonl --tokenizer-dir /models/GLM-5.3-Flash \
    --out data/items/templates.jsonl
```

Drops junk generations, validates structured rows, renders each completion
with the target's chat template and tokenizer, and leaves the prompt to Atlas's
`/tokenize` at capture time.

## 4. Capture

```
python scripts/capture.py --server http://127.0.0.1:8893 --items data/items/templates.jsonl \
    --capture-dir /capture --dest data/rows/run1 --out data/capture-logs/run1 --minutes 240 --gates
python -m enntity_drafter.fp8 watch --root data/rows --log data/rows/fp8-log.jsonl --stop-file data/rows/STOP
python scripts/check_capture.py --rows data/rows/run1 \
    --records data/capture-logs/run1/capture-records.jsonl --model /models/GLM-5.3-Flash --out data/checks/run1.json
```

Run the FP8 converter beside the capture (CPU only; it converts each verified
row and removes its marker). Do not run the trainer on a node that serves the
target.

## 5. Index and train

```
python -m enntity_drafter.capture_io index --data-root data --fast 'rows/run1' \
    --meta data/capture-logs/run1/capture-records.jsonl --heldout data/heldout-ids.jsonl --out data/index
python -m enntity_drafter.capture_io partition --index data/index/index.json --world 2 --stratify source \
    --out data/index/index-w2.json
torchrun ... scripts/train.py ...     # see docs/training.md
```

`partition` also writes `files-rank<r>.txt`: the files each rank needs, to
copy to that node's local disk.

## 6. Export and check

```
python scripts/export.py --config configs/full.json --checkpoint runs/r1/checkpoints/step-0001000 --out exports/r1
# start Atlas with --draft-model exports/r1 (and once with the reference drafter), then per visit:
python scripts/atlas_check.py probe exact --tag s1 --out checks/s1
python scripts/atlas_check.py probe multi --tag s1 --out checks/s1
# save rank 0's engine log as checks/s1/rank0.log, repeat for the other arm, then:
python scripts/atlas_check.py analyze checks/s1 checks/d1 checks/d2 checks/s2
```

Atlas reads the BF16 export as is and builds its own NVFP4 copies at load.
Make sure the export is readable by the engine's user.

## Storage model

A capture-train round is self-contained: capture shards on the training
nodes' disks, train, keep the checkpoints, delete the shards. The prompts and
generations (a few GB) are what to keep: any row can be recaptured from them.
FP8 shards are raw memory-mapped arrays; general-purpose compression saves
only about 18%.
