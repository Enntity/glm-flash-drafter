# Bring your own data

This page takes you, or your coding agent, from raw data (chat logs, agent traffic, a list of prompts, or existing
GLM-5.3-Flash conversation logs) to a volunteer training site that trains on that data. One command,
`scripts/byo_prepare.py`, chains every step and stops with a message that says what to fix. The data stays on your
machine. Only weight deltas and a few counters are uploaded (see [Privacy](#7-privacy)).

```
your files ─► validate ─► generate ─► items ─► capture ─► check ─► index ─► drafter-join run --mode byo
              (rules,     (GLM-5.3-   (GLM chat  (Atlas      (alignment, (capture
               scrub,      Flash, for  template,  capture     GLM         index)
               dedupe)     prompts     token ids) build, FP8) agreement)
                           without a
                           completion)
```

Before you start, you need:

- a registered site ([JOIN.md](../JOIN.md) steps 1-2) and the install from JOIN.md step 1;
- your own copy of the GLM-5.3-Flash checkpoint (for example `nvidia/GLM-5.3-Flash-NVFP4`). The scripts read its
  `tokenizer.json` and `chat_template.jinja`, and the check reads its final norm and `lm_head`;
- for prompts that have no completion: a GLM-5.3-Flash endpoint (your own server, or an API key; see
  [section 2](#2-teacher-text-rules));
- for capture: the Atlas capture build serving GLM-5.3-Flash on one or two GB10-class GPUs
  ([section 4](#4-capture-with-the-atlas-capture-build)). Training then needs one GPU on a node that is **not**
  serving the target;
- disk for the captured features, about 24.6 KB per stored token ([estimate](#disk)).

Every command below runs from the repository root.

## Quick start

1. Check the input and see the plan. A dry run writes nothing:

   ```
   python scripts/byo_prepare.py --input my-chats.jsonl --work data/byo --teacher /models/GLM-5.3-Flash \
       --teacher-url http://<glm-host>:8000/v1/chat/completions --capture-dir /capture --dry-run
   ```

   Expected output (your counts will differ; long command lines are shortened here):

   ```
   [byo] validate: 7 records -> 7 prompts, 1 reused GLM-5.3-Flash completions, 6 prompts without one
   [byo] scrub: email 1 in 1 records, not changed (review, then rerun with --scrub redact or --scrub drop)
   [byo] estimate (GLM tokenizer; 3,000 tokens per completion still to generate): 7 sequences, 18,356 stored tokens -> 0.45 GB FP8 on disk (rows are BF16, 49,152 B/token, until converted)
   [byo] generate: 6 prompts x1 -> http://<glm-host>:8000/v1/chat/completions (spend cap $10 per run)
       python scripts/api_generate.py --prompts data/byo/gen-prompts.jsonl --out data/byo/gen --sources byo ...
   [byo] items: render + tokenize with the GLM chat template
       python scripts/prepare_items.py --prompts data/byo/prompts.jsonl --generations ... --out data/byo/items.jsonl
   [byo] capture: Atlas capture build at http://127.0.0.1:8893 (docs/capture.md), FP8 conversion alongside
       python scripts/capture.py --server http://127.0.0.1:8893 --model glm-5.3-flash-atlas --items ...
       python -m enntity_drafter.fp8 watch --root data/byo/rows --log data/byo/fp8-log.jsonl --stop-file ...
   [byo] check: token alignment (every row) + GLM argmax agreement (sampled rows)
       python scripts/check_capture.py --rows data/byo/check-sample --records ... --model /models/GLM-5.3-Flash
   [byo] index: the capture index drafter-join reads
       python -m enntity_drafter.capture_io index --data-root data/byo --fast rows --meta ... --out data/byo/index
   [byo] dry run: nothing was written
   ```

   A problem in a later stage (no endpoint configured, missing checkpoint files) shows up here as
   `[byo] <stage> would stop: ... (fix: ...)`.

2. Run it. On the capture node, give the commands that start and stop your Atlas capture server, so the target is
   down before anything else uses that memory:

   ```
   python scripts/byo_prepare.py --input my-chats.jsonl --work data/byo --teacher /models/GLM-5.3-Flash \
       --teacher-url http://<glm-host>:8000/v1/chat/completions --capture-dir /capture \
       --before-capture './start-atlas-capture.sh' --after-capture './stop-atlas.sh'
   ```

   The output has one line per stage (placeholders in angle brackets; your counts depend on your data):

   ```
   [byo] validate: <records> records -> <prompts> prompts, <reused> reused GLM-5.3-Flash completions, <missing> prompts without one
   [byo] generate: <missing> prompts x1 -> <teacher-url> (spend cap $10 per run)
   [byo] generate: <missing>/<missing> prompts now have completions
   [byo] items: <sequences> sequences, <tokens> tokens (locally rendered)
   [byo] capture: <n> of <n> sequences to capture
       <row-name> T=<len> from=<start> <rate> tok/s
       collected <n> of <n> rows; bad: []
   [byo] check: alignment: <n> rows kept, 0 rejected; sampled <k>: <k>/<k> tile [from, T) with the capture records
   [byo] check: agreement: <pct>% over <positions> generated positions (genuine GLM-5.3-Flash text at T=1: about 83-86%)
   [byo] index: <rows> rows, <stored> stored tokens, <anchors> eligible anchors, <gb> GB
   [byo] ready. First time on this site, then every round:
       drafter-join setup --teacher /models/GLM-5.3-Flash --mode byo --byo-capture-index <abs>/index/index.json --byo-capture-root <abs>
       drafter-join run   --teacher /models/GLM-5.3-Flash --mode byo --byo-capture-index <abs>/index/index.json --byo-capture-root <abs>
   ```

   The one number to check is agreement: well below the 83-86% band means the completions are not
   GLM-5.3-Flash output (fix that before training). Measured throughput on GB10 hardware is in
   [docs/results.md](results.md).

3. Run the two printed commands (JOIN.md steps 4-6) on a node that is not serving GLM-5.3-Flash.

**If your only GLM-5.3-Flash is the capture build itself**, let it be the teacher as well (one Atlas server can
serve chat completions, tokenize and capture) and never start a second GLM server beside it. Three steps:

```
# 1. start the Atlas capture build yourself (section 4), then generate and capture against it
python scripts/byo_prepare.py --input my-chats.jsonl --work data/byo --teacher /models/GLM-5.3-Flash \
    --teacher-url http://127.0.0.1:8893/v1/chat/completions --teacher-model glm-5.3-flash-atlas \
    --capture-dir /capture --until capture
# 2. stop the Atlas server (frees the memory)
# 3. the same command without --until: generation and capture are done, so it checks and indexes
python scripts/byo_prepare.py --input my-chats.jsonl --work data/byo --teacher /models/GLM-5.3-Flash \
    --teacher-url http://127.0.0.1:8893/v1/chat/completions --teacher-model glm-5.3-flash-atlas \
    --capture-dir /capture
```

The command is **resumable**. Rerun it unchanged after an interruption, a time window that ran out
(`--gen-minutes`, `--capture-minutes`) or a fix. Finished work is never redone: prompts that already have a good
completion are not sent again, captured rows are not captured again, and a fully prepared directory only re-checks
and re-indexes. `--until STAGE` stops after a stage, for example `--until items` on a machine without a GPU. Exit
codes: 0 ready (or the requested stage done), 2 something to fix (the message ends with `[byo] fix: ...`).

## 1. Accepted input formats

Pass one or more files with `--input` (repeatable). The format is chosen per line or file:

| input | example | used as |
|---|---|---|
| chat JSONL, one conversation per line | `{"messages": [...], "tools": [...], "reasoning_effort": "high"}` | prompt; a final assistant turn is a reused completion |
| request/response log JSONL (what an OpenAI-compatible proxy records) | `{"request": {"model": ..., "messages": [...]}, "response": {"model": ..., "provider": ..., "choices": [...]}}` | same; `response.model` and `response.provider` are checked |
| prompt JSONL | `{"prompt": "...", "system": "optional"}` or a bare JSON string per line | prompt to generate |
| plain text, `.txt` | one prompt per non-empty line | prompt to generate |
| JSON array, `.json` | `[record, record, ...]` | each element as above |

Chat records (also the `request` object of a log):

| field | required | meaning |
|---|---|---|
| `messages` | yes | OpenAI format. Roles `system` (`developer` is read as `system`), `user`, `assistant`, `tool`. `content` is a string or a list of `{"type": "text", "text": ...}` parts; image or audio parts are refused |
| assistant `reasoning_content` or `reasoning` | for reused completions | GLM-5.3-Flash's thinking trace. A turn whose `content` holds `<think>...</think>` (a server without a reasoning parser) is split automatically |
| assistant `tool_calls` | no | `[{"id", "type": "function", "function": {"name", "arguments"}}]`; `arguments` may be a JSON string or an object |
| tool `tool_call_id` | no | as in the OpenAI API |
| `tools`, `tool_choice`, `response_format` | no | as sent to the model; they are part of the rendered prompt |
| `reasoning_effort` (or `reasoning: {"effort": ...}`) | no | the effort the turn was generated with; `--effort` fills rows that have none |
| `model` (or `teacher`) | for reused completions | the model that wrote the assistant text; see section 2 |
| `provider` | no | the host that served it (OpenRouter logs carry this) |
| `finish_reason` (on the assistant turn, or `choices[0].finish_reason`) | no | `length` marks a truncated reply (no stop token is added); default `tool_calls` when the turn calls tools, else `stop` |
| `id`, `category`, `source` | no | labels, kept locally only. `category` defaults to `agentic` (tools present), `structured` (`response_format`) or `chat` |

A full chat record with a reused GLM-5.3-Flash completion:

```json
{"id": "support-17", "model": "glm-5.3-flash", "reasoning_effort": "high",
 "tools": [{"type": "function", "function": {"name": "lookup_order", "parameters": {"type": "object",
           "properties": {"id": {"type": "integer"}}}}}],
 "messages": [
  {"role": "system", "content": "You are the support agent for a hardware store."},
  {"role": "user", "content": "Where is order 7?"},
  {"role": "assistant", "content": "", "reasoning_content": "The user wants order 7. I should look it up.",
   "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "lookup_order", "arguments": "{\"id\": 7}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "{\"status\": \"shipped\", \"eta\": \"Friday\"}"},
  {"role": "assistant", "content": "Order 7 has shipped and should arrive on Friday.",
   "reasoning_content": "The tool says shipped, ETA Friday. Answer briefly."}]}
```

What becomes a training sequence:

- **Prompts without a completion** (a conversation ending with a `user` or `tool` turn, or a prompt list) are sent
  to GLM-5.3-Flash, `--samples N` times each (default 1).
- **Conversations ending with an assistant turn** reuse that turn as the completion (`--turns last`, the default).
  With `--turns all`, every assistant turn of the conversation is a completion with the conversation before it as
  the prompt, so a 10-step agent session gives 10 sequences. Each sequence is captured with its whole prompt, so
  `all` costs more capture time on long sessions; it is the best use of agent logs.
- **Duplicates** are dropped: identical prompts (same messages, tools and effort, after scrubbing) collapse into
  one; identical completions of a prompt count once; distinct completions of one prompt are kept as samples.
- Each prompt gets the id `<your id or "byo">-<first 12 hex digits of the SHA-256 of its content>`. Adding input
  files later processes only the new prompts. Rows of prompts you removed are moved to `<work>/stale/` and not
  indexed.

A record that breaks a rule stops the run with its file and line and a fix, for example
`chats.jsonl:41: assistant text model 'gpt-x' is not GLM-5.3-Flash; use --regenerate to replace it ...`.
`--skip-bad` leaves such records out instead (they are counted).

## 2. Teacher text rules

The drafter learns to predict what GLM-5.3-Flash will write next, and the network's attestation, signed with every
upload, states that "Teacher completions and captured features came only from GLM-5.3-Flash" (participation terms
in `coordinator/terms.py`). So **every completion must be GLM-5.3-Flash output**. Three ways to get it:

1. **Your own GLM-5.3-Flash server** (vLLM, SGLang or Atlas serving your checkpoint, OpenAI-compatible):
   `--teacher-url http://<server>/v1/chat/completions` and `--teacher-model <served name>`. Start the server with
   its GLM reasoning parser, so replies return the thinking as `reasoning_content` (or `reasoning`). Generation
   stops with an error if most completions come back without a thinking trace.
2. **An API key of your choice**: without `--teacher-url`, scripts/api_generate.py uses OpenRouter with the key in
   `OPENROUTER_API_KEY` (or `--teacher-key-env NAME`). The Z.AI endpoint is excluded from routing (its terms bar
   using the service to develop competing models) and `data_collection: deny` is requested. Read each host's terms
   before a large run. `--spend-cap` (default $10) caps each run; rerunning continues with the prompts that are
   still missing.
3. **Existing logs**, only if they are GLM-5.3-Flash output. A record's `model` (or `response.model`) must name
   GLM-5.3-Flash (`glm-5.3-flash`, `z-ai/glm-5.3-flash` and similar spellings match). Records without a model field
   need `--completions-are-glm`, your statement that their assistant text is GLM-5.3-Flash output.

Never use:

- **outputs of any other model**, not even as earlier turns of a conversation. With `--regenerate`, a record whose
  assistant text is foreign or unattested is cut before its first assistant turn and that turn is generated fresh;
  without it, the run stops;
- **anything from `incoai/GLM-5.3-Flash-DFlash2`** (CC BY-NC-ND): not its weights, not text generated with it, not
  data derived from it. Records whose model names `incoai` or `DFlash` are refused;
- **text served by the Z.AI endpoint**, or by any host whose terms forbid training competing models. Records whose
  `provider` is `Z.AI` are refused;
- data you are not allowed to use for training an Apache-2.0 model, or other people's private data without their
  permission (the attestation covers this as well).

The checks above read fields; the [agreement check](#5-build-and-verify-the-store) after capture tests the text
itself, and text that is not GLM-5.3-Flash output fails it.

**Thinking.** GLM-5.3-Flash is a thinking-only model: a completion is its reasoning, `</think>`, then the answer or
tool calls, and the drafter is trained on all of it. A reused turn logged without its thinking does not match what
the model wrote and is refused unless you pass `--allow-missing-reasoning`. Prefer `--regenerate` instead.

**Effort and sampling.** Match the reasoning effort your own traffic uses. Higher effort means longer thinking
(the default effort produced 91% reasoning tokens and truncated 44% of code answers at 16K tokens,
[pipeline.md](pipeline.md)). Use the effort names your checkpoint's `chat_template.jinja` accepts (for example
`high`). Give every row an effort (`--effort` fills rows that have none): a row without one is generated at the
server's default but rendered at capture with the template's default (`max`), and Atlas renders a tool request
without an effort as thinking-off. Generation uses the serving defaults: temperature 1.0, top_p 0.95, `--max-tokens` 16384. A reply cut
at the limit is kept without a stop token.

## 3. Rendering and tokenization

The training sequence must be token-for-token what GLM-5.3-Flash sees when serving. Nothing about the template is
vendored in this repository: everything comes from your checkpoint directory (`--teacher`, or `--tokenizer-dir`
if the template files live elsewhere).

- `enntity_drafter/glm_render.py` loads the checkpoint's `tokenizer.json` and `chat_template.jinja` through
  transformers' chat-template renderer. `completion_text()` renders the conversation with and without the new
  assistant turn and keeps what follows the generation prompt `<|assistant|><think>`: the reasoning, `</think>`,
  the answer and any tool calls, exactly as the template writes a past assistant turn.
- `scripts/prepare_items.py` turns each prompt and completion into a capture item:
  - **completion ids**: that completion text tokenized with your tokenizer, followed by the stop token the model
    would emit: `<|observation|>` after tool calls, `<|user|>` after a natural stop, none when the reply was
    truncated;
  - **prompt ids**: rendered by the Atlas server's own `/tokenize` at capture time (messages, tools, effort), so the
    prompt is the serving template's tokens, not a local re-implementation;
  - sequences longer than 49,152 tokens are truncated; junk generations (errors, host error text, empty answers)
    are dropped and listed in `<work>/items-validity.jsonl`.

Output: `<work>/items.jsonl` (token ids plus the rendering request; your text stays in this local file).

## 4. Capture with the Atlas capture build

The drafter is conditioned on the target's hidden states, so they are captured from the serving engine itself, in
a prefill of each sequence. The protocol, formats and measurements are in [capture.md](capture.md); this section
is what you need to run it.

**Engine.** [Enntity/atlas](https://github.com/Enntity/atlas), branch `exp/drafter-capture-20261014` (research
only, not in releases). The serving recipe for GLM-5.3-Flash on two DGX Sparks (image, model conversion,
profiles) is [SparkGLM](https://github.com/Enntity/sparkglm). Start the server with your usual SparkGLM
profile plus:

| setting | value |
|---|---|
| `ATLAS_HIDDEN_CAPTURE_DIR` | a directory on each rank's host, for example `/capture` (pass each with `--capture-dir`) |
| `ATLAS_HIDDEN_CAPTURE_BUFFERS` | `1` (each pinned buffer is about 450 MB; two pushed a GB10 host past its memory floor) |
| `ATLAS_GLM_DFLASH=1`, `ATLAS_GLM_PREFILL_SP=1`, `ATLAS_NO_MTP_DRAFTER_CONTEXT=1`, `--max-prefill-tokens=8191` | serving-equivalent numerics; without them the features differ from serving |
| no `--dflash` / `--draft-model`, prefix cache off | no drafter loaded; every kept row is computed by its own request |

Put the start and stop commands in two scripts and pass them as `--before-capture` and `--after-capture`. The
start command must return once the server is launching (for example `docker run -d ...`), not run in the foreground.
`byo_prepare.py` runs the first, waits up to `--server-wait` seconds (default 900) for `GET /v1/models` to answer,
captures, and always runs the second, also after an error or Ctrl-C. Without `--before-capture` it expects a
server that is already running at `--atlas-server` (default `http://127.0.0.1:8893`, served name
`--atlas-model glm-5.3-flash-atlas`) and waits at most 15 seconds for it.

**Two nodes.** With TP2 each rank writes its own segments. Run `byo_prepare.py` on a host that sees both capture
directories (mount the second rank's directory, for example over NFS) and pass `--capture-dir` once per rank.

**Memory safety on GB10.** CPU and GPU share memory, and exhausting it can hang the host.

- Never run the GLM-5.3-Flash server and the trainer on the same node at the same time. Use `--after-capture` to
  stop the server, and run `drafter-join` on a node that is not serving.
- Never run two GLM-5.3-Flash servers on one node: a separate teacher server for generation and the capture build
  do not fit together. Generate on another machine, through an API key, or with the capture build itself (see
  the quick start).
- The check stage reads the target's final norm, `lm_head` and embedding (a few GB). It runs after
  `--after-capture` has stopped the server.
- `drafter-join` adds the trainer's guards (`--min-free-gb 12 --probe`) unless you pass your own `--min-free-gb`.

**FP8 storage.** A row is written in BF16 (49,152 bytes per stored token) and converted in place to FP8 e4m3 with
per-row scales (24,600 bytes per token) by `python -m enntity_drafter.fp8 watch`, which runs beside the capture.
Teacher top-1 from the FP8 final slot matches BF16 98.9% of the time (capture.md). A row that fails the
conversion check stays BF16 and still trains.

<a id="disk"></a>**Disk.** Only the last 2,048 prompt tokens (the drafter's window) and the completion are stored:

```
stored tokens per sequence = min(prompt tokens, 2048) + completion tokens
disk = stored tokens x 24.6 KB (FP8), plus BF16 headroom for rows not yet converted
```

| data | stored tokens | disk (FP8) |
|---|---|---|
| 1,000 chats, 500-token prompts, 2,000-token completions | 2.5M | 62 GB |
| 2,000 agent steps (`--turns all`), long prompts, 1,500-token completions | 7.1M | 175 GB |
| 10,000 mixed prompts, 3,000-token completions, long prompts | 50.5M | 1.24 TB |

`--dry-run` prints this estimate for your input, using your tokenizer when `--teacher` is given.

**Time.** On two GB10 nodes without prefix cache, capture runs at about 950 sequence tokens per second for rows
under 2K tokens and about 1,900-2,550 for longer rows (about 0.3 s fixed cost per request). Round 2 captured 38.6M
sequence tokens in 7.7 hours. `--capture-minutes` (default 240) bounds one run; rerun to continue.

Optional capture flags pass through with `--capture-arg`, for example `--capture-arg=--gates` on your first run
(repeat and causal-prefix gates, checked by the check stage) or `--capture-arg=--split-prefix` for long shared
prefixes with the prefix cache on (faster, but not the validated default; see capture.md).

## 5. Build and verify the store

**Alignment, every row.** As each request finishes, the capture collector checks that the row's segments cover
positions `[from, T)` exactly once and carry the item's token ids. A row that fails (`token_ids_differ`,
`coverage`, `writer_error`, or `unverified` when a killed capture left no record) is moved to `<work>/rejected/` and never indexed. To retry such rows, delete their
directories under `rejected/` and rerun.

**Agreement, sampled rows.** `scripts/check_capture.py` reloads `--check-rows` rows (default 200, a fixed sample),
checks that they tile their capture records, and computes argmax agreement: how often
`lm_head(norm(final_hidden[i-1]))`, the target's own top prediction, equals the token actually at position `i`,
over every generated position. This tests the text. For genuine GLM-5.3-Flash completions sampled at T=1,
agreement is about 83-86% (template rows measured 83.5%, equal to the model's expected top-1 probability); text
sampled at a lower temperature scores higher. Text from another model, edited text, or a mismatched template or
effort scores clearly lower.

| result | verdict | what happens |
|---|---|---|
| all sampled rows tile and agreement >= 80% | ok | index built, command printed |
| agreement 78-80% | warn | index built; check the input's origin and effort |
| agreement < `--min-agreement` (default 0.78), a row that does not tile, or a failed `--gates` check | fail | stops before indexing: `[byo] fix: ...` |
| `--skip-agreement` | warn | unverified; use only to test the pipeline |

Rows of at least 128 generated tokens with agreement under 60% are listed (worst first) in the output and in
`<work>/check-verdict.json`, so you can find the inputs they came from (`origin` in `<work>/prompts.jsonl`).

**Index.** `python -m enntity_drafter.capture_io index` scans the row sidecars and writes `<work>/index/index.json`
and `tokens.i32`: per row its segment files, stored range, first generated position (from the capture records),
eligible anchors and FP8/BF16 storage. Every row is marked `train`. `drafter-join` carves its own public
self-check set from these rows (a fixed hash carve of at most 64 rows, as in recipe shards), so you do not hold
any out yourself. The index reads the captures in place: keep the work directory where it is while you train.

Files in the work directory:

| path | contents |
|---|---|
| `prompts.jsonl`, `validate-report.json` | normalized prompts (with their input `origin`); counts, scrub hits, problems |
| `reused/generations.jsonl`, `gen/generations.jsonl`, `gen/gen-summary.json` | completions from your logs; completions generated here |
| `items.jsonl`, `items-validity.jsonl` | capture items; dropped generations and why |
| `rows/<name>/` | captured features (FP8); `rejected/`, `stale/` rows that are not indexed |
| `capture-logs/capture-records.jsonl`, `capture-logs/alignment.jsonl` | per-request records; per-row alignment results |
| `check.json`, `check-verdict.json` | check output and verdict |
| `index/index.json` | what `drafter-join` reads |
| `byo-ready.json`, `logs/byo.log` | the printed commands; full output of every step |

## 6. Run

Use the printed commands (JOIN.md steps 4-6):

```
drafter-join setup --teacher /models/GLM-5.3-Flash --mode byo --byo-capture-index <work>/index/index.json \
    --byo-capture-root <work> [--tag <label>]
drafter-join run   --teacher /models/GLM-5.3-Flash --mode byo --byo-capture-index <work>/index/index.json \
    --byo-capture-root <work> [--tag <label>]
```

Bring-your-own options of `drafter-join` (as of this commit):

| option | meaning |
|---|---|
| `--mode byo` | train on your data; nothing is generated or captured by `drafter-join` |
| `--byo-capture-index FILE` and `--byo-capture-root DIR` | the index this page builds and the directory its paths are relative to (`--work`) |
| `--byo-features DIR` | instead: an `enntity.drafter_rows/v1` row store (`enntity_drafter/data.py`), if you already have one |
| `--byo-exclude-ids FILE` | JSONL `{"id": ...}` of rows never to train or self-check on (repeatable); for example private held-out rows on the same disks |
| `--tag LABEL` | an opaque label for your data, shown to the operator (pass `--tag` to `byo_prepare.py` and it is added to the printed commands) |

`drafter-join` writes its own index under `<work dir of drafter-join>/byo-index/` and never deletes or moves your
data. Each round trains about 30 minutes and uploads one delta. To add data later, add input files, rerun
`byo_prepare.py` with the same `--work` (only new prompts are generated and captured), then restart
`drafter-join run`.

## 7. Privacy

- **What leaves your machine for the network:** weight deltas and the metadata the protocol defines: mode, your
  optional tag, token and step counts, trainer version, base hash and a signed data-source attestation; heartbeats
  carry only phase, round, version, token counters, throughput, GPU power, MFU and GPU count. No prompts,
  completions, activations or files are uploaded (participation terms, item 1).
- **Teacher generation is the exception you control.** Prompts that need completions are sent to the endpoint
  you choose. On your own server they stay on your machines; with an API key they go to that host under its terms.
  Use your own server, or scrub first, for anything sensitive.
- **Deltas can memorize.** A delta is a function of your data. A drafter is small and trained on token
  prediction, but in principle rare strings it saw often (a key, a name, an address repeated across many rows) can
  be recovered from weights. Remove what you would not want memorized before you prepare the data.
- **Scrub helper.** Every run scans prompts and completions with a regex backstop for private keys, API keys
  (`sk-`, `ghp_`, `hf_`, `AKIA` and others), JWTs and bearer tokens, credentials in URLs, `password=`-style
  values, e-mail addresses (except `example.com` and other reserved domains), IPv4 addresses (except loopback and
  documentation ranges), phone numbers with a country code, card numbers (Luhn-checked) and US SSNs. It reports
  counts by kind and never prints the matches.
  - `--scrub report` (default): count only;
  - `--scrub redact`: replace each match with `[REDACTED:<kind>]` before anything is generated or rendered.
    Best for prompts that still need a completion, which GLM-5.3-Flash then writes from the redacted text;
  - `--scrub drop`: leave out every record with a match. Best for reused completions, because editing GLM's own
    text lowers agreement;
  - `--scrub off`.

  Regexes do not find names, street addresses or free-form personal details, and they flag some harmless strings
  (version numbers that look like IP addresses, long digit runs). Review the report, and for personal data use a
  dedicated PII tool (for example Microsoft Presidio) or manual review before preparing.
- **The work directory holds your text** (prompts, completions, items) and the features. Keep it private, and
  delete it when you stop contributing. `drafter-join` deletes only activations it generated itself, never yours.

## 8. What data helps most

- **The dashboard's "wanted" list.** The coordinator's public page (`https://<host>/<path>/`, or JSON at
  `https://<host>/<path>/api/public/dashboard`, key `wanted`) shows the categories where the merged drafter is
  weakest on the held-out set (`wanted.weakest`, lowest expected accepted length first) and the operator's notes
  (`wanted.notes`). Data in those categories moves the network most.
- **Your real traffic first.** The drafter is only as good as its match to what GLM-5.3-Flash is asked in
  practice. Agent sessions with tool calls (use `--turns all`), code, structured output (JSON, configs, tables)
  and prose all matter. Relative to the DFlash2 reference, round 2b was furthest behind on structured output, then
  code, then prose (0.65x / 0.73x / 0.76x, [results.md](results.md)); absolute acceptance is lowest on prose.
  Multilingual text and long contexts are under-represented in the public builders.
- **Mix.** Do not let one template or one task dominate. Many near-identical prompts (the same form with
  different names) teach less than varied ones; cap them. Keep categories roughly in proportion to the traffic you
  want the drafter to serve.
- **Size per round.** A round is H inner steps (default 500). With round 2b's settings (512 anchors x 4
  micro-steps per step) it draws about 1M anchor positions, and every generated token is one eligible anchor per
  pass. So about **1M generated tokens** (roughly 300-500 thinking completions of 2-3K tokens) is one pass per
  round. More is better; more than about two passes over the same data stops helping (round 2b capped small
  sources at 2 epochs). The canary (20 steps) runs on any non-empty index.
- **Dedupe and leakage.** Exact duplicates are removed for you. Do not include benchmark or evaluation prompts:
  the Atlas check's prompts and topics are listed in `enntity_drafter/prompts/templates.py` (`EXCLUDED_EXACT`,
  `EXCLUDED_TOPIC`). Training on them would inflate the published numbers.

## 9. Troubleshooting

| message | cause | fix |
|---|---|---|
| `assistant text unattested; pass --completions-are-glm ...` | a record has assistant turns and no `model` field | if it is GLM-5.3-Flash output, `--completions-are-glm`; otherwise `--regenerate` |
| `assistant text model '...' is not GLM-5.3-Flash` / `non-commercial` / `served by Z.AI` | another model, the incoai checkpoint, or the Z.AI endpoint wrote it | `--regenerate` (cuts it, even as context) or remove the records |
| `assistant turn without its thinking trace` | the logger dropped `reasoning_content` | log the reasoning, or `--regenerate` |
| `N input record(s) failed validation` | malformed lines (listed with file:line) | fix them, or `--skip-bad` |
| `scrub: ... not changed` | the regex backstop found secrets or contact data | review, then `--scrub redact` or `--scrub drop` |
| `no endpoint is configured` | prompts need completions | `--teacher-url` (your server) or set `OPENROUTER_API_KEY` |
| `only N of M generated completions carry a thinking trace` | your server returns thinking inside `content`, or not at all | enable the GLM reasoning parser, delete `<work>/gen/`, rerun |
| `generate: K/N prompts now have completions; ... failed` | errors, junk replies, spend cap or time window | rerun to retry; see `gen/gen-summary.json` |
| `lacks tokenizer.json, chat_template.jinja` | `--teacher` is not the checkpoint directory | point `--teacher` (or `--tokenizer-dir`) at it |
| `items: ... dropped {'render_error': ...}` | a conversation the template cannot render (for example a tool reply without its call) | fix those records (`items-validity.jsonl` names them) |
| `capture needs --capture-dir` / `capture dir ... does not exist` | capture is not configured | pass each rank's `ATLAS_HIDDEN_CAPTURE_DIR` as seen from this host |
| `Atlas server not reachable` | the server is not up, still loading, or on another port | check `--before-capture`, `--server-wait`, `--atlas-server` |
| `prompt does not end with the generation prompt` (in `logs/byo.log`) | the server is not the GLM-5.3-Flash capture build or its template differs | serve your checkpoint with the capture branch (section 4) |
| `collected X of Y rows; bad: [...]` / `rejected {'coverage': ...}` | an engine writer error or a missing rank directory | check every rank's `--capture-dir`; delete `rejected/<name>` and rerun |
| `rejected {'unverified': ...}` | the capture was killed before its collector recorded the row | delete `rejected/<name>` and rerun |
| `rejected {'token_ids_differ': ...}` | the engine computed different tokens than the item | the template or checkpoint differs between `--teacher` and the server; use the same copy |
| `agreement X% < 78%` | the text is not GLM-5.3-Flash output, or the effort or template does not match | regenerate those inputs with GLM-5.3-Flash; the listed rows point at them |
| `the index has no trainable rows` | every row was rejected, or no generated span fits the window | read `index/index.json` `skipped_rows` |
| out of memory or a hung host | target server and trainer on one node | stop the server (`--after-capture`); keep the trainer guards |
| old rows after changing the checkpoint or template | row ids hash the prompt, not the template | start a fresh `--work` |
