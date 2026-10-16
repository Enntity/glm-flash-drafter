# Hidden-state capture

The drafter is conditioned on the target's hidden states. Training features
must be the ones the engine feeds the drafter when serving, so they are
captured from the serving engine itself, in a prefill of each training
sequence (prompt + teacher completion).

## Engine

[`Enntity/atlas`](https://github.com/Enntity/atlas), branch
[`exp/drafter-capture-20261014`](https://github.com/Enntity/atlas/tree/exp/drafter-capture-20261014)
(commits `327655e2`, `77dc4954`, `b4251dc4`; research only, not in releases).
It adds `crates/spark-model/src/model/hidden_capture.rs`: after each capture
layer of a prefill chunk it copies the rows through the same mHC stream
contraction the DFlash feature path uses (`dflash_capture_rows`), and after
the last layer it copies the contracted final hidden (the input of the final
norm). Copies are asynchronous (device staging, pinned host buffer, writer
thread); the measured cost is about zero.

Environment:

| variable | value | meaning |
|---|---|---|
| `ATLAS_HIDDEN_CAPTURE_DIR` | e.g. `/capture` | enables the feature; segments go to `<dir>/chunks/<tag>/` |
| `ATLAS_HIDDEN_CAPTURE_LAYERS` | `5,14,24,33,42` (default) | the DFlash2 target layers, engine indexing (output of layer i) |
| `ATLAS_HIDDEN_CAPTURE_BUFFERS` | `1` | pinned host buffers (~450 MB each); two pushed a GB10 host past its memory floor |

Serving-equivalent settings (all were needed; see "Finding" below):

| setting | why |
|---|---|
| `ATLAS_GLM_DFLASH=1` | serving with a DFlash drafter runs a BF16 mHC highway; without this flag the engine runs it in FP32 and the features differ |
| `ATLAS_GLM_PREFILL_SP=1` | sequence-parallel prefill over the two ranks (needs the BF16 highway) |
| `ATLAS_NO_MTP_DRAFTER_CONTEXT=1` | SP needs no MTP prompt-hidden buffer |
| `--max-prefill-tokens=8191` | SP takes even chunks: 8192-row chunks instead of 8193 |
| no `--dflash` / `--draft-model` | no drafter is loaded, so no drafter file is opened |
| prefix cache off (template and structured rows) | every kept row is computed by the request itself |

With the prefix cache on (agentic sessions with long shared prefixes),
`scripts/capture.py --split-prefix` first prefills `ids[:from]` disarmed; the
engine caps prefix reuse at `from` while capture is armed. This was 1.4x
faster on long agentic rows but moved decode-vs-prefill agreement by up to
-0.54 points, so the validated default is no prefix cache.

The serving recipe for GLM-5.3-Flash on two DGX Sparks (image, model
conversion, profiles) is [SparkGLM](https://github.com/Enntity/sparkglm).

## Protocol

Capture is armed per request by a file on every rank's host:
`<dir>/enable` = `{"tag": "<name>", "from": <first absolute position to keep>}`.
The engine re-reads it at the start of every prefill chunk, so one server can
serve, tokenize and capture. `from = max(0, P - 2048)` where `P` is the prompt
length and 2048 is the drafter's sliding window: a drafter at anchor `a` only
reads features in `[a - 2048, a)`, and anchors are generated positions.
Rows before `from` never leave the GPU.

`scripts/capture.py` writes the enable files, sends
`POST /v1/completions {prompt: <token ids>, max_tokens: 1, temperature: 0}`,
disarms, and hands the tag to a collector that moves every rank's finished
segments into `<dest>/<tag>/` and verifies them.

## Segment format (`atlas.hidden_capture.segment/v2`)

One directory per request tag. Files:

```
r<rank>-<L|F>-p<pos0:09d>-n<rows>.bin    BF16, slot-major [slots, rows, hidden]
r<rank>-<L|F>-p<pos0:09d>-n<rows>.json   sidecar
```

- `L` segments hold the capture layers (`slots` = 5), `F` segments the final
  hidden (`slots` = 1). `hidden` = 4096.
- Under sequence-parallel prefill each rank writes the layer rows it owns
  (rank 0 the upper half of a chunk); rank 0 writes the final rows.
- A request's `L` segments, merged by absolute position, tile `[from, T)`
  exactly once; so do its `F` segments. Token ids must agree across ranks.
- A segment is written as `.part` and renamed when complete; `.err` marks a
  writer failure.

Sidecar fields: `schema`, `tag`, `rank`, `kind` (`layers` | `final`),
`layers`, `pos0`, `rows`, `hidden_size`, `dtype`, `layout`, `final_slot`,
`chunk` ([chunk start, chunk rows]), `sp`, `from`, `seq_tokens`, `token_ids`.

Size: 6 slots x 4096 x 2 bytes = **49,152 bytes per stored token** in BF16.

## FP8 storage

`python -m enntity_drafter.fp8 watch|convert|measure` converts a verified row
in place:

- the `.bin` becomes `.fp8`: e4m3 bytes `[slots, rows, hidden]` followed by
  float32 scales `[slots, rows]`, one per row per slot, `scale = amax / 448`
  (1 for an all-zero row), round to nearest even;
- the sidecar gets `dtype: fp8_e4m3`, `source_dtype: bf16`, a `scale`
  description and the measured `fp8_rel_l2` per slot;
- each segment is written as `.fp8.tmp`, read back and compared, checked
  against `--max-rel` (default 10% relative L2 per slot), renamed, and only
  then is the `.bin` deleted. An interrupted conversion resumes safely; a
  segment with non-finite values stays BF16.

Size: 6 x (4096 + 4) = **24,600 bytes per stored token** (0.50x BF16).

Measured on nine BF16 rows (2,150 generated positions; `scripts/check_capture.py --fp8`):

| | value |
|---|---|
| relative L2 per slot, layers 5 / 14 / 24 / 33 / 42 / final | 2.65 / 2.66 / 2.64 / 2.64 / 2.64 / 2.45 % |
| teacher top-1 (from the final slot), BF16 vs FP8 | 98.9% (97.7-100% per row) |
| KL(p_bf16 ‖ p_fp8), mean / worst-row p99 | 0.00075 / 0.0096 nats |
| argmax agreement with the captured next token | 83.86% BF16, 83.95% FP8 |

The trainer reads BF16 and FP8 segments (even mixed in one row);
`--fp8-device` dequantizes on the GPU, bit-identical to the CPU codec.

## Index

`python -m enntity_drafter.capture_io index` scans sidecars only and writes
`index.json`: per row the segment files, `[lo, hi)`, the first generated
position `P` (joined from `--meta` capture records or `--manifest` files), the
eligible anchor range, source, category and split (rows listed in `--heldout`
by name, id or prompt id are held out). `partition` assigns rows to ranks
(optionally stratified by source, and only rows a rank already holds when
`--resident` lists them), and `bench` measures loader throughput.

## Validation

`scripts/check_capture.py` checks, per row: tiling and token ids against the
capture records; argmax agreement of `lm_head(norm(final[i - 1]))` with token
`i` over the generated span; and, for `capture.py --gates` rows, that a repeat
is bit-exact, that mutating token 20 leaves positions < 20 bit-exact
(causality) and changes the rest.

Measured on the serving-equivalent configuration:

| check | result |
|---|---|
| token ids and positions, segments tile `[from, T)` | 228/228 rows |
| gates (32-token and 8K, SP-sharded): repeat, causal prefix, mutated suffix | pass |
| repeat capture without prefix cache | 12/12 bit-exact |
| argmax agreement, templates | 83.47% (pilot 83.42%) |

Teacher text sampled at temperature 1 agrees with the target's own argmax as
often as the target's expected top-1 probability (83.1% vs 82.9% on the pilot),
so the API text is on-policy enough to train on.

### Finding: features depend on engine numerics

A pilot run without the DFlash lane used an FP32 mHC highway; the serving
configuration uses BF16. Same tokens, relative L2 between the two per slot
(layers 5 / 14 / 24 / 33 / 42 / final): 4.7 / 7.9 / 3.3 / 5.2 / 8.4 / 10.3% on
template rows up to 24K tokens, and up to 47-52% on 9-32K-token agentic rows.
Changing only the prefill chunking (8192 vs 8193 rows) moves long-row
features by a similar amount, so a 32-token bound (0.7-3.5%) does not bound
long rows. Argmax agreement is unchanged by either. Capture with the exact
serving settings, and use agreement as the functional check.

## Throughput (two GB10 nodes, no prefix cache)

Sequence tokens per wall second over capture requests:

| row length | tok/s |
|---|---|
| < 2K | 942 |
| 2-8K | 1,890 |
| 8-16K | 2,484 |
| 16-24K | 2,559 |

Short rows are bound by the engine's fixed cost per request (about 0.3 s).
Round 2 captured 23,942 structured rows (38.6M sequence tokens) in 7.74 h of
requests.
