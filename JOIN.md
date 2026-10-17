# JOIN.md: run a volunteer training site

A runbook for a person or a coding agent. Inputs: this repository and an **invite link** from the operator
(`https://<host>/<path>/join#invite=inv_...`). Run every command from the repository root on the machine that
will train. Stop at the first step whose check fails and report its output.

Background: [docs/volunteer.md](docs/volunteer.md). Only weight deltas and counters leave the site.

## Safety rules (always)

1. **GB10 memory is shared by CPU and GPU; exhausting it can hang the host.** Never start the GLM-5.3-Flash
   server and the trainer on the same node at the same time. Keep the trainer's memory guards (the client adds
   `--min-free-gb 12 --probe` unless you pass your own `--min-free-gb`).
2. Activations are deleted after each round; do not copy them elsewhere.
3. Never paste the site token (in `~/.enntity-drafter/config.json`) into chat, logs or issues.
4. Do not train on data you are not allowed to use, or on anything derived from incoai or other non-commercial
   checkpoints or data. Every upload attests to this.

## 1. Install

```
python3 -m venv .venv && . .venv/bin/activate
pip install 'torch>=2.11,<2.13' numpy safetensors transformers cryptography pyyaml tokenizers jinja2
bash scripts/install_speculators.sh
pip install -e .
drafter-join --help
```

Check: the last command prints usage starting with `usage: drafter-join`.

## 2. Register (once)

Read the terms the command prints, then:

```
drafter-join redeem '<invite link>' --accept-terms --name "<display name>"
```

Omit `--name` to appear as "anonymous". Check: it prints `registered as ...; token saved to
~/.enntity-drafter/config.json (mode 0600)`. The config also pins the network's publishing key; every global the
site downloads must carry that key's signature. An `invite_invalid` error means the link was used, revoked or
expired: ask the operator for a new one.

## 3. Decide the mode

- Two GB10-class GPUs with the 200G link, a GLM-5.3-Flash copy and the Atlas capture build: **recipe mode**
  (default). The site claims an Enntity shard recipe, generates teacher text and captures it locally.
- Your own captured data (an `enntity.drafter_rows/v1` store or a capture index): **bring-your-own mode**,
  `--mode byo --byo-features DIR` (or `--byo-capture-index FILE --byo-capture-root DIR`), optionally
  `--tag <label>`. Your data never leaves the machine. Nothing is generated or captured in this mode.

With a capture index (for example one built by `python -m enntity_drafter.capture_io index` over captures you
already have), the client writes its own index under `<work>/byo-index/` and reads the capture files in place:
only rows your index marks `train` are used; rows it marks held-out are left alone; up to 64 of the remaining rows
(a fixed hash carve, as in recipe shards) become the public check set the self-check evaluates; and rows whose
name, id or prompt id is listed in a `--byo-exclude-ids FILE` (JSONL `{"id": ...}`, repeatable) are never trained or
evaluated, for example an operator's private held-out rows that sit on the same disks. The run log prints how many
rows were kept and dropped.

Recipe mode also needs: `--teacher DIR` (your GLM-5.3-Flash, for example `nvidia/GLM-5.3-Flash-NVFP4`),
`--tokenizer-dir DIR` (the same directory), an OpenAI-compatible GLM-5.3-Flash endpoint (`--teacher-url URL`, or an
`OPENROUTER_API_KEY` in the environment), the Atlas capture server (`--atlas-server URL`, `--capture-dir DIR` per
rank; engine branch `exp/drafter-capture-20261014`, see docs/capture.md), and `--before-capture` /
`--after-capture` commands that start and stop it, so the trainer never shares a node with it.

## 4. Preflight, smoke and canary

```
drafter-join setup --teacher <GLM dir> --model-dir <GLM dir> [--peer <second node>] [--atlas-image <image>] \
    [mode flags from step 3]
```

Expected: one line per check, `PASS`, `WARN` or `FAIL`, then `PASS smoke (tiny train + delta checks)`, then a
canary (20 training steps from the current global, uploaded and fully validated, a few minutes) ending in
`canary ... accepted: this site is ready`. Fix every `FAIL` with the hint on its line and rerun. `WARN` lines are
advisory (for example no RDMA device in bring-your-own mode). A `version mismatch` or `terms changed` message:
run `drafter-join update --yes`, then rerun setup.

## 5. Run

```
drafter-join run --teacher <GLM dir> [mode flags]
```

It loops until stopped: check versions, claim (recipe) or use your data (byo), pull the global, train one round
(about 30 minutes), self-check, upload, clean up. Progress lines show the phase and step. Stop with Ctrl-C at any
time; rerunning resumes the interrupted phase (training resumes from its last checkpoint, uploads resume
part by part).

## 6. Verify

```
drafter-join status
```

Check: `coordinator.deltas` lists your deltas with `state` `pending`, `accepted` or `merged`. `quarantined` comes
with `reason_code` and `reason`; quarantined deltas are kept and re-validated if the validator is fixed, and your
tokens stay credited. The operator's dashboard (`https://<host>/<path>/`) shows your site and totals.

## Troubleshooting

| message | action |
|---|---|
| `version mismatch` | `drafter-join update --yes`, then `drafter-join run` (a new canary runs first) |
| `self-check failed before upload` | nothing was uploaded; read the code (layout, non_finite, norm_high ...) and report it with the trainer log path it prints |
| `training failed ... see <log>` | read the log; rerun to resume; out-of-memory: lower `--anchors` via `--trainer-arg` or stop other processes |
| `no recipe shard is open` | the client waits and retries; or switch to bring-your-own mode |
| `waiting (...)` | nothing to do: the finished delta is kept and the client continues by itself (for example it runs a canary first, then uploads the delta) |
| `... The delta is kept at <work>/unmergeable/...` | the network changed in a way that makes that delta unusable; it is kept for inspection and the next round starts |
| `coordinator unreachable` | network issue; the client retries with backoff; rerun later |
