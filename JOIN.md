# JOIN.md: run a volunteer training site

A runbook for a person or a coding agent. Inputs: this repository and an **invite link** from the operator
(`https://<host>/<path>/join#invite=inv_...`). Run every command from the repository root on the machine that
will train. Stop at the first step whose check fails and report its output.

Background: [docs/volunteer.md](docs/volunteer.md). Only weight deltas and counters leave the site.

## Safety rules (always)

1. **GB10 memory is shared by CPU and GPU; exhausting it can hang the host.** Never start the GLM-5.3-Flash
   server and the trainer on the same node at the same time. Keep the trainer's memory guards (on a GPU the client
   adds `--min-free-gb 12 --probe-headroom-gb 4 --low-memory pause` unless you pass your own values, and always
   `--probe`). Below the floor the trainer pauses: it saves a checkpoint and exits, and the client resumes the
   round once memory has been back for a minute, so a dip costs at most the unfinished step.
2. **Use a dedicated host if you can.** On a shared GB10, other users' containers and services come and go and
   move tens of GB (a ~28 GB build container is enough to push a training host from 113 GB available to under
   8 GB). Before joining, check what else runs (`free -g`, `nvidia-smi`, `docker stats --no-stream`) and expect
   their peaks while you train; `drafter-join setup` samples memory for a minute and counts other GPU and large
   processes. A shared host trains, but pauses through every peak.
3. Activations are deleted after each round; do not copy them elsewhere.
4. Never paste the site token (in `~/.enntity-drafter/config.json`) into chat, logs or issues.
5. Do not train on data you are not allowed to use, or on anything derived from incoai or other non-commercial
   checkpoints or data. Every upload attests to this.

## 1. Install

Prerequisites: Python 3.10 or newer **with its development headers** (`Python.h`) and a C compiler. Triton compiles
the trainer's GPU kernels on the first training step and needs both; the CPU smoke test cannot notice their absence,
`drafter-join setup` checks them. One of:

- the system Python plus headers: `sudo apt install python3-dev build-essential` (Debian/Ubuntu);
- without root, a uv-managed CPython (ships its headers):
  `pip install uv && uv python install 3.12` (or uv's standalone installer), then create the venv below with
  `"$(uv python find 3.12)" -m venv .venv` instead of `python3 -m venv .venv`;
- a container whose Python ships its headers and a compiler, for example the official `python:3.12` image (not
  `-slim`) started with `docker run --gpus all -it -v "$PWD":/work -w /work python:3.12 bash` (needs the NVIDIA
  Container Toolkit), then the steps below inside it.

```
python3 -m venv .venv && . .venv/bin/activate
pip install 'torch>=2.11,<2.13' numpy safetensors transformers cryptography pyyaml tokenizers jinja2
bash scripts/install_speculators.sh
pip install -e .
drafter-join --help
```

Check: the last command prints usage starting with `usage: drafter-join`. `install_speculators.sh` keeps the
installed torch and does not install `torchaudio` or `torchvision` (speculators lists them but does not use them;
`pip check` mentions them, which is expected). It does move `transformers` into speculators' tested range
(below 5.17), downgrading a newer one on purpose.

In every shell that runs `drafter-join`, raise the open-file limit to the host's hard limit (the loader
memory-maps many files): `ulimit -n $(ulimit -Hn)`. A fixed large value fails on hosts whose hard limit is lower.

## 2. Register (once)

Read the terms the command prints, then:

```
drafter-join redeem '<invite link>' --accept-terms --name "<display name>"
```

Without a terminal (for example a coding agent), `--accept-terms` is required: without it the command prints the
terms and stops with exit code 2. Omit `--name` to appear as "anonymous". Check: it prints `registered as ...; token
saved to ~/.enntity-drafter/config.json (mode 0600)`. The config also pins the network's publishing key; every
global the site downloads must carry that key's signature. An `invite_invalid` error means the link was used,
revoked or expired: ask the operator for a new one.

## 3. Decide the mode

- Two GB10-class GPUs with the 200G link, a GLM-5.3-Flash copy and the Atlas capture build: **recipe mode**
  (default). The site claims an Enntity shard recipe, generates teacher text and captures it locally.
- Your own captured data (an `enntity.drafter_rows/v1` store or a capture index): **bring-your-own mode**,
  `--mode byo --byo-features DIR` (or `--byo-capture-index FILE --byo-capture-root DIR`), optionally
  `--tag <label>`. Your data never leaves the machine. Nothing is generated or captured in this mode.

**Captures you already have in this repository's format** (Atlas capture build output, for example
`<root>/fast-*/p*/` row directories, plus the capture manifest or capture records that give each row's metadata):
let the client index them,

```
drafter-join setup --teacher <GLM dir> --mode byo --byo-capture-root <root> --byo-captures 'fast-*/p*' \
    --byo-manifest <root>/manifest-train.jsonl [--byo-exclude-ids <file>] [--tag <label>]
```

(the same flags for `run`; `--byo-captures` globs are relative to `--byo-capture-root`, both flags repeat, and the
index is rebuilt under `<work>/byo-source-index/` by every `setup` and `run`), or build the index yourself once and
pass it with `--byo-capture-index`:

```
python -m enntity_drafter.capture_io index --data-root <root> --fast 'fast-*/p*' \
    --manifest <root>/manifest-train.jsonl --out <root>/index
drafter-join setup --teacher <GLM dir> --mode byo --byo-capture-index <root>/index/index.json \
    --byo-capture-root <root>
```

A manifest is also the allowlist: only rows it lists are indexed. Rows it marks held-out stay out of training.

With a capture index the client writes its own index under `<work>/byo-index/` and reads the capture files in place:
only rows your index marks `train` are used; rows it marks held-out are left alone; up to 64 of the remaining rows
(a fixed hash carve, as in recipe shards) become the public check set the self-check evaluates; and rows whose
name, id or prompt id is listed in a `--byo-exclude-ids FILE` (JSONL `{"id": ...}`, repeatable) are never trained or
evaluated, for example an operator's private held-out rows that sit on the same disks (pass
`--byo-exclude-ids heldout-ids.jsonl` to both `setup` and `run`). The run log prints how many rows were kept and
dropped.

**Starting from raw data** (your own prompts, chat logs, agent traffic or GLM-5.3-Flash logs) rather than
captures: follow [docs/byo-data.md](docs/byo-data.md). `python scripts/byo_prepare.py --input FILE --work DIR
--teacher <GLM dir> ...` validates the input against the teacher rules (completions must be GLM-5.3-Flash output),
generates missing completions, captures with the Atlas capture build, checks token alignment and GLM agreement, builds
the capture index and prints the exact `drafter-join` command. Start with `--dry-run`.

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

Expected: one line per check, `PASS`, `WARN` or `FAIL` (including `Triton build`, which compiles a trivial GPU
kernel, `open-file limit`, `memory steadiness`, a minute of MemAvailable samples, and `other workloads`), then `PASS smoke (tiny train + delta checks)`, then a canary (20 training steps
from the current global, uploaded and validated, a few minutes) ending in `canary ... accepted: this site is
ready`. Fix every `FAIL` with the hint on its line and rerun. `WARN` lines are advisory (for example no RDMA device
in bring-your-own mode). A `version mismatch` or `terms changed` message: run `drafter-join update --yes`, then
rerun setup.

The canary proves the pipeline; it is never merged, so it is checked for sanity (finite, shapes, base hash, size)
and rejected only if it collapses the held-out score, not for a small dip. Setup ends with the canary: if it fails
or is interrupted, fix the cause and rerun this same `drafter-join setup` command, which checks the host again and
resumes or reruns the canary. (`drafter-join run` also runs a canary first whenever one is owed: long rounds never
start before a canary is accepted.)

`the network has not published training settings yet`: the operator has not finished setting up; nothing was
trained. Rerun later (`drafter-join run` waits and retries by itself).

## 5. Run

```
drafter-join run --teacher <GLM dir> [mode flags]
```

It loops until stopped: check versions, claim (recipe) or use your data (byo), pull the global, train one round
(about 30 minutes), self-check, upload, clean up. Progress lines show the phase and step, and once per run the
micro-batch the memory probe chose (`micro-batch: 256 anchors x grad_accum 8 x 1 rank(s) = 2048 anchors per
step`). Stop with Ctrl-C (or SIGTERM) at any time: the trainer stops too and the coordinator is told the site
stopped. Rerunning resumes the interrupted phase (training resumes from its last checkpoint, uploads resume part
by part).

**Keep it running after you log out.** A plain `drafter-join run` stops with the shell that started it. Use one of:

```
tmux new -s drafter 'ulimit -n $(ulimit -Hn); drafter-join run --teacher <GLM dir> [mode flags]'   # detach: Ctrl-b d

setsid nohup bash -c 'ulimit -n $(ulimit -Hn); exec drafter-join run --teacher <GLM dir> [mode flags]' \
    >> ~/.enntity-drafter/run.log 2>&1 < /dev/null &                                     # stop: pkill -f 'drafter-join run'
```

or a systemd user unit, `~/.config/systemd/user/drafter-join.service` (absolute paths; `loginctl enable-linger
$USER` keeps it running without a login):

```
[Unit]
Description=drafter-join volunteer site

[Service]
WorkingDirectory=<repository>
ExecStart=/bin/bash -c 'ulimit -n $(ulimit -Hn) && exec <repository>/.venv/bin/drafter-join run --teacher <GLM dir> [mode flags]'
Restart=on-failure
RestartSec=300

[Install]
WantedBy=default.target
```

then `systemctl --user daemon-reload && systemctl --user enable --now drafter-join` (logs: `journalctl --user -u
drafter-join -f`; stop: `systemctl --user stop drafter-join`).

### Moving the site to another machine

Stop `drafter-join run` (Ctrl-C or SIGTERM; an interrupted round is checkpointed), then copy `~/.enntity-drafter`
to the new machine with its permissions (`config.json` holds the token and must stay mode 0600; the default work
directory `~/.enntity-drafter/work` holds the round in progress, its checkpoints and any delta not yet received):

```
rsync -a ~/.enntity-drafter/ <new host>:.enntity-drafter/      # then on the new host: chmod 600 ~/.enntity-drafter/config.json
```

On the new machine install this repository (step 1) and run `drafter-join setup` with `--no-canary` to check the
host, then `drafter-join run` with the same flags. The interrupted round resumes where it stopped when its files
are at the same paths there (the same user name and home, or the same `--work` and data directories; symlinks
work). Otherwise the client stops and names the missing files: make them available at those paths, or start the
round over with `drafter-join run --restart-cycle` (its progress is lost; finished deltas are kept and still
uploaded). Do not run the same site on two machines at once.

## 6. Verify

```
drafter-join status
```

Check: `coordinator.deltas` lists your deltas with `state` `pending`, `accepted` or `merged`. `quarantined` comes
with `reason_code` and `reason`; quarantined deltas are kept and re-validated if the validator is fixed, and your
tokens stay credited. The network's public dashboard shows your site and totals: open the coordinator URL from your
invite link (the part before `/join`, `https://<host>/<path>/`).

## Troubleshooting

| message | action |
|---|---|
| `version mismatch` | `drafter-join update --yes`, then `drafter-join run` (a new canary runs first) |
| `self-check failed before upload` | nothing was uploaded; read the code (layout, non_finite, norm_high ...) and report it with the trainer log path it prints; the message ends with the command to run next |
| `canary not accepted` | fix the cause named in the message, then rerun the same `drafter-join setup` command (it runs a new canary) |
| `training failed ... see <log>` with `Python.h` or a compiler error | Triton cannot compile: see the prerequisites in step 1, then rerun setup |
| `paused: low memory` | nothing to do: another process took memory; the round is checkpointed and resumes by itself once memory is back (`drafter-join status`; the dashboard shows `paused`). Frequent pauses: see safety rule 2 |
| `memory stayed below ... for 6 h` | the round is kept; free memory on the host or move the site, then rerun `drafter-join run` |
| `uses files that are not on this host` | after moving a site: see "Moving the site to another machine" |
| `the network has not published training settings yet` | nothing to do on your side; `run` waits and retries, or rerun setup later |
| `training failed ... see <log>` | read the log; rerun to resume (the memory probe runs again first); out-of-memory or `probe_low_memory`: stop other processes, or as a last resort cap the micro-batch with `--trainer-arg=--anchors=256` (the client warns that it overrides the network's setting; with a network `global_anchors` the probe already picks a smaller micro-batch by itself) |
| `no recipe shard is open` | the client waits and retries; or switch to bring-your-own mode |
| `waiting (...)` | nothing to do: the finished delta is kept and the client continues by itself (for example it runs a canary first, then uploads the delta) |
| `... The delta is kept at <work>/unmergeable/...` | the network changed in a way that makes that delta unusable; it is kept for inspection and the next round starts |
| `coordinator unreachable` | network issue; the client retries with backoff; rerun later |
