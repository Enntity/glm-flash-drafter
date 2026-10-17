# Volunteer training

Several sites train one drafter together, exchanging only weights, never data. Think SETI@home for a
speculative-decoding drafter. This page covers the design, what a site spends, the trust model and what is not
built yet. To run a site, follow [JOIN.md](../JOIN.md); to operate the network, see [operators.md](operators.md).

Status: **built and tested on CPU, not deployed.** No hosted coordinator exists yet. Everything below runs in this
repository's tests on a tiny model; the full-size path (capture, GB10 training, object storage) has not been run end
to end across sites.

## How it works

```
                 site (2 x GB10)                                  operator
  ┌────────────────────────────────────────────┐      ┌───────────────────────────────────┐
  │ drafter-join run                           │      │ coordinator (control plane)       │
  │  1 check pins (trainer, config, validator) │◄────►│  invites, tokens, shard leases,   │
  │  2 claim a recipe shard  | or use own data │      │  rounds, audit log, dashboard     │
  │  3 generate + capture (GLM-5.3-Flash TP2)  │      │  signs short-lived storage URLs   │
  │  4 pull global weights ──────────────┐     │      └───────────────┬───────────────────┘
  │  5 train H steps (one GPU)           │     │                      │ signed URLs
  │  6 self-check (validator's own code) │     │      ┌───────────────▼───────────────────┐
  │  7 upload delta ─────────────────────┼─────┼─────►│ object storage (S3 / GCS)         │
  │  8 delete activations                │     │      │  globals/vN, deltas/<site>/<id>   │
  └──────────────────────────────────────┼─────┘      └───────────────▲───────────────────┘
                                         └────────────────────────────┤
                                                      ┌───────────────┴───────────────────┐
                                                      │ validator (operator GPU host)     │
                                                      │  checks + private held-out eval   │
                                                      │  token-weighted plain average     │
                                                      │  publishes global vN+1            │
                                                      └───────────────────────────────────┘
```

- **Every site trains the whole drafter** on its own data shard; nothing is partitioned by model section.
- **Rounds.** A site starts from the current global weights, takes H inner AdamW steps (default 500, about 30
  minutes on one GB10 at round 2b's step time) with the trainer's `--sync-mode diloco --diloco-transport external`
  mode, and uploads `delta = global - local`. Its inner optimizer state stays on the site between rounds.
- **Merge.** The validator averages accepted deltas weighted by tokens trained and applies the outer step with
  learning rate 1 and no momentum: plain averaging. Tokens are self-reported, so they only count across sites:
  each site's total in a round is capped at twice the median site total, every site weighs the same when fewer
  than three sites are in the round, and a site may have at most 3 unmerged deltas. A delta nearly identical to
  another one of the round (cosine above 0.995) is quarantined.
- **Merge gate.** A merged global is evaluated on the private held-out set before it is published. If any group
  is more than 1% below the best published global, or clearly below version 0, it is not published and its
  deltas are held for the operator (not discarded). This stops small accepted drops from compounding. The momentum option (OpenDiLoCo's lr 0.7 / Nesterov 0.9) stays
  in the code, off.
- **Why plain averaging.** In a round-3 A/B on the full model (two sites, three syncs, H = 500, Enntity's own
  data including internal agentic sessions), the momentum outer step fell well behind synchronous training, while
  plain averaging matched or beat it on held-out groups: agentic +2.3%, template +1.0%, structured +1.1%, code
  -0.6%, prose -2.1% expected accepted length. One draw per arm.
- **Equivalence.** `tests/test_volunteer.py` runs two sites (one recipe, one bring-your-own data) through the
  coordinator, validator and `drafter-join` for three rounds and checks that the merged global equals, bitwise,
  the trainer's own two-site DiLoCo run.

### Site modes

- **recipe**: claim an Enntity shard recipe (prompt builder, seed, row range, generation settings), generate
  teacher text with GLM-5.3-Flash (your own server, or an API key of your choice), capture hidden states with the
  Atlas capture build, train. Spot checks can replay recipe shards.
- **bring your own data (byo)**: train on your own captured data. It never leaves your machine; you may give an
  opaque tag. No spot checks are possible in this mode; the held-out check still applies.

In both modes only weight deltas and metadata are uploaded: mode, recipe id or tag, token and step counts,
trainer version, base hash, and a signed data-source attestation (no incoai or other non-commercial or derived
data; GLM-5.3-Flash as the only teacher). The attestation is an HMAC keyed by the hash of the site's token: only
the token holder can make it and the coordinator can check it. It is a commitment, not a proof.

## Requirements

- Recipe mode: two GB10-class GPUs (two DGX Spark nodes, or equivalent) with the 200G link, because capture
  runs GLM-5.3-Flash in TP2. Training itself fits one GPU. Bring-your-own mode needs one GPU.
- Python 3.10+, torch 2.11-2.12, speculators at the pinned revision (`scripts/install_speculators.sh`), your
  own GLM-5.3-Flash copy (for example `nvidia/GLM-5.3-Flash-NVFP4`), and for recipe mode the Atlas capture build
  (branch `exp/drafter-capture-20261014`, [capture.md](capture.md)).
- Memory safety on GB10 (CPU and GPU share memory): never run the target server and the trainer on the same node;
  the client adds the trainer's `--min-free-gb 12 --probe` guards on CUDA unless you set your own.

## What a site spends per round

| | per round (defaults) |
|---|---|
| download | the global weights: 2.34 GB (bf16) |
| upload | one delta: 1.18 GB (int8, the default; 0.79% error on a real delta) or 2.34 GB if the network sets bf16 |
| training | H = 500 steps, about 30 minutes on one GB10 (round 2b: ~3.6 s/step) |
| memory | the round-2b trainer peaked at 33 GB allocated; external DiLoCo mode adds a 4.7 GB fp32 copy of the global |
| disk | FP8 activations at 24,600 bytes per captured token, deleted after the round; the delta until it has a verdict |
| teacher text (recipe) | your own API key or server; the client's default spend cap is $10 per shard |

At 18 MB/s the int8 upload takes about 65 s and the bf16 download about 130 s.

## Trust model and validation

Joining is by **one-use invite**. The code travels in the link's URL fragment, which browsers never send to
servers. Redeeming it records acceptance of the [participation terms](../coordinator/terms.py) and returns a site
token that the coordinator stores only as a SHA-256 hash. The first sites are the ones the operator invites.

Checks, in order:

1. **Before training, on the site:** the client compares its trainer version, portable config hash and validator
   version with the network's pins and refuses to train on a mismatch (`drafter-join update`).
2. **At upload initiation, before any bytes move:** the delta's trainer version and config hash must equal the
   pins (the validator version need not: the current validator re-checks every delta); its base must be a global
   this coordinator published and still retains; size cap; attestation signature; for recipe mode, the site must
   have leased that shard. Mismatches answer with an actionable message. If the network changes in the middle of a
   round, the client keeps the delta locally and says why.
3. **On the site, before upload** (`enntity_drafter/validation.py`, the validator's own code): provenance stamp,
   header, key/shape/dtype layout against the base, finite values, RMS and max-abs bounds published by the
   validator, base hash, and a held-out comparison on a small public check set carved from the site's own shard.
   A failing delta is never uploaded.
4. **On the validator:** size and SHA-256 of the stored file, the same checks, an optional spot check that replays
   a recipe shard and compares the update's direction (cosine), then the held-out evaluation of `base - delta`
   against the base on a **private, rotating held-out set** that is not in the repository or on the coordinator.
5. **Conservative verdicts** (`validation.Thresholds`): accept unless a held-out group's expected accepted length
   drops; a drop of more than 1% is borderline (merged at half weight, or held for the operator); only a drop of
   more than 3% is a rejection. A missing or non-finite score is a rejection (the check fails closed).
   Per-category numbers are recorded but too small to decide on. Sites see only the decision and a reason code
   with a fixed explanation, never held-out numbers, so the held-out set cannot be probed through verdicts.
6. **Quarantine, not deletion.** Rejected deltas stay in storage with a machine-readable code and a human reason
   the site can read (`drafter-join status`). When the validator's version changes, the coordinator re-queues every
   quarantined or held delta automatically.

Every published global is signed with the operator's ed25519 key; sites pin the key when they register and
refuse unsigned or wrongly signed globals. Uploads are safetensors tensors only; nothing uploaded is executed or
unpickled. A delta the validator cannot process is quarantined as `validator_error` (re-validated after the next
validator update) instead of stopping the queue. Revoking a site quarantines its unmerged deltas. The coordinator never loads
weights (a test enforces that it imports no tensor library).

## What we promise contributors

- **Canary first.** A site's first cycle, and the first after any trainer, config or validator version change, is
  a short run (20 steps by default, at most 50) whose delta goes through the full validation path. The coordinator,
  not the client, decides that a delta is a canary, and refuses a long delta from a site that still owes one. Long rounds start only
  after it is accepted; `drafter-join setup` ends with it.
- **No late version surprises.** Pins are checked by the client before training and by the coordinator at upload
  initiation, never after the bytes have moved.
- **No timing rejections.** Deltas against older globals are accepted, down-weighted by 0.5 per version behind,
  as long as the base is retained. The coordinator keeps the newest 8 globals, every base of an unmerged or
  quarantined delta, and every version a site reported training from in the last 12 hours. Shard leases are
  extended by heartbeats, have a 2-hour grace period, and work on an expired lease is still accepted.
- **Credit at receipt.** The dashboard counts tokens as soon as a delta is received (pending, accepted, held or
  quarantined); validation never erases credit.
- **Resilience.** The trainer checkpoints within a round and resumes; uploads are chunked, resumable and retried
  with backoff; upload ids are idempotent (a retry or a duplicate file never counts twice); the site keeps its
  delta until the coordinator has received it and returned a verdict; coordinator state is SQLite (WAL) and
  survives restarts; a validator crash leaves the delta claimed until a timeout, then it is validated again.
- **Short rounds** (about 30 minutes of training) bound the worst case.

These are covered by `tests/test_coordinator.py` and `tests/test_volunteer.py` (interrupted chunk then resume,
coordinator restart mid-round, validator crash mid-evaluation, stale bases 1 and 2 versions behind, version mismatch
at initiation, a validator bug fixed and re-validated, duplicate uploads, lease expiry).

## Visibility

The coordinator serves a public dashboard at its base path: current global version, live sites (display name or
"anonymous", status, throughput), a leaderboard of tokens received and deltas accepted, the held-out curve per
group over versions (rounded, at least an hour old), the operator-published Atlas benchmark table, the weakest
groups ("wanted"),
aggregate reasons for held or quarantined deltas, and network totals. A "how it works" page explains validation.
Heartbeats carry only phase, round, version, token counters, throughput, GPU power, MFU and GPU count; any other
field is refused. The audit log (registrations, invite use, verdicts, revocations) is admin-only.

## Not built yet

- A hosted coordinator and validator. Nothing is deployed.
- A full-size run across real sites. The recipe pipeline in `drafter-join` (prompts, teacher text, capture, FP8,
  index) calls the repository's scripts but has not been run end to end; `setup`'s GPU, peer and Atlas checks have
  not been run on GB10 hardware. The S3 and GCS backends are tested for request signing only, not against real
  buckets.
- Spot checks for real recipes: the operator has to generate and capture the shard itself, and hosted sampling
  is not deterministic, so a replay can only be compared approximately (by direction). The replay hook exists and
  is tested on a stand-in.
- Overlapped (streaming) syncs, robust aggregation (median or trimmed mean).
