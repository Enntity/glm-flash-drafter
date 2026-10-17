# Operating the volunteer network

Two operator-run pieces: the **coordinator** (control plane, small container) and the **validator** (trusted GPU
host with the private held-out set). Object storage holds weights and deltas. Design and guarantees:
[volunteer.md](volunteer.md).

## Coordinator

Stdlib Python (`coordinator/`), one SQLite file, no weights. It fits a 3 vCPU / 3 GB host; the compose file caps
it at 512 MB.

```
cd coordinator
DRAFTER_PUBLIC_URL=https://<your host>/drafter docker compose up -d --build
```

- Put a TLS reverse proxy in front of `127.0.0.1:8080` and route `<base path>/*` to it with the prefix kept.
  **Set `DRAFTER_TRUSTED_PROXIES`** to the proxy's address as the container sees it (for Traefik: its address on
  the shared Docker network), or to that network's CIDR range so a proxy restart with a new address keeps working;
  comma-separate several entries. Only requests from that address have their rightmost `X-Forwarded-For` entry used
  as the client address for rate limits; without it every client would share one bucket. The compose file sets
  `DRAFTER_PROXY_MODE=1`, which refuses to start until the variable is set; outside proxy mode a missing value is
  logged as a warning and limits are per connection. IPs are only held in memory. Invite redemption allows a
  burst of 20 per client address, then one per minute.
- The public router serves the dashboard, the public JSON API, the site and validator APIs and, with the local
  backend only, signed blob transfers. **The admin API listens on 127.0.0.1:8081 inside the container and is never
  published**; it also requires the admin token (written to `/data/admin-token`, mode 0600, on first start;
  `python -m coordinator admin-token` replaces it) and a loopback `Host`. The CLI,
  `docker compose exec coordinator python -m coordinator <command>`, acts on the database directly.
- Review queue: `python -m coordinator attention` lists held deltas and unreviewed quarantines (also
  `GET /admin/attention`); `deltas --state quarantined` lists them all; `release ID accept|reject|revalidate`
  decides one (`--publish` makes the next merge skip the merge gate once, recorded as `merge.gate_overridden`).
  Validator errors are retried automatically with backoff (1 min doubling to 1 h, 5 attempts) before they need a
  review. Anything held or unreviewed for 24 h is escalated (`ATTENTION` in the log, `delta.needs_review` in the
  audit log). Quarantined objects are deleted only after an operator reviewed them.
- Maintenance runs every 10 minutes (`python -m coordinator maintenance` by hand): it aborts uploads left open for
  more than `DRAFTER_UPLOAD_TTL_S` (24 h; S3 multipart uploads are aborted, GCS sessions lapse), deletes delta
  objects 7 days after their merge and 30 days after quarantine (`DRAFTER_DELTA_RETENTION_S`,
  `DRAFTER_QUARANTINE_RETENTION_S`), and retires unused globals. Each site may have 2 open uploads and 3 unmerged
  deltas. Below `DRAFTER_MIN_FREE_BYTES` (512 MB) free on the database volume the coordinator refuses new work
  and stops recording heartbeat history, so SQLite never runs out of space.
- Back up the `/data` volume (SQLite in WAL mode: copy with `sqlite3 .backup` or stop the container first).

Environment (credentials only as mounted files):

| variable | meaning |
|---|---|
| `DRAFTER_PUBLIC_URL` | public URL including the base path; used in invite links |
| `DRAFTER_BASE_PATH` | default `/drafter` |
| `DRAFTER_STORAGE` | `s3`, `gcs` or `local` (local: development only; bytes then flow through the coordinator) |
| `DRAFTER_BUCKET` | the dedicated bucket |
| `DRAFTER_S3_ENDPOINT`, `DRAFTER_S3_REGION`, `DRAFTER_S3_ADDRESSING` | S3-compatible endpoint, region (default `fsn1`), `path` (default) or `virtual` |
| `DRAFTER_S3_ACCESS_KEY_FILE`, `DRAFTER_S3_SECRET_KEY_FILE` | secret files |
| `DRAFTER_GCS_CREDENTIALS_FILE` | service-account JSON file (GCS) |
| `DRAFTER_URL_EXPIRY_S`, `DRAFTER_LEASE_S`, `DRAFTER_LEASE_GRACE_S`, `DRAFTER_KEEP_GLOBALS`, `DRAFTER_CLAIM_TIMEOUT_S`, `DRAFTER_MAX_DELTA_BYTES`, ... | `coordinator/service.py: Config` |

### Object storage

- Use a **dedicated bucket** and a **dedicated credential scoped to that bucket only**: on GCS a service account
  with `roles/storage.objectAdmin` on that bucket (or narrower), on S3-compatible stores a key limited to it.
  Never reuse a credential that can read anything else, such as an application's file-storage service account,
  which may be able to read user files.
- GCS: the coordinator signs V4 URLs with the service account's key (`pip install cryptography`, in the image).
  Uploads are resumable sessions whose exact size is pinned by a signed `x-goog-content-length-range`.
- S3-compatible (for example Hetzner Object Storage): presigned multipart uploads; each part's length is part of
  its signature, so the declared size holds.
- Every file's SHA-256 is declared by its uploader and checked by whoever downloads it (validator for deltas,
  sites for globals); the coordinator checks sizes after completion.
- **Egress cost.** Per round each site downloads the 2.3 GB global and the validator downloads every 2.3 GB delta.
  GCS internet egress is about $0.12/GB: fine for a small pilot, expensive at many sites with hourly syncs.
  S3-compatible stores such as Hetzner Object Storage are far cheaper at scale; switching is a configuration change.
- **Lifecycle.** Suggested rule: delete objects under `deltas/` after 30 days (quarantined deltas must survive long
  enough to be re-validated after a validator fix). Do not put a lifecycle rule on `globals/`: the coordinator
  deletes globals itself once nothing references them (it keeps the newest `DRAFTER_KEEP_GLOBALS`, default 8, and
  every base still in use).

### Admin commands

```
python -m coordinator invite --label "lab name" [--hours 72]   # one-use link, printed once (code in the fragment)
python -m coordinator invites | revoke-invite ID
python -m coordinator sites  | revoke-site PUBLIC_ID           # revocation stops the token at once
python -m coordinator validator --label gpu-1                  # validator token, printed once
python -m coordinator add-shards --builder coverage --builder-args '{"seed": 31, "profile": "long", "prefix": "v1"}' \
    --generation '{"samples": 2}' --rows-per-shard 500 --total 8000
python -m coordinator set training '{"lr": 1e-4, "warmup": 50, "min_lr_ratio": 0.3, "anchors": 512,
    "global_anchors": 2048, "inner_steps": <see below>, "steps": 10000000, "canary_steps": 20, "delta_dtype": "int8"}'
python -m coordinator set requirements '{"trainer": "scratch-v1", "config_sha256": "<printed by validator init>"}'
python -m coordinator set benchmarks @bench.json               # {"columns": [...], "rows": [[...]], "note": "..."}
python -m coordinator set wanted_notes '["long-form prose", "multilingual prose"]'
python -m coordinator set publish_key '"<hex printed by validator init>"'   # sites pin it at registration
python -m coordinator release DELTA_ID accept|reject [--weight-factor 0.5]   # held deltas (borderline, merge gate)
python -m coordinator attention                                # what needs you (see below)
python -m coordinator audit --limit 100
```

**Setting up a network, in this order (every step is mandatory):**

1. `python -m validator init ...` (see [Validator](#validator)): publishes version 0, prints the publishing key and
   the `requirements`.
2. `set publish_key` and `set requirements` with what `init` printed.
3. `set training`. Sites never train on the trainer's command-line defaults (anchors 64 and a learning rate for
   training from scratch): a site refuses to train until `training` holds at least `lr`, `warmup`, `min_lr_ratio`,
   `anchors`, `global_anchors` (or, as before, `grad_accum`) and `inner_steps`, and waits instead. Recommended
   values for continued training from a trained version 0: `lr` 1e-4, `warmup` 50, `min_lr_ratio` 0.3,
   `global_anchors` 2048 (anchors per optimizer step), `anchors` 512. `global_anchors` is the batch every site
   trains with; `anchors` is only the largest micro-batch a site may use: each site's memory probe picks the
   largest micro-batch that leaves its memory floor free (a GB10 shared with other services may get 256 or 128)
   and sets `grad_accum` so micro x accum x GPUs = `global_anchors`. Without `global_anchors`, `anchors` and
   `grad_accum` are used as given on every site (the old behaviour). Size `inner_steps` so a round is about 30
   minutes of training on one GB10: 1800 / `step_s`, where `step_s` is the seconds per optimizer step the trainer
   logs (`train.jsonl`) at this batch (for example 500 at 3.6 s/step).
4. Invites (`python -m coordinator invite`).

The coordinator enforces the order: until `requirements` (with `trainer` and `config_sha256`), `publish_key` and
`training` (with the fields above) are set, it refuses invites and shard leases with `not_ready`, naming what is
missing, and `attention` lists the missing settings under `missing_settings`. A site that registered before
`publish_key` was set would adopt the first key it is offered.

`attention` also lists `self_check_failures` (sites whose deltas failed their own pre-upload check, so nothing
arrived: for example "site X: canary self-check failed: heldout_regression x2", with the worst public check-set
drop; reported in heartbeats, no data) and `silent_sites` (sites seen in the last day whose heartbeats stopped
without a clean stop: `stale` after two missed heartbeats, `offline` after `DRAFTER_ONLINE_S`, 15 minutes). The
dashboard shows the same statuses (and `paused` while a site waits for memory), so a stopped process never looks
like it is training.

`training` is the only trainer configuration a site takes from the network, and the client accepts only typed
numeric keys from it (`lr`, `warmup`, `min_lr_ratio`, `anchors`, `global_anchors`, `weight_decay`, `beta1`,
`beta2`, `clip`, `grad_accum`, `crop`, plus `inner_steps`, `steps`, `canary_steps`, `delta_dtype`). A site's own
`--trainer-arg` still wins (for example a lower `--anchors` cap); the client prints a warning when it overrides a
network setting. `min_lr_ratio` 1 keeps the
learning rate constant after warmup. Shard recipes must use builders and seeds whose prompts are disjoint from the
held-out set.

## Validator

Runs on trusted GPU hardware, next to the private held-out set. The validator host should be:

- **always on**: deltas wait for it, a canary blocks a new site until its verdict, and merges happen only while it
  runs (a laptop that sleeps stalls the network);
- **a GPU node that does not serve the target** (one GB10 is enough; `--eval-args "... --device mps"` works on an
  Apple GPU for trials, much slower);
- **next to the held-out set on local disk**: every verdict evaluates it, so it must not cross a slow link, and it
  never leaves the operator's machines;
- **the only place the signing key exists**: `init` creates it there (0600); keep the validator token beside it.

`--teacher` needs only the target's embedding, final norm and lm_head (a directory with those tensors and the
target `config.json`), not the full model. Order of the first steps: see "Setting up a network" above.

Canaries (a site's first delta, and the first after a pin change) are judged for sanity only: the delta checks
(provenance, header, base hash, layout, finite values, `max_abs` and `max_rms`; not `min_rms`) and a catastrophic
bound, rejecting (`canary_collapse`) only if a held-out group falls below half of the base score
(`Thresholds.canary_floor`). A few steps from a fresh optimizer can legitimately dip, and canaries are never merged.
Deltas that would be merged keep the conservative `--hold-drop` / `--reject-drop` gate.

```
python -m validator init --coordinator https://<host>/drafter --token-file ~/.drafter-validator-token \
    --workdir /srv/drafter-validator --config configs/full.json --teacher /models/GLM-5.3-Flash \
    --from-checkpoint runs/<start>/checkpoints/step-N \
    --heldout-capture-index /srv/heldout/index.json --heldout-capture-root /srv/heldout \
    --eval-args "--eval-groups template=template,structured=structured --eval-breakdown category"
python -m validator run   (same flags)
```

- `--signing-key FILE` (created by `init` if missing, keep it on the validator host only): every published global
  is signed with this ed25519 key; `init` prints the public key to set as `publish_key`.
- `init` publishes version 0 from a trainer checkpoint (a fresh one: `train.py ... --lr 0 --steps 1 --stop-after 1`)
  and prints the `requirements` to set on the coordinator.
- `run` loops: re-announce its version, validate pending deltas (canaries first), merge once `--min-deltas`
  (default 2) are accepted and `--min-round-s` (default 1800) has passed since the last merge, or the oldest accepted
  delta has waited `--round-timeout-s`. The merge streams one tensor at a time; the fp32 master stays in the workdir
  and the published copy (`--publish-dtype bf16`) goes to storage. The merge inputs (deltas and weights) are saved
  before merging and outer momentum is kept per version, so a crash and re-run produce the same file.
- **Merge gate.** Each merged global is evaluated on the held-out set before publishing and compared with the
  best published version (one version's whole score vector, not a per-group maximum) using a margin of
  max(1%, measured eval noise; `init` evaluates version 0 three times to measure it), and with version 0. A
  regressing merge is not published: its deltas are held (`python -m coordinator release ID accept --publish`
  overrides the gate once, audited).
- **The held-out set is private**: never in the repository, never on the coordinator. Build it from prompts no
  shard recipe can produce (for example builder runs with an unpublished seed) and rotate it; its manifest hash is
  recorded with every version, so a rotation is visible in the curve, and cached base scores are recomputed.
- Thresholds: `--hold-drop 0.01`, `--reject-drop 0.03`, `--borderline reduce|hold`. Any change to the validation
  rules must bump `enntity_drafter.validation.VALIDATOR_VERSION`: the coordinator then re-validates every
  quarantined or held delta, and sites run a new canary.
- A crash leaves a delta claimed; it returns to the queue after `DRAFTER_CLAIM_TIMEOUT_S` (default 2 h).
- Memory: the validator materializes `base - delta` (4.7 GB fp32) and runs the trainer's held-out eval in a
  subprocess; give it a GPU node that does not serve the target.

## Security notes

- Tokens (site and validator) and invite codes are stored as SHA-256 hashes; the token is shown once.
- Every authenticated route is rate-limited per site; redeem and public routes per client address.
- Heartbeats accept a fixed set of numeric fields and a version string; display names are restricted to letters,
  digits and a few punctuation marks and rendered as text.
- Deltas are safetensors only; the validator reads them through the license guard (`enntity_drafter/guard.py`).
- The dashboard sends a strict Content-Security-Policy and loads nothing from third parties.
- Keep host-specific wiring (proxy labels, networks, paths) out of this repository.
