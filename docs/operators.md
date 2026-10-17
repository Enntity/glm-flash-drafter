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

- Put a TLS reverse proxy in front of `127.0.0.1:8080` and route `<base path>/*` to it with the prefix kept. Set
  `DRAFTER_TRUST_PROXY=1` (the compose file does) so per-client rate limits use `X-Forwarded-For`; IPs are only
  held in memory.
- The public router serves the dashboard, the public JSON API, the site and validator APIs and, with the local
  backend only, signed blob transfers. **The admin API listens on 127.0.0.1:8081 inside the container and is never
  published**; use `docker compose exec coordinator python -m coordinator <command>`, which acts on the database
  directly.
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
python -m coordinator set training '{"inner_steps": 500, "steps": 10000000, "canary_steps": 20, "lr": 3e-4,
    "warmup": 50, "min_lr_ratio": 1.0, "delta_dtype": "bf16"}'
python -m coordinator set requirements '{"trainer": "scratch-v1", "config_sha256": "<printed by validator init>"}'
python -m coordinator set benchmarks @bench.json               # {"columns": [...], "rows": [[...]], "note": "..."}
python -m coordinator set wanted_notes '["long-form prose", "multilingual prose"]'
python -m coordinator release DELTA_ID accept|reject [--weight-factor 0.5]   # held deltas
python -m coordinator audit --limit 100
```

`training` is the only trainer configuration a site takes from the network, and the client accepts only typed
numeric keys from it (`lr`, `warmup`, `min_lr_ratio`, `anchors`, `weight_decay`, `beta1`, `beta2`, `clip`,
`grad_accum`, `crop`, plus `inner_steps`, `steps`, `canary_steps`, `delta_dtype`). Use `min_lr_ratio` 1 (constant
after warmup) for open-ended runs. Shard recipes must use builders and seeds whose prompts are disjoint from the
held-out set.

## Validator

Runs on trusted GPU hardware, next to the private held-out set:

```
python -m validator init --coordinator https://<host>/drafter --token-file ~/.drafter-validator-token \
    --workdir /srv/drafter-validator --config configs/full.json --teacher /models/GLM-5.3-Flash \
    --from-checkpoint runs/<start>/checkpoints/step-N \
    --heldout-capture-index /srv/heldout/index.json --heldout-capture-root /srv/heldout \
    --eval-args "--eval-groups template=template,structured=structured --eval-breakdown category"
python -m validator run   (same flags)
```

- `init` publishes version 0 from a trainer checkpoint (a fresh one: `train.py ... --lr 0 --steps 1 --stop-after 1`)
  and prints the `requirements` to set on the coordinator.
- `run` loops: re-announce its version, validate pending deltas (canaries first), merge once `--min-deltas`
  (default 2) are accepted and `--min-round-s` (default 1800) has passed since the last merge, or the oldest accepted
  delta has waited `--round-timeout-s`. The merge streams one tensor at a time; the fp32 master stays in the workdir
  and the published copy (`--publish-dtype bf16`) goes to storage.
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
