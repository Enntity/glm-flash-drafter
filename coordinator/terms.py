"""Participation terms and the per-delta data attestation (versioned; their SHA-256 is what a site accepts/signs).

The attestation is signed with HMAC-SHA256 keyed by SHA-256(site token): only the token holder can produce it and
the coordinator, which stores only that hash, can check it.  It is a commitment by the site, not a proof about the
data (nothing about the data is uploaded).  Stdlib only: imported by the coordinator and the site client.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

TERMS_VERSION = "2026-10-16"
TERMS_TEXT = """GLM-5.3-Flash drafter volunteer training: participation terms (2026-10-16)

1. What you contribute. Your site trains copies of the open drafter and uploads weight deltas (differences between
   weights) and the metadata the protocol defines: mode, recipe id or a self-chosen opaque tag, token and step
   counts, trainer version, base hash, heartbeats (phase, counters, throughput) and this attestation. No prompts,
   completions, activations or files from your machine are uploaded.
2. License. You license each delta you upload under the Apache License 2.0, the license of the drafter, and confirm
   you have the right to do so.
3. Data. You attest, per delta, to the data-source statement below.
4. Validation. Deltas are validated on a private held-out set and may be accepted, held for review, merged at
   reduced weight or quarantined. Received work is credited on receipt. Quarantined deltas are kept for weeks and
   re-validated when the validator is fixed.
5. Display. Your display name (or "anonymous"), status, throughput and contribution totals are public on the
   dashboard. Do not put personal or identifying information in a display name.
6. Conduct. Do not upload deltas designed to degrade or backdoor the drafter, or deltas trained on data you are not
   allowed to use. The operator may revoke a site's token. Participation is voluntary and unpaid; you can stop at
   any time.
7. No warranty. The software is provided as is (Apache-2.0); you run it on your own hardware at your own risk.
"""
ATTESTATION_VERSION = "1"
ATTESTATION_TEXT = """Data-source attestation v1: the delta was trained only on (a) an Enntity shard recipe generated as published, or
(b) data I am allowed to use for training an Apache-2.0 model. It contains no incoai weights or text, no data under
non-commercial or no-derivatives terms, and nothing derived from such data. Teacher completions and captured
features came only from GLM-5.3-Flash. No private or personal data of others was used without permission.
"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


TERMS_SHA256 = sha256_text(TERMS_TEXT)
ATTESTATION_SHA256 = sha256_text(ATTESTATION_TEXT)


def token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def attestation_fields(delta: dict[str, Any]) -> dict[str, Any]:
    """What the signature covers (canonical JSON of these fields)."""

    return {"statement_sha256": ATTESTATION_SHA256, "client_id": delta["client_id"], "sha256": delta["sha256"],
            "base_sha256": delta["base_sha256"], "mode": delta["mode"], "shard_id": delta.get("shard_id"),
            "tag": delta.get("tag")}


def sign_attestation(token: str, delta: dict[str, Any]) -> dict[str, Any]:
    fields = attestation_fields(delta)
    mac = hmac.new(bytes.fromhex(token_hash(token)), json.dumps(fields, sort_keys=True).encode(), hashlib.sha256)
    return {"version": ATTESTATION_VERSION, "statement_sha256": ATTESTATION_SHA256, "signature": mac.hexdigest()}


def global_message(version: int, sha256: str) -> bytes:
    """What the operator's ed25519 key signs for every published global (sites verify it with the key they
    pinned when they registered)."""

    return f"enntity-drafter-global|v1|{int(version)}|{sha256}".encode()


def verify_attestation(stored_token_hash: str, delta: dict[str, Any], attestation: dict[str, Any]) -> bool:
    if not isinstance(attestation, dict) or attestation.get("statement_sha256") != ATTESTATION_SHA256:
        return False
    fields = attestation_fields(delta)
    mac = hmac.new(bytes.fromhex(stored_token_hash), json.dumps(fields, sort_keys=True).encode(), hashlib.sha256)
    return hmac.compare_digest(mac.hexdigest(), str(attestation.get("signature", "")))
