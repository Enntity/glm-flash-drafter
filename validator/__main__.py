"""Validator CLI (operator, trusted GPU host).  The token is read from a file, never from arguments.

    python -m validator init --coordinator URL --token-file vtoken --workdir /srv/validator \\
        --config configs/full.json --teacher /models/GLM-5.3-Flash --from-checkpoint runs/r2b/checkpoints/step-N \\
        --heldout-capture-index /srv/heldout/index.json --heldout-capture-root /srv/heldout \\
        --eval-args "--eval-groups agentic=a,template=template --eval-breakdown category --attn sdpa"
    python -m validator run  (same data flags; loops)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path

from coordinator.client import ApiClient
from enntity_drafter import diloco as dl
from enntity_drafter import model as sm
from enntity_drafter.guard import guarded_torch_load, require_scratch_stamp
from enntity_drafter.train import TRAINER_VERSION
from enntity_drafter.validation import Thresholds, evaluate, metrics_of

from .worker import Validator


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m validator", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["init", "run", "once"])
    p.add_argument("--coordinator", required=True, help="base URL, e.g. https://<host>/drafter")
    p.add_argument("--token-file", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--heldout-features", type=Path, help="private held-out row store")
    data.add_argument("--heldout-capture-index", type=Path, help="private held-out capture index")
    p.add_argument("--heldout-capture-root", type=Path)
    p.add_argument("--eval-args", default="", help="extra trainer eval flags (groups, breakdown, attention, device)")
    p.add_argument("--from-checkpoint", type=Path, help="init: version 0 from a trainer checkpoint directory")
    p.add_argument("--publish-dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--min-deltas", type=int, default=2)
    p.add_argument("--min-sites", type=int, default=1, help="distinct sites required, even after round timeout")
    p.add_argument("--min-round-s", type=float, default=1800)
    p.add_argument("--round-timeout-s", type=float, default=3600)
    p.add_argument("--hold-drop", type=float, default=Thresholds.hold_drop)
    p.add_argument("--reject-drop", type=float, default=Thresholds.reject_drop)
    p.add_argument("--borderline", choices=["reduce", "hold"], default="reduce")
    p.add_argument("--interval", type=float, default=60)
    p.add_argument("--signing-key", type=Path, default=None,
                   help="ed25519 private key (hex seed file) that signs every published global; created by init if "
                        "missing")
    a = p.parse_args(argv)
    cfg = sm.load_config(a.config)
    eval_args = ["--config", str(a.config), "--teacher", str(a.teacher), *shlex.split(a.eval_args)]
    if a.heldout_features:
        eval_args += ["--features", str(a.heldout_features)]
        heldout_manifest = a.heldout_features / "manifest.json"
    else:
        eval_args += ["--capture-index", str(a.heldout_capture_index), "--capture-root", str(a.heldout_capture_root)]
        heldout_manifest = a.heldout_capture_index
    heldout_id = hashlib.sha256(heldout_manifest.read_bytes()).hexdigest()[:16]   # rotation changes it
    requirements = {"trainer": TRAINER_VERSION, "config_sha256": dl.config_sha256(cfg)}
    signing_key = None
    if a.signing_key is not None:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        if not a.signing_key.is_file():
            if a.cmd != "init":
                p.error(f"--signing-key {a.signing_key} does not exist")
            a.signing_key.touch(mode=0o600)
            a.signing_key.write_text(Ed25519PrivateKey.generate().private_bytes_raw().hex() + "\n")
        signing_key = Ed25519PrivateKey.from_private_bytes(bytes.fromhex(a.signing_key.read_text().strip()))
    validator = Validator(
        ApiClient(a.coordinator, a.token_file.read_text().strip()), a.workdir,
        lambda path: metrics_of(evaluate(path, eval_args, inprocess=False)), requirements=requirements,
        thresholds=Thresholds(hold_drop=a.hold_drop, reject_drop=a.reject_drop, borderline=a.borderline),
        publish_dtype=a.publish_dtype, min_deltas=a.min_deltas, min_sites=a.min_sites, min_round_s=a.min_round_s,
        round_timeout_s=a.round_timeout_s, heldout_id=heldout_id, signing_key=signing_key)
    if a.cmd == "init":
        if a.from_checkpoint is None:
            p.error("init needs --from-checkpoint (a fresh init: train.py --lr 0 --steps 1 --stop-after 1)")
        path = sorted(a.from_checkpoint.glob("rank*.pt"))[0]
        payload = guarded_torch_load(path, "drafter")
        params = (payload.get("diloco") or {}).get("global") or payload["params"]
        stamp = require_scratch_stamp(json.loads((a.from_checkpoint / "PROVENANCE.json").read_text()), path)
        validator.bootstrap({k: v.float() for k, v in params.items()}, stamp)
        print("Version 0 published. Pin the network to this trainer and config (coordinator host):")
        print("  python -m coordinator set requirements '" + json.dumps(requirements) + "'")
        if signing_key is not None:
            public = signing_key.public_key().public_bytes_raw().hex()
            print("Publish the signing key (sites pin it when they register):")
            print(f"  python -m coordinator set publish_key '\"{public}\"'")
        return
    while True:
        print(json.dumps(validator.step()), flush=True)
        if a.cmd == "once":
            return
        time.sleep(a.interval)


if __name__ == "__main__":
    main()
