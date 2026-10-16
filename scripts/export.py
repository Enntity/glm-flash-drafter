#!/usr/bin/env python3
"""Export one training checkpoint (``<run>/checkpoints/step-N``) in the Atlas BF16 drafter format.

``train.py --export`` exports the final weights; use this for an earlier checkpoint (round 2a shipped
step 2,910 of a stopped run).  The checkpoint is read through the license guard (it must carry this
trainer's scratch provenance stamp), its fp32 masters are cast to BF16, written with a generated
``config.json`` and ``PROVENANCE.json``, and reloaded to verify the engine contract (81 bare keys, no
target tensors).

    python scripts/export.py --config configs/full.json --checkpoint runs/r2a/checkpoints/step-0002910 \\
        --out exports/r2a-step2910
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.export import export_engine_checkpoint, verify_export  # noqa: E402
from enntity_drafter.guard import guarded_torch_load, require_scratch_stamp  # noqa: E402
from enntity_drafter.train import gather_body  # noqa: E402


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True, help="checkpoints/step-N directory")
    ap.add_argument("--out", type=Path, required=True, help="new export directory")
    a = ap.parse_args(argv)
    cfg = sm.load_config(a.config)
    stamp = require_scratch_stamp(json.loads((a.checkpoint / "PROVENANCE.json").read_text()), a.checkpoint)
    digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    # the trainer stamps the config with target_name = the --teacher path; compare everything else
    if stamp.get("config_sha256") not in (None, digest):
        cfg_with_target = dict(cfg, target_name=stamp.get("target", {}).get("target_root"))
        if hashlib.sha256(json.dumps(cfg_with_target, sort_keys=True).encode()).hexdigest() != stamp["config_sha256"]:
            raise SystemExit(f"{a.config} does not match the checkpoint's config digest")
    path = next(iter(sorted(a.checkpoint.glob("rank*.pt"))), None)
    if path is None:
        raise SystemExit(f"{a.checkpoint}: no rank*.pt")
    payload = guarded_torch_load(path, "drafter")
    body = gather_body(list(payload["params"].items()))
    model = sm.build_model(cfg, device="meta")
    expected = {n: tuple(p.shape) for n, p in model.named_parameters() if n in set(sm.body_parameter_names(model))}
    if {n: tuple(t.shape) for n, t in body.items()} != expected:
        raise SystemExit("checkpoint parameters do not match the config's body")
    provenance = {**stamp, "exported_from": {"checkpoint": a.checkpoint.name, "step": int(payload["step"])}}
    receipt = export_engine_checkpoint(body, cfg, a.out, provenance)
    receipt["verify"] = verify_export(a.out, expected)
    print(json.dumps({k: receipt[k] for k in ("model_sha256", "body_keys", "format")} | {
        "verify_keys": receipt["verify"]["keys"], "dtypes": receipt["verify"]["dtypes"]}, indent=1))


if __name__ == "__main__":
    main()
