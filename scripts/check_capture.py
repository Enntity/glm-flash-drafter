#!/usr/bin/env python3
"""Check captured rows: gates, token ids, decode-vs-prefill agreement, FP8 storage error.

For every row directory under ``--rows`` (``<tag>/r<rank>-<L|F>-p<pos0>-n<rows>.{bin,fp8,json}``):

* segments tile ``[from, T)`` once per kind and carry the item's token ids
  (``--records``: capture-records.jsonl from scripts/capture.py);
* argmax agreement: ``lm_head(norm(final[i - 1]))`` vs the captured token ``i``
  over the generated span (``--model``: the target directory; only the final
  norm and lm_head are read, through the license guard);
* ``--fp8``: relative L2 of the FP8 round trip per slot, and on up to
  ``--positions`` generated positions the KL between the teacher distributions
  from the BF16 and FP8 final slot plus both argmax agreements;
* gates (``gate32-*``, ``gate8k-*`` from ``capture.py --gates``): a repeat is
  bit-exact, a mutated token 20 leaves positions < 20 bit-exact and changes the rest.

    python scripts/check_capture.py --rows data/rows/run1 --records data/capture-logs/run1/capture-records.jsonl \\
        --model /models/GLM-5.3-Flash --out data/checks/run1.json [--fp8]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.capture_io import FP8, SUFFIX, fp8_dequantize, fp8_quantize  # noqa: E402

EPS = 1e-5


def load_row(directory: Path) -> tuple[int, int, list[int], torch.Tensor, torch.Tensor]:
    """(lo, hi, ids, layers [5, n, h], final [n, h]) of one row directory (BF16 or FP8 segments)."""

    out = {}
    for kind in ("L", "F"):
        metas = sorted((json.loads(p.read_text()) for p in directory.glob(f"r*-{kind}-p*.json")),
                       key=lambda m: m["pos0"])
        parts, ids, position = [], [], None
        for meta in metas:
            if position is not None and meta["pos0"] != position:
                raise ValueError(f"{directory}: {kind} segments do not tile at {position}")
            position = meta["pos0"] + meta["rows"]
            slots, rows, hidden = (len(meta["layers"]) if kind == "L" else 1), meta["rows"], meta["hidden_size"]
            stem = f"r{meta['rank']}-{kind}-p{meta['pos0']:09d}-n{rows}"
            raw = np.fromfile(directory / (stem + SUFFIX[meta.get("dtype", "bf16")]), dtype=np.uint8)
            if meta.get("dtype") == FP8:
                bits = fp8_dequantize(raw[: slots * rows * hidden].reshape(slots, rows, hidden),
                                      raw[slots * rows * hidden:].view(np.float32).reshape(slots, rows))
            else:
                bits = raw.view(np.uint16).reshape(slots, rows, hidden)
            parts.append(torch.from_numpy(np.ascontiguousarray(bits).view(np.int16)).view(torch.bfloat16))
            ids += meta["token_ids"] or [None] * rows
        out[kind] = (metas[0]["pos0"], position, ids, torch.cat(parts, dim=1))
    (lo, hi, layer_ids, layers), (flo, fhi, final_ids, final) = out["L"], out["F"]
    if (lo, hi) != (flo, fhi) or any(a is not None and a != b for a, b in zip(layer_ids, final_ids)):
        raise ValueError(f"{directory}: layer and final segments disagree")
    return lo, hi, final_ids, layers, final[0]


def roundtrip(tensor: torch.Tensor) -> torch.Tensor:
    bits = tensor.contiguous().view(torch.int16).numpy().view(np.uint16)
    return torch.from_numpy(fp8_dequantize(*fp8_quantize(bits)).view(np.int16)).view(torch.bfloat16)


class Head:
    """Target final norm + lm_head (the teacher distribution the trainer derives from the final slot)."""

    def __init__(self, model_dir: Path, device: str):
        tensors, _ = sm.load_target_frozen(model_dir)
        self.norm = tensors["norm.weight"].float().to(device)
        self.head = tensors["lm_head.weight"].to(device, torch.bfloat16)
        self.device = device

    def logprobs(self, final: torch.Tensor) -> torch.Tensor:
        x = final.to(self.device).float()
        x = (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + EPS) * self.norm).to(torch.bfloat16)
        return (x @ self.head.T).float().log_softmax(-1)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--records", type=Path, action="append", default=[])
    ap.add_argument("--model", type=Path, help="target directory (agreement / --fp8 KL); omit to skip")
    ap.add_argument("--fp8", action="store_true")
    ap.add_argument("--positions", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    records = {r["name"]: r for path in a.records for r in map(json.loads, path.open()) if "error" not in r}
    head = Head(a.model, device) if a.model else None
    result: dict = {"rows": [], "gates": {}}
    rows = sorted(p for p in a.rows.iterdir() if p.is_dir())
    for directory in rows:
        if directory.name.startswith("gate"):
            continue
        lo, hi, ids, layers, final = load_row(directory)
        row: dict = {"name": directory.name, "stored": hi - lo}
        record = records.get(directory.name)
        if record is not None:
            row.update(P=record["P"], tiles=(lo, hi) == (record["from"], record["T"]))
        start = max(row.get("P", lo + 1), lo + 1)
        if head is not None and hi > start:
            positions = torch.arange(start, hi)
            targets = torch.tensor([ids[p - lo] for p in positions.tolist()], device=device)
            logp = head.logprobs(final[positions - 1 - lo])
            row.update(generated=len(positions), agree=float((logp.argmax(-1) == targets).float().mean()))
        if a.fp8:
            slots = torch.cat([layers, final[None]], 0)
            restored = roundtrip(slots)
            row["fp8_rel_l2"] = [round(float((restored[s].float() - slots[s].float()).norm()
                                             / slots[s].float().norm()), 5) for s in range(slots.shape[0])]
            if head is not None and hi > start:
                pick = torch.arange(start, hi)
                if len(pick) > a.positions:
                    pick = pick[torch.randperm(len(pick), generator=torch.Generator().manual_seed(0))[: a.positions]]
                pick = pick.sort().values
                p_bf16, p_fp8 = head.logprobs(final[pick - 1 - lo]), head.logprobs(restored[-1][pick - 1 - lo])
                kl = (p_bf16.exp() * (p_bf16 - p_fp8)).sum(-1)
                row.update(fp8_top1_equal=float((p_bf16.argmax(-1) == p_fp8.argmax(-1)).float().mean()),
                           fp8_kl_mean=float(kl.mean()), fp8_kl_p99=float(kl.quantile(0.99)))
        result["rows"].append(row)
        print(json.dumps(row), flush=True)
    for size in ("32", "8k"):
        names = [a.rows / f"gate{size}-{x}" for x in ("a", "b", "repeat")]
        if all(n.exists() for n in names):
            (_, _, _, la, fa), (_, _, _, lb, fb), (_, _, _, lr, fr) = (load_row(n) for n in names)
            result["gates"][size] = {
                "repeat_exact": torch.equal(la, lr) and torch.equal(fa, fr),
                "causal_prefix_exact": torch.equal(la[:, :20], lb[:, :20]) and torch.equal(fa[:20], fb[:20]),
                "mutated_suffix_differs": not torch.equal(fa[20:], fb[20:])}
    agreed = [r for r in result["rows"] if "agree" in r]
    if agreed:
        total = sum(r["generated"] for r in agreed)
        result["agreement"] = sum(r["agree"] * r["generated"] for r in agreed) / total
    result["tiles_ok"] = f"{sum(r.get('tiles', False) for r in result['rows'])}/{len(result['rows'])}"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: result[k] for k in result if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
