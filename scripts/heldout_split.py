#!/usr/bin/env python3
"""Hold out prompts by id hash, so every sample of a prompt lands in the same split.

``H(p) = sha256(salt + p) / 2**256``.  Rows of ``--count-sources`` sources are held out per category as the
``round(frac x category size)`` prompts with the smallest ``H``; rows of every other source when ``H < frac``.
The selection depends only on the prompt lists, never on what was captured, so rerunning it after more capture
gives the same prompts.  Pass the output as ``--heldout`` to ``python -m enntity_drafter.capture_io index``
(a row is held out when its name, id or ``prompt_id`` is listed).  Round 2b used the default salt, frac 0.03 and
``--count-sources template``.

    python scripts/heldout_split.py --prompts data/prompts/templates/train.jsonl \\
        --prompts data/prompts/coverage1/train.jsonl --count-sources template --out data/heldout-ids.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, action="append", required=True, help="prompt rows (repeatable)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--frac", type=float, default=0.03)
    ap.add_argument("--salt", default="r2b-heldout/")
    ap.add_argument("--count-sources", default="", help="comma list of sources carved per category by count")
    a = ap.parse_args(argv)

    def H(prompt_id: str) -> float:
        return int(hashlib.sha256(f"{a.salt}{prompt_id}".encode()).hexdigest(), 16) / 2**256

    rows = {r["id"]: r for path in a.prompts for r in map(json.loads, path.open())}
    counted = set(filter(None, a.count_sources.split(",")))
    held: list[dict] = []
    groups: dict[tuple[str, str], list[str]] = defaultdict(list)
    for pid, row in rows.items():
        if row.get("source") in counted:
            groups[(row["source"], row.get("category", ""))].append(pid)
        elif H(pid) < a.frac:
            held.append(row)
    for ids in groups.values():
        held += [rows[p] for p in sorted(ids, key=H)[: round(a.frac * len(ids))]]
    held.sort(key=lambda r: r["id"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text("".join(json.dumps({"id": r["id"], "source": r.get("source"), "category": r.get("category")})
                             + "\n" for r in held))
    print(json.dumps({"prompts": len(rows), "held_out": len(held),
                      "by_source": dict(Counter(r.get("source") for r in held)),
                      "by_category": dict(Counter(f"{r.get('source')}/{r.get('category')}" for r in held))}, indent=1))


if __name__ == "__main__":
    main()
