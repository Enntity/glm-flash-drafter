"""Training samples straight from Atlas hidden-state captures (no row-store rewrite).

Formats read
------------
* **pilot** (``atlas.hidden_capture.chunk/v1``, engine 327655e2): one directory
  per request with ``p<pos0>-n<rows>.{bin,json}``.  Each ``.bin`` is BF16
  slot-major ``[6, rows, 4096]``: layers 5,14,24,33,42 then the contracted
  pre-final-norm hidden.  The sidecar holds ``pos0``, ``rows`` and the token
  ids.  A directory may hold only a suffix of the chunks (e.g. a partial copy).
* **fast** (``atlas.hidden_capture.segment/v2``, engine 77dc4954): one
  directory per request tag with ``r<rank>-<L|F>-p<pos0>-n<rows>.{bin,json}``.
  ``L`` files are ``[5, rows, 4096]`` (the capture layers), ``F`` files
  ``[1, rows, 4096]`` (final).  Under sequence-parallel prefill each rank
  writes the layer rows it owns, so a request's ``L`` segments come from
  several ranks; they are merged by absolute position and must tile
  ``[from, T)`` exactly once (same for ``F``).  Only the last ``W`` context
  rows plus the generated span are stored (per-request window, ``from``).
* **fast FP8**: a fast segment converted after capture
  (``enntity_drafter.fp8``) keeps its sidecar with ``dtype`` ``fp8_e4m3`` and a
  ``.fp8`` file instead of the ``.bin``: e4m3 bytes ``[slots, rows, 4096]``
  followed by float32 scales ``[slots, rows]`` (one per row per slot,
  ``amax / 448``).  Reads dequantize to BF16 (``fp8_dequantize``); a row may mix
  BF16 and FP8 segments.

Neither format records where the generated span starts, so the index joins a
metadata file (capture records / items) per row name to get ``P``.

Sample rule (matches serving)
-----------------------------
Atlas feeds the drafter the target features of the last ``max_ctx`` = sliding
window ``W`` (2048) positions before the anchor (``impl_b3_dflash.rs``
accumulator) and Speculators trains with exactly that window (base KV in
``[a - W, a)``, ``dflash/attention.py``).  Anchors are the bonus token, i.e.
any generated position ``a`` in ``[P, T - B]`` (all ``B`` = 8 block slots are
real tokens).  When the stored rows start at ``lo > 0`` an anchor also needs
``a - W >= lo``; anchors without the full window are skipped, never padded
(serving never sees a truncated window except at true sequence start, which
is ``lo == 0``).

A sample is one row: a random crop of eligible anchors (``--crop`` anchors
wide, default 4096), up to ``max_anchors`` (512) anchors drawn uniformly
without replacement inside it, and the union of their windows
``[a - W, a + B)`` read from disk.  Disjoint windows become separate
"documents" in one packed sequence, so the index-space window of the
Speculators mask equals the absolute-position window; ``position_ids`` carry
absolute positions (RoPE as served).  Final-hidden rows are read only where a
target needs them (``[a - 1, a + B - 1)``).  ``loss_mask`` is the generated
span; the optional decode-vs-prefill agreement mask (:func:`agreement_keep`)
is computed on the device from the final slot.

I/O
---
Reads are ``np.memmap`` slices (no full loads); file handles are cached per
worker process; every capture file is passed through ``incoai_guard.check_path``
once.  A rank-local partition (``partition``) lets each FSDP rank read its own
NVMe; ``bench`` measures loader throughput.

CLI::

    python -m enntity_drafter.capture_io index --data-root R --pilot pilot-rows --fast fast-rows/<run>/p<i> ...
                                 --meta records.jsonl ... --heldout heldout.jsonl --out IDX
    python -m enntity_drafter.capture_io partition --index IDX/index.json --world 2 --out IDX/index-w2.json
    python -m enntity_drafter.capture_io bench --index IDX/index.json --data-root R --workers 8 --samples 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import threading
import time
from collections import Counter, OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import torch

from .guard import check_path

INDEX_SCHEMA = "enntity.capture_index/v1"
PILOT_SCHEMA = "atlas.hidden_capture.chunk/v1"
FAST_SCHEMA = "atlas.hidden_capture.segment/v2"
DEFAULT_LAYERS = [5, 14, 24, 33, 42]
NO_TRAIN_SPLITS = ("heldout", "heldout_extra")
FP8 = "fp8_e4m3"
FP8_MAX = 448.0
SUFFIX = {"bf16": ".bin", FP8: ".fp8"}


# --------------------------------------------------------------------------
# FP8 storage (e4m3, one float32 scale per row per slot)
# --------------------------------------------------------------------------

def fp8_quantize(bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """BF16 bits ``[..., hidden]`` (uint16) -> (e4m3 bytes, same shape; float32 scales ``[...]``).

    Scale = row amax / 448 (1 for an all-zero row); round to nearest even.  Raises on
    non-finite input (a capture with inf/nan stays BF16)."""

    x = torch.from_numpy(np.ascontiguousarray(bits)).view(torch.bfloat16).float()
    amax = x.abs().amax(-1)
    if not bool(torch.isfinite(amax).all()):
        raise ValueError("non-finite hidden state")
    scale = torch.where(amax > 0, amax / FP8_MAX, torch.ones_like(amax))
    q = (x / scale[..., None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(torch.uint8).numpy(), scale.numpy()


def fp8_dequantize(q: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """(e4m3 bytes ``[..., hidden]``, float32 scales ``[...]``) -> BF16 bits (uint16)."""

    x = torch.from_numpy(np.array(q, dtype=np.uint8)).view(torch.float8_e4m3fn).float()
    x *= torch.from_numpy(np.array(scale, dtype=np.float32))[..., None]
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def segment_bytes(dtype: str, slots: int, rows: int, hidden: int) -> int:
    if dtype == FP8:
        return slots * rows * (hidden + 4)
    return slots * rows * hidden * 2


# --------------------------------------------------------------------------
# Index building (reads JSON sidecars only)
# --------------------------------------------------------------------------

def _tiles(segments: list[list[Any]], kind: str, name: str) -> tuple[int, int]:
    """Segments ``[file, pos0, rows, ...]`` sorted by pos0 must tile one range once."""

    if not segments:
        raise ValueError(f"{name}: no {kind} segments")
    position = segments[0][1]
    for segment in segments:
        if segment[1] != position:
            raise ValueError(f"{name}: {kind} segments do not tile (gap/overlap at {position}, "
                             f"next starts at {segment[1]})")
        position += segment[2]
    return segments[0][1], position


def scan_pilot_dir(directory: Path, rel: str) -> dict[str, Any]:
    metas = []
    for sidecar in directory.glob("p*-n*.json"):
        meta = json.loads(sidecar.read_text())
        if meta.get("schema") != PILOT_SCHEMA or meta.get("dtype") != "bf16":
            raise ValueError(f"{sidecar}: not a bf16 {PILOT_SCHEMA} chunk")
        binary = sidecar.with_suffix(".bin")
        slots = len(meta["layers"]) + 1
        expected = slots * meta["rows"] * meta["hidden_size"] * 2
        if not binary.is_file() or binary.stat().st_size != expected:
            raise ValueError(f"{binary}: missing or size != {expected}")
        metas.append((meta, f"{rel}/{binary.name}"))
    metas.sort(key=lambda item: item[0]["pos0"])
    if not metas:
        raise ValueError(f"{directory}: no chunks")
    layers, hidden = metas[0][0]["layers"], metas[0][0]["hidden_size"]
    layer_segs, final_segs, ids = [], [], []
    for meta, file in metas:
        if meta["layers"] != layers or meta["hidden_size"] != hidden:
            raise ValueError(f"{directory}: layer/hidden mismatch")
        slots = len(layers) + 1
        layer_segs.append([file, meta["pos0"], meta["rows"], slots, 0])
        final_segs.append([file, meta["pos0"], meta["rows"], slots, slots - 1])
        if len(meta["token_ids"]) != meta["rows"]:
            raise ValueError(f"{file}: token ids != rows")
        ids += meta["token_ids"]
    lo, hi = _tiles(layer_segs, "chunk", str(directory))
    return {"format": "pilot", "layers": layers, "hidden_size": hidden, "lo": lo, "hi": hi,
            "layer_segs": layer_segs, "final_segs": final_segs, "ids": ids,
            "seq_tokens": metas[-1][0].get("seq_tokens")}


def scan_fast_dir(directory: Path, rel: str) -> dict[str, Any]:
    """Merge one tag's per-rank segments by absolute position."""

    if list(directory.glob("*.part")) or list(directory.glob("*.err")):
        raise ValueError(f"{directory}: incomplete (.part/.err present)")
    by_kind: dict[str, list[tuple[dict[str, Any], str]]] = {"L": [], "F": []}
    for sidecar in directory.glob("r*-*-p*-n*.json"):
        kind = sidecar.name.split("-")[1]
        meta = json.loads(sidecar.read_text())
        if meta.get("schema") != FAST_SCHEMA or meta.get("dtype") not in SUFFIX:
            raise ValueError(f"{sidecar}: not a bf16/{FP8} {FAST_SCHEMA} segment")
        binary = sidecar.with_suffix(SUFFIX[meta["dtype"]])
        slots = len(meta["layers"]) if kind == "L" else 1
        expected = segment_bytes(meta["dtype"], slots, meta["rows"], meta["hidden_size"])
        if not binary.is_file() or binary.stat().st_size != expected:
            raise ValueError(f"{binary}: missing or size != {expected}")
        by_kind[kind].append((meta, f"{rel}/{binary.name}"))
    for kind in by_kind:
        by_kind[kind].sort(key=lambda item: item[0]["pos0"])
    if not by_kind["L"] or not by_kind["F"]:
        raise ValueError(f"{directory}: missing L or F segments")
    layers = by_kind["L"][0][0]["layers"]
    hidden = by_kind["L"][0][0]["hidden_size"]
    layer_segs = [[f, m["pos0"], m["rows"], len(m["layers"]), 0] for m, f in by_kind["L"]]
    final_segs = [[f, m["pos0"], m["rows"], 1, 0] for m, f in by_kind["F"]]
    if any(m["layers"] != layers or m["hidden_size"] != hidden for m, _ in by_kind["L"]):
        raise ValueError(f"{directory}: layer/hidden mismatch across segments")
    lo, hi = _tiles(layer_segs, "L", str(directory))
    if _tiles(final_segs, "F", str(directory)) != (lo, hi):
        raise ValueError(f"{directory}: L and F ranges differ")
    ids: list[int | None] = [None] * (hi - lo)
    for meta, _ in by_kind["F"] + by_kind["L"]:
        for offset, token in enumerate(meta.get("token_ids") or []):
            slot = meta["pos0"] - lo + offset
            if ids[slot] is None:
                ids[slot] = token
            elif ids[slot] != token:
                raise ValueError(f"{directory}: rank token ids disagree at {lo + slot}")
    if any(token is None for token in ids):
        raise ValueError(f"{directory}: token ids missing for some positions")
    dtypes = {m["dtype"] for m, _ in by_kind["L"] + by_kind["F"]}
    return {"format": "fast", "layers": layers, "hidden_size": hidden, "lo": lo, "hi": hi,
            "layer_segs": layer_segs, "final_segs": final_segs, "ids": ids,
            "dtype": dtypes.pop() if len(dtypes) == 1 else "mixed",
            "seq_tokens": by_kind["F"][-1][0].get("seq_tokens"),
            "from": by_kind["F"][0][0].get("from"),
            "ranks": sorted({m["rank"] for m, _ in by_kind["L"] + by_kind["F"]}),
            "sp": any(m.get("sp") for m, _ in by_kind["L"])}


def _norm_name(name: str) -> str:
    return name.replace("/", "_").replace("#", "_")


# Alternative key spellings accepted in manifests / records / items.
_ALIASES = {
    "P": ("P", "prompt_tokens"),
    "generated": ("generated", "generated_tokens", "gen_tokens"),
    "stored": ("stored", "stored_tokens", "stored_rows"),
    "dir": ("dir", "path", "rel_dir"),
}


def read_records(path: str | Path) -> list[dict[str, Any]]:
    """JSONL, a JSON list, or a JSON object (a ``rows``/``items`` list, else one record)."""

    text = Path(path).read_text()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(value, list):
        return value
    rows = value.get("rows", value.get("items"))
    return rows if isinstance(rows, list) else [value]


def load_meta(paths: Iterable[str | Path]) -> dict[str, dict[str, Any]]:
    """Per-row metadata by request name, merged over manifests, capture records and items.

    ``P`` (first generated position) resolves, in order, to: ``P``/``prompt_tokens``;
    ``len(prompt_ids_used)`` (prompt re-tokenized by Atlas at capture);
    ``tokens - completion_tokens``; ``len(prompt_ids)``; else, at index time,
    ``T - generated`` (``generated``/``generated_tokens``/``completion_tokens``).
    Expected token ids (checked against the capture) are ``ids``, or the
    used/local prompt ids plus ``completion_ids``.  Earlier files win per key.
    """

    raw: dict[str, dict[str, Any]] = {}
    keys = ("P", "ids", "prompt_ids_used", "prompt_ids", "completion_ids", "tokens", "completion_tokens",
            "generated", "stored", "bytes", "dir", "from", "split",
            "id", "prompt_id", "source", "kind", "category", "on_policy", "group", "onpolicy_model")
    for path in paths:
        for record in read_records(path):
            if "error" in record:
                continue
            for canonical, aliases in _ALIASES.items():
                for alias in aliases:
                    if record.get(alias) is not None and canonical not in record:
                        record[canonical] = record[alias]
            name = record.get("name") or record.get("tag") or record["id"]
            entry = raw.setdefault(_norm_name(name), {})
            for key in keys:
                if record.get(key) is not None and key not in entry:
                    entry[key] = record[key]
    meta = {}
    for name, entry in raw.items():
        prompt = entry.get("prompt_ids_used") or entry.get("prompt_ids")
        info = {k: entry[k] for k in keys[7:] if k in entry}
        if "generated" not in info and "completion_tokens" in entry:
            info["generated"] = entry["completion_tokens"]
        if "P" in entry:
            info["P"] = int(entry["P"])
        elif entry.get("prompt_ids_used"):
            info["P"] = len(entry["prompt_ids_used"])
        elif "tokens" in entry and "completion_tokens" in entry:
            info["P"] = int(entry["tokens"]) - int(entry["completion_tokens"])
        elif entry.get("prompt_ids") is not None:
            info["P"] = len(entry["prompt_ids"])
        if "ids" in entry:
            info["ids"] = entry["ids"]
        elif prompt is not None and "completion_ids" in entry:
            info["ids"] = prompt + entry["completion_ids"]
        meta[name] = info
    return meta


def eligible_range(lo: int, hi: int, P: int, window: int, block: int) -> tuple[int, int]:
    """Inclusive anchor range ``[a_min, a_max]`` (empty when a_min > a_max)."""

    a_min = max(P, 1, lo + window if lo > 0 else 0)
    a_max = hi - block
    return a_min, a_max


def expand_roots(data_root: Path, roots: list[str]) -> list[str]:
    """Roots may be globs relative to the data root (e.g. ``fast-rows/round1-*/p*``)."""

    expanded = []
    for root in roots:
        if any(ch in root for ch in "*?["):
            expanded += sorted(str(p.relative_to(data_root)) for p in data_root.glob(root) if p.is_dir())
        else:
            expanded.append(root)
    return expanded


def build_index(data_root: Path, pilot: list[str], fast: list[str], meta_paths: list[str],
                heldout_path: str | None, out: Path, *, window: int, block: int,
                layers: list[int], manifest: list[str] | None = None,
                pilot_as_data: bool = False) -> dict[str, Any]:
    """Scan capture dirs and write the index.

    * ``manifest`` files (e.g. the round-1 capture manifest) are metadata and,
      when given, also the allowlist: only rows they list are indexed, and a
      listed row that is missing on disk is reported.
    * Pilot rows (FP32 mHC highway, off-serving numerics) are indexed under
      split ``pilot_offserving`` and never trained or evaluated unless
      ``pilot_as_data`` (loader tests only).
    * A request captured more than once (pilot and fast, or repeated runs) is
      indexed once: a fast (serving-equivalent) copy beats a pilot copy, then
      the most eligible anchors wins (ties: first root given).
    """

    meta = load_meta(list(manifest or []) + list(meta_paths))
    allow = set(load_meta(manifest)) if manifest else None
    pilot, fast = expand_roots(data_root, pilot), expand_roots(data_root, fast)
    heldout_ids = set()
    if heldout_path:
        with open(heldout_path) as stream:
            heldout_ids = {_norm_name(json.loads(line)["id"]) for line in stream}
    rows: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    skipped: Counter = Counter()
    skipped_rows: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    token_chunks: list[np.ndarray] = []
    token_offset = 0
    hidden_size = None
    candidates = [("pilot", root, d) for root in pilot for d in sorted((data_root / root).iterdir()) if d.is_dir()]
    candidates += [("fast", root, d) for root in fast for d in sorted((data_root / root).iterdir()) if d.is_dir()]
    for fmt, root, directory in candidates:
        name = directory.name
        if name.startswith("gate") or (allow is not None and name not in allow):
            continue
        rel = f"{root}/{name}"
        listed_dir = (meta.get(name) or {}).get("dir")
        if allow is not None and listed_dir and str(listed_dir).strip("/").split("/")[-3:] != rel.split("/")[-3:]:
            continue   # the manifest names this row's directory: other copies (older runs) are ignored
        try:
            check_path(directory / "x.bin", "features")
            scan = (scan_pilot_dir if fmt == "pilot" else scan_fast_dir)(directory, rel)
        except ValueError as error:
            skipped["unreadable"] += 1
            skipped_rows.append({"dir": rel, "reason": str(error)[:300]})
            continue
        if scan["layers"] != layers or scan["hidden_size"] != (hidden_size or scan["hidden_size"]):
            raise ValueError(f"{rel}: capture layers/hidden {scan['layers']}/{scan['hidden_size']} differ")
        hidden_size = scan["hidden_size"]
        info = meta.get(name)
        if info is not None and "P" not in info and info.get("generated") is not None:
            info = dict(info, P=scan["hi"] - int(info["generated"]))
        if info is None or "P" not in info:
            skipped["no_meta"] += 1
            skipped_rows.append({"dir": rel, "reason": "no metadata (P unknown)"})
            continue
        lo, hi, P = scan["lo"], scan["hi"], int(info["P"])
        problems = []
        if info.get("stored") is not None and int(info["stored"]) != hi - lo:
            problems.append(f"stored {info['stored']} != {hi - lo}")
        if fmt == "fast" and info.get("from") is not None and int(info["from"]) != lo:
            problems.append(f"from {info['from']} != {lo}")
        if problems:
            skipped["manifest_mismatch"] += 1
            skipped_rows.append({"dir": rel, "reason": "; ".join(problems)})
            continue
        if info.get("ids") is not None:
            expect = info["ids"]
            if len(expect) != hi or expect[lo:hi] != scan["ids"]:
                skipped["ids_mismatch"] += 1
                skipped_rows.append({"dir": rel, "reason": f"token ids differ from metadata ({len(expect)} vs T={hi})"})
                continue
        a_min, a_max = eligible_range(lo, hi, P, window, block)
        if fmt == "pilot" and not pilot_as_data:
            split = "pilot_offserving"   # FP32-highway pilot features: loader tests only
        elif (info.get("split") == "heldout"
              or heldout_ids & {name, _norm_name(str(info.get("id", ""))), _norm_name(str(info.get("prompt_id", "")))}):
            split = "heldout"
        elif info.get("kind") == "heldout" or info.get("split") in ("heldout_extra", "heldout-session"):
            split = "heldout_extra"   # captured from a held-out session but not in the current held-out set
        else:
            split = "train"
        anchors = max(0, a_max - a_min + 1)
        if name in seen:
            skipped["duplicate"] += 1
            if (seen[name]["format"] == "fast", seen[name]["anchors"]) >= (fmt == "fast", anchors):
                continue
            rows.remove(seen[name])
        token_chunks.append(np.asarray(scan["ids"], dtype=np.int32))
        nbytes = sum((data_root / s[0]).stat().st_size for s in scan["layer_segs"])
        if fmt == "fast":
            nbytes += sum((data_root / s[0]).stat().st_size for s in scan["final_segs"])
        seen[name] = {
            "name": name, "id": info.get("id", name), "prompt_id": info.get("prompt_id"), "format": fmt, "dir": rel, "lo": lo, "hi": hi, "P": P,
            "a_min": a_min, "a_max": a_max, "anchors": anchors, "split": split,
            "source": info.get("source"), "kind": info.get("kind"), "category": info.get("category"),
            "on_policy": info.get("on_policy"), "tok_off": token_offset, "bytes": nbytes,
            "layer_segs": scan["layer_segs"], "final_segs": scan["final_segs"],
            "dtype": scan.get("dtype", "bf16"),
            **({"sp": scan["sp"], "ranks": scan["ranks"]} if fmt == "fast" else {}),
        }
        rows.append(seen[name])
        token_offset += hi - lo
    if allow is not None:
        missing = sorted(allow - set(seen))
        skipped["manifest_rows_not_indexed"] = len(missing)
        skipped_rows += [{"name": n, "reason": "listed in manifest, not indexed"} for n in missing[:200]]
    tokens = np.concatenate(token_chunks) if token_chunks else np.zeros(0, np.int32)
    tokens.tofile(out / "tokens.i32")
    no_anchor = [r["name"] for r in rows if r["anchors"] == 0]
    summary = summarize(rows)
    index = {
        "schema": INDEX_SCHEMA, "window": window, "block": block, "capture_layers": layers,
        "hidden_size": hidden_size,
        "final_hidden": "pre-final-norm", "data_roots": {"pilot": pilot, "fast": fast},
        "tokens_file": "tokens.i32", "tokens_sha256": hashlib.sha256(tokens.tobytes()).hexdigest(),
        "manifest": [str(m) for m in manifest or []], "pilot_as_data": pilot_as_data,
        "rows": rows, "skipped": dict(skipped), "skipped_rows": skipped_rows,
        "rows_without_full_window_anchor": no_anchor, "summary": summary,
    }
    (out / "index.json").write_text(json.dumps(index))
    return index


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        for key in (row["split"], f"{row['split']}/{row['source']}"):
            group = groups[key]
            group["rows"] += 1
            group["stored_tokens"] += row["hi"] - row["lo"]
            group["eligible_anchors"] += row["anchors"]
            group["rows_with_anchors"] += row["anchors"] > 0
            group["gb"] += row["bytes"] / 1e9
            group[f"rows_{row.get('dtype', 'bf16')}"] += 1
    return {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in sorted(groups.items())}


def _needed_files(row: dict[str, Any], window: int) -> set[str]:
    """Files a sample of this row can read: those overlapping ``[a_min - W, hi)``."""

    start = max(row["lo"], row["a_min"] - window)
    needed = set()
    for file, pos0, rows, *_ in row["layer_segs"] + row["final_segs"]:
        if pos0 + rows > start:
            needed |= {file, file[:-4] + ".json"}
    return needed


def partition_index(index: dict[str, Any], world: int,
                    max_gb: dict[int, float] | None = None, stratify: str | None = None,
                    resident: dict[int, set[str]] | None = None) -> dict[str, Any]:
    """Assign train rows to ranks (greedy, balanced by eligible anchors).

    Held-out rows are needed on every rank (all ranks evaluate the same rows).
    ``max_gb`` caps a rank's local bytes (e.g. the free disk on a host the data
    must be copied to); rows that do not fit go to an uncapped rank, and the
    capped rank then simply cycles its smaller set more often.  ``stratify``
    (a row key, e.g. ``source``) balances the anchor load within each value of
    that key, so every rank holds about the same share of every source.
    ``resident`` restricts a rank to rows whose needed files it already holds
    (a list of files on that host): nothing new is copied to it.  Returns the
    index with ``rank`` set on train rows plus per-rank copy lists.
    """

    window = index["window"]
    max_gb = max_gb or {}
    resident = resident or {}
    groups: dict[Any, list[int]] = defaultdict(lambda: [0] * world)
    heldout_gb = sum(r["bytes"] for r in index["rows"] if r["split"] in NO_TRAIN_SPLITS and r["anchors"]) / 1e9
    used = [heldout_gb] * world
    train = sorted((r for r in index["rows"] if r["split"] == "train" and r["anchors"] > 0),
                   key=lambda r: -r["anchors"])
    for row in train:
        gb = row["bytes"] / 1e9
        needed = _needed_files(row, window) if resident else set()
        fits = [k for k in range(world) if used[k] + gb <= max_gb.get(k, float("inf"))
                and (k not in resident or needed <= resident[k])]
        if not fits:
            raise ValueError(f"row {row['name']} fits no rank under {max_gb}")
        group = groups[row.get(stratify) if stratify else None]
        rank = min(fits, key=lambda k: group[k])
        row["rank"] = rank
        group[rank] += row["anchors"]
        used[rank] += gb
    load = [sum(group[k] for group in groups.values()) for k in range(world)]
    files: dict[int, set[str]] = {rank: set() for rank in range(world)}
    for row in index["rows"]:
        if row["anchors"] == 0 or row["split"] not in ("train",) + NO_TRAIN_SPLITS:
            continue
        for rank in (range(world) if row["split"] != "train" else [row["rank"]]):
            files[rank] |= _needed_files(row, window)
    return dict(index, world=world, rank_anchor_load=load, rank_gb_upper_bound=used,
                **({"rank_anchor_load_by": {str(k): v for k, v in groups.items()}} if stratify else {}),
                copy_lists={str(k): sorted(v) for k, v in files.items()})


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

class SegmentReader:
    """Memory-mapped segment reads with a per-process LRU of open maps."""

    def __init__(self, data_root: str | Path, hidden: int, max_open: int = 256):
        self.root = Path(data_root)
        self.hidden = hidden
        self.max_open = max_open
        self.maps: OrderedDict[str, np.memmap] = OrderedDict()
        self.checked: set[str] = set()
        self.bytes_read = 0

    def _map(self, file: str) -> np.memmap | tuple[np.memmap, np.memmap]:
        """BF16: ``[slots * rows, hidden]`` uint16; FP8: (``[slots * rows, hidden]`` uint8, scales)."""

        mapped = self.maps.get(file)
        if mapped is not None:
            self.maps.move_to_end(file)
            return mapped
        path = self.root / file
        if file not in self.checked:
            check_path(path, "features")
            self.checked.add(file)
        if file.endswith(SUFFIX[FP8]):
            raw = np.memmap(path, dtype=np.uint8, mode="r")
            n = raw.size // (self.hidden + 4)
            mapped = (raw[: n * self.hidden].reshape(n, self.hidden), raw[n * self.hidden:].view(np.float32))
        else:
            mapped = np.memmap(path, dtype=np.uint16, mode="r").reshape(-1, self.hidden)
        self.maps[file] = mapped
        while len(self.maps) > self.max_open:
            self.maps.popitem(last=False)
        return mapped

    def read(self, segments: list[list[Any]], slots: int, start: int, end: int,
             out: np.ndarray, out_offset: int, scales: np.ndarray | None = None) -> None:
        """Copy rows ``[start, end)`` of ``slots`` slots into ``out[out_offset:, s, :]``.

        ``out`` uint16 (BF16 bits): FP8 segments are dequantized here.  ``out`` uint8 (FP8 kept, dequantized
        on the device by :func:`materialize_fp8`): the e4m3 bytes go to ``out`` and their scales to
        ``scales[out_offset:, s]``; every segment read must then be FP8."""

        for file, pos0, rows, nslots, slot0 in segments:
            a, b = max(start, pos0), min(end, pos0 + rows)
            if a >= b:
                continue
            mapped = self._map(file)
            target = out[out_offset + a - start: out_offset + b - start]
            for s in range(slots):
                base = (slot0 + s) * rows - pos0
                if out.dtype == np.uint8:
                    if not isinstance(mapped, tuple):
                        raise ValueError(f"{file}: BF16 segment read into an FP8 sample")
                    target[:, s, :] = mapped[0][base + a: base + b]
                    scales[out_offset + a - start: out_offset + b - start, s] = mapped[1][base + a: base + b]
                elif isinstance(mapped, tuple):
                    target[:, s, :] = fp8_dequantize(mapped[0][base + a: base + b], mapped[1][base + a: base + b])
                else:
                    target[:, s, :] = mapped[base + a: base + b]
            self.bytes_read += (b - a) * slots * (self.hidden + 4 if isinstance(mapped, tuple) else self.hidden * 2)


def merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def choose_anchors(row: dict[str, Any], rng: random.Random, max_anchors: int, crop: int) -> list[int]:
    """Random crop of the eligible anchor range, then up to ``max_anchors`` anchors inside it."""

    a_min, a_max = row["a_min"], row["a_max"]
    count = a_max - a_min + 1
    if count <= 0:
        return []
    if crop and count > crop:
        a_min = rng.randint(a_min, a_max - crop + 1)
        count = crop
    chosen = rng.sample(range(a_min, a_min + count), min(max_anchors, count))
    return sorted(chosen)


class CaptureIndex:
    def __init__(self, index_path: str | Path, data_root: str | Path):
        self.path = Path(index_path)
        check_path(self.path, "features")
        raw = self.path.read_bytes()
        self.digest = hashlib.sha256(raw).hexdigest()
        self.index = json.loads(raw)
        if self.index.get("schema") != INDEX_SCHEMA:
            raise ValueError(f"{index_path}: schema {self.index.get('schema')!r} != {INDEX_SCHEMA}")
        self.rows = self.index["rows"]
        self.window = self.index["window"]
        self.block = self.index["block"]
        self.hidden = self.index["hidden_size"]
        self.layers = self.index["capture_layers"]
        self.data_root = Path(data_root)
        self.tokens = np.memmap(self.path.parent / self.index["tokens_file"], dtype=np.int32, mode="r")
        self.final_hidden = self.index["final_hidden"]

    def split(self, name: str, rank: int | None = None) -> list[dict[str, Any]]:
        rows = [r for r in self.rows if r["split"] == name and r["anchors"] > 0]
        if rank is not None and name == "train" and "world" in self.index:
            rows = [r for r in rows if r.get("rank") == rank]
        return rows

    def row_tokens(self, row: dict[str, Any], start: int, end: int) -> np.ndarray:
        offset = row["tok_off"] - row["lo"]
        return self.tokens[offset + start: offset + end]


def _row_spans(row: dict[str, Any], anchors: list[int], window: int, block: int):
    """(context spans, final-slot spans) of one row's sample: the union of ``[a - W, a + B)`` and ``[a - 1, a + B - 1)``."""

    spans = merge_intervals((max(row["lo"], a - window), a + block) for a in anchors)
    finals = merge_intervals((max(row["lo"], a - 1), a + block - 1) for a in anchors)
    return spans, finals


def build_sample(index: CaptureIndex, reader: SegmentReader, row: dict[str, Any],
                 anchors: list[int], fp8_device: bool = False) -> dict[str, Any]:
    """Packed sample for the Speculators forward plus the compact anchor indices (one row)."""

    return build_bundle(index, reader, [(row, anchors)], fp8_device)


def build_bundle(index: CaptureIndex, reader: SegmentReader, parts: list[tuple[dict[str, Any], list[int]]],
                 fp8_device: bool = False) -> dict[str, Any]:
    """One packed sample from one or more (row, anchors) parts.

    Every disjoint context window of every part is its own document (document ids are unique over the bundle),
    ``position_ids`` are each row's absolute positions, ``loss_mask`` each row's generated span, so the
    Speculators mask -- and every loss term -- of the bundle equals the sum of the parts' own samples.

    ``fp8_device``: when every part is an FP8 row, the e4m3 bytes and scales are handed over as they are
    (``hidden_states_fp8`` / ``hidden_scale`` and ``verifier_last_hidden_states_fp8`` / ``verifier_scale``) and
    dequantized on the device (:func:`materialize_fp8`, same arithmetic as :func:`fp8_dequantize`): half the
    bytes per read and no CPU codec work in the reader threads.  Otherwise FP8 segments are dequantized here."""

    window, block, hidden = index.window, index.block, index.hidden
    nl = len(index.layers)
    plans = [(row, anchors, *_row_spans(row, anchors, window, block)) for row, anchors in parts]
    total = sum(b - a for _, _, spans, _ in plans for a, b in spans)
    fp8 = fp8_device and all(row.get("dtype") == FP8 for row, _ in parts)
    dtype = np.uint8 if fp8 else np.uint16
    layer_rows = np.empty((total, nl, hidden), dtype=dtype)
    final_rows = np.zeros((total, 1, hidden), dtype=dtype)
    layer_scale = np.ones((total, nl), dtype=np.float32) if fp8 else None
    final_scale = np.ones((total, 1), dtype=np.float32) if fp8 else None
    ids = np.empty(total, dtype=np.int64)
    positions = np.empty(total, dtype=np.int64)
    documents = np.empty(total, dtype=np.int64)
    loss_mask = np.empty(total, dtype=bool)
    compact: list[int] = []
    offset = doc = 0
    for row, anchors, spans, finals in plans:
        offsets = []
        for start, end in spans:
            reader.read(row["layer_segs"], nl, start, end, layer_rows, offset, layer_scale)
            ids[offset: offset + end - start] = index.row_tokens(row, start, end)
            positions[offset: offset + end - start] = np.arange(start, end)
            documents[offset: offset + end - start] = doc
            loss_mask[offset: offset + end - start] = np.arange(start, end) >= row["P"]
            for f_start, f_end in finals:
                a, b = max(start, f_start), min(end, f_end)
                if a < b:
                    reader.read(row["final_segs"], 1, a, b, final_rows, offset + a - start, final_scale)
            offsets.append((start, end, offset))
            offset += end - start
            doc += 1
        for a in anchors:
            for start, end, base in offsets:
                if start <= a < end:
                    compact.append(base + a - start)
                    break
    supervised = sum(int(loss_mask[c + 1: c + block].sum()) for c in compact)
    inputs = {
        "input_ids": torch.from_numpy(ids).unsqueeze(0),
        "loss_mask": torch.from_numpy(loss_mask).unsqueeze(0),
        "document_ids": torch.from_numpy(documents).unsqueeze(0),
        "position_ids": torch.from_numpy(positions).unsqueeze(0),
    }
    if fp8:
        inputs.update(hidden_states_fp8=torch.from_numpy(layer_rows.reshape(1, total, nl * hidden)),
                      hidden_scale=torch.from_numpy(layer_scale).unsqueeze(0),
                      verifier_last_hidden_states_fp8=torch.from_numpy(final_rows.reshape(1, total, hidden)),
                      verifier_scale=torch.from_numpy(final_scale).unsqueeze(0))
    else:
        inputs.update(
            hidden_states=torch.from_numpy(layer_rows.reshape(total, nl * hidden)).view(torch.bfloat16).unsqueeze(0),
            verifier_last_hidden_states=torch.from_numpy(final_rows.reshape(total, hidden)).view(torch.bfloat16)
            .unsqueeze(0))
    names = [row["name"] for row, _ in parts]
    return {
        "inputs": inputs,
        "anchors": compact,
        "row": names[0] if len(names) == 1 else f"{names[0]}+{len(names) - 1}",
        "rows": names,
        "on_policy": all(row.get("on_policy") is not False for row, _ in parts),
        "context_rows": total,
        "supervised": supervised,
    }


FP8_INPUTS = (("hidden_states_fp8", "hidden_scale", "hidden_states"),
              ("verifier_last_hidden_states_fp8", "verifier_scale", "verifier_last_hidden_states"))


def materialize_fp8(inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Replace FP8 inputs (e4m3 bytes ``[1, n, slots * hidden]`` + float32 scales ``[1, n, slots]``) by BF16
    tensors, on whatever device they are: ``(e4m3 -> fp32) * scale -> bf16``, one slot at a time (bit-identical
    to :func:`fp8_dequantize`)."""

    for q_key, s_key, out_key in FP8_INPUTS:
        if q_key not in inputs:
            continue
        q, scale = inputs.pop(q_key), inputs.pop(s_key)
        slots = scale.shape[-1]
        q = q.view(torch.float8_e4m3fn).view(*q.shape[:-1], slots, -1)
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        for s in range(slots):
            out[..., s, :] = (q[..., s, :].float() * scale[..., s, None].float()).to(torch.bfloat16)
        inputs[out_key] = out.view(*q.shape[:-2], -1)
    return inputs


# --------------------------------------------------------------------------
# Deterministic training stream (DataLoader workers)
# --------------------------------------------------------------------------

def parse_source_weights(spec: str) -> list[tuple[tuple[str, ...], float]]:
    """``template=0.4,code+agentic=0.4,structured=0.2`` -> [((sources), share)], shares summing to 1.
    A ``+`` group shares its weight between its sources in proportion to their eligible anchors."""

    groups = []
    for part in spec.split(","):
        names, _, share = part.partition("=")
        groups.append((tuple(n.strip() for n in names.split("+")), float(share)))
    total = sum(share for _, share in groups)
    if total <= 0 or any(share < 0 for _, share in groups):
        raise ValueError(f"bad source weights {spec!r}")
    return [(names, share / total) for names, share in groups]


class CaptureTrainStream(torch.utils.data.Dataset):
    """Map-style: item ``i`` is this rank's ``i``-th sample, a pure function of
    (seed, rank, i).  Each epoch is a shuffled multiset in which a row appears
    ``ceil(eligible anchors / crop)`` times, so rows are visited in proportion
    to their generated tokens; resume replays the same stream exactly.

    ``source_weights`` (:func:`parse_source_weights`) switches to source-balanced
    coverage sampling: every source cycles its own items, and the source of item
    ``i`` follows a deterministic smooth weighted round robin whose frequencies
    make each source's share of *anchors* equal its weight.  An item is one
    group of a row chunk: a row's eligible range is cut into ``ceil(N / crop)``
    equal chunks, each chunk's positions are shuffled once per source epoch and
    dealt into ``ceil(len / max_anchors)`` strided groups, so one source epoch
    uses every eligible position of every row as an anchor exactly once.

    ``pack`` makes an item a fixed bundle of such groups (short rows
    share a micro-step, :meth:`_bundles`, :func:`build_bundle`); :meth:`cap_epochs`
    re-weights from a given item on so that no source exceeds a number of epochs
    over the planned run; ``fp8_device`` hands FP8 rows to the device undecoded.
    """

    def __init__(self, index_path: str | Path, data_root: str | Path, *, rank: int, world: int,
                 seed: int, max_anchors: int, crop: int, source_weights: str | None = None,
                 pack: bool = False, fp8_device: bool = False):
        self.index_path, self.data_root = str(index_path), str(data_root)
        self.rank, self.world, self.seed = rank, world, seed
        self.max_anchors, self.crop = max_anchors, crop
        self.pack, self.fp8_device = pack, fp8_device
        self._index: CaptureIndex | None = None
        self._reader: SegmentReader | None = None
        index = self.index
        partitioned = "world" in index.index
        if partitioned and index.index["world"] != world:
            raise ValueError(f"index partitioned for world {index.index['world']}, running {world}")
        self.rows = index.split("train", rank if partitioned else None)
        self.stride = 1 if partitioned else world
        self.offset = 0 if partitioned else rank
        if source_weights and not partitioned:   # unpartitioned: deal each source's rows to the ranks
            by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in self.rows:
                by_source[str(row.get("source"))].append(row)
            self.rows = [row for source in sorted(by_source) for row in by_source[source][rank::world]]
            self.stride, self.offset = 1, 0
        if not self.rows:
            raise ValueError(f"rank {rank}: no train rows with eligible anchors")
        self.epoch_items = [i for i, row in enumerate(self.rows)
                            for _ in range(max(1, math.ceil(row["anchors"] / crop)) if crop else 1)]
        self._order_cache: tuple[int, list[int]] | None = None
        self._lock = threading.Lock()
        self.balanced = None
        if source_weights:
            self.balanced = self._balanced_plan(parse_source_weights(source_weights))

    # -- source-balanced coverage ------------------------------------------------------------------------
    def _chunks(self, row: dict[str, Any]) -> list[tuple[int, int]]:
        count = row["a_max"] - row["a_min"] + 1
        pieces = max(1, math.ceil(count / self.crop)) if self.crop else 1
        return [(row["a_min"] + c * count // pieces, row["a_min"] + (c + 1) * count // pieces) for c in range(pieces)]

    def _balanced_plan(self, groups: list[tuple[tuple[str, ...], float]]) -> dict[str, Any]:
        by_source: dict[str, list[int]] = defaultdict(list)
        for i, row in enumerate(self.rows):
            by_source[str(row.get("source"))].append(i)
        named = {name for names, _ in groups for name in names}
        if set(by_source) - named:
            raise ValueError(f"rank {self.rank}: sources {sorted(set(by_source) - named)} have no weight")
        sources, items, pool, share = [], {}, {}, {}
        for names, weight in groups:
            present = [n for n in names if by_source.get(n)]
            if not present:
                if weight > 0:
                    raise ValueError(f"rank {self.rank}: no train rows for {'+'.join(names)}")
                continue
            anchors = {n: sum(self.rows[i]["anchors"] for i in by_source[n]) for n in present}
            for n in present:
                sources.append(n)
                share[n] = weight * anchors[n] / sum(anchors.values())
                pool[n] = anchors[n]
                entries = [(i, c, j, groups_)
                           for i in by_source[n] for c, (a, b) in enumerate(self._chunks(self.rows[i]))
                           for groups_ in [max(1, math.ceil((b - a) / self.max_anchors))] for j in range(groups_)]
                items[n] = self._bundles(n, entries) if self.pack else [(e,) for e in entries]
        plan = {"sources": sources, "items": items, "pool": pool, "share": share, "target_share": dict(share),
                "schedule": [], "counts": [], "credit": [0.0] * len(sources), "orders": {}, "switch": None}
        plan["weights"] = self._frequencies(plan, share)
        return plan

    @staticmethod
    def _frequencies(plan: dict[str, Any], share: dict[str, float]) -> list[float]:
        """Draw frequencies whose anchor shares equal ``share``: frequency ~ share / mean anchors per item."""

        frequency = {n: share[n] / (plan["pool"][n] / len(plan["items"][n])) for n in plan["sources"]}
        total = sum(frequency.values())
        return [frequency[n] / total for n in plan["sources"]]

    # -- packing short rows ------------------------------------------------------------------------------
    def _entry_size(self, entry: tuple[int, int, int, int]) -> tuple[int, int]:
        """(anchors, upper bound on context rows) of one (row, chunk, group, groups) entry."""

        row_i, chunk, group, groups = entry
        row = self.rows[row_i]
        a, b = self._chunks(row)[chunk]
        window, block = self.index.window, self.index.block
        return len(range(a + group, b, groups)), min(row["hi"], b - 1 + block) - max(row["lo"], a - window)

    def _bundles(self, source: str, entries: list[tuple[int, int, int, int]]) -> list[tuple]:
        """Pack a source's entries into bundles of at most ``max_anchors`` anchors and ``crop + W + B`` context rows
        (the probe's worst case), FP8 and BF16 rows apart (an FP8 bundle stays FP8 to the device).  Fixed for the
        run: entries are shuffled once (seeded), then placed first-fit into the last 16 open bundles.  An epoch
        still uses every eligible position exactly once (anchors are re-dealt inside each entry per epoch)."""

        cap_anchors = self.max_anchors
        cap_context = (self.crop or 0) + self.index.window + self.index.block
        order = list(entries)
        random.Random(f"{self.seed}/{self.rank}/{source}/bundles").shuffle(order)
        bundles: list[list] = []
        open_: dict[bool, list[list]] = {True: [], False: []}   # per dtype class: [entries, anchors, context]
        for entry in order:
            anchors, context = self._entry_size(entry)
            kind = self.rows[entry[0]].get("dtype") == FP8
            target = next((b for b in open_[kind] if b[1] + anchors <= cap_anchors and b[2] + context <= cap_context),
                          None)
            if target is None:
                target = [[], 0, 0]
                bundles.append(target)
                open_[kind].append(target)
                if len(open_[kind]) > 16:
                    open_[kind].pop(0)
            target[0].append(entry)
            target[1] += anchors
            target[2] += context
        return [tuple(b[0]) for b in bundles]

    # -- per-source epoch cap ----------------------------------------------------------------------------
    def cap_epochs(self, total_items: int, max_epochs: float, switch_item: int,
                   fill: set[str] | None = None) -> dict[str, Any]:
        """Re-weight from item ``switch_item`` on so that no source is drawn more than ``max_epochs`` epochs over
        ``total_items`` items (the run's plan); the share a capped source gives up goes to the uncapped sources in
        ``fill`` (default: all uncapped) in proportion to their target shares; other uncapped sources keep their
        target share.  Deterministic in (plan, switch_item): a resumed run calls it again with the same arguments
        before reading.  Returns the planned shares/epochs."""

        plan = self.balanced
        if plan is None:
            return {}
        with self._lock:
            if len(plan["schedule"]) > switch_item:
                raise RuntimeError(f"cap_epochs: schedule already at {len(plan['schedule'])} > switch {switch_item}")
            target, pool = plan["target_share"], plan["pool"]
            mean = {n: pool[n] / len(plan["items"][n]) for n in plan["sources"]}
            share = dict(target)
            capped: set[str] = set()
            for _ in range(100):
                anchors_total = total_items / sum(share[n] / mean[n] for n in plan["sources"])
                new_caps = {n for n in plan["sources"] if n not in capped
                            and share[n] * anchors_total > max_epochs * pool[n] * (1 + 1e-9)}
                capped |= new_caps
                fixed = {n: min(target[n], max_epochs * pool[n] / anchors_total) for n in capped}
                fixed.update({n: target[n] for n in plan["sources"]
                              if n not in capped and fill is not None and n not in fill})
                free = [n for n in plan["sources"] if n not in fixed]
                rest = 1.0 - sum(fixed.values())
                weight = sum(target[n] for n in free)
                new = {**fixed, **{n: rest * target[n] / weight for n in free}} if free else fixed
                if all(abs(new[n] - share[n]) < 1e-12 for n in plan["sources"]) and not new_caps:
                    break
                share = new
            plan["switch"] = (switch_item, self._frequencies(plan, share))
            plan["share"] = share
            anchors_total = total_items / sum(share[n] / mean[n] for n in plan["sources"])
            return {"switch_item": switch_item, "total_items": total_items, "max_epochs": max_epochs,
                    "capped": sorted(capped), "infeasible": len(capped) == len(plan["sources"]),
                    "share": {n: round(share[n], 4) for n in plan["sources"]},
                    "planned_epochs": {n: round(share[n] * anchors_total / pool[n], 3) for n in plan["sources"]}}

    def _schedule_to(self, item: int) -> None:
        """Extend the smooth weighted round robin (source, index within source) through ``item``."""

        plan = self.balanced
        schedule, credit = plan["schedule"], plan["credit"]
        seen = plan["counts"][-1][:] if plan["counts"] else [0] * len(plan["weights"])
        while len(schedule) <= item:
            if plan["switch"] is not None and len(schedule) == plan["switch"][0]:
                plan["weights"] = plan["switch"][1]
            weights = plan["weights"]
            for k, weight in enumerate(weights):
                credit[k] += weight
            pick = max(range(len(weights)), key=lambda k: credit[k])
            credit[pick] -= 1.0
            schedule.append((pick, seen[pick]))
            seen[pick] += 1
            plan["counts"].append(seen[:])

    def _balanced_item(self, item: int) -> list[tuple[dict[str, Any], list[int]]]:
        """The (row, anchors) parts of item ``item`` (one part unless ``pack``)."""

        plan = self.balanced
        with self._lock:
            self._schedule_to(item)
            source_k, within = plan["schedule"][item]
            source = plan["sources"][source_k]
            items = plan["items"][source]
            epoch, offset = divmod(within, len(items))
            key = (source, epoch)
            order = plan["orders"].get(key)
            if order is None:
                order = list(range(len(items)))
                random.Random(f"{self.seed}/{self.rank}/{source}/{epoch}").shuffle(order)
                plan["orders"] = {k: v for k, v in plan["orders"].items() if k[0] != source}
                plan["orders"][key] = order
        parts = []
        for row_i, chunk, group, groups in items[order[offset]]:
            row = self.rows[row_i]
            a, b = self._chunks(row)[chunk]
            positions = list(range(a, b))
            random.Random(f"{self.seed}/{row['name']}/{chunk}/{epoch}").shuffle(positions)
            parts.append((row, sorted(positions[group::groups])))
        return parts

    def source_report(self, items: int) -> dict[str, dict[str, float]]:
        """Anchors this rank draws per source in its first ``items`` items, and the epochs that makes."""

        plan = self.balanced
        if plan is None or items <= 0:
            return {}
        with self._lock:
            self._schedule_to(items - 1)
            counts = plan["counts"][items - 1]
        report = {}
        for k, source in enumerate(plan["sources"]):
            entries = plan["items"][source]
            per_epoch = len(entries)
            full, rest = divmod(counts[k], per_epoch)
            # anchors: full epochs use the whole pool; the partial epoch ~ its item share
            anchors = full * plan["pool"][source] + rest * plan["pool"][source] / per_epoch
            report[source] = {"items": counts[k], "anchors": round(anchors), "pool_anchors": plan["pool"][source],
                              "epochs": round(anchors / plan["pool"][source], 3), "target_share": round(plan["share"][source], 4)}
        total = sum(r["anchors"] for r in report.values())
        for r in report.values():
            r["anchor_share"] = round(r["anchors"] / total, 4)
        return report

    @property
    def index(self) -> CaptureIndex:
        if self._index is None:
            self._index = CaptureIndex(self.index_path, self.data_root)
        return self._index

    @property
    def reader(self) -> SegmentReader:
        """One reader (memmap LRU) per thread: prefetch threads share nothing mutable."""

        if self._reader is None:
            self._reader = threading.local()
        if not hasattr(self._reader, "value"):
            self._reader.value = SegmentReader(self.data_root, self.index.hidden)
        return self._reader.value

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_index"] = state["_reader"] = state["_order_cache"] = None
        state.pop("_lock", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return 1 << 40

    def row_for(self, item: int) -> dict[str, Any]:
        global_index = item * self.stride + self.offset
        epoch, offset = divmod(global_index, len(self.epoch_items))
        with self._lock:
            cache = self._order_cache
            if cache is None or cache[0] != epoch:
                order = list(self.epoch_items)
                random.Random(self.seed * 1_000_003 + epoch * 7 + self.rank * (self.stride == 1)).shuffle(order)
                cache = self._order_cache = (epoch, order)
        return self.rows[cache[1][offset]]

    def __getitem__(self, item: int) -> dict[str, Any]:
        if self.balanced is not None:
            parts = self._balanced_item(item)
        else:
            row = self.row_for(item)
            rng = random.Random(f"{self.seed}/{self.rank}/{item}")
            parts = [(row, choose_anchors(row, rng, self.max_anchors, self.crop))]
        started = time.perf_counter()
        sample = build_bundle(self.index, self.reader, parts, self.fp8_device)
        sample["read_s"] = time.perf_counter() - started
        sample["item"] = item
        return sample


class _Range(torch.utils.data.Sampler):
    def __init__(self, start: int, stop: int):
        self.start, self.stop = start, stop

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.start, self.stop))

    def __len__(self) -> int:
        return self.stop - self.start


def prefetch_map(fn, items: Iterable[Any], threads: int, depth: int | None = None,
                 stats: dict[str, float] | None = None) -> Iterator[Any]:
    """``fn(item)`` for each item, in order, built ahead by a bounded thread pool.

    Samples are hundreds of MB; worker processes would copy each one through
    shared memory, while threads hand it over by reference.  The memmap copies
    release the GIL, so reads overlap with the training step.  ``stats`` (optional)
    accumulates the queue depth seen at each hand-over: ``takes``, ``ready_sum``
    (finished samples waiting) and ``head_not_ready`` (the consumer had to wait).
    """

    from concurrent.futures import ThreadPoolExecutor

    threads = max(1, threads)
    depth = depth or 2 * threads
    items = iter(items)
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="capture-read") as pool:
        pending: list[Any] = []
        for item in items:
            pending.append(pool.submit(fn, item))
            if len(pending) >= depth:
                break
        while pending:
            if stats is not None:
                stats["takes"] = stats.get("takes", 0) + 1
                stats["ready_sum"] = stats.get("ready_sum", 0) + sum(f.done() for f in pending)
                stats["head_not_ready"] = stats.get("head_not_ready", 0) + (not pending[0].done())
            future = pending.pop(0)
            for item in items:
                pending.append(pool.submit(fn, item))
                break
            yield future.result()


def pin_sample(sample: dict[str, Any]) -> dict[str, Any]:
    """Page-lock the input tensors (host-side, in the reader thread) so the device copy is async."""

    sample["inputs"] = {k: v.pin_memory() for k, v in sample["inputs"].items()}
    return sample


def thread_prefetch(stream: CaptureTrainStream, start: int, stop: int, threads: int,
                    depth: int | None = None, pin: bool = False,
                    stats: dict[str, float] | None = None) -> Iterator[dict[str, Any]]:
    """Samples ``start..stop`` in order (see ``prefetch_map``); ``pin``: page-locked inputs."""

    fn = (lambda item: pin_sample(stream[item])) if pin else stream.__getitem__
    return prefetch_map(fn, range(start, stop), threads, depth, stats)


def train_loader(stream: CaptureTrainStream, start: int, stop: int, workers: int,
                 pin_memory: bool, mode: str = "thread", pin_threads: bool = False,
                 stats: dict[str, float] | None = None):
    """``thread`` (default): in-process prefetch threads (``pin_threads``: they also page-lock
    the inputs; ``stats``: queue depth, see ``prefetch_map``); ``process``: DataLoader workers."""

    if mode == "thread":
        return thread_prefetch(stream, start, stop, max(1, workers), pin=pin_threads, stats=stats)
    kwargs: dict[str, Any] = {}
    if workers:
        kwargs = {"prefetch_factor": 2, "persistent_workers": False,
                  "multiprocessing_context": "spawn" if sys.platform == "darwin" else "fork"}
    return torch.utils.data.DataLoader(stream, batch_size=None, sampler=_Range(start, stop),
                                       num_workers=workers, pin_memory=pin_memory, **kwargs)


def eval_samples(index: CaptureIndex, rows: list[dict[str, Any]], *, seed: int, max_anchors: int,
                 crop: int, threads: int = 1, fp8_device: bool = False) -> Iterator[dict[str, Any]]:
    """Deterministic held-out samples (same anchors on every rank and every eval),
    in row order, read ahead by ``threads`` threads (one reader per thread)."""

    readers = threading.local()

    def build(row: dict[str, Any]) -> dict[str, Any] | None:
        anchors = choose_anchors(row, random.Random(f"eval/{seed}/{row['name']}"), max_anchors, crop)
        if not anchors:
            return None
        if not hasattr(readers, "value"):
            readers.value = SegmentReader(index.data_root, index.hidden)
        return build_sample(index, readers.value, row, anchors, fp8_device)

    for sample in prefetch_map(build, rows, threads):
        if sample is not None:
            yield sample


# --------------------------------------------------------------------------
# Optional decode-vs-prefill agreement mask (device side)
# --------------------------------------------------------------------------

@torch.no_grad()
def agreement_keep(model: Any, inputs: dict[str, torch.Tensor], anchors: list[int], mode: str,
                   chunk: int = 2048) -> torch.Tensor:
    """Return ``loss_mask`` with slot positions dropped where the stored token
    disagrees with the target's own prefill prediction from the final slot.

    ``argmax``: keep positions where argmax(target(final[i-1])) == token[i].
    ``minprob:<p>``: keep positions where the target gives the token >= p
    (only real outliers; the pilot measured argmax agreement 83% at T=1,
    which equals E[p1], so ``argmax`` would drop ~17% of good positions).
    """

    block = model.block_size
    mask = inputs["loss_mask"].clone()
    slots = torch.tensor([a + k for a in anchors for k in range(1, block)], device=mask.device)
    slots = slots[mask[0, slots]]
    if slots.numel() == 0:
        return mask
    keep = torch.empty(slots.numel(), dtype=torch.bool, device=mask.device)
    final = inputs["verifier_last_hidden_states"]
    tokens = inputs["input_ids"][0, slots]
    threshold = float(mode.split(":", 1)[1]) if mode.startswith("minprob:") else None
    for start in range(0, slots.numel(), chunk):
        part = slots[start: start + chunk]
        logits = model.verifier_lm_head(model.verifier_norm(final[:, part - 1]))[0].float()
        if threshold is None:
            keep[start: start + chunk] = logits.argmax(-1) == tokens[start: start + chunk]
        else:
            logp = logits.log_softmax(-1).gather(-1, tokens[start: start + chunk, None])[:, 0]
            keep[start: start + chunk] = logp >= math.log(threshold)
    mask[0, slots[~keep]] = False
    return mask


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _bench(args: argparse.Namespace) -> dict[str, Any]:
    stream = CaptureTrainStream(args.index, args.data_root, rank=args.rank, world=args.world, seed=args.seed,
                                max_anchors=args.anchors, crop=args.crop)
    loader = train_loader(stream, 0, args.samples, args.workers, pin_memory=False, mode=args.mode)
    started = time.perf_counter()
    rows = context = supervised = 0
    first = None
    for sample in loader:
        if first is None:
            first = time.perf_counter() - started
        rows += 1
        context += sample["context_rows"]
        supervised += sample["supervised"]
    wall = time.perf_counter() - started
    hidden = stream.index.hidden
    gbytes = context * (len(stream.index.layers) * hidden * 2) / 1e9
    return {"samples": rows, "wall_s": round(wall, 2), "first_sample_s": round(first or 0, 2),
            "samples_per_s": round(rows / wall, 2), "context_rows_per_s": round(context / wall),
            "supervised_tokens_per_s": round(supervised / wall), "layer_gb_per_s": round(gbytes / wall, 3),
            "mean_context_rows": round(context / max(rows, 1)), "mean_supervised": round(supervised / max(rows, 1)),
            "workers": args.workers, "mode": args.mode, "crop": args.crop, "anchors": args.anchors}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("index")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--pilot", action="append", default=[], help="dir (or glob) of pilot request dirs, relative to root")
    p.add_argument("--fast", action="append", default=[], help="dir (or glob) of fast tag dirs, relative to root")
    p.add_argument("--meta", action="append", default=[], help="capture records / items jsonl")
    p.add_argument("--manifest", action="append", default=[],
                   help="capture manifest (JSON/JSONL): metadata + allowlist of rows to index")
    p.add_argument("--pilot-as-data", action="store_true",
                   help="train/eval on pilot rows (off-serving FP32-highway features; loader tests only)")
    p.add_argument("--heldout", help="held-out prompt jsonl (ids)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window", type=int, default=2048)
    p.add_argument("--block", type=int, default=8)
    p = sub.add_parser("partition")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--world", type=int, required=True)
    p.add_argument("--max-gb", action="append", default=[], help="RANK:GB cap on a rank's local data")
    p.add_argument("--stratify", default=None, help="row key to balance within (e.g. source)")
    p.add_argument("--resident", action="append", default=[],
                   help="RANK:FILE listing data files already on that rank's host (relative to the data root, first "
                        "token per line); that rank only gets rows it fully holds")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("bench")
    p.add_argument("--index", type=Path, required=True)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--mode", choices=["thread", "process"], default="thread")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--anchors", type=int, default=512)
    p.add_argument("--crop", type=int, default=4096)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261014)
    args = parser.parse_args(argv)
    if args.cmd == "index":
        index = build_index(args.data_root, args.pilot, args.fast, args.meta, args.heldout, args.out,
                            window=args.window, block=args.block, layers=DEFAULT_LAYERS,
                            manifest=args.manifest, pilot_as_data=args.pilot_as_data)
        print(json.dumps({"rows": len(index["rows"]), "skipped": index["skipped"],
                          "rows_without_full_window_anchor": len(index["rows_without_full_window_anchor"]),
                          "summary": index["summary"]}, indent=1))
    elif args.cmd == "partition":
        caps = {int(k): float(v) for k, v in (item.split(":") for item in args.max_gb)}
        resident = {int(k): {line.split()[0] for line in Path(v).read_text().splitlines() if line.strip()}
                    for k, v in (item.split(":", 1) for item in args.resident)}
        index = partition_index(json.loads(args.index.read_text()), args.world, caps, args.stratify, resident)
        args.out.write_text(json.dumps(index))
        for rank, files in index["copy_lists"].items():
            (args.out.parent / f"files-rank{rank}.txt").write_text("\n".join(files) + "\n")
        print(json.dumps({"world": args.world, "rank_anchor_load": index["rank_anchor_load"],
                          "rank_anchor_load_by": index.get("rank_anchor_load_by"),
                          "rank_gb_upper_bound": [round(g, 1) for g in index["rank_gb_upper_bound"]],
                          "files": {k: len(v) for k, v in index["copy_lists"].items()}}))
    else:
        print(json.dumps(_bench(args)))


if __name__ == "__main__":
    main()
