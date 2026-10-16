"""Convert captured fast segments from BF16 to FP8 e4m3 storage, in place (e4m3 + per-row scale).

Format (read by ``capture_io``): the segment's ``.bin`` is replaced by a
``.fp8`` file holding e4m3 bytes ``[slots, rows, hidden]`` followed by float32
scales ``[slots, rows]`` (row amax / 448), and its sidecar gets ``dtype``
``fp8_e4m3`` plus the measured round-trip error.  Per segment: write
``.fp8.tmp``, re-read and compare it, check the relative L2 error per slot
(< ``--max-rel``), rename, rewrite the sidecar, then delete the ``.bin``.  An
interrupted conversion is resumed safely; a segment with non-finite values
stays BF16.

    python -m enntity_drafter.fp8 watch --root ROWS --log ROWS/fp8-log.jsonl [--stop-file F]
        convert every row dir under ``ROWS`` (any depth) that holds the
        ``_collected`` marker (written by scripts/capture.py after a verified row),
        then remove the marker; exit once the stop file exists and no marker is left
    python -m enntity_drafter.fp8 convert DIR ...      convert these row dirs now
    python -m enntity_drafter.fp8 measure DIR ...      error only (writes nothing)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .capture_io import FAST_SCHEMA, FP8, FP8_MAX, fp8_dequantize, fp8_quantize

MARKER = "_collected"
ROWS_PER_PASS = 2048        # bounds the float32 temporaries to ~100 MB


def _segment(sidecar: Path, write: bool, max_rel: float) -> dict:
    meta = json.loads(sidecar.read_text())
    if meta.get("schema") != FAST_SCHEMA:
        raise ValueError(f"{sidecar}: not a {FAST_SCHEMA} segment")
    bin_path, fp8_path = sidecar.with_suffix(".bin"), sidecar.with_suffix(".fp8")
    if meta["dtype"] == FP8:          # converted earlier; finish an interrupted cleanup
        if write and bin_path.exists():
            bin_path.unlink()
        return {"kind": sidecar.name.split("-")[1], "skipped": "already fp8"}
    slots = len(meta["layers"]) if sidecar.name.split("-")[1] == "L" else 1
    rows, hidden = meta["rows"], meta["hidden_size"]
    bits = np.fromfile(bin_path, dtype=np.uint16)
    if bits.size != slots * rows * hidden:
        raise ValueError(f"{bin_path}: size {bits.size * 2} != {slots * rows * hidden * 2}")
    bits = bits.reshape(slots, rows, hidden)
    q = np.empty((slots, rows, hidden), np.uint8)
    scale = np.empty((slots, rows), np.float32)
    err, ref = np.zeros(slots), np.zeros(slots)
    try:
        for s in range(slots):
            for a in range(0, rows, ROWS_PER_PASS):
                b = min(rows, a + ROWS_PER_PASS)
                q[s, a:b], scale[s, a:b] = fp8_quantize(bits[s, a:b])
                x = torch.from_numpy(np.ascontiguousarray(bits[s, a:b])).view(torch.bfloat16).double()
                y = torch.from_numpy(fp8_dequantize(q[s, a:b], scale[s, a:b]).view(np.int16)).view(torch.bfloat16).double()
                err[s] += float(((y - x) ** 2).sum())
                ref[s] += float((x ** 2).sum())
    except ValueError as error:       # non-finite: keep this segment in BF16
        return {"kind": sidecar.name.split("-")[1], "skipped": str(error)}
    rel = [float(np.sqrt(e / r)) if r > 0 else 0.0 for e, r in zip(err, ref)]
    out = {"kind": sidecar.name.split("-")[1], "rows": rows, "slots": slots, "err2": err.tolist(),
           "ref2": ref.tolist(), "rel_l2": rel, "bytes_in": bits.nbytes, "bytes_out": q.nbytes + scale.nbytes}
    if max(rel) >= max_rel:
        raise ValueError(f"{sidecar}: FP8 relative L2 {max(rel):.4f} >= {max_rel}")
    if not write:
        return out
    tmp = fp8_path.with_name(fp8_path.name + ".tmp")
    with open(tmp, "wb") as stream:
        stream.write(q.tobytes())
        stream.write(scale.tobytes())
        stream.flush()
        os.fsync(stream.fileno())
    back = np.fromfile(tmp, dtype=np.uint8)
    if back.size != q.nbytes + scale.nbytes or not np.array_equal(back[: q.nbytes], q.reshape(-1)) \
            or not np.array_equal(back[q.nbytes:].view(np.float32), scale.reshape(-1)):
        tmp.unlink()
        raise ValueError(f"{tmp}: read-back differs")
    tmp.rename(fp8_path)
    meta.update(dtype=FP8, source_dtype="bf16", fp8_rel_l2=[round(r, 6) for r in rel],
                scale={"dtype": "f32", "per": "row_slot", "value": "amax/448", "fmax": FP8_MAX,
                       "layout": "tail [slots, rows] after the e4m3 payload [slots, rows, hidden]"})
    side_tmp = sidecar.with_name(sidecar.name + ".tmp")
    side_tmp.write_text(json.dumps(meta))
    side_tmp.rename(sidecar)
    bin_path.unlink()
    return out


def convert_dir(directory: Path, *, write: bool = True, max_rel: float = 0.1) -> dict:
    """Convert (or, ``write=False``, only measure) every segment of one row dir."""

    started = time.monotonic()
    segs = [_segment(p, write, max_rel) for p in sorted(directory.glob("r*-*-p*-n*.json"))]
    done = [s for s in segs if "rel_l2" in s]
    # per slot: layers 5, 14, 24, 33, 42 (L segments) then the final hidden (F)
    err, ref = np.zeros(6), np.zeros(6)
    for s in done:
        idx = slice(0, 5) if s["kind"] == "L" else slice(5, 6)
        err[idx] += s["err2"]
        ref[idx] += s["ref2"]
    return {"dir": str(directory), "segments": len(segs), "converted": len(done),
            "kept_bf16": [s["skipped"] for s in segs if s.get("skipped") and s["skipped"] != "already fp8"],
            "bytes_in": sum(s["bytes_in"] for s in done), "bytes_out": sum(s["bytes_out"] for s in done),
            "err2": err.tolist(), "ref2": ref.tolist(),
            "rel_l2": [float(np.sqrt(e / r)) if r > 0 else None for e, r in zip(err, ref)],
            "seconds": round(time.monotonic() - started, 3)}


def watch(root: Path, log: Path, stop_file: Path | None, max_rel: float, poll: float) -> None:
    while True:
        pending = sorted(root.rglob(MARKER), key=lambda p: p.stat().st_mtime)
        for marker in pending:
            row = marker.parent
            try:
                record = convert_dir(row, max_rel=max_rel)
            except Exception as error:  # noqa: BLE001  keep BF16, report, go on
                record = {"dir": str(row), "error": repr(error)[:400]}
            record.update(name=row.name, rel=str(row.relative_to(root)), t=time.time())
            with open(log, "a") as stream:
                stream.write(json.dumps(record) + "\n")
            marker.unlink()
        if not pending:
            if stop_file is not None and stop_file.exists():
                return
            time.sleep(poll)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("watch")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--log", type=Path, required=True)
    p.add_argument("--stop-file", type=Path)
    p.add_argument("--poll", type=float, default=2.0)
    for name in ("convert", "measure"):
        p2 = sub.add_parser(name)
        p2.add_argument("dirs", nargs="+", type=Path)
    for q in sub.choices.values():
        q.add_argument("--max-rel", type=float, default=0.1)
        q.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    if args.cmd == "watch":
        watch(args.root, args.log, args.stop_file, args.max_rel, args.poll)
        return
    for directory in args.dirs:
        print(json.dumps(convert_dir(directory, write=args.cmd == "convert", max_rel=args.max_rel)), flush=True)


if __name__ == "__main__":
    main()
