#!/usr/bin/env python3
"""Hidden-state capture client for an Atlas target built with the capture patch.

Engine: ``github.com/Enntity/atlas`` branch ``exp/drafter-capture-20261014``
(``ATLAS_HIDDEN_CAPTURE_DIR``; see docs/capture.md for the server settings).
Each rank of the target writes segments under ``<capture dir>/chunks/<tag>/``
while ``<capture dir>/enable`` holds ``{"tag", "from"}``.  This client, run on
any host that can see every rank's capture directory (one directory on a
single node; for two nodes, mount the second rank's directory here, e.g. over
NFS), does for each item in order until ``--minutes`` run out:

1. resolve the item's token ids (``atlas_tokenize`` items get their prompt ids
   from the server's own ``/tokenize``; completion ids come from the item);
2. arm capture on every rank (``enable`` = tag + ``from`` = max(0, P - W));
3. ``POST /v1/completions`` with the ids as the prompt and ``max_tokens: 1``
   (a prefill: the features are what the engine computes for those tokens);
4. hand the tag to a background collector, which moves every rank's finished
   segments into ``<dest>/<tag>/``, checks that layer and final segments each
   tile ``[from, T)`` exactly once with the item's token ids, and marks a
   verified row ``_collected`` (``python -m enntity_drafter.fp8 watch`` converts
   marked rows to FP8).

``--gates`` first captures the repeat / causal-prefix gates (a 32-token and an
8K-token prefix of the first long item, a copy with token 20 mutated, and a
repeat), checked afterwards by scripts/check_capture.py.  ``--no-capture``
sends the same requests disarmed (engine throughput ceiling).
``--split-prefix`` (prefix cache on) first prefills ``ids[:from]`` disarmed so
the captured request restores a cached snapshot just below ``from``.

Output: ``<out>/capture-records.jsonl`` (one record per item: name, P, from,
T, timing, plus the item's metadata; no text), ``<out>/move-log.json``.  Pass
the records as ``--meta`` to ``python -m enntity_drafter.capture_io index``.

Stdlib only.

    python scripts/capture.py --server http://127.0.0.1:8893 --items data/items/templates.jsonl \\
        --capture-dir /capture --dest data/rows/run1 --out data/capture-logs/run1 --minutes 120
"""
from __future__ import annotations

import argparse
import json
import queue
import shutil
import threading
import time
import urllib.request
from pathlib import Path

VOCAB = 154880
WINDOW = 2048                  # DFlash2 sliding_window: rows kept before the generated span
GEN_PROMPT = [154828, 154841]  # <|assistant|><think>
MARKER = "_collected"


class Server:
    def __init__(self, base: str, model: str):
        self.base, self.model = base.rstrip("/"), model

    def post(self, path: str, body: dict, timeout: float = 3600) -> dict:
        request = urllib.request.Request(self.base + path, json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def prefill(self, ids: list[int]) -> float:
        started = time.monotonic()
        self.post("/v1/completions", {"model": self.model, "prompt": ids, "max_tokens": 1, "temperature": 0})
        return time.monotonic() - started


def resolve(server: Server, item: dict, max_seq: int) -> None:
    """Items with ``atlas_tokenize`` get their prompt ids from the server's renderer (sets ids, P, from, T)."""

    spec = item.pop("atlas_tokenize", None)
    if spec is not None:
        body = {"model": server.model, "messages": spec["messages"]}
        if spec.get("tools"):
            body["tools"] = spec["tools"]
        # No effort -> the template's default ('max').  Atlas would otherwise render a tool request
        # without an effort as thinking-off, which none of the API generations were.
        effort = spec.get("reasoning_effort") or "max"
        body.update(reasoning_effort=effort, chat_template_kwargs={"reasoning_effort": effort})
        prompt = server.post("/tokenize", body)["tokens"]
        if prompt[-2:] != GEN_PROMPT:
            raise ValueError(f"prompt does not end with the generation prompt: {prompt[-4:]}")
        completion = item.pop("completion_ids")
        item.update(ids=(prompt + completion)[:max_seq], P=len(prompt), effort_rendered=effort,
                    truncated=len(prompt) + len(completion) > max_seq)
    elif "ids" not in item:
        item["ids"] = (item.pop("prompt_ids") + item.pop("completion_ids"))[:max_seq]
        item["P"] = item.get("P", item.get("prompt_tokens"))
    item.setdefault("from", max(0, item["P"] - WINDOW))
    item["T"] = len(item["ids"])
    item.setdefault("name", str(item.get("id", "item")).replace("/", "_").replace("#", "_"))


def arm(dirs: list[Path], tag: str | None, start: int = 0) -> None:
    for directory in dirs:
        if tag is None:
            (directory / "enable").unlink(missing_ok=True)
            continue
        tmp = directory / "enable.tmp"
        tmp.write_text(json.dumps({"tag": tag, "from": start}))
        tmp.chmod(0o666)
        tmp.rename(directory / "enable")


def segments(directory: Path, kind: str) -> list[dict]:
    found = []
    for sidecar in directory.glob(f"r*-{kind}-p*.json"):
        if sidecar.with_suffix(".bin").exists():
            found.append(json.loads(sidecar.read_text()))
    return sorted(found, key=lambda m: m["pos0"])


def tiles(segs: list[dict], lo: int, hi: int) -> bool:
    position = lo
    for meta in segs:
        if meta["pos0"] != position:
            return False
        position += meta["rows"]
    return position == hi


def collect(dirs: list[Path], dest: Path, tag: str, ids: list[int], start: int, timeout: float) -> dict:
    """Move every rank's finished segments of ``tag`` into ``dest/tag`` and verify them."""

    target = dest / tag
    target.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    while True:
        for directory in dirs:
            source = directory / "chunks" / tag
            if not source.exists():
                continue
            if any(source.glob("*.err")):
                return {"ok": False, "err": f"engine writer error in {source}"}
            for binary in sorted(source.glob("*.bin")):   # .part files are renamed to .bin when complete
                sidecar = binary.with_suffix(".json")
                shutil.move(str(sidecar), target / sidecar.name)
                shutil.move(str(binary), target / binary.name)
        layer, final = segments(target, "L"), segments(target, "F")
        if tiles(layer, start, len(ids)) and tiles(final, start, len(ids)):
            break
        if time.monotonic() - started > timeout:
            return {"ok": False, "err": "coverage", "L": [(m["pos0"], m["rows"], m["rank"]) for m in layer],
                    "F": [(m["pos0"], m["rows"], m["rank"]) for m in final]}
        time.sleep(0.05)
    mismatch = sum(meta["token_ids"] is None or meta["token_ids"] != ids[meta["pos0"]: meta["pos0"] + meta["rows"]]
                   for meta in layer + final)
    for directory in dirs:
        try:
            (directory / "chunks" / tag).rmdir()
        except OSError:
            pass
    record = {"ok": True, "ids_mismatch": mismatch, "ranks": sorted({m["rank"] for m in layer + final}),
              "sp": any(m.get("sp") for m in layer), "wait_s": round(time.monotonic() - started, 3),
              "bytes": sum(p.stat().st_size for p in target.glob("*.bin"))}
    if not mismatch:
        (target / MARKER).write_text("")
    return record


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server", default="http://127.0.0.1:8893", help="Atlas OpenAI-compatible base URL")
    ap.add_argument("--model", default="glm-5.3-flash-atlas", help="served model name")
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--capture-dir", type=Path, action="append", default=[],
                    help="each rank's ATLAS_HIDDEN_CAPTURE_DIR as seen from this host (repeat per rank)")
    ap.add_argument("--dest", type=Path, help="row destination (<dest>/<tag>/)")
    ap.add_argument("--out", type=Path, required=True, help="records directory")
    ap.add_argument("--minutes", type=float, default=60.0, help="no request starts after this")
    ap.add_argument("--max-seq", type=int, default=49151, help="server --max-seq-len minus the generated token")
    ap.add_argument("--collect-timeout", type=float, default=60.0)
    ap.add_argument("--no-capture", action="store_true")
    ap.add_argument("--gates", action="store_true")
    ap.add_argument("--split-prefix", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv)
    if not a.no_capture and (not a.capture_dir or a.dest is None):
        ap.error("capture needs --capture-dir (per rank) and --dest")
    server, dirs = Server(a.server, a.model), a.capture_dir
    deadline = time.time() + a.minutes * 60
    a.out.mkdir(parents=True, exist_ok=True)
    for directory in dirs:
        (directory / "chunks").mkdir(parents=True, exist_ok=True)
    arm(dirs, None)
    items = [json.loads(line) for line in a.items.open()][: a.limit or None]
    pending: queue.Queue = queue.Queue()
    moved: dict[str, dict] = {}

    def collector() -> None:
        while (job := pending.get()) is not None:
            tag, ids, start = job
            try:
                moved[tag] = collect(dirs, a.dest, tag, ids, start, a.collect_timeout)
            except Exception as error:  # noqa: BLE001 - recorded per row
                moved[tag] = {"ok": False, "err": repr(error)[:400]}

    worker = threading.Thread(target=collector, daemon=True)
    worker.start()

    def one(name: str, ids: list[int], start: int) -> tuple[float, float]:
        warm = 0.0
        if a.split_prefix and start >= 256:
            arm(dirs, None)
            warm = server.prefill(ids[:start])
        if not a.no_capture:
            arm(dirs, name, start)
        wall = server.prefill(ids)
        if not a.no_capture:
            arm(dirs, None)
            pending.put((name, ids, start))
        return wall, warm

    records = a.out / ("measure-records.jsonl" if a.no_capture else "capture-records.jsonl")
    try:
        if a.gates and not a.no_capture:
            for it in items:
                resolve(server, it, a.max_seq)
                if it["T"] >= 8200:
                    break
            else:
                raise SystemExit("--gates needs an item with at least 8,200 tokens")
            for size, base in (("32", it["ids"][:32]), ("8k", it["ids"][:8192])):
                mutated = list(base)
                mutated[20] = (mutated[20] + 1) % VOCAB
                for name, seq in ((f"gate{size}-a", base), (f"gate{size}-b", mutated), (f"gate{size}-repeat", base)):
                    one(name, seq, 0)
        with records.open("a") as stream:
            for it in items:
                if time.time() >= deadline:
                    print("deadline reached", flush=True)
                    break
                began = time.time()
                try:
                    if "T" not in it:
                        resolve(server, it, a.max_seq)
                    wall, warm = one(it["name"], it["ids"], it["from"])
                except Exception as error:  # noqa: BLE001 - recorded, next item
                    arm(dirs, None)
                    stream.write(json.dumps({"name": it.get("name"), "error": repr(error)[:400]}) + "\n")
                    continue
                record = {k: v for k, v in it.items() if k != "ids"}
                record.update(tokens=it["T"], stored_rows=it["T"] - it["from"], wall_s=round(wall, 4),
                              prefix_wall_s=round(warm, 4), t=began)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                print(f"{it['name']} T={it['T']} from={it['from']} {it['T'] / (wall + warm):.0f} tok/s", flush=True)
    finally:
        arm(dirs, None)
        pending.put(None)
        worker.join()
        if moved:
            (a.out / "move-log.json").write_text(json.dumps(moved, indent=1))
            bad = [k for k, m in moved.items() if not m.get("ok") or m.get("ids_mismatch")]
            print(f"collected {len(moved) - len(bad)} of {len(moved)} rows; bad: {bad[:10]}", flush=True)


if __name__ == "__main__":
    main()
