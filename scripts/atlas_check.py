#!/usr/bin/env python3
"""Drafter check against a running Atlas server: exactness + acceptance (tok/step) per category.

``probe`` sends requests one at a time (concurrency 1, greedy) and records each request's UTC start/end;
``analyze`` matches each request to the engine's per-request ``Done:`` log line (tokens, tok/s, ``tok_step``,
per-position survival ``surv=``) by time and tabulates per visit and category, plus text agreement across
visits (the same prompt must give the same text with any drafter: speculation must not change output).

Probe modes:
  exact    4 prompts, non-streamed with token logprobs, 256 tokens (exactness record)
  multi    the matched 12 prose + 12 code prompts, 256 forced tokens (``min_tokens``), streamed
  bodies   BODIES = your own JSONL of {name, category, body} (OpenAI chat bodies), streamed

A visit = one server start with one drafter (``--draft-model``); run each arm twice in ABBA order and save the
engine log of rank 0 as ``<rundir>/rank0.log`` next to the probe outputs.  RigMark (github.com/alexellis/rigmark)
decode rows are matched too when ``rig.json``, ``rig.t0`` and ``rig.t1`` (epoch seconds) are in the run dir.

    python scripts/atlas_check.py probe exact  --tag s1 --out runs/s1
    python scripts/atlas_check.py probe multi  --tag s1 --out runs/s1
    python scripts/atlas_check.py analyze runs/s1 runs/d1 runs/d2 runs/s2 [--json analysis.json]
"""
import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CFG = {"url": "http://127.0.0.1:8893/v1/chat/completions", "model": "glm-5.3-flash-atlas"}

PROSE = ["a hash map", "TCP congestion control", "a B-tree", "garbage collection", "public-key cryptography",
         "the water cycle", "photosynthesis", "a compiler", "the stock market", "vaccines", "plate tectonics",
         "an operating system scheduler"]
CODE = ["an LRU cache with TTL", "a JSON tokenizer", "a thread pool", "Dijkstra's shortest path",
        "a trie with prefix search", "a rate limiter (token bucket)", "an interval tree",
        "a markdown-to-HTML converter for headings and lists", "a CSV parser that handles quotes",
        "a priority queue with decrease-key", "a simple regex matcher (. and *)", "a bloom filter"]
MULTI = {**{f"prose-{i:02d}": "Explain, in flowing prose with no lists, how %s works." % t for i, t in enumerate(PROSE)},
         **{f"code-{i:02d}": "Write a complete Python implementation of %s with type hints. Code only." % t
            for i, t in enumerate(CODE)}}
EXACT4 = ["Write a red-black tree in Python with insert and delete. Code only.",
          "List the planets of the solar system in order from the sun, one sentence each.",
          "Explain the difference between processes and threads in three paragraphs.",
          "Write a SQL schema for a library with books, members and loans, with comments."]


def sha(x):
    return hashlib.sha1(json.dumps(x).encode()).hexdigest()[:12]


def simple_body(prompt, max_tokens, forced):
    body = {"model": CFG["model"], "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, "chat_template_kwargs": {"enable_thinking": False, "thinking": False}}
    if forced:
        body["min_tokens"] = max_tokens
    return body


def post(body):
    return urllib.request.urlopen(urllib.request.Request(CFG["url"], json.dumps(body).encode(),
                                                         {"Content-Type": "application/json"}), timeout=1800)


def stream(body):
    body = dict(body, stream=True, stream_options={"include_usage": True})
    t0 = time.time()
    reason, text, tools, first, last, usage, finish = [], [], [], None, None, None, None
    with post(body) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                usage = d["usage"]
            for c in d.get("choices") or []:
                delta = c.get("delta") or {}
                got = False
                for key, acc in (("reasoning_content", reason), ("content", text)):
                    if delta.get(key):
                        acc.append(delta[key])
                        got = True
                if delta.get("tool_calls"):
                    tools.append(json.dumps(delta["tool_calls"], sort_keys=True))
                    got = True
                if got:
                    now = time.time()
                    first = first or now
                    last = now
                finish = c.get("finish_reason") or finish
    t1 = time.time()
    full = "".join(reason) + "\n</think>\n" + "".join(text) + "\n<tools>\n" + "".join(tools)
    n = (usage or {}).get("completion_tokens", 0)
    dec = (last - first) if first and last and last > first else None
    return {"n": n, "text_sha": sha(full), "text": full, "t0": t0, "t1": t1,
            "ttft_s": round(first - t0, 3) if first else None, "decode_s": round(dec, 3) if dec else None,
            "decode_tok_s": round((n - 1) / dec, 2) if dec and n > 1 else None, "finish": finish,
            "prompt_tokens": (usage or {}).get("prompt_tokens")}


def whole(body):
    body = dict(body, logprobs=True, top_logprobs=1)
    t0 = time.time()
    r = json.load(post(body))
    t1 = time.time()
    c = r["choices"][0]
    m = c["message"]
    full = (m.get("reasoning_content") or "") + "\n</think>\n" + (m.get("content") or "")
    lps = [(t.get("token"), t.get("logprob")) for t in ((c.get("logprobs") or {}).get("content") or [])]
    return {"n": r["usage"]["completion_tokens"], "text_sha": sha(full), "text": full, "lp_sha": sha(lps),
            "lp_n": len(lps), "t0": t0, "t1": t1, "finish": c.get("finish_reason"),
            "prompt_tokens": r["usage"].get("prompt_tokens")}


def report(name, r):
    print("%-28s %5d tok  ttft %6.2f s  decode %7.2f tok/s  text %s %s" % (
        name, r["n"], r.get("ttft_s") or -1, r.get("decode_tok_s") or -1, r["text_sha"], r.get("lp_sha") or ""),
        flush=True)



DONE = re.compile(r"^(\S+Z)\s+INFO\s+\S*mtp_accept_debug: Done: (\d+) tokens \((\w+)\) ([\d.]+) tok/s, TTFT=([\d.]+)ms.*?"
                  r"tok_step=([\d.]+)")


def ts(s):
    return datetime.strptime(s[:26].rstrip("Z"), "%Y-%m-%dT%H:%M:%S.%f").replace(tzinfo=timezone.utc).timestamp()


SURV = re.compile(r"surv=([\d.,]+)")


def done_lines(log):
    out = []
    for line in open(log, errors="replace"):
        m = DONE.match(line)
        if m:
            surv = SURV.search(line)
            out.append({"t": ts(m.group(1)), "n": int(m.group(2)), "why": m.group(3), "tok_s": float(m.group(4)),
                        "ttft_ms": float(m.group(5)), "tok_step": float(m.group(6)),
                        "surv": [float(x) for x in surv.group(1).strip(",").split(",")] if surv else None})
    return out


def load(rundir):
    rundir = Path(rundir)
    dones = done_lines(rundir / "rank0.log")
    reqs = {}
    for mode in ("exact", "multi", "bodies"):
        for f in rundir.glob(f"{mode}-*.json"):
            for name, r in json.load(open(f))["req"].items():
                if r.get("t0") is None:
                    continue
                # back-to-back requests: the previous Done lands just before t0, this one within ~0.1 s of t1
                cand = [d for d in dones if r["t0"] + 0.05 < d["t"] <= r["t1"] + 0.5]
                r["done"] = cand[0] if len(cand) == 1 and cand[0]["n"] == r["n"] else None
                r["n_done_matches"] = len(cand)
                reqs[name] = r
    rig = rundir / "rig.json"
    if rig.exists():
        t0 = float((rundir / "rig.t0").read_text())
        t1 = float((rundir / "rig.t1").read_text())
        rd = json.load(open(rig))
        cand = [d for d in dones if t0 <= d["t"] <= t1 + 0.5]
        dec = rd.get("decode") or rd.get("results", {}).get("decode") or {}
        names = list(dec) if dec else ["code", "prose", "structured"]
        # decode runs workloads in prompts.json order, one request each (--runs 1); take the last len(names)
        for name, d in zip(names, cand[-len(names):]):
            row = (dec.get(name) or {}).get("runs", [{}])[0]
            reqs[f"rig-{name}"] = {"category": f"rig-{name}", "n": row.get("completion_tokens", d["n"]),
                                   "decode_tok_s": row.get("decode_tokens_per_second"), "done": d,
                                   "text_sha": None, "n_done_matches": len(cand)}
    return reqs


def summarize(reqs):
    cats = {}
    for name, r in reqs.items():
        cats.setdefault(r["category"], []).append(r)
    rows = {}
    for c, rs in sorted(cats.items()):
        ok = [r for r in rs if r.get("done")]
        steps = sum(r["done"]["n"] / r["done"]["tok_step"] for r in ok)
        rows[c] = {"requests": len(rs), "matched": len(ok),
                   "tok_step_mean": round(statistics.mean(r["done"]["tok_step"] for r in ok), 3) if ok else None,
                   "tok_step_pooled": round(sum(r["done"]["n"] for r in ok) / steps, 3) if steps else None,
                   "engine_tok_s_mean": round(statistics.mean(r["done"]["tok_s"] for r in ok), 2) if ok else None,
                   "client_decode_tok_s_mean": round(statistics.mean(r["decode_tok_s"] for r in rs if r.get("decode_tok_s")), 2)
                   if any(r.get("decode_tok_s") for r in rs) else None,
                   "tokens": sum(r["done"]["n"] for r in ok)}
        weighted = [(r["done"]["n"] / r["done"]["tok_step"], r["done"]["surv"]) for r in ok if r["done"].get("surv")]
        if weighted:   # engine per-position survival, weighted by each request's verify steps
            total = sum(w for w, _ in weighted)
            depth = min(len(v) for _, v in weighted)
            rows[c]["surv_step_weighted"] = [round(sum(w * v[k] for w, v in weighted) / total, 3) for k in range(depth)]
    return rows


def probe(args):
    CFG.update(url=args.server.rstrip("/") + "/v1/chat/completions", model=args.model)
    out = {"mode": args.mode, "tag": args.tag, "start": time.time(), "req": {}}
    if args.mode == "exact":
        for i, p in enumerate(EXACT4):
            r = out["req"][f"exact-{i}"] = dict(whole(simple_body(p, 256, False)), category="exact")
            report(f"exact-{i}", r)
    elif args.mode == "multi":
        for name, p in MULTI.items():
            r = out["req"][name] = dict(stream(simple_body(p, 256, True)), category=name.split("-")[0])
            report(name, r)
    else:
        for line in open(args.bodies):
            it = json.loads(line)
            body = dict(it["body"], model=CFG["model"])
            try:
                r = dict(stream(body), category=it["category"])
            except Exception as e:  # keep going; record the failure
                r = {"error": repr(e)[:300], "category": it["category"], "n": 0, "text_sha": None, "t0": None}
            out["req"][it["name"]] = r
            report(it["name"], r) if "error" not in r else print(it["name"], "ERROR", r["error"], flush=True)
    out["end"] = time.time()
    args.out.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out / f"{args.mode}-{args.tag}.json", "w"), indent=1)
    print("PROBE-DONE", args.mode, args.tag, flush=True)


def analyze(args):
    allreqs = {Path(a).name: load(a) for a in args.rundirs}
    summ = {k: summarize(v) for k, v in allreqs.items()}
    print(json.dumps(summ, indent=1))
    # exactness: same request name across visits must give the same text sha
    # tool-call ids are a server-wide counter (one visit may have made one more call): not model text
    for v in allreqs.values():
        for r in v.values():
            if r.get("text") is not None:
                r["text_sha"] = hashlib.sha1(re.sub(r'"id": "[^"]*"', '"id": ""', r["text"]).encode()).hexdigest()[:12]
    names = sorted({n for v in allreqs.values() for n, r in v.items() if r.get("text_sha")})
    agree = {}
    for n in names:
        shas = {k: v[n]["text_sha"] for k, v in allreqs.items() if n in v and v[n].get("text_sha")}
        lps = {k: v[n].get("lp_sha") for k, v in allreqs.items() if n in v and v[n].get("lp_sha")}
        agree[n] = {"visits": len(shas), "text_equal": len(set(shas.values())) == 1,
                    "lp_equal": len(set(lps.values())) <= 1}
    bad = {n: a for n, a in agree.items() if not a["text_equal"]}
    print("EXACT", len(agree), "names;", sum(a["visits"] > 1 for a in agree.values()), "seen in >1 visit;",
          len(bad), "texts differ:", json.dumps(bad))
    if args.json:
        json.dump({"summary": summ, "agree": agree}, open(args.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("mode", choices=["exact", "multi", "bodies"])
    p.add_argument("--tag", required=True, help="visit name (output file suffix)")
    p.add_argument("--out", type=Path, required=True, help="run directory")
    p.add_argument("--bodies", help="bodies mode: JSONL of {name, category, body}")
    p.add_argument("--server", default="http://127.0.0.1:8893")
    p.add_argument("--model", default="glm-5.3-flash-atlas")
    p = sub.add_parser("analyze")
    p.add_argument("rundirs", nargs="+")
    p.add_argument("--json", help="also write the summary here")
    args = ap.parse_args()
    if args.cmd == "probe" and args.mode == "bodies" and not args.bodies:
        ap.error("bodies mode needs --bodies")
    (probe if args.cmd == "probe" else analyze)(args)


if __name__ == "__main__":
    main()
