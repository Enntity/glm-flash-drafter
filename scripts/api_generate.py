#!/usr/bin/env python3
#!/usr/bin/env python3
"""Teacher text for drafter training: GLM-5.3-Flash completions via OpenRouter.

Reads prompt rows (``enntity_drafter.prompts.templates`` / ``.coverage`` output:
``id, source, category, messages, tools?, tool_choice?, response_format?,
reasoning_effort, temperature, split, prompt_sha, provider_extra?``) and sends
only rows with ``split == "train"`` whose ``source`` is named by ``--sources``.

Terms: Z.ai's own Terms of Use bar using its service to develop competing
models, so the Z.AI endpoint is excluded from routing.  The GLM-5.3-Flash
weights are MIT; generation goes through third-party OpenRouter hosts with
``data_collection: deny``.  Read each host's terms before a large run.

Issues requests for ``--minutes`` of wall clock (or until ``--target-tokens``
good completion tokens), then lets in-flight requests finish for at most
``--drain`` minutes.  A prompt's samples are issued back to back (they share a
prefix for capture).

Settings: thinking mode (hosted GLM-5.3-Flash does not offer reasoning off),
max_tokens 16384, temperature = serving default 1.0 / top_p 0.95 (what Atlas
uses from the checkpoint's generation_config.json when a client sends none),
one retry per request, 600 s timeout, hard spend cap (usage.cost plus a
worst-case reserve for in-flight requests, cross-checked against the key's
usage counter).  Every reply is recorded; ``junk`` marks rows to drop (errors,
host error text returned as content, empty answers).

Key: ``OPENROUTER_API_KEY`` (or ``--key-env``) from the environment; never printed or written.  ``--url`` and
``--model`` point at another OpenAI-compatible GLM-5.3-Flash endpoint (for example your own server); the
OpenRouter-only fields (provider routing, usage accounting, reasoning switch, key usage check) are then not sent
and thinking is left at the server's default.
Output: ``<out>/generations.jsonl`` and ``<out>/gen-summary.json``.

    python scripts/api_generate.py --prompts data/prompts/templates/train.jsonl --out data/gen/templates \
        --sources template --minutes 30 --spend-cap 10
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

OPENROUTER_URL = URL = "https://openrouter.ai/api/v1/chat/completions"
KEY_URL = "https://openrouter.ai/api/v1/key"
MODEL = "z-ai/glm-5.3-flash"
#: Z.AI: excluded by its terms (above).  AtlasCloud: returned quota-error text as content in a probe.
PROVIDER = {"ignore": ["Z.AI", "AtlasCloud"], "data_collection": "deny", "allow_fallbacks": True}
ALLOWED_SOURCES = {"template"}
JUNK = re.compile(r"(?i)(quota (?:is )?exhausted|rate.?limit|insufficient (?:credits|balance)|internal server error|"
                  r"service unavailable|upstream (?:error|connect)|bad gateway|model is overloaded|"
                  r"no endpoints found|context length exceeded|request timed out)")
OUT_PRICE_RESERVE = 2.5e-6   # $/output token reserve per in-flight request (observed ~0.44e-6 incl. input)


def api_messages(msgs):
    out = []
    for m in msgs:
        mm = {"role": m["role"], "content": m.get("content") if m.get("content") is not None else ""}
        if m.get("tool_calls"):
            mm["tool_calls"] = m["tool_calls"]
        if m["role"] == "tool":
            mm["tool_call_id"] = m.get("tool_call_id") or "call_unknown"
        out.append(mm)
    return out


def body_for(row, max_tokens):
    b = {"model": MODEL, "messages": api_messages(row["messages"]), "max_tokens": max_tokens,
         "temperature": row.get("temperature", 1.0), "top_p": 0.95}
    if URL == OPENROUTER_URL:
        b.update(usage={"include": True}, provider={**PROVIDER, **(row.get("provider_extra") or {})},
                 reasoning={"effort": row["reasoning_effort"]} if row.get("reasoning_effort") else {"enabled": True})
    if row.get("tools"):
        b["tools"] = row["tools"]
    for k in ("tool_choice", "response_format"):
        if row.get(k):
            b[k] = row[k]
    return b


def post(body, key, timeout):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = f"Bearer {key}"
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def key_usage(key):
    if URL != OPENROUTER_URL:
        return None
    try:
        req = urllib.request.Request(KEY_URL, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.load(r).get("data", {})
        return {k: d.get(k) for k in ("usage", "usage_daily")}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:200]}


def classify(rec):
    if rec.get("error"):
        return "error"
    if rec["finish"] in (None, "error"):
        return "finish_error"
    # Structured rows legitimately mention rate limits, 503s and timeouts; only short replies can be host error text.
    junk_len = 300 if rec.get("source") == "structured" else 2000
    if JUNK.search(rec["content"] or "") and len(rec["content"] or "") < junk_len:
        return "provider_error_text"
    if not (rec["content"] or "").strip() and not rec["tool_calls"] and rec["finish"] != "length":
        return "empty_answer"
    return None


def build_schedule(rows, pairs_frac, seed, samples=None):
    """Interleave categories round-robin; a prompt's samples are adjacent.
    samples=K gives every prompt K samples; otherwise a pairs_frac share of
    prompts gets two."""
    rng = random.Random(seed)
    seen, pools = set(), collections.defaultdict(list)
    for r in rows:
        if r["prompt_sha"] not in seen:
            seen.add(r["prompt_sha"])
            pools[r["category"]].append(r)
    for p in pools.values():
        rng.shuffle(p)
    total = sum(len(p) for p in pools.values())
    weight = {c: len(p) / total for c, p in pools.items()}
    credit = {c: 0.0 for c in pools}
    sched = []
    while any(pools.values()):
        for c in pools:
            credit[c] += weight[c]
        c = max((c for c in pools if pools[c]), key=lambda c: credit[c])
        credit[c] -= 1
        r = pools[c].pop()
        k = samples or (2 if rng.random() < pairs_frac else 1)
        sched += [(r, i) for i in range(k)]
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", required=True, help="prompt rows (jsonl)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--drain", type=float, default=10)
    ap.add_argument("--concurrency", type=int, default=96)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--spend-cap", type=float, default=25.0)
    ap.add_argument("--pairs", type=float, default=0.35, help="fraction of prompts sampled twice")
    ap.add_argument("--seed", type=int, default=20261014)
    ap.add_argument("--sources", default="template,structured",
                    help="comma list of row sources to send (others are skipped)")
    ap.add_argument("--samples", type=int, default=None, help="samples per prompt (overrides --pairs)")
    ap.add_argument("--target-tokens", type=float, default=None,
                    help="stop issuing once good completion tokens reach this (e.g. 20e6)")
    ap.add_argument("--effort", default="",
                    help="effort for rows that carry none, per source, e.g. template=high")
    ap.add_argument("--url", default=OPENROUTER_URL, help="OpenAI-compatible chat completions URL")
    ap.add_argument("--model", default=MODEL, help="model name at --url")
    ap.add_argument("--key-env", default="OPENROUTER_API_KEY", help="environment variable holding the API key")
    a = ap.parse_args()
    global ALLOWED_SOURCES, URL, MODEL
    ALLOWED_SOURCES = set(a.sources.split(","))
    URL, MODEL = a.url, a.model
    effort = dict(kv.split("=", 1) for kv in a.effort.split(",") if kv)
    key = os.environ.get(a.key_env)
    if not key and URL == OPENROUTER_URL:
        raise SystemExit(f"{a.key_env} not set")
    rows = [json.loads(l) for l in open(a.prompts)]
    rows = [r for r in rows if r["source"] in ALLOWED_SOURCES and r["split"] == "train"]
    for r in rows:
        if not r.get("reasoning_effort") and r["source"] in effort:
            r["reasoning_effort"] = effort[r["source"]]
    sched = build_schedule(rows, a.pairs, a.seed, a.samples)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    gen_path = out / "generations.jsonl"
    lock = threading.Lock()
    state = {"spent": 0.0, "inflight": 0, "issued": 0, "retries": 0, "stop_reason": None, "good_tokens": 0}
    key_before = key_usage(key)
    t0 = time.time()
    issue_deadline = t0 + a.minutes * 60
    drain_deadline = issue_deadline + a.drain * 60
    sem = threading.Semaphore(a.concurrency)
    fout = open(gen_path, "a")

    def run(row, sample, seq):
        assert row["source"] in ALLOWED_SOURCES
        body = body_for(row, a.max_tokens)
        attempt = 0
        while True:
            attempt += 1
            ts = time.time()
            rec = {"id": f"{row['id']}#s{sample}", "prompt_id": row["id"], "sample": sample, "seq": seq,
                   "source": row["source"], "category": row["category"], "attempt": attempt,
                   "issued_at": round(ts - t0, 2), "temperature": body["temperature"], "top_p": body["top_p"],
                   "reasoning_param": body.get("reasoning"), "max_tokens": a.max_tokens}
            try:
                d = post(body, key, a.timeout)
                if not d.get("choices"):
                    raise RuntimeError("no choices: " + json.dumps(d)[:300])
                ch = d["choices"][0]
                msg = ch.get("message") or {}
                usage = d.get("usage") or {}
                rec.update(provider=d.get("provider"), finish=ch.get("finish_reason"),
                           native_finish=ch.get("native_finish_reason"), wall_s=round(time.time() - ts, 2),
                           prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                           reasoning_tokens=(usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
                           cost=usage.get("cost") or 0.0, reasoning=msg.get("reasoning") or "",
                           content=msg.get("content") or "", tool_calls=msg.get("tool_calls"))
                rec["junk"] = classify(rec)
            except urllib.error.HTTPError as e:
                rec.update(error=f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}", wall_s=round(time.time() - ts, 2))
            except Exception as e:  # noqa: BLE001 - every failure is recorded
                rec.update(error=repr(e)[:300], wall_s=round(time.time() - ts, 2))
            with lock:
                state["spent"] += rec.get("cost") or 0.0
                if not rec.get("error") and not rec.get("junk"):
                    state["good_tokens"] += rec.get("completion_tokens") or 0
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
            if rec.get("error") and attempt == 1 and time.time() < issue_deadline:
                with lock:
                    state["retries"] += 1
                continue
            return

    def worker(row, sample, seq):
        try:
            run(row, sample, seq)
        finally:
            with lock:
                state["inflight"] -= 1
            sem.release()

    for seq, (row, sample) in enumerate(sched):
        sem.acquire()
        with lock:
            over = state["spent"] + (state["inflight"] + 1) * a.max_tokens * OUT_PRICE_RESERVE > a.spend_cap
        reached = a.target_tokens is not None and state["good_tokens"] >= a.target_tokens
        if time.time() >= issue_deadline or over or reached:
            sem.release()
            state["stop_reason"] = "spend_cap" if over else "target_tokens" if reached else "issue_window_closed"
            break
        with lock:
            state["inflight"] += 1
            state["issued"] += 1
        threading.Thread(target=worker, args=(row, sample, seq), daemon=True).start()
    else:
        state["stop_reason"] = "schedule_exhausted"
    issue_end = time.time()
    print(f"issue stopped ({state['stop_reason']}) after {issue_end - t0:.0f}s: issued {state['issued']}", flush=True)
    last = 0
    while time.time() < drain_deadline:
        with lock:
            n = state["inflight"]
        if n == 0:
            break
        if time.time() - last > 60:
            print(f"  draining: {n} in flight, spent ${state['spent']:.3f}", flush=True)
            last = time.time()
        time.sleep(2)
    with lock:
        abandoned = state["inflight"]
        fout.flush()
    end = time.time()
    key_after = key_usage(key)

    recs = {}
    for line in open(gen_path):
        r = json.loads(line)
        recs[r["id"]] = r            # last attempt wins
    good = [r for r in recs.values() if not r.get("error") and not r.get("junk")]
    by_source = collections.defaultdict(list)
    for r in good:
        by_source[r["source"]].append(r)
    by = collections.defaultdict(list)
    for r in recs.values():
        by[r["category"]].append(r)
    summary = {
        "sources_sent": sorted(ALLOWED_SOURCES), "model": MODEL, "provider_policy": PROVIDER,
        "max_tokens": a.max_tokens, "timeout_s": a.timeout, "temperature_default": 1.0, "top_p": 0.95,
        "temperature_note": "serving default: Atlas uses generation_config.json (temperature 1.0, top_p 0.95) when the client sends none.",
        "concurrency": a.concurrency, "issue_minutes": a.minutes, "drain_minutes_cap": a.drain,
        "stop_reason": state["stop_reason"], "issued_requests": state["issued"], "retries": state["retries"],
        "abandoned_in_flight": abandoned, "issue_window_s": round(issue_end - t0, 1), "total_wall_s": round(end - t0, 1),
        "samples_recorded": len(recs), "good_samples": len(good),
        "spend_usage_cost_usd": round(state["spent"], 4),
        "key_usage_before": key_before, "key_usage_after": key_after,
        "output_tokens_good": sum(r["completion_tokens"] or 0 for r in good),
        "aggregate_out_tok_s_good": round(sum(r["completion_tokens"] or 0 for r in good) / (end - t0), 1),
        "per_category": {c: {"samples": len(v), "good": sum(1 for r in v if not r.get("error") and not r.get("junk")),
                             "errors": sum(1 for r in v if r.get("error")),
                             "junk": dict(collections.Counter(r["junk"] for r in v if r.get("junk"))),
                             "finish": dict(collections.Counter(r.get("finish") for r in v if not r.get("error")))}
                         for c, v in sorted(by.items())},
        "per_source": {s: {"good": len(v),
                           "completion_tokens": sum(r.get("completion_tokens") or 0 for r in v),
                           "mean_completion": round(sum(r.get("completion_tokens") or 0 for r in v) / len(v)),
                           "reasoning_share": round(sum(r.get("reasoning_tokens") or 0 for r in v)
                                                    / max(1, sum(r.get("completion_tokens") or 0 for r in v)), 3),
                           "finish_length": sum(1 for r in v if r.get("finish") == "length")}
                       for s, v in sorted(by_source.items())},
        "providers": dict(collections.Counter(r.get("provider") for r in good).most_common()),
        "error_samples": [r["error"][:160] for r in recs.values() if r.get("error")][:6],
    }
    (out / "gen-summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
