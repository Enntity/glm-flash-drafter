"""Build the structured-coverage prompt set (local; sends nothing).

Round 1's drafter data was mostly long-reasoning template answers; this set is
Enntity-authored synthetic prompts for the structured output shapes a serving
deployment sees:
strict JSON (response_format json_schema), JSON extraction, OpenAI-format tool
calls (single, parallel, tool-result follow-ups, multi-turn), config/data
files (YAML, TOML, SQL, CSV, ...), code edits (unified diff, SEARCH/REPLACE,
whole function/file), Markdown tables and short code in many languages.

Every prompt is generated here from domains.py / codegen.py vocabularies and
templates with a fixed seed: license "Enntity-authored synthetic", no
third-party text. Prompts resembling the RigMark benchmark's are excluded (no
index/square sequence, no Go token-bucket rate limiter) so benchmark numbers
are not trained on.

Each row carries `check` (what the validator needs: kind, schema, expected
tool use, language, original code) next to the fields scripts/api_generate.py
reads.  ``--profile base`` reproduces round 2b's coverage-1 set, ``--profile
long --prefix cov2`` its coverage-2 set (longer outputs).

Outputs: <out>/train.jsonl and <out>/manifest.json.

    python -m enntity_drafter.prompts.coverage --out data/prompts/coverage1 [--no-tokens]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
from pathlib import Path

from . import codegen
from .domains import (DOMAINS, VOCAB, CITIES, describe_fields, dumps, entity_schema, fields,
                      flat_fields, human, ident, person, plural, record, title, top_entities, company)

SEED = 20261015
LICENSE = "Enntity-authored synthetic"
EFFORT = "low"
COUNTS = {"json_schema": 650, "json_extract": 600, "tool_call": 650, "config": 600,
          "code_edit": 550, "table_md": 400, "short_code": 550}
SIZES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60)
SIZE_W = (4, 5, 6, 6, 7, 6, 7, 7, 6, 6, 6, 5, 5, 4, 3, 2)
SIZES_L = (3, 5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80)
SIZE_W_L = (2, 3, 4, 5, 6, 7, 8, 8, 7, 6, 4, 3, 2)
LONG = False   # --profile long (coverage-2): longer outputs, see LONG_ALT
#: OpenRouter provider routing for response_format rows: only hosts that implement the parameter.
RF_PROVIDER = {"require_parameters": True}


def size(rng, cap=60):
    if LONG:
        cap = min(int(cap * 1.5), 80)
    while True:
        n = rng.choices(*((SIZES_L, SIZE_W_L) if LONG else (SIZES, SIZE_W)))[0]
        if n <= cap:
            return n


def pick(rng):
    dom = rng.choice(list(DOMAINS))
    return dom, rng.choice(top_entities(dom))


def sys_prompt(rng, dom, kind):
    label = DOMAINS[dom][0]
    pool = [None, None, None, "You are a helpful assistant.",
            f"You are the internal assistant for {label}. Be precise and concise.",
            "You are a meticulous data engineer. Follow output-format instructions exactly."]
    if kind == "json":
        pool += ["You are a JSON API. Respond with a single JSON value and nothing else.",
                 f"You are a structured-data service for {label}. Output must be valid JSON that matches the requested schema."]
    if kind == "code":
        pool += ["You are an expert software engineer. Prefer small, correct, idiomatic changes.",
                 "You are a coding assistant inside an IDE. Answer with code first; keep prose minimal."]
    return rng.choice(pool)


def msgs(system, user):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]


def csv_text(rows, cols, sep=","):
    def cell(v):
        v = ";".join(v) if isinstance(v, list) else ("true" if v is True else "false" if v is False else str(v))
        return f'"{v}"' if (sep in v or '"' in v) else v
    return "\n".join([sep.join(cols)] + [sep.join(cell(r[c]) for c in cols) for r in rows])


def wrap_root(name, item_schema, n=None):
    arr = {"type": "array", "items": item_schema}
    if n:
        arr["minItems"] = arr["maxItems"] = n
    return {"type": "object", "properties": {name: arr}, "required": [name], "additionalProperties": False}


def json_request(rng, row, schema, name, intro, force=None):
    """Attach response_format / schema instructions. Returns final user text."""
    mode = force or rng.choices(["strict", "json_object", "instructed"], [0.62, 0.08, 0.30])[0]
    show = rng.choices(["schema", "prose", "none"], [0.6, 0.25, 0.15] if mode == "strict" else [0.65, 0.35, 0])[0]
    text = intro
    if show == "schema":
        sj = json.dumps(schema, indent=rng.choice((None, 2)))
        text += rng.choice(["\n\nJSON Schema:\n" + sj, "\n\nThe response must conform to this JSON Schema:\n" + sj,
                            "\n\nSchema:\n```json\n" + sj + "\n```"])
    elif show == "prose":
        text += "\n\n" + row.pop("_prose")
    row.pop("_prose", None)
    if mode == "strict":
        row["response_format"] = {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}
        row["provider_extra"] = RF_PROVIDER
    elif mode == "json_object":
        row["response_format"] = {"type": "json_object"}
        row["provider_extra"] = RF_PROVIDER
        text += "\n\nReturn a JSON object."
    else:
        text += "\n\n" + rng.choice(["Respond with JSON only, no prose and no code fences.",
                                     "Output only the JSON.", "Return valid JSON and nothing else.",
                                     "Reply with a single ```json code block."])
    row["check"] = {"kind": "json", "schema": schema, "mode": mode, "schema_shown": show}
    return text


# ---------------------------------------------------------------- json_schema

SEQ_TASKS = [
    ("cube", "each integer n from {a} to {b} and its cube", lambda n: {"n": n, "cube": n ** 3}),
    ("tri", "the first {k} triangular numbers with their position", lambda n: {"position": n, "triangular": n * (n + 1) // 2}),
    ("pow2", "powers of two for exponents {a} through {b}", lambda n: {"exponent": n, "value": 2 ** n}),
    ("c2f", "a Celsius-to-Fahrenheit table for every {s} degrees from {a0} to {b0}", None),
    ("km", "kilometres {a} to {b} converted to miles (two decimals)", lambda n: {"km": n, "miles": round(n * 0.621371, 2)}),
    ("bin", "the integers {a} to {b} with their binary and hexadecimal representations",
     lambda n: {"decimal": n, "binary": bin(n)[2:], "hex": hex(n)[2:]}),
    ("mult", "the multiplication table of {m} from 1 to {k}", None),
    ("fib", "the first {k} Fibonacci numbers with their index (F(1)=1, F(2)=1)", None),
    ("prime", "the first {k} prime numbers with their index", None),
    ("leap", "every year from {y0} to {y1} with a flag saying whether it is a leap year", None),
    ("interest", "year-by-year balances for {p} invested at {r}% compounded annually for {k} years", None),
    ("ascii", "uppercase letters A through {L} with their ASCII codes", None),
]


def seq_task(rng):
    name, desc, fn = rng.choice(SEQ_TASKS)
    k = rng.randint(10, 70)
    a = rng.randint(0, 40)
    b = a + k - 1
    p = {"a": a, "b": b, "k": k, "s": rng.choice((5, 10)), "a0": -20, "b0": rng.choice((40, 60, 100)),
         "m": rng.randint(3, 19), "y0": 1960 + rng.randint(0, 40), "p": rng.choice((1000, 2500, 10000)),
         "r": rng.choice((3, 4.5, 6, 7.25)), "L": rng.choice("HMRTZ")}
    p["y1"] = p["y0"] + k
    item_fields = {"cube": {"n": "integer", "cube": "integer"}, "tri": {"position": "integer", "triangular": "integer"},
                   "pow2": {"exponent": "integer", "value": "integer"}, "c2f": {"celsius": "number", "fahrenheit": "number"},
                   "km": {"km": "integer", "miles": "number"}, "bin": {"decimal": "integer", "binary": "string", "hex": "string"},
                   "mult": {"factor": "integer", "product": "integer"}, "fib": {"index": "integer", "value": "integer"},
                   "prime": {"index": "integer", "prime": "integer"}, "leap": {"year": "integer", "leap": "boolean"},
                   "interest": {"year": "integer", "balance": "number"}, "ascii": {"letter": "string", "code": "integer"}}[name]
    item = {"type": "object", "properties": {k_: {"type": t} for k_, t in item_fields.items()},
            "required": list(item_fields), "additionalProperties": False}
    return desc.format(**p), item, item_fields


PLAN_SCHEMAS = {
    "itinerary": ("a {d}-day itinerary for a team offsite in {city}", {
        "type": "object", "properties": {
            "city": {"type": "string"}, "days": {"type": "array", "items": {"type": "object", "properties": {
                "day": {"type": "integer"}, "theme": {"type": "string"}, "activities": {"type": "array", "items": {
                    "type": "object", "properties": {"time": {"type": "string"}, "title": {"type": "string"},
                                                     "duration_minutes": {"type": "integer"}, "indoor": {"type": "boolean"}},
                    "required": ["time", "title", "duration_minutes", "indoor"], "additionalProperties": False}}},
                "required": ["day", "theme", "activities"], "additionalProperties": False}},
            "estimated_budget_usd": {"type": "number"}},
        "required": ["city", "days", "estimated_budget_usd"], "additionalProperties": False}),
    "test_plan": ("a test plan for the {service} release", {
        "type": "object", "properties": {
            "service": {"type": "string"}, "suites": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "kind": {"type": "string", "enum": ["unit", "integration", "e2e", "load", "security"]},
                "cases": {"type": "array", "items": {"type": "object", "properties": {
                    "id": {"type": "string"}, "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["p0", "p1", "p2"]}, "automated": {"type": "boolean"}},
                    "required": ["id", "description", "priority", "automated"], "additionalProperties": False}}},
                "required": ["name", "kind", "cases"], "additionalProperties": False}},
            "exit_criteria": {"type": "array", "items": {"type": "string"}}},
        "required": ["service", "suites", "exit_criteria"], "additionalProperties": False}),
    "migration": ("a phased plan to migrate {service} from {a} to {b}", {
        "type": "object", "properties": {
            "phases": {"type": "array", "items": {"type": "object", "properties": {
                "name": {"type": "string"}, "steps": {"type": "array", "items": {"type": "string"}},
                "rollback": {"type": "string"}, "estimated_days": {"type": "integer"}},
                "required": ["name", "steps", "rollback", "estimated_days"], "additionalProperties": False}},
            "risks": {"type": "array", "items": {"type": "object", "properties": {
                "risk": {"type": "string"}, "likelihood": {"type": "string", "enum": ["low", "medium", "high"]},
                "mitigation": {"type": "string"}}, "required": ["risk", "likelihood", "mitigation"], "additionalProperties": False}}},
        "required": ["phases", "risks"], "additionalProperties": False}),
    "error_catalog": ("an error catalog for the public API of {label}", {
        "type": "object", "properties": {"errors": {"type": "array", "items": {"type": "object", "properties": {
            "code": {"type": "string"}, "http_status": {"type": "integer"}, "message": {"type": "string"},
            "retryable": {"type": "boolean"}, "remediation": {"type": "string"}},
            "required": ["code", "http_status", "message", "retryable", "remediation"], "additionalProperties": False}}},
        "required": ["errors"], "additionalProperties": False}),
    "roadmap": ("a two-quarter product roadmap for {label}", {
        "type": "object", "properties": {"quarters": {"type": "array", "items": {"type": "object", "properties": {
            "quarter": {"type": "string"}, "milestones": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "owner": {"type": "string"}, "status": {"type": "string", "enum": ["planned", "in_progress", "done"]},
                "tasks": {"type": "array", "items": {"type": "string"}}}, "required": ["title", "owner", "status", "tasks"],
                "additionalProperties": False}}}, "required": ["quarter", "milestones"], "additionalProperties": False}}},
        "required": ["quarters"], "additionalProperties": False}),
    "runbook": ("an incident runbook for {service} degraded latency", {
        "type": "object", "properties": {"title": {"type": "string"}, "severity": {"type": "string", "enum": ["sev1", "sev2", "sev3"]},
                                         "detection": {"type": "array", "items": {"type": "string"}},
                                         "steps": {"type": "array", "items": {"type": "object", "properties": {
                                             "order": {"type": "integer"}, "action": {"type": "string"}, "command": {"type": "string"},
                                             "expected": {"type": "string"}}, "required": ["order", "action", "command", "expected"],
                                             "additionalProperties": False}},
                                         "escalation": {"type": "array", "items": {"type": "string"}}},
        "required": ["title", "severity", "detection", "steps", "escalation"], "additionalProperties": False}),
}

CLASSIFY = {
    "sentiment": (["positive", "neutral", "negative"], {
        "positive": ["The {p} arrived early and works perfectly.", "Honestly the best {p} I've owned, would buy again.",
                     "Great value; the {p} exceeded what I expected.", "Setup took two minutes and the {p} feels solid."],
        "neutral": ["The {p} is fine, nothing special.", "It's a {p}. Does the job, packaging was plain.",
                    "Received the {p}; haven't used it enough to judge.", "Average {p}, about what the price suggests."],
        "negative": ["The {p} broke after a week.", "Returned the {p}: it never matched the photos.",
                     "Support ignored my emails about the faulty {p}.", "The {p} smells like plastic and wobbles."]}, "product"),
    "ticket_route": (["billing", "bug", "feature_request", "account_access", "security"], {
        "billing": ["I was charged twice for the {p} plan this month.", "Can you resend the invoice for our {p} subscription?",
                    "Our card was declined renewing {p}; what now?"],
        "bug": ["Exports crash when the report has more than 500 rows.", "The dashboard shows yesterday's numbers after refresh.",
                "Uploading a PNG returns a 500 error since this morning."],
        "feature_request": ["Could you add dark mode to the {p} dashboard?", "We'd love SSO support on the {p} tier.",
                            "Please let us schedule exports weekly."],
        "account_access": ["I can't log in after resetting my password.", "Our admin left; how do we transfer ownership?",
                           "The two-factor code never arrives."],
        "security": ["We think an API key leaked in a public repo.", "Someone logged in from a country we don't operate in.",
                     "Please confirm whether you were affected by the reported breach."]}, "plan"),
    "log_level": (["info", "warning", "error", "critical"], {
        "info": ["{p} started on port 8080", "{p} processed 1200 jobs in 4.2s", "cache warmed for {p}"],
        "warning": ["{p} retrying upstream call (attempt 3/5)", "{p} disk usage at 82%", "{p} slow query took 2300ms"],
        "error": ["{p} failed to parse payload: unexpected token", "{p} upstream returned 502", "{p} write timed out after 30s"],
        "critical": ["{p} out of memory, process killed", "{p} lost quorum; cluster unavailable", "{p} data corruption detected in shard 4"]},
        "service"),
    "bank_intent": (["check_balance", "transfer", "card_lost", "dispute_charge", "other"], {
        "check_balance": ["how much is left in my checking account", "what's my current balance?", "balance please"],
        "transfer": ["send 200 to my savings tomorrow", "move 50 euros to Ana", "can I wire money to {p} today"],
        "card_lost": ["I think my card was stolen at {p}", "lost my debit card, freeze it", "my wallet is gone, block the card"],
        "dispute_charge": ["there's a charge from {p} I don't recognize", "I was billed twice by {p}", "refund never arrived from {p}"],
        "other": ["what are your branch hours", "do you offer student accounts", "change my mailing address"]}, "company"),
}


def gen_json_schema(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    row = {"_prose": ""}
    v = rng.choices(["records", "aggregate", "classify", "sequence", "plan", "transform"], [0.26, 0.16, 0.14, 0.14, 0.15, 0.15])[0]
    if v == "records":
        n = size(rng)
        subset = None
        if rng.random() < 0.35:
            fs = [f.name for f in fields(dom, ent)]
            subset = [fs[0]] + rng.sample(fs[1:], max(2, len(fs) // 2))
        schema = wrap_root(plural(ent), entity_schema(dom, ent, rng.random() < 0.4, subset), n if rng.random() < 0.5 else None)
        row["_prose"] = f"Return an object with a `{plural(ent)}` array; each item has: " + describe_fields(dom, ent, subset) + "."
        extra = rng.choice(["", " Use a realistic mix of values.", f" Sort them by `{fields(dom, ent)[0].name}`.",
                            " Make the records internally consistent.", " Dates should fall in 2026.",
                            " Avoid repeating the same city or name twice."])
        intro = f"Generate {n} synthetic {human(plural(ent) if n > 1 else ent)} for {label}.{extra}"
    elif v == "aggregate":
        k = rng.randint(5, 40)
        fs = flat_fields(dom, ent)
        rows = [record(rng, dom, ent) for _ in range(k)]
        cols = [f.name for f in fs]
        spec = codegen.Spec(dom, ent)
        g, m = spec.group.name, spec.measure
        props = {"record_count": {"type": "integer"},
                 f"by_{g}": {"type": "array", "items": {"type": "object", "properties": {
                     g: {"type": "string"}, "count": {"type": "integer"}, **({f"total_{m.name}": {"type": "number"}} if m else {})},
                     "required": [g, "count"] + ([f"total_{m.name}"] if m else []), "additionalProperties": False}}}
        if m:
            props[f"max_{m.name}"] = {"type": "object", "properties": {"key": {"type": "string"}, "value": {"type": "number"}},
                                      "required": ["key", "value"], "additionalProperties": False}
        props["observations"] = {"type": "array", "items": {"type": "string"}}
        schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
        row["_prose"] = (f"Fields: `record_count`; `by_{g}` (one entry per {human(g)} with `count`"
                         + (f" and `total_{m.name}`" if m else "") + ")" + (f"; `max_{m.name}` with the record `key` and `value`" if m else "")
                         + "; `observations` (2-4 short strings).")
        fmt = rng.choice(["CSV", "JSON"])
        data = csv_text(rows, cols) if fmt == "CSV" else dumps(rows, rng)
        intro = (f"Here are {k} {human(plural(ent))} from {label} as {fmt}:\n\n{data}\n\n"
                 f"Summarize them: count the records, group by `{g}`" + (f", total `{m.name}` per group, and find the record with the highest `{m.name}` (key = `{spec.key}`)" if m else "")
                 + ", and add a few observations.")
    elif v == "classify":
        task = rng.choice(list(CLASSIFY))
        labels, bank, slot = CLASSIFY[task]
        k = rng.randint(4, 30)
        items = []
        for j in range(k):
            lab = rng.choice(labels)
            items.append((f"{task[:2]}-{j + 1}", rng.choice(bank[lab]).replace("{p}", rng.choice(VOCAB[slot]) if slot in VOCAB else company(rng)), lab))
        item = {"type": "object", "properties": {"id": {"type": "string"}, "label": {"type": "string", "enum": labels},
                                                 "confidence": {"type": "number"}, "reason": {"type": "string"}},
                "required": ["id", "label", "confidence", "reason"], "additionalProperties": False}
        schema = wrap_root("results", item, k)
        row["_prose"] = f"Return `results`: one object per input with `id`, `label` (one of {', '.join(labels)}), `confidence` (0-1) and a short `reason`."
        listing = "\n".join(f"{a}: {b}" for a, b, _ in items) if rng.random() < 0.6 else dumps([{"id": a, "text": b} for a, b, _ in items], rng)
        intro = f"Classify each message ({task.replace('_', ' ')}).\n\n{listing}"
        row["labels_hidden"] = {a: c for a, _, c in items}
    elif v == "sequence":
        desc, item, item_fields = seq_task(rng)
        root = rng.random() < 0.3
        if root:
            schema = {"type": "array", "items": item}
        else:
            schema = wrap_root("rows", item)
        row["_prose"] = ("Return a JSON array" if root else "Return an object with a `rows` array") + " of objects with " + ", ".join(f"`{k}` ({t})" for k, t in item_fields.items()) + "."
        intro = f"Produce {desc} as JSON."
        if root and rng.random() < 0.7:
            row["_force_instructed"] = True   # array roots are not portable for strict schemas
    elif v == "plan":
        pname = rng.choice(list(PLAN_SCHEMAS))
        tmpl, schema = PLAN_SCHEMAS[pname]
        intro = "Create " + tmpl.format(d=rng.randint(2, 5), city=rng.choice(CITIES), service=rng.choice(VOCAB["service"]),
                                        a=rng.choice(["MySQL 5.7", "a monolith", "self-managed Kafka", "cron jobs", "REST polling"]),
                                        b=rng.choice(["PostgreSQL 16", "three services", "a managed queue", "a workflow engine", "webhooks"]),
                                        label=label) + "." + rng.choice(["", " Be specific.", " Keep each string short.", " Aim for depth over breadth."])
        row["_prose"] = "Use this structure: " + json.dumps(schema)
    else:  # transform
        k = size(rng, 25)
        fs = flat_fields(dom, ent)
        rows = [record(rng, dom, ent) for _ in range(k)]
        spec = codegen.Spec(dom, ent)
        g = spec.group.name
        how = rng.choice(["camel", "group", "subset"])
        if how == "camel":
            cam = {f.name: codegen.camel(f.name) for f in fs}
            item = {"type": "object", "properties": {cam[f.name]: entity_schema(dom, ent)["properties"][f.name] for f in fs},
                    "required": list(cam.values()), "additionalProperties": False}
            schema = wrap_root("items", item, k)
            intro = f"Convert every key in these {human(plural(ent))} to camelCase and wrap them in an `items` array. Keep values unchanged."
        elif how == "group":
            item = {"type": "object", "properties": {g: {"type": "string"}, "keys": {"type": "array", "items": {"type": "string"}},
                                                     "count": {"type": "integer"}},
                    "required": [g, "keys", "count"], "additionalProperties": False}
            schema = wrap_root("groups", item)
            intro = f"Group these {human(plural(ent))} by `{g}`: for each group list the `{spec.key}` values as `keys` and the `count`."
        else:
            sub = [fs[0].name] + rng.sample([f.name for f in fs[1:]], min(3, len(fs) - 1))
            schema = wrap_root("items", entity_schema(dom, ent, False, sub), k)
            intro = f"Keep only the fields {', '.join('`' + s + '`' for s in sub)} of each record and return them under `items`."
        row["_prose"] = "Target shape: " + json.dumps(schema)
        intro += "\n\n" + dumps(rows, rng)
    force = "instructed" if row.pop("_force_instructed", False) else None
    text = json_request(rng, row, schema, f"{v}_{ent}"[:60], intro, force)
    row["messages"] = msgs(sys_prompt(rng, dom, "json"), text)
    row["check"]["variant"] = v
    row.pop("labels_hidden", None)
    return row, dom


# ---------------------------------------------------------------- json_extract

FILLER = ["Thanks again for the quick turnaround.", "Let me know if anything looks off.", "Sorry for the long message.",
          "Copying the team for visibility.", "We can discuss on Thursday's call.", "Ignore the earlier draft.",
          "This is from the export someone pasted into the channel.", "Numbers are from the system of record."]


def mention(rng, f, v):
    h = human(f.name)
    if isinstance(v, bool):
        v = rng.choice((("yes", "no"), ("true", "false"), ("Y", "N")))[0 if v else 1]
    elif isinstance(v, list):
        v = ", ".join(v)
    return rng.choice([f"{h}: {v}", f"the {h} is {v}", f"{h} = {v}", f"{v} ({h})", f"{h} — {v}", f"with {h} {v}"])


def render_doc(rng, dom, ent, recs, missing):
    style = rng.choice(["email", "notes", "log", "chat", "html", "narrative", "ticket"])
    fs = flat_fields(dom, ent)
    lines = []
    if style == "email":
        lines += [f"From: {person(rng)} <{ident(rng, 6)}@{rng.choice(['ops', 'team', 'desk'])}.example>",
                  f"Subject: {human(plural(ent)).capitalize()} update", "", "Hi all,", ""]
    elif style == "ticket":
        lines += [f"Ticket #{rng.randint(1000, 99999)} — {human(ent)} data from {DOMAINS[dom][0]}", ""]
    for j, r in enumerate(recs):
        parts = [mention(rng, f, r[f.name]) for f in fs if (j, f.name) not in missing]
        tail = parts[1:]
        rng.shuffle(tail)
        parts = parts[:1] + tail
        if style == "log":
            lines.append(f"{r.get('placed_at') or '2026-0%d-1%dT0%d:00:00Z' % (rng.randint(1, 9), rng.randint(0, 9), rng.randint(0, 9))} "
                         f"{rng.choice(['INFO', 'INFO', 'WARN'])} {ent} " + " ".join(
                             f"{f.name}={json.dumps(r[f.name]) if not isinstance(r[f.name], str) else r[f.name]!r}"
                             for f in fs if (j, f.name) not in missing))
        elif style == "html":
            lines.append("<ul>" + "".join(f"<li><b>{human(f.name)}</b>: {r[f.name]}</li>" for f in fs if (j, f.name) not in missing) + "</ul>")
        elif style == "chat":
            lines.append(f"[{rng.randint(8, 18)}:{rng.randint(10, 59)}] {rng.choice(FIRST_NAMES)}: " + "; ".join(parts))
        elif style == "notes":
            lines.append(f"- {human(ent).capitalize()} {j + 1}: " + ", ".join(parts))
        else:
            lines.append(rng.choice([f"Next, a {human(ent)} ", f"We also have one ", f"Another {human(ent)}: ", "Then "])
                         + ", ".join(parts) + ". " + (rng.choice(FILLER) if rng.random() < 0.25 else ""))
    if style == "email":
        lines += ["", rng.choice(FILLER), "", "Best,", person(rng)]
    return "\n".join(lines), style


FIRST_NAMES = ["Ana", "Ben", "Chi", "Dev", "Eli", "Fay", "Gus", "Ida"]


def gen_json_extract(rng, i):
    dom, ent = pick(rng)
    fs = flat_fields(dom, ent)
    k = size(rng, 16)
    recs = [record(rng, dom, ent) for _ in range(k)]
    missing = set()
    nullable = rng.random() < 0.3
    if nullable:
        for j in range(k):
            if rng.random() < 0.5:
                missing.add((j, rng.choice(fs[1:]).name))
    doc, style = render_doc(rng, dom, ent, recs, missing)
    subset = None
    if rng.random() < 0.4:
        subset = [fs[0].name] + rng.sample([f.name for f in fs[1:]], min(len(fs) - 1, rng.randint(2, 4)))
    item = entity_schema(dom, ent, rng.random() < 0.3, subset or [f.name for f in fs])
    if nullable:
        for p in item["properties"].values():
            if isinstance(p.get("type"), str) and p["type"] != "array":
                p["type"] = [p["type"], "null"]
                if "enum" in p:
                    p["enum"] = p["enum"] + [None]
    schema = wrap_root(plural(ent), item)
    row = {"_prose": f"Return `{plural(ent)}`: one object per {human(ent)} with " + describe_fields(dom, ent, subset or [f.name for f in fs])
           + "." + (" Use null when a value is not stated." if nullable else "")}
    intro = (rng.choice(["Extract every {e} mentioned in the text below.", "Pull the {e} records out of this {s}.",
                         "Parse the following {s} into structured {e} data.", "Turn this {s} into JSON, one entry per {e}."])
             .format(e=human(ent), s={"log": "log excerpt", "html": "HTML fragment", "chat": "chat transcript"}.get(style, style))
             + (" Use null for anything not stated." if nullable else "") + "\n\n---\n" + doc + "\n---")
    text = json_request(rng, row, schema, f"extract_{ent}", intro)
    row["messages"] = msgs(sys_prompt(rng, dom, "json"), text)
    row["check"]["variant"] = style
    row["check"]["expected_items"] = k
    return row, dom


# ---------------------------------------------------------------- tool_call

def fn(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": required, "additionalProperties": False}}}


GENERIC_TOOLS = [
    fn("get_weather", "Current weather and short forecast for a location.",
       {"location": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["location"]),
    fn("get_current_time", "Current local time for an IANA timezone.", {"timezone": {"type": "string"}}, ["timezone"]),
    fn("convert_currency", "Convert an amount between ISO 4217 currencies at today's rate.",
       {"amount": {"type": "number"}, "from_currency": {"type": "string"}, "to_currency": {"type": "string"}}, ["amount", "from_currency", "to_currency"]),
    fn("send_email", "Send an email from the shared operations mailbox.",
       {"to": {"type": "array", "items": {"type": "string"}}, "subject": {"type": "string"}, "body": {"type": "string"},
        "cc": {"type": "array", "items": {"type": "string"}}}, ["to", "subject", "body"]),
    fn("create_calendar_event", "Create a calendar event.",
       {"title": {"type": "string"}, "start": {"type": "string", "description": "ISO 8601"}, "end": {"type": "string", "description": "ISO 8601"},
        "attendees": {"type": "array", "items": {"type": "string"}}, "location": {"type": "string"}}, ["title", "start", "end"]),
    fn("web_search", "Search the web.", {"query": {"type": "string"}, "max_results": {"type": "integer"}}, ["query"]),
    fn("run_sql", "Run a read-only SQL query against the analytics warehouse.",
       {"query": {"type": "string"}, "database": {"type": "string", "enum": ["analytics", "billing", "events"]}}, ["query"]),
    fn("read_file", "Read a UTF-8 file from the project workspace.", {"path": {"type": "string"}}, ["path"]),
    fn("write_file", "Create or overwrite a file in the project workspace.",
       {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
    fn("list_directory", "List entries of a workspace directory.", {"path": {"type": "string"}, "recursive": {"type": "boolean"}}, ["path"]),
    fn("http_request", "Make an HTTP request to an internal service.",
       {"method": {"type": "string", "enum": ["GET", "POST", "PUT", "PATCH", "DELETE"]}, "url": {"type": "string"},
        "headers": {"type": "object", "additionalProperties": {"type": "string"}}, "body": {"type": "string"}}, ["method", "url"]),
    fn("calculate", "Evaluate an arithmetic expression exactly.", {"expression": {"type": "string"}}, ["expression"]),
    fn("translate_text", "Translate text.", {"text": {"type": "string"}, "target_language": {"type": "string"}}, ["text", "target_language"]),
    fn("create_issue", "Open an issue in the team tracker.",
       {"title": {"type": "string"}, "body": {"type": "string"}, "priority": {"type": "string", "enum": ["low", "medium", "high", "urgent"]},
        "labels": {"type": "array", "items": {"type": "string"}}}, ["title", "body"]),
    fn("post_message", "Post a message to a chat channel.", {"channel": {"type": "string"}, "text": {"type": "string"}}, ["channel", "text"]),
    fn("schedule_reminder", "Schedule a reminder.", {"text": {"type": "string"}, "at": {"type": "string", "description": "ISO 8601"}}, ["text", "at"]),
]
GENERIC = {t["function"]["name"]: t for t in GENERIC_TOOLS}


def domain_tools(dom):
    out = {}
    for ent in top_entities(dom):
        spec = codegen.Spec(dom, ent)
        fs = flat_fields(dom, ent)
        sch = entity_schema(dom, ent, True, [f.name for f in fs])["properties"]
        g = spec.group
        gsch = sch[g.name]
        out[f"search_{plural(ent)}"] = fn(f"search_{plural(ent)}", f"Search {human(plural(ent))} in {DOMAINS[dom][0]}.",
                                          {"query": {"type": "string"}, g.name: gsch, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, [])
        out[f"get_{ent}"] = fn(f"get_{ent}", f"Fetch one {human(ent)} by `{spec.key}`.", {spec.key: {"type": "string"}}, [spec.key])
        req = [f.name for f in fs[1:] if f.kind not in ("bool", "list")]
        out[f"create_{ent}"] = fn(f"create_{ent}", f"Create a {human(ent)}.", {f.name: sch[f.name] for f in fs[1:]}, req)
        out[f"update_{ent}"] = fn(f"update_{ent}", f"Update a {human(ent)}'s {human(g.name)} and attach a note.",
                                  {spec.key: {"type": "string"}, g.name: gsch, "note": {"type": "string"}}, [spec.key, g.name])
    return out


def tc(name, args, rng):
    return {"id": "call_" + ident(rng, 24), "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def gen_tool_call(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    dt = domain_tools(dom)
    spec = codegen.Spec(dom, ent)
    g = spec.group.name
    sc = rng.choices(["single", "parallel", "create_many", "result_answer", "result_chain", "write_file", "multi_turn", "no_tool"],
                     [0.18, 0.14, 0.11, 0.17, 0.13, 0.09, 0.12, 0.06])[0]
    recs = [record(rng, dom, ent) for _ in range(rng.randint(2, 6))]
    keys = [r[spec.key] for r in recs]
    gv = rng.choice(spec.group.arg.split("|")) if spec.group.kind == "enum" else recs[0][g]
    needed = set()
    expect = "call"
    hist = []
    if sc == "single":
        ask = rng.choice([
            (f"Find {human(plural(ent))} with {human(g)} {gv}, up to {rng.choice((5, 10, 20))} results.", f"search_{plural(ent)}"),
            (f"Pull up {human(ent)} {keys[0]}.", f"get_{ent}"),
            (f"Set {human(ent)} {keys[0]} to {gv} and note that it was {rng.choice(VOCAB['note'])}.", f"update_{ent}"),
            (f"What's the weather in {rng.choice(CITIES)} right now, in {rng.choice(('celsius', 'fahrenheit'))}?", "get_weather"),
            (f"Convert {rng.randint(10, 9000)} {rng.choice(('USD', 'EUR', 'JPY'))} to {rng.choice(('GBP', 'CAD', 'CHF'))}.", "convert_currency"),
            (f"Email {person(rng).split()[0].lower()}@ops.example a summary saying the {human(ent)} backlog for {rng.choice(CITIES)} is cleared.", "send_email"),
            (f"Book a 45-minute review called '{title(ent)} audit' next Tuesday at 10:00 UTC with {person(rng)} and {person(rng)}.", "create_calendar_event"),
            (f"How many {human(plural(ent))} per {human(g)} did we have last month? Query the warehouse table `{plural(ent)}`.", "run_sql"),
        ])
        user, t = ask
        needed.add(t)
    elif sc == "parallel":
        n = rng.randint(2, 5)
        if rng.random() < 0.5:
            user = f"Get me the details for these {human(plural(ent))}: " + ", ".join(keys[:n]) + "."
            needed.add(f"get_{ent}")
        else:
            cs = rng.sample(CITIES, n)
            user = "Compare the weather in " + ", ".join(cs[:-1]) + f" and {cs[-1]}, and tell me the local time in {cs[0]}."
            needed |= {"get_weather", "get_current_time"}
    elif sc == "create_many":
        n = min(len(recs), rng.randint(1, 5))
        lines = []
        for r in recs[:n]:
            lines.append("- " + "; ".join(f"{human(f.name)}: {', '.join(r[f.name]) if isinstance(r[f.name], list) else r[f.name]}"
                                          for f in flat_fields(dom, ent)[1:]))
        user = f"Add these {n} {human(plural(ent) if n > 1 else ent)} to the system:\n" + "\n".join(lines)
        needed.add(f"create_{ent}")
    elif sc in ("result_answer", "result_chain"):
        q = f"Show me {human(plural(ent))} with {human(g)} {gv}."
        call = tc(f"search_{plural(ent)}", {g: gv, "limit": 10}, rng)
        for r in recs:
            r[g] = gv if isinstance(gv, str) else r[g]
        result = {"results": recs, "total": len(recs)}
        hist = [{"role": "user", "content": q}, {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)}]
        needed.add(f"search_{plural(ent)}")
        if sc == "result_answer":
            user = None
            expect = "answer"
            if rng.random() < 0.5:
                hist[0]["content"] += rng.choice([" Put them in a markdown table.", " Summarize as a table with the key fields.",
                                                  " Just list the ids and one line each."])
        else:
            new = rng.choice(spec.group.arg.split("|")) if spec.group.kind == "enum" else "reviewed"
            user = f"Thanks. Now move all of those to {new} with the note '{rng.choice(VOCAB['note'])}'."
            needed.add(f"update_{ent}")
    elif sc == "write_file":
        kind = rng.choice(["yaml", "json", "toml", "env"])
        path = {"yaml": f"config/{dom}.yaml", "json": f"fixtures/{plural(ent)}.json", "toml": f"config/{dom}.toml", "env": ".env.example"}[kind]
        user = (f"Create `{path}` containing " + {"yaml": f"a config for the {dom} service: database, cache, feature flags and per-region rate limits",
                                                 "json": f"{len(recs)} sample {human(plural(ent))} with fields " + ", ".join(f.name for f in flat_fields(dom, ent)),
                                                 "toml": f"settings for the {dom} worker (queues, retries, timeouts, logging)",
                                                 "env": f"every environment variable the {dom} service needs, with placeholder values and comments"}[kind] + ".")
        needed.add("write_file")
    elif sc == "multi_turn":
        call = tc(f"get_{ent}", {spec.key: keys[0]}, rng)
        hist = [{"role": "user", "content": f"What's the status of {human(ent)} {keys[0]}?"},
                {"role": "assistant", "content": "", "tool_calls": [call]},
                {"role": "tool", "tool_call_id": call["id"], "content": json.dumps(recs[0])},
                {"role": "assistant", "content": f"{human(ent).capitalize()} {keys[0]} has {human(g)} **{recs[0][g]}**."}]
        user = rng.choice([f"Change it to {gv} and email {person(rng).split()[0].lower()}@ops.example about it.",
                           f"Also look up {keys[1]} and {keys[2] if len(keys) > 2 else keys[1]}.",
                           f"Open a high-priority issue about {keys[0]} and post a heads-up in #{dom}-ops."])
        needed |= {f"update_{ent}", "send_email", f"get_{ent}", "create_issue", "post_message"} & set(list(dt) + list(GENERIC))
        needed.add(f"get_{ent}")
    else:
        user = rng.choice([f"In one paragraph, what does a {human(ent)} represent in {label}?",
                           "Thanks, that's all for now!", f"What fields does a {human(ent)} have? No need to look anything up."])
        expect = "answer_or_call"
    pool = dict(dt)
    tools = {k: v for k, v in pool.items() if k in needed}
    tools.update({k: GENERIC[k] for k in sorted(needed) if k in GENERIC})
    distract = [k for k in list(pool) + list(GENERIC) if k not in tools]
    for k in rng.sample(distract, min(len(distract), rng.randint(1, 9))):
        tools[k] = pool.get(k) or GENERIC[k]
    tl = list(tools.values())
    rng.shuffle(tl)
    system = rng.choice([None, f"You are an operations assistant for {label}. Use the tools to act; never invent record data.",
                         f"You help staff at {label}. Today is 2026-0{rng.randint(1, 9)}-1{rng.randint(0, 9)}. Call tools when you need data or to make changes.",
                         "You are a helpful assistant with access to tools."])
    messages = ([{"role": "system", "content": system}] if system else []) + hist + ([{"role": "user", "content": user}] if user else [])
    row = {"messages": messages, "tools": tl,
           "check": {"kind": "tool", "expect": expect, "variant": sc, "needed": sorted(needed)}}
    if sc in ("single", "create_many", "write_file") and rng.random() < 0.25:
        row["tool_choice"] = "required"
        row["check"]["expect"] = "call"
    return row, dom


# ---------------------------------------------------------------- config

LANG_DEPS = {"rust": ["serde", "tokio", "clap", "anyhow", "reqwest", "tracing", "sqlx", "axum", "thiserror", "chrono", "uuid", "rand"],
             "python": ["fastapi", "pydantic", "httpx", "sqlalchemy", "alembic", "redis", "celery", "structlog", "pytest", "ruff", "mypy"]}


def ddl(rng, dom, dialect):
    """Enntity-authored DDL text for a domain (given to SQL query prompts)."""
    out = []
    for ent in top_entities(dom):
        cols = []
        for fi, f in enumerate(flat_fields(dom, ent)):
            t = {"int": "INTEGER", "num": "NUMERIC(12,2)", "bool": "BOOLEAN", "date": "DATE",
                 "datetime": "TIMESTAMP" if dialect != "sqlite" else "TEXT", "list": "TEXT"}.get(f.kind, "TEXT" if dialect == "sqlite" else "VARCHAR(120)")
            c = f"  {f.name} {t}"
            if fi == 0:
                c += " PRIMARY KEY"
            elif f.kind == "enum":
                c += " NOT NULL CHECK (" + f.name + " IN (" + ", ".join(f"'{v}'" for v in f.arg.split("|")) + "))"
            cols.append(c)
        out.append(f"CREATE TABLE {plural(ent)} (\n" + ",\n".join(cols) + "\n);")
    return "\n\n".join(out)


def gen_config(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    fs = flat_fields(dom, ent)
    spec = codegen.Spec(dom, ent)
    svc = rng.choice(VOCAB["service"])
    fmt = rng.choices(["yaml", "toml", "sql", "csv", "other"], [0.27, 0.15, 0.25, 0.15, 0.18])[0]
    only = rng.choice(["", " Output only the file.", " Return just the file in one fenced code block.",
                       " Add brief comments where useful.", " No explanation needed."])
    chk = {"kind": "config", "format": fmt}
    if fmt == "yaml":
        v = rng.choice(["k8s", "compose", "gha", "openapi", "fixture", "alerts", "ansible"])
        if v == "k8s":
            parts = ["Deployment", "Service"] + rng.sample(["ConfigMap", "HorizontalPodAutoscaler", "PodDisruptionBudget", "Ingress"], rng.randint(0, 3))
            env = rng.sample(["DATABASE_URL", "REDIS_URL", "LOG_LEVEL", "REGION", "FEATURE_FLAGS", "OTEL_ENDPOINT", "MAX_WORKERS"], rng.randint(2, 5))
            user = (f"Write Kubernetes manifests ({', '.join(parts)}) for `{svc}`: image registry.example/{svc}:{rng.choice(['1.4.2', '2.0.0', '2.3.1'])}, "
                    f"{rng.randint(2, 12)} replicas, container port {rng.choice([8080, 3000, 9000, 5000])}, env vars {', '.join(env)}, "
                    f"CPU/memory requests and limits, readiness and liveness probes on /healthz, namespace `{dom}`.")
        elif v == "compose":
            s = rng.sample(VOCAB["service"], rng.randint(2, 5)) + rng.sample(["postgres", "redis", "nats", "minio", "otel-collector"], rng.randint(1, 3))
            user = f"Write a docker-compose.yml for local development of {label} with services: {', '.join(s)}. Include healthchecks, named volumes, a shared network and sensible depends_on."
        elif v == "gha":
            lang = rng.choice(["Python", "Node.js", "Go", "Rust", "Java"])
            user = (f"Write a GitHub Actions workflow for a {lang} repository: run on pushes and pull requests to main, a test matrix over "
                    f"{rng.randint(2, 4)} {lang} versions and ubuntu/macos, cache dependencies, lint, test with coverage, and a deploy job to "
                    f"{rng.choice(VOCAB['region'])} that only runs on version tags.")
        elif v == "openapi":
            user = (f"Write an OpenAPI 3.0 spec (YAML) for a CRUD API on {human(plural(ent))} for {label}. Fields: "
                    + describe_fields(dom, ent, [f.name for f in fs]) + ". Include list with pagination, get, create, update, delete, error responses and component schemas.")
        elif v == "fixture":
            n = size(rng, 40)
            user = f"Write a YAML fixture file with {n} synthetic {human(plural(ent))} under a top-level `{plural(ent)}` key. Fields: " + ", ".join(f.name for f in fs) + "."
        elif v == "alerts":
            user = f"Write Prometheus alerting rules (YAML) for {', '.join(rng.sample(VOCAB['service'], rng.randint(2, 4)))}: error rate, p99 latency, saturation and an absent-metrics alert, with severity labels and runbook annotations."
        else:
            user = f"Write an Ansible playbook that installs and configures {rng.choice(['nginx', 'postgresql', 'node_exporter', 'redis', 'chrony'])} on the `{dom}_servers` group, templating its config and restarting via a handler."
        chk["variant"] = v
    elif fmt == "toml":
        v = rng.choice(["cargo", "pyproject", "app", "fixture"])
        if v == "cargo":
            user = (f"Write a Cargo.toml for a crate `{dom}-{rng.choice(['worker', 'api', 'cli', 'sync'])}` (edition 2021) depending on "
                    + ", ".join(rng.sample(LANG_DEPS["rust"], rng.randint(3, 7))) + ", with features, a dev-dependency section and an optimized release profile.")
        elif v == "pyproject":
            user = (f"Write a pyproject.toml for a Python package `{dom}_service` using " + ", ".join(rng.sample(LANG_DEPS["python"], rng.randint(3, 7)))
                    + ", with optional dev dependencies, a console script, and ruff, mypy and pytest configuration.")
        elif v == "app":
            user = f"Write a TOML config for the {svc} of {label}: server, database, cache, logging, feature flags and per-region limits for {', '.join(rng.sample(VOCAB['region'], 3))}."
        else:
            n = size(rng, 25)
            user = f"Write {n} synthetic {human(plural(ent))} as a TOML array of tables `[[{plural(ent)}]]` with fields " + ", ".join(f.name for f in fs) + "."
        chk["variant"] = v
    elif fmt == "sql":
        dialect = rng.choice(["PostgreSQL", "SQLite", "MySQL"])
        v = rng.choice(["ddl", "query", "query", "inserts", "migration"])
        schema_txt = ddl(rng, dom, dialect.lower())
        if v == "ddl":
            user = (f"Write {dialect} DDL for {label}. Tables: " + "; ".join(f"{plural(e)} (" + describe_fields(dom, e, [f.name for f in flat_fields(dom, e)]) + ")"
                                                                             for e in top_entities(dom))
                    + ". Add primary keys, CHECK constraints for enumerations, NOT NULL where sensible and useful indexes.")
        elif v == "query":
            m = spec.measure.name if spec.measure else spec.key
            task = rng.choice([f"returns the top 3 {human(plural(ent))} by `{m}` within each `{spec.group.name}`",
                               f"returns, per `{spec.group.name}`, the count and average `{m}`, only for groups with at least 5 rows",
                               f"returns the running total of `{m}` ordered by `{spec.key}`",
                               f"finds `{spec.group.name}` values whose total `{m}` is above the overall average",
                               f"returns the {rng.randint(5, 20)} most recent {human(plural(ent))} with their rank by `{m}`"])
            user = f"Given this {dialect} schema:\n\n```sql\n{schema_txt}\n```\n\nWrite a query that {task}. Use CTEs where they help."
        elif v == "inserts":
            n = size(rng, 30)
            user = f"Given this {dialect} schema:\n\n```sql\n{schema_txt}\n```\n\nWrite INSERT statements adding {n} realistic rows to `{plural(ent)}`."
        else:
            col = rng.choice(["archived_at", "priority", "external_ref", "updated_by", "region_code"])
            user = f"Given this {dialect} schema:\n\n```sql\n{schema_txt}\n```\n\nWrite an up and down migration adding `{col}` to `{plural(ent)}`, backfilling existing rows and adding an index."
        chk.update(variant=v, dialect=dialect, ddl=schema_txt)
    elif fmt == "csv":
        n = size(rng)
        sep = rng.choice([",", ",", ",", "\t", ";"])
        sname = {",": "CSV", "\t": "TSV", ";": "semicolon-separated CSV"}[sep]
        cols = [f.name for f in fs]
        if rng.random() < 0.6:
            user = f"Produce a {sname} file with header `{(sep if sep != chr(9) else ', ').join(cols)}` and {n} {'row' if n == 1 else 'rows'} of synthetic {human(plural(ent) if n > 1 else ent)} for {label}. Quote fields that contain the separator."
        else:
            rows = [record(rng, dom, ent) for _ in range(min(n, 25))]
            user = f"Convert this JSON to {sname} with a header row (lists joined with `|`):\n\n" + dumps(rows, rng)
            n = len(rows)
        chk.update(variant="csv", sep=sep, columns=len(cols), rows=n)
    else:
        v = rng.choice(["nginx", "dockerfile", "makefile", "systemd", "terraform", "proto", "graphql", "dotenv", "ini", "xml"])
        user = {
            "nginx": f"Write an nginx server block for {dom}.example that terminates TLS, redirects HTTP to HTTPS, proxies /api to {svc}:8080 and /ws with websocket upgrade, adds security headers and gzip.",
            "dockerfile": f"Write a multi-stage Dockerfile for a {rng.choice(['Go', 'Rust', 'Node.js 22', 'Python 3.12', 'Java 21'])} service called {svc}: build stage, slim runtime stage, non-root user, healthcheck.",
            "makefile": f"Write a Makefile for the {dom} project with targets build, test, lint, fmt, run, docker-build, docker-push, migrate and clean, with a help target that lists them.",
            "systemd": f"Write a systemd service unit and timer that run `/opt/{dom}/bin/{svc} sync` every {rng.choice([5, 10, 15, 30])} minutes with restart on failure, hardening options and an environment file.",
            "terraform": f"Write Terraform (HCL) for {label}: an object-storage bucket with versioning, a managed PostgreSQL instance, a private network and outputs. Use variables for region and instance size.",
            "proto": f"Write a protobuf (proto3) definition for a {title(ent)}Service with the {title(ent)} message (fields: " + ", ".join(f.name for f in fs) + "), enums where values are fixed, and Get/List/Create/Update RPCs.",
            "graphql": f"Write a GraphQL SDL schema for {human(plural(ent))}: the {title(ent)} type (fields: " + ", ".join(f.name for f in fs) + "), enums, a paginated connection, queries and mutations.",
            "dotenv": f"Write a .env.example for the {svc} of {label}, grouping variables by concern with comments and safe placeholder values.",
            "ini": f"Write an INI config for a legacy {dom} batch job with sections [general], [database], [paths], [retry] and [notifications].",
            "xml": f"Write an XML document listing {size(rng, 20)} synthetic {human(plural(ent))} (fields: " + ", ".join(f.name for f in fs) + ") under a <{plural(ent)}> root.",
        }[v]
        chk["variant"] = v
    row = {"messages": msgs(sys_prompt(rng, dom, "code"), user + only), "check": chk}
    return row, dom


# ---------------------------------------------------------------- code_edit

NEW_FIELDS = [("notes", "optional free-text string"), ("external_ref", "optional string"), ("created_by", "string, required"),
              ("priority", "integer from 1 to 5"), ("updated_at", "ISO 8601 timestamp string")]


def edit_instruction(rng, s: codegen.Spec, nm, lang, bug):
    if bug:
        lo, hi = s.lohi()
        return "Users report that " + codegen.BUG_SYMPTOM[bug].format(
            list_fn=nm["list_fn"], total_fn=nm["total_fn"], group=human(s.group.name),
            measure=human(s.measure.name) if s.measure else "", range=human(s.range.name) if s.range else "", lo=lo) + ". Fix it.", "fix"
    opts = [("add_field", "Add a `{f}` field ({d}) to the {T} model and validate it."),
            ("count", f"Add a function that counts records per `{s.group.name}` (same style as the existing code)."),
            ("paginate", f"Add `offset` and `limit` parameters to `{nm['list_fn']}` and apply them after filtering."),
            ("duplicate", "Make adding a record whose key already exists fail with a clear error instead of overwriting it."),
            ("rename", f"Rename `{nm['list_fn']}` to `find_by_{s.group.name}` (keep the language's naming convention) and update every reference."),
            ("update", f"Add a method that changes a stored record's `{s.group.name}`, re-validating the record before saving it."),
            ("sort", f"Make `{nm['list_fn']}` return results sorted by `{s.measure.name if s.measure else s.key}` in descending order.")]
    if s.measure:
        opts.append(("nonneg", f"Validation should also reject a negative `{s.measure.name}`."))
    k, text = rng.choice(opts)
    f, d = rng.choice(NEW_FIELDS)
    return text.format(f=f, d=d, T=s.T), k


def gen_code_edit(rng, i):
    dom, ent = pick(rng)
    s = codegen.Spec(dom, ent)
    lang = rng.choices(codegen.LANGS, [0.3, 0.22, 0.16, 0.12, 0.1, 0.1])[0]
    bug = rng.choice(codegen.BUGS) if rng.random() < 0.35 else None
    if bug == "range_exclusive" and not s.range or bug == "overwrite_total" and not s.measure:
        bug = "filter_inverted"
    src = codegen.source(lang, s, bug)
    nm = codegen.names(lang, s)
    path = codegen.path_for(lang, s)
    instr, ek = edit_instruction(rng, s, nm, lang, bug)
    fmt = rng.choices(["diff", "search_replace", "function", "file"], [0.3, 0.3, 0.18, 0.22])[0]
    if fmt == "function" and ek in ("add_field", "count", "duplicate", "update", "rename"):
        fmt = "search_replace"
    fmt_text = {
        "diff": rng.choice([f"Respond with a unified diff against `{path}` only.", "Reply with a git-style unified diff, nothing else.",
                            f"Output the change as a unified diff (---/+++ headers, @@ hunks) for `{path}`."]),
        "search_replace": (f"Answer with SEARCH/REPLACE blocks in this format, one per change:\n\n{path}\n<<<<<<< SEARCH\n(exact existing lines)\n=======\n(new lines)\n>>>>>>> REPLACE\n\n"
                           "The SEARCH part must match the file exactly."),
        "function": "Return only the corrected/updated function (or method) in one code block.",
        "file": rng.choice(["Return the complete updated file.", "Show the whole file after the change, in one code block."]),
    }[fmt]
    user = f"`{path}`:\n\n```{lang}\n{src}\n```\n\n{instr}\n\n{fmt_text}"
    if rng.random() < 0.3:
        user = f"{instr}\n\n{fmt_text}\n\n`{path}`:\n```{lang}\n{src}\n```"
    row = {"messages": msgs(sys_prompt(rng, dom, "code"), user),
           "check": {"kind": "edit", "format": fmt, "lang": lang, "original": src, "edit": ek, "bug": bug, "path": path}}
    return row, dom


# ---------------------------------------------------------------- table_md

COMPARE = [("PostgreSQL, MySQL, SQLite and MongoDB", "a small multi-tenant SaaS backend"),
           ("Kafka, RabbitMQ, NATS and Redis Streams", "an event pipeline handling 20k messages/s"),
           ("REST, GraphQL and gRPC", "a mobile app talking to a dozen services"),
           ("cron, systemd timers and Kubernetes CronJobs", "scheduled batch jobs"),
           ("Go, Rust, Java and Python", "a latency-sensitive API gateway"),
           ("S3-style object storage, a network file system and block volumes", "storing user uploads"),
           ("blue-green, canary and rolling deployments", "a payments service"),
           ("JWT sessions, opaque server sessions and mTLS", "service and user authentication"),
           ("SQLite, DuckDB and pandas", "local analytics on 5 GB of CSV files"),
           ("Terraform, Pulumi and CloudFormation", "a team managing three environments")]
CRITERIA = ["operational complexity", "throughput", "latency", "learning curve", "ecosystem", "cost", "durability",
            "horizontal scaling", "tooling", "failure modes", "best fit"]
COMMIT_KIND = ["feat", "fix", "perf", "docs", "refactor", "chore", "test", "build"]
COMMIT_WHAT = ["add retry budget to {s}", "handle empty payloads in {s}", "speed up cold start of {s}", "document {s} env vars",
               "split {s} handlers into modules", "bump base image for {s}", "cover pagination edge cases in {s}",
               "drop legacy v1 endpoints from {s}", "cache tenant lookups in {s}", "fix race in {s} shutdown"]


def gen_table_md(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    v = rng.choices(["data", "compare", "report", "changelog", "schedule", "cli"], [0.3, 0.2, 0.15, 0.12, 0.11, 0.12])[0]
    if v == "data":
        k = size(rng, 30)
        rows = [record(rng, dom, ent) for _ in range(k)]
        fs = flat_fields(dom, ent)
        cols = [f.name for f in fs]
        spec = codegen.Spec(dom, ent)
        want = rng.choice([f"a Markdown table of all rows sorted by `{(spec.measure or spec.group).name}`",
                           f"one Markdown table per `{spec.group.name}` value, each under a heading",
                           f"a Markdown table with columns {', '.join(rng.sample(cols, min(4, len(cols))))}" + (f" plus a totals row for `{spec.measure.name}`" if spec.measure else ""),
                           "a Markdown table followed by three bullet-point takeaways"])
        data = csv_text(rows, cols) if rng.random() < 0.5 else dumps(rows, rng)
        user = f"Turn this {human(ent)} data from {label} into {want}.\n\n{data}"
    elif v == "compare":
        opts, ctx = rng.choice(COMPARE)
        crit = rng.sample(CRITERIA, rng.randint(4, 7))
        user = f"Compare {opts} for {ctx} in a Markdown table with rows for {', '.join(crit)}. Then give a two-sentence recommendation."
    elif v == "report":
        team = rng.choice(["platform", "payments", "data", "mobile", "support tooling"])
        facts = [f"- {rng.choice(VOCAB['service'])}: availability {round(rng.uniform(98.5, 99.99), 2)}%, p99 {rng.randint(80, 900)} ms" for _ in range(rng.randint(2, 5))]
        facts += [f"- incident {rng.choice(['sev2', 'sev3'])}: {rng.choice(VOCAB['note'])} ({rng.randint(10, 300)} min)" for _ in range(rng.randint(0, 3))]
        facts += [f"- shipped: {rng.choice(COMMIT_WHAT).format(s=rng.choice(VOCAB['service']))}" for _ in range(rng.randint(1, 4))]
        user = (f"Write the weekly status report for the {team} team in Markdown: Summary, Metrics (table), Incidents (table, or 'None'), "
                f"Shipped (bullets) and Next week (checklist). Facts:\n" + "\n".join(facts))
    elif v == "changelog":
        n = rng.randint(6, 30)
        commits = [f"{ident(rng, 7)} {rng.choice(COMMIT_KIND)}({rng.choice(VOCAB['service']).split('-')[0]}): {rng.choice(COMMIT_WHAT).format(s=rng.choice(VOCAB['service']))}" for _ in range(n)]
        user = f"Write release notes for v{rng.randint(1, 4)}.{rng.randint(0, 20)}.0 in Markdown from these commits, grouped by type with headings, plus a table of breaking changes (or a note that there are none):\n\n" + "\n".join(commits)
    elif v == "schedule":
        people = [person(rng) for _ in range(rng.randint(3, 8))]
        user = f"Make a {rng.choice(['two-week', 'four-week', 'one-week'])} on-call rota starting Monday 2026-{rng.randint(1, 12):02d}-0{rng.randint(1, 7)} as a Markdown table (date, primary, secondary) for: {', '.join(people)}. Nobody is primary two days in a row. Add a short table of per-person counts."
    else:
        tool = f"{dom}ctl"
        flags = rng.sample(["--config PATH", "--region NAME", "--dry-run", "--output json|yaml|table", "--limit N", "--since DURATION",
                            "--verbose", "--profile NAME", "--timeout SECONDS", "--filter EXPR", "--yes", "--no-color"], rng.randint(4, 9))
        user = f"Write the README reference section for the `{tool}` CLI ({label}): a Markdown table of flags (flag, argument, default, description) for {', '.join(flags)}, then three usage examples in a bash code block."
    row = {"messages": msgs(sys_prompt(rng, dom, None), user), "check": {"kind": "markdown", "variant": v}}
    return row, dom


# ---------------------------------------------------------------- short_code

CODE_LANGS = ["python", "javascript", "typescript", "go", "rust", "java", "c", "cpp", "csharp", "kotlin", "swift", "ruby", "php", "bash", "lua", "scala"]
CODE_W = [16, 9, 10, 9, 8, 6, 5, 6, 5, 4, 3, 4, 3, 5, 2, 2]
CODE_TASKS = [
    "an LRU cache with get and put in O(1)", "a debounce function", "binary search returning the insertion point",
    "slugify a title (ASCII, lowercase, hyphens)", "retry a function with exponential backoff and jitter",
    "merge overlapping intervals", "topological sort that reports cycles", "parse an ISO 8601 duration like PT1H30M into seconds",
    "convert camelCase to snake_case", "flatten a nested dictionary/map using dotted keys", "split a list into chunks of size n",
    "a ring buffer with fixed capacity", "a trie with insert and prefix search", "Dijkstra's shortest path on an adjacency list",
    "Levenshtein distance", "convert integers to Roman numerals and back", "parse a URL query string into a map of lists",
    "transpose a matrix", "count word frequencies and return the top k", "a moving average over a sliding window",
    "validate and normalize a phone number into E.164", "a thread-safe counter", "deep-merge two JSON-like objects",
    "group items by a key function", "a minimal INI file parser", "base64url encode/decode without padding",
    "generate a UUIDv4-like random identifier", "a priority queue (binary heap)", "read a CSV file and sum one column",
    "a token-bucket rate limiter", "pretty-print a byte size (KiB/MiB/GiB)", "check whether a string is a valid IPv4 address",
    "a simple event emitter with on/off/emit", "paginate a list (page, page_size) returning items and total pages",
    "compute a SHA-256 hex digest of a file", "rotate an array by k positions in place", "find all anagram groups in a word list",
    "parse simple command-line flags of the form --name=value", "a memoize helper with a max size", "run-length encode and decode a string",
]
CODE_STYLE = ["Code only, no explanation.", "Include a short usage example.", "Keep it under 40 lines.", "Add brief doc comments.",
              "Include a few unit tests.", "Explain the complexity in one sentence after the code.", ""]


def gen_short_code(rng, i):
    dom, ent = pick(rng)
    lang = rng.choices(CODE_LANGS, CODE_W)[0]
    if rng.random() < 0.3:
        s = codegen.Spec(dom, ent)
        fs = flat_fields(dom, ent)
        task = rng.choice([f"a function that parses one CSV line into a {s.T} record with fields " + ", ".join(f.name for f in fs),
                           f"a validator for {human(ent)} records ({describe_fields(dom, ent, [f.name for f in fs])}) returning a list of errors",
                           f"a function that groups {human(plural(ent))} by `{s.group.name}`" + (f" and sums `{s.measure.name}`" if s.measure else ""),
                           f"a type/struct for a {human(ent)} with JSON (de)serialization for fields " + ", ".join(f.name for f in fs)])
    else:
        task = rng.choice(CODE_TASKS)
        while lang == "go" and "rate limit" in task:   # RigMark's code prompt
            task = rng.choice(CODE_TASKS)
    lname = {"cpp": "C++", "csharp": "C#", "javascript": "JavaScript", "typescript": "TypeScript"}.get(lang, lang.capitalize())
    user = rng.choice([f"Write {task} in {lname}.", f"{lname}: {task}.", f"I need {task}. Use {lname}.",
                       f"Implement {task} in idiomatic {lname}."]) + " " + rng.choice(CODE_STYLE)
    row = {"messages": msgs(sys_prompt(rng, dom, "code"), user.strip()), "check": {"kind": "code", "lang": lang}}
    return row, dom


# ---------------------------------------------------------------- long profile (coverage-2)
# Longer outputs where real usage is long: nested schemas, multi-entity extraction,
# multi-round tool chains, multi-file/combined code edits, full configs, bigger programs.

def nested_item(rng, dom, ent, describe=False):
    s = entity_schema(dom, ent, describe)
    for a in rng.sample(["history", "related", "metadata", "metrics"], rng.randint(1, 3)):
        if a == "history":
            key, sch = "history", {"type": "array", "items": {"type": "object", "properties": {
                "at": {"type": "string"}, "event": {"type": "string", "enum": ["created", "updated", "reviewed", "escalated", "archived"]},
                "actor": {"type": "string"}, "changes": {"type": "array", "items": {"type": "object", "properties": {
                    "field": {"type": "string"}, "from": {"type": "string"}, "to": {"type": "string"}},
                    "required": ["field", "from", "to"], "additionalProperties": False}}},
                "required": ["at", "event", "actor", "changes"], "additionalProperties": False}}
        elif a == "related":
            o = rng.choice([e for e in top_entities(dom) if e != ent] or [ent])
            fs = [f.name for f in flat_fields(dom, o)]
            key, sch = "related_" + plural(o), {"type": "array", "items": entity_schema(dom, o, False, [fs[0]] + rng.sample(fs[1:], min(3, len(fs) - 1)))}
        elif a == "metadata":
            key, sch = "metadata", {"type": "object", "properties": {
                "source": {"type": "string", "enum": ["import", "api", "manual", "sync"]}, "version": {"type": "integer"},
                "labels": {"type": "array", "items": {"type": "string"}},
                "audit": {"type": "object", "properties": {"created_by": {"type": "string"}, "created_at": {"type": "string"},
                                                           "checksum": {"type": "string"}},
                          "required": ["created_by", "created_at", "checksum"], "additionalProperties": False}},
                "required": ["source", "version", "labels", "audit"], "additionalProperties": False}
        else:
            key, sch = "metrics", {"type": "object", "properties": {
                "score": {"type": "number"}, "percentile": {"type": "integer", "minimum": 0, "maximum": 100},
                "trend": {"type": "string", "enum": ["up", "flat", "down"]}, "weekly": {"type": "array", "items": {"type": "number"}}},
                "required": ["score", "percentile", "trend", "weekly"], "additionalProperties": False}
        if key not in s["properties"]:
            s["properties"][key] = sch
            s["required"].append(key)
    return s


def gen_json_nested(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    spec = codegen.Spec(dom, ent)
    g = spec.group.name
    item = nested_item(rng, dom, ent, rng.random() < 0.4)
    row = {}
    if rng.random() < 0.5:
        n = size(rng, 30)
        schema = wrap_root(plural(ent), item, n if rng.random() < 0.4 else None)
        intro = (f"Generate {n} detailed synthetic {human(plural(ent) if n > 1 else ent)} for {label}, each with its nested "
                 + ", ".join(k for k in item["properties"] if k not in {f.name for f in fields(dom, ent)}) + ". Keep values consistent across the nesting.")
        v = "nested_records"
    else:
        secs = rng.sample(["Overview", "Exceptions", "Top performers", "Needs review", "Recently changed", "At risk", "Archive candidates"],
                          rng.randint(2, 4))
        per = rng.randint(2, 7)
        schema = {"type": "object", "properties": {
            "report_id": {"type": "string"}, "generated_at": {"type": "string"}, "scope": {"type": "string"},
            "sections": {"type": "array", "items": {"type": "object", "properties": {
                "title": {"type": "string"}, "summary": {"type": "string"}, plural(ent): {"type": "array", "items": item}},
                "required": ["title", "summary", plural(ent)], "additionalProperties": False}},
            "totals": {"type": "object", "properties": {"records": {"type": "integer"}, f"by_{g}": {"type": "array", "items": {
                "type": "object", "properties": {g: {"type": "string"}, "count": {"type": "integer"}},
                "required": [g, "count"], "additionalProperties": False}}}, "required": ["records", f"by_{g}"], "additionalProperties": False},
            "warnings": {"type": "array", "items": {"type": "string"}}},
            "required": ["report_id", "generated_at", "scope", "sections", "totals", "warnings"], "additionalProperties": False}
        intro = (f"Build a synthetic {human(ent)} report for {label} with sections {', '.join(secs)}; put {per} {human(plural(ent))} in each "
                 f"section, compute totals by `{g}` over every listed record, and add any warnings.")
        v = "nested_report"
    row["_prose"] = "Use exactly this structure: " + json.dumps(schema)
    text = json_request(rng, row, schema, f"{v}_{ent}"[:60], intro)
    row["messages"] = msgs(sys_prompt(rng, dom, "json"), text)
    row["check"]["variant"] = v
    return row, dom


def gen_json_extract_multi(rng, i):
    dom = rng.choice([d for d in DOMAINS if len(top_entities(d)) >= 2])
    ents = top_entities(dom)[:2]
    parts, props, prose, total = [], {}, [], 0
    for ent in ents:
        fs = flat_fields(dom, ent)
        k = size(rng, 14)
        total += k
        recs = [record(rng, dom, ent) for _ in range(k)]
        doc, style = render_doc(rng, dom, ent, recs, set())
        parts.append(f"## {human(plural(ent)).capitalize()}\n\n{doc}")
        props[plural(ent)] = {"type": "array", "items": entity_schema(dom, ent, rng.random() < 0.3, [f.name for f in fs])}
        prose.append(f"`{plural(ent)}`: one object per {human(ent)} with " + describe_fields(dom, ent, [f.name for f in fs]))
    schema = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    row = {"_prose": "Return " + "; ".join(prose) + "."}
    intro = (f"This export from {DOMAINS[dom][0]} mixes {human(plural(ents[0]))} and {human(plural(ents[1]))}. "
             "Extract every record of both kinds.\n\n---\n" + "\n\n".join(parts) + "\n---")
    text = json_request(rng, row, schema, f"extract_{dom}", intro)
    row["messages"] = msgs(sys_prompt(rng, dom, "json"), text)
    row["check"]["variant"] = "multi_entity"
    row["check"]["expected_items"] = total
    return row, dom


def gen_tool_chain(rng, i):
    """2-4 rounds of calls and results (some parallel), then an answer or another round."""
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    dt = domain_tools(dom)
    spec = codegen.Spec(dom, ent)
    g, key = spec.group.name, spec.key
    gv = rng.choice(spec.group.arg.split("|")) if spec.group.kind == "enum" else None
    recs = [record(rng, dom, ent) for _ in range(rng.randint(3, 9))]
    if gv:
        for r in recs:
            r[g] = gv
    needed = {f"search_{plural(ent)}", f"get_{ent}"}
    want_table = rng.random() < 0.5
    q = (f"Find {human(plural(ent))}" + (f" with {human(g)} {gv}" if gv else "") + ", look at the details of the top "
         f"{min(3, len(recs))}, and " + rng.choice(["summarize what you find.", "give me a table of them.", "tell me which ones need attention.",
                                                    "draft a short status note."]))
    hist = [{"role": "user", "content": q}]
    call = tc(f"search_{plural(ent)}", ({g: gv} if gv else {"query": human(ent)}) | {"limit": 25}, rng)
    flat = [{f.name: r[f.name] for f in flat_fields(dom, ent)} for r in recs]
    hist += [{"role": "assistant", "content": "", "tool_calls": [call]},
             {"role": "tool", "tool_call_id": call["id"], "content": json.dumps({"results": flat, "total": len(recs)})}]
    top = rng.sample(recs, min(3, len(recs)))
    calls = [tc(f"get_{ent}", {key: r[key]}, rng) for r in top]
    hist.append({"role": "assistant", "content": rng.choice(["", f"Found {len(recs)}. Fetching details for {len(top)} of them.", ""]),
                 "tool_calls": calls})
    for c, r in zip(calls, top):
        hist.append({"role": "tool", "tool_call_id": c["id"], "content": json.dumps(r | {"notes": rng.choice(VOCAB["note"])})})
    if rng.random() < 0.5:   # a generic third round
        t = rng.choice(["run_sql", "convert_currency", "get_weather", "read_file"])
        needed.add(t)
        args, res = {
            "run_sql": ({"query": f"SELECT {g}, count(*) FROM {plural(ent)} GROUP BY {g}", "database": "analytics"},
                        {"rows": [{g: v, "count": rng.randint(1, 400)} for v in (spec.group.arg.split("|") if spec.group.kind == "enum" else ["a", "b"])]}),
            "convert_currency": ({"amount": rng.randint(100, 9000), "from_currency": "USD", "to_currency": "EUR"},
                                 {"converted": round(rng.uniform(90, 8500), 2), "rate": 0.92}),
            "get_weather": ({"location": rng.choice(CITIES)}, {"temp_c": rng.randint(-5, 33), "conditions": rng.choice(["rain", "clear", "fog", "wind"])}),
            "read_file": ({"path": f"docs/{dom}/policy.md"}, {"content": f"# {title(ent)} policy\n\n- Escalate {human(plural(ent))} older than 14 days.\n- Notes are required on every status change.\n"}),
        }[t]
        c3 = tc(t, args, rng)
        hist += [{"role": "assistant", "content": "", "tool_calls": [c3]},
                 {"role": "tool", "tool_call_id": c3["id"], "content": json.dumps(res)}]
    end = rng.choices(["answer", "update_all", "email", "write_report", "issues"], [0.4, 0.2, 0.15, 0.15, 0.1])[0]
    expect, user = "answer", None
    if end != "answer":
        summary = (f"I found {len(recs)} {human(plural(ent))}" + (f" with {human(g)} {gv}" if gv else "") + ". Details for "
                   + ", ".join(r[key] for r in top) + " are above.")
        hist.append({"role": "assistant", "content": summary})
        expect = "call"
        if end == "update_all":
            new = rng.choice(spec.group.arg.split("|")) if spec.group.kind == "enum" else "reviewed"
            user = f"Move every one of the {len(recs)} to {new}, with a note explaining why."
            needed.add(f"update_{ent}")
        elif end == "email":
            user = f"Email a full summary with a per-record breakdown to {person(rng).split()[0].lower()}@ops.example and cc {person(rng).split()[0].lower()}@ops.example."
            needed.add("send_email")
        elif end == "write_report":
            user = rng.choice([f"Save all of them as CSV at reports/{dom}-{plural(ent)}.csv.", f"Write a Markdown report to reports/{dom}-review.md with a table and next steps.",
                               f"Dump the detailed records as JSON to exports/{plural(ent)}.json."])
            needed.add("write_file")
        else:
            user = "Open an issue for each of the detailed ones, with the relevant fields in the body."
            needed.add("create_issue")
    tools = {k: dt.get(k) or GENERIC[k] for k in sorted(needed)}
    distract = [k for k in list(dt) + list(GENERIC) if k not in tools]
    for k in rng.sample(distract, min(len(distract), rng.randint(2, 10))):
        tools[k] = dt.get(k) or GENERIC[k]
    tl = list(tools.values())
    rng.shuffle(tl)
    system = rng.choice([None, f"You are an operations agent for {label}. Use tools for every lookup or change; never invent data.",
                         f"You assist staff at {label}. Call tools in parallel when calls are independent."])
    messages = ([{"role": "system", "content": system}] if system else []) + hist + ([{"role": "user", "content": user}] if user else [])
    return {"messages": messages, "tools": tl,
            "check": {"kind": "tool", "expect": expect, "variant": f"chain_{end}", "needed": sorted(needed)}}, dom


def gen_config_long(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    svc = rng.choice(VOCAB["service"])
    ents = top_entities(dom)
    v = rng.choice(["multi_file", "k8s_full", "openapi_full", "sql_suite", "big_fixture", "gha_full", "compose_full"])
    chk = {"kind": "config", "format": "multi" if v == "multi_file" else {"sql_suite": "sql"}.get(v, "yaml"), "variant": v}
    if v == "multi_file":
        files = rng.sample(["Dockerfile", "docker-compose.yml", ".env.example", "Makefile", ".github/workflows/ci.yml",
                            "k8s/deployment.yaml", "config/app.toml", "nginx.conf", "pyproject.toml", "Cargo.toml"], rng.randint(3, 5))
        user = (f"Set up the {svc} of {label} for deployment. Create these files: " + ", ".join(f"`{f}`" for f in files)
                + ". Put each file in its own fenced code block preceded by its path.")
    elif v == "k8s_full":
        user = (f"Write the complete Kubernetes manifests for `{svc}` in namespace `{dom}`: Namespace, ServiceAccount, ConfigMap, Secret (placeholder values), "
                f"Deployment ({rng.randint(3, 12)} replicas, probes, resources, anti-affinity, rolling update), Service, Ingress with TLS, "
                "HorizontalPodAutoscaler, PodDisruptionBudget and a NetworkPolicy. One multi-document YAML.")
    elif v == "openapi_full":
        user = (f"Write a full OpenAPI 3.0 spec (YAML) for {label} covering " + " and ".join(human(plural(e)) for e in ents)
                + ": list (filters, pagination), get, create, update, delete for each, shared error schema, security scheme and examples. Fields: "
                + "; ".join(f"{plural(e)}: " + describe_fields(dom, e, [f.name for f in flat_fields(dom, e)]) for e in ents) + ".")
    elif v == "sql_suite":
        dialect = rng.choice(["PostgreSQL", "SQLite"])
        chk.update(dialect=dialect, variant="ddl")
        k = rng.randint(4, 7)
        user = (f"For {label}, write {dialect} SQL: (1) DDL for tables " + "; ".join(f"{plural(e)} (" + ", ".join(f.name for f in flat_fields(dom, e)) + ")" for e in ents)
                + " with keys, CHECK constraints and indexes; (2) INSERT statements with 8 rows per table; (3) "
                + f"{k} reporting queries over them (group totals, top-N per group, running totals, anti-joins). One SQL code block, queries commented.")
    elif v == "big_fixture":
        n = size(rng, 60)
        fmt = rng.choice(["yaml", "csv"])
        chk["format"] = fmt
        if fmt == "csv":
            chk.update(variant="csv", sep=",", columns=len(flat_fields(dom, ent)), rows=n)
            user = f"Produce a CSV with header `{','.join(f.name for f in flat_fields(dom, ent))}` and {n} rows of varied synthetic {human(plural(ent))} for {label}."
        else:
            user = f"Write a YAML seed file with {n} varied synthetic {human(plural(ent))} for {label} under `{plural(ent)}` (fields: " + ", ".join(f.name for f in fields(dom, ent)) + ")."
    elif v == "gha_full":
        user = (f"Write a GitHub Actions workflow for the {dom} monorepo: path-filtered jobs for {', '.join(rng.sample(VOCAB['service'], 3))}, "
                "lint/test matrices, docker build with cache, SBOM upload, staging deploy on main, production deploy on tags with manual approval, and Slack notification on failure.")
    else:
        s = rng.sample(VOCAB["service"], rng.randint(4, 7))
        user = (f"Write docker-compose.yml for the full {label} stack: {', '.join(s)}, postgres, redis, nats, minio, prometheus, grafana and an otel-collector, "
                "with healthchecks, depends_on conditions, resource limits, named volumes, two networks and env files.")
    return {"messages": msgs(sys_prompt(rng, dom, "code"), user), "check": chk}, dom


def fmt_instructions(rng, fmt, paths):
    multi = len(paths) > 1
    return {
        "diff": "Respond with a unified diff" + (" covering every file you change (one ---/+++ header per file)." if multi else f" against `{paths[0]}`."),
        "search_replace": ("Answer with SEARCH/REPLACE blocks, one per change, each preceded by the file path:\n\npath/to/file\n<<<<<<< SEARCH\n(exact existing lines)\n"
                           "=======\n(new lines)\n>>>>>>> REPLACE\n\nThe SEARCH part must match the file exactly."),
        "file": "Return every changed file in full, each in its own code block preceded by its path." if multi else "Return the complete updated file.",
    }[fmt]


def gen_code_edit_long(rng, i):
    """Multi-file edits (2-3 files) or several combined edits to one file."""
    lang = rng.choices(codegen.LANGS, [0.3, 0.22, 0.16, 0.12, 0.1, 0.1])[0]
    dom = rng.choice(list(DOMAINS))
    specs = [codegen.Spec(dom, e) for e in top_entities(dom)]
    multi = rng.random() < 0.6
    if multi and len(specs) < 2:
        d2 = rng.choice([d for d in DOMAINS if d != dom])
        specs.append(codegen.Spec(d2, rng.choice(top_entities(d2))))
    specs = specs[:rng.randint(2, 3)] if multi else [rng.choice(specs)]
    bug_at = rng.randrange(len(specs)) if rng.random() < 0.4 else None
    files, instrs, edits = {}, [], []
    for j, s in enumerate(specs):
        bug = rng.choice(codegen.BUGS) if j == bug_at else None
        if bug == "range_exclusive" and not s.range or bug == "overwrite_total" and not s.measure:
            bug = "filter_inverted"
        p = codegen.path_for(lang, s)
        files[p] = codegen.source(lang, s, bug)
        if bug:
            t, k = edit_instruction(rng, s, codegen.names(lang, s), lang, bug)
            instrs.append(f"In `{p}`: {t}")
            edits.append(k)
    if multi:
        f, d = rng.choice(NEW_FIELDS)
        cross = rng.sample([f"Add a `{f}` field ({d}) to every model and validate it.",
                            "Add `offset`/`limit` pagination to every list function.",
                            "Make every store reject a duplicate key with a clear error instead of overwriting.",
                            "Add a function to each file that counts records per group field (the field its list function filters on).",
                            "Add a method to each store returning all records sorted by key.",
                            "Add structured logging (key, operation) to every add/remove, using the language's standard logger."], rng.randint(1, 2))
        instrs += cross
        edits.append("cross_file")
    else:
        s = specs[0]
        seen_k = set(edits)
        for _ in range(rng.randint(2, 3)):
            t, k = edit_instruction(rng, s, codegen.names(lang, s), lang, None)
            if k not in seen_k:
                seen_k.add(k)
                instrs.append(t)
                edits.append(k)
    fmt = rng.choices(["diff", "search_replace", "file"], [0.35, 0.35, 0.3])[0]
    body = "\n\n".join(f"`{p}`:\n\n```{lang}\n{src}\n```" for p, src in files.items())
    task = "\n".join(f"{n + 1}. {t}" for n, t in enumerate(instrs)) if len(instrs) > 1 else instrs[0]
    user = f"{body}\n\nPlease make these changes:\n{task}\n\n{fmt_instructions(rng, fmt, list(files))}"
    original = files if multi else next(iter(files.values()))
    return {"messages": msgs(sys_prompt(rng, dom, "code"), user),
            "check": {"kind": "edit", "format": fmt, "lang": lang, "original": original, "edit": "+".join(edits),
                      "bug": bug_at is not None, "variant": f"{'multi' if multi else 'combined'}_{fmt}"}}, dom


BIG_TASKS = [
    "a command-line todo app that stores tasks in a JSON file, with add, list, done and remove subcommands",
    "a small HTTP JSON API (standard library only) with in-memory CRUD for {e} records",
    "an arithmetic expression tokenizer, recursive-descent parser and evaluator supporting variables and parentheses",
    "a finite-state machine for {e} status transitions with allowed-transition validation and tests",
    "an LRU cache with per-entry TTL, plus unit tests",
    "a CSV-to-JSON converter command that infers column types",
    "a job queue with a worker pool, retries with backoff and graceful shutdown",
    "a validator for a subset of JSON Schema (type, properties, required, enum, items, minimum/maximum) with tests",
    "a line-based text diff using longest common subsequence, printing unified-diff style output",
    "a tiny template engine supporting {{{{var}}}} substitution, if blocks and for loops, with tests",
    "a double-entry ledger with accounts, transactions, balance checks and a trial-balance report, with tests",
    "a config loader that merges defaults, a TOML-like file and environment overrides, with validation errors",
    "a Sudoku solver using backtracking with constraint propagation",
    "a matrix type with add, multiply, transpose and determinant, plus tests",
    "an event-sourced store for {e} records (append events, rebuild state, snapshots)",
    "a URL-shortener core: base62 ids, collision handling and a storage interface with an in-memory implementation, plus tests",
    "a markdown table formatter that aligns columns and supports left/right/center alignment",
    "a bounded concurrent web-page fetcher with a concurrency limit, timeouts and a results summary",
    "an inventory module with reservations, releases and low-stock alerts, plus tests",
    "a priority-based task scheduler with dependencies (topological order) and cycle detection, plus tests",
]


def gen_short_code_big(rng, i):
    dom, ent = pick(rng)
    lang = rng.choices(CODE_LANGS, CODE_W)[0]
    task = rng.choice(BIG_TASKS).format(e=human(ent))
    lname = {"cpp": "C++", "csharp": "C#", "javascript": "JavaScript", "typescript": "TypeScript"}.get(lang, lang.capitalize())
    user = rng.choice([f"Write {task} in {lname}.", f"Implement {task} in idiomatic {lname}.", f"{lname}: build {task}."]) + " " + rng.choice(
        ["Organize it into functions/types with doc comments.", "Include tests.", "Code only.", "Keep it to a single file.",
         "Include a short usage example at the end.", ""])
    return {"messages": msgs(sys_prompt(rng, dom, "code"), user.strip()), "check": {"kind": "code", "lang": lang, "variant": "big"}}, dom


def gen_table_md_long(rng, i):
    dom, ent = pick(rng)
    label = DOMAINS[dom][0]
    v = rng.choice(["big_data", "multi_table", "adr", "runbook", "changelog_big"])
    if v == "big_data":
        k = size(rng, 50)
        rows = [record(rng, dom, ent) for _ in range(k)]
        cols = [f.name for f in flat_fields(dom, ent)]
        spec = codegen.Spec(dom, ent)
        user = (f"Here are {k} {human(plural(ent))} from {label}. Produce a Markdown document with: a full table of all rows, a summary table per `{spec.group.name}`"
                + (f" with count and total `{spec.measure.name}`" if spec.measure else "") + ", and a short list of anomalies.\n\n"
                + (csv_text(rows, cols) if rng.random() < 0.5 else dumps(rows, rng)))
    elif v == "multi_table":
        opts, ctx = rng.choice(COMPARE)
        user = (f"For {ctx}, write a Markdown evaluation of {opts}: a feature comparison table ({', '.join(rng.sample(CRITERIA, 6))}), a rough monthly cost table "
                "for small/medium/large deployments, a risk table (risk, likelihood, mitigation), and a migration checklist.")
    elif v == "adr":
        user = (f"Write an architecture decision record in Markdown for {label}: moving {rng.choice(VOCAB['service'])} to "
                f"{rng.choice(['an event-driven design', 'a managed database', 'a separate read model', 'a multi-region setup'])}. Sections: Context, Decision, "
                "Options considered (table), Consequences (table of positive/negative), Rollout plan (table with phases and owners), Open questions.")
    elif v == "runbook":
        svc = rng.choice(VOCAB["service"])
        user = (f"Write an on-call runbook in Markdown for `{svc}` at {label}: service overview, dashboards table, alerts table (alert, meaning, first action), "
                "step-by-step procedures for three failure modes with commands in bash blocks, and an escalation contacts table.")
    else:
        n = rng.randint(30, 60)
        commits = [f"{ident(rng, 7)} {rng.choice(COMMIT_KIND)}({rng.choice(VOCAB['service']).split('-')[0]}): "
                   f"{rng.choice(COMMIT_WHAT).format(s=rng.choice(VOCAB['service']))}" for _ in range(n)]
        user = ("Write release notes in Markdown from these commits: highlights, sections per type, a table of affected services with change counts, "
                "and an upgrade-notes table.\n\n" + "\n".join(commits))
    return {"messages": msgs(sys_prompt(rng, dom, None), user), "check": {"kind": "markdown", "variant": v}}, dom


LONG_ALT = {"json_schema": (gen_json_nested, 0.55), "json_extract": (gen_json_extract_multi, 0.4), "tool_call": (gen_tool_chain, 0.6),
            "config": (gen_config_long, 0.5), "code_edit": (gen_code_edit_long, 0.6), "short_code": (gen_short_code_big, 0.55),
            "table_md": (gen_table_md_long, 0.5)}


GENS = {"json_schema": gen_json_schema, "json_extract": gen_json_extract, "tool_call": gen_tool_call, "config": gen_config,
        "code_edit": gen_code_edit, "table_md": gen_table_md, "short_code": gen_short_code}
RIGMARK_EXCLUDE = ("\"square\"", "its square", "squares of", "ratelimit.go")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output directory (train.jsonl, manifest.json)")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--no-tokens", action="store_true", help="skip GLM rendering / token counts (no tokenizer needed)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prefix", default="cov1", help="row id prefix")
    ap.add_argument("--profile", choices=("base", "long"), default="base")
    ap.add_argument("--exclude", action="append", default=[], help="earlier train.jsonl whose prompts must not recur")
    a = ap.parse_args()
    global LONG
    LONG = a.profile == "long"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    msg_key = lambda m: hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest()[:16]  # noqa: E731
    seen_msgs = set()   # the same conversation with other tools/response_format still counts as a repeat
    for ex in a.exclude:
        for l in open(ex):
            r = json.loads(l)
            seen.add(r["prompt_sha"])
            seen_msgs.add(msg_key(r["messages"]))
    excluded = len(seen)
    stats = collections.defaultdict(collections.Counter)
    for cat, n in COUNTS.items():
        rng = random.Random(f"{a.seed}-{cat}")
        made = tries = 0
        while made < round(n * a.scale) and tries < n * 20:
            tries += 1
            alt, p_long = LONG_ALT[cat]
            row, dom = (alt if LONG and rng.random() < p_long else GENS[cat])(rng, made)
            blob = json.dumps({"m": row["messages"], "t": row.get("tools"), "r": row.get("response_format")}, sort_keys=True)
            if any(x in blob.lower() for x in RIGMARK_EXCLUDE):
                stats[cat]["benchmark_excluded"] += 1
                continue
            h = hashlib.sha256(blob.encode()).hexdigest()[:16]
            mk = msg_key(row["messages"])
            if h in seen or mk in seen_msgs:
                stats[cat]["duplicate"] += 1
                continue
            seen.add(h)
            seen_msgs.add(mk)
            row.pop("_prose", None)
            row.update(id=f"{a.prefix}-{cat}-{made:05d}", source="structured", category=cat, domain=dom, split="train",
                       reasoning_effort=EFFORT, temperature=1.0, license=LICENSE, prompt_sha=h)
            rows.append(row)
            made += 1
            stats[cat]["rows"] += 1
            stats[cat]["variant:" + str(row["check"].get("variant") or row["check"].get("format") or row["check"].get("lang"))] += 1
            if row.get("response_format"):
                stats[cat]["rf:" + row["response_format"]["type"]] += 1
    if not a.no_tokens:
        from .. import glm_render
        for r in rows:
            r["prompt_tokens"] = glm_render.count(glm_render.render(r["messages"], r.get("tools"), r["reasoning_effort"], True))
    path = out / "train.jsonl"
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    toks = [r.get("prompt_tokens", 0) for r in rows]
    manifest = {
        "schema": "enntity.drafter-distill.coverage1-prompts/v1", "seed": a.seed, "profile": a.profile, "scale": a.scale,
        "excluded_prior_prompts": excluded, "exclude_files": [Path(e).name for e in a.exclude], "license": LICENSE,
        "license_note": "Every prompt, record, document, tool definition and code file is generated by "
                        "enntity_drafter.prompts (coverage, domains, codegen) from Enntity-authored vocabularies and "
                        "templates; no third-party text.",
        "benchmark_exclusion": "No RigMark benchmark prompt is reproduced: index/square sequences and the Go token-bucket "
                               f"rate limiter are excluded; rows matching {list(RIGMARK_EXCLUDE)} are dropped.",
        "rows": len(rows), "jsonl": path.name, "jsonl_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "reasoning_effort": EFFORT, "by_category": {c: dict(v) for c, v in stats.items()},
        "by_domain": dict(collections.Counter(r["domain"] for r in rows)),
        "prompt_tokens": {"total": sum(toks), "mean": round(sum(toks) / max(1, len(toks))),
                          "max": max(toks) if toks else 0, "p50": sorted(toks)[len(toks) // 2] if toks else 0},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps({k: manifest[k] for k in ("rows", "prompt_tokens", "by_category")}, indent=1))


if __name__ == "__main__":
    main()
