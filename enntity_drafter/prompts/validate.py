"""Validators for structured-coverage generations (``enntity_drafter.prompts.coverage`` rows).

``validate(row, generation)`` returns ``(validity, detail, keep)``; the row's
``check`` field says what to verify.  Validity classes (first word "ok" = passes):

  json     ok | ok_fenced (parsed after removing a code fence) | schema_mismatch | invalid | truncated
  tool     ok | ok_text (expected answer/either, answered in text) | no_call (answered in text when a call was expected)
           | args_schema_mismatch | unknown_tool | args_invalid
  config   ok (parses: YAML/TOML/CSV/SQLite) | format_ok (unchecked format, code present) | parse_error | ragged | empty
  edit     ok (SEARCH matches / diff applies / code parses) | ok_fuzz / ok_lenient (diff applies with fuzz /
           with header line counts ignored) | format_ok | search_mismatch | no_apply | no_blocks | syntax_error
  code     ok (python AST / node --check) | format_ok | syntax_error | no_code
  markdown ok | ragged | no_table

Dropped (``keep`` False): API errors and junk, empty answers without tool calls, JSON rows that are invalid or
truncated (and schema mismatches under a strict response_format: the host's grammar failed), tool rows whose calls
don't parse or name unknown tools.

Needs PyYAML (``pip install .[validate]``); diff checks call ``patch`` and JavaScript checks ``node`` when present
(a missing binary yields ``check_error`` for that row, not a crash).
"""
from __future__ import annotations

import ast
import csv
import io
import json
import os
import re
import sqlite3
import subprocess
import tempfile
import textwrap
import tomllib
from pathlib import Path

import yaml

FENCE = re.compile(r"```([\w+#.-]*)[^\n]*\n(.*?)(?:```|\Z)", re.S)
SR = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)
DROP = {"json": {"invalid", "truncated"}, "tool": {"args_invalid", "unknown_tool", "empty"}}


# ---------------------------------------------------------------- helpers

def blocks(text):
    return [(m.group(1).lower(), m.group(2)) for m in FENCE.finditer(text or "")]


def main_block(text, tags=()):
    bs = blocks(text)
    for t, b in bs:
        if t in tags:
            return b
    return bs[0][1] if bs else (text or "").strip()


def schema_errors(x, s, path="$", out=None):
    """Subset validator for the schemas build_coverage_prompts.py emits."""
    out = [] if out is None else out
    if len(out) > 5:
        return out
    t = s.get("type")
    types = t if isinstance(t, list) else [t] if t else []
    ok = {"object": lambda v: isinstance(v, dict), "array": lambda v: isinstance(v, list),
          "string": lambda v: isinstance(v, str), "boolean": lambda v: isinstance(v, bool), "null": lambda v: v is None,
          "integer": lambda v: isinstance(v, int) and not isinstance(v, bool) or isinstance(v, float) and v.is_integer(),
          "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)}
    if types and not any(ok[tt](x) for tt in types):
        out.append(f"{path}: type {type(x).__name__} != {t}")
        return out
    if "enum" in s and x not in s["enum"]:
        out.append(f"{path}: {str(x)[:40]!r} not in enum")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        if "minimum" in s and x < s["minimum"] or "maximum" in s and x > s["maximum"]:
            out.append(f"{path}: {x} out of range")
    if isinstance(x, dict):
        props = s.get("properties", {})
        for r in s.get("required", []):
            if r not in x:
                out.append(f"{path}: missing {r}")
        for k, v in x.items():
            if k in props:
                schema_errors(v, props[k], f"{path}.{k}", out)
            elif s.get("additionalProperties") is False:
                out.append(f"{path}: extra {k}")
            elif isinstance(s.get("additionalProperties"), dict):
                schema_errors(v, s["additionalProperties"], f"{path}.{k}", out)
    if isinstance(x, list):
        if "minItems" in s and len(x) < s["minItems"] or "maxItems" in s and len(x) > s["maxItems"]:
            out.append(f"{path}: {len(x)} items")
        if "items" in s:
            for i, v in enumerate(x[:200]):
                schema_errors(v, s["items"], f"{path}[{i}]", out)
    return out


def parse_json(text):
    t = (text or "").strip()
    try:
        return json.loads(t), False
    except json.JSONDecodeError:
        pass
    b = main_block(t, ("json",))
    try:
        return json.loads(b), True
    except json.JSONDecodeError:
        return None, None


# ---------------------------------------------------------------- per kind

def check_json(row, g):
    if g["finish"] == "length":
        return "truncated", None
    x, fenced = parse_json(g["content"])
    if fenced is None:
        return "invalid", None
    errs = schema_errors(x, row["check"]["schema"])
    if errs:
        return "schema_mismatch", errs[:3]
    return ("ok_fenced" if fenced else "ok"), None


def check_tool(row, g):
    tools = {t["function"]["name"]: t["function"]["parameters"] for t in row["tools"]}
    calls = g.get("tool_calls") or []
    exp = row["check"]["expect"]
    if not calls:
        if not (g["content"] or "").strip():
            return "empty", None
        return ("no_call" if exp == "call" else "ok_text"), None
    detail = []
    cls = "ok"
    for c in calls:
        f = c.get("function") or {}
        if f.get("name") not in tools:
            return "unknown_tool", [f.get("name")]
        try:
            args = json.loads(f.get("arguments") or "{}")
        except json.JSONDecodeError:
            return "args_invalid", [f.get("name")]
        if not isinstance(args, dict):
            return "args_invalid", [f.get("name")]
        errs = schema_errors(args, tools[f["name"]])
        if errs:
            cls, detail = "args_schema_mismatch", errs[:3]
    return cls, {"calls": len(calls), "names": sorted({c["function"]["name"] for c in calls}), "errors": detail}


def check_config(row, g):
    c = row["check"]
    text = g["content"] or ""
    if not text.strip():
        return "empty", None
    fmt = c["format"]
    try:
        if fmt == "multi":   # several files, each in its own block: every YAML/TOML/JSON block must parse
            bs = blocks(text)
            for t, b in bs:
                if t in ("yaml", "yml"):
                    list(yaml.safe_load_all(b))
                elif t == "toml":
                    tomllib.loads(b)
                elif t == "json":
                    json.loads(b)
            return ("ok" if len(bs) >= 2 else "format_ok"), {"files": len(bs)}
        if fmt == "yaml":
            docs = list(yaml.safe_load_all(main_block(text, ("yaml", "yml"))))
            return ("ok" if any(d is not None for d in docs) else "parse_error"), None
        if fmt == "toml":
            tomllib.loads(main_block(text, ("toml",)))
            return "ok", None
        if fmt == "csv":
            b = main_block(text, ("csv", "tsv", "text", ""))
            rows = list(csv.reader(io.StringIO(b.strip()), delimiter=c["sep"]))
            if not rows:
                return "empty", None
            n = len(rows[0])   # nested JSON may add columns, so only consistency with the header is checked
            return ("ok" if n >= c["columns"] and all(len(r) == n for r in rows) else "ragged"), {"rows": len(rows) - 1}
        if fmt == "sql":
            sql = "\n".join(b for t, b in blocks(text) if t in ("sql", "")) or text
            if c["dialect"] != "SQLite":
                return ("format_ok" if re.search(r"(?i)\b(select|create|insert|alter)\b", sql) else "parse_error"), None
            db = sqlite3.connect(":memory:")
            if c["variant"] in ("inserts", "query"):
                db.executescript(c["ddl"])
            db.executescript(sql)
            return "ok", None
    except Exception as e:  # noqa: BLE001
        return "parse_error", [repr(e)[:120]]
    return ("format_ok" if blocks(text) or len(text) > 40 else "empty"), None


def _norm(t):
    return "\n".join(l.rstrip() for l in t.splitlines())


def apply_sr(files, text):
    """files: {path: source}. Each SEARCH must match (exactly or modulo trailing spaces) in one file."""
    out = dict(files)
    pairs = SR.findall(text)
    if not pairs:
        return None, "no_blocks"
    for s, r in pairs:
        for p, src in out.items():
            if s in src:
                out[p] = src.replace(s, r, 1)
                break
            if s.strip() and _norm(s) in _norm(src):
                out[p] = _norm(src).replace(_norm(s), r, 1)
                break
        else:
            return None, "search_mismatch"
    return out, "ok"


def _apply_one(orig, d):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "f"
        p.write_text(orig)
        d = d if d.endswith("\n") else d + "\n"
        for flags, cls in ((["-F", "0"], "ok"), ([], "ok_fuzz")):
            r = subprocess.run(["patch", "--dry-run", "-s", "-f", *flags, str(p)], input=d, capture_output=True, text=True, timeout=20)
            if r.returncode == 0:
                return cls
    # Lenient (header line counts ignored, as udiff appliers do): every hunk's old side occurs in the file.
    src = _norm(orig)
    hunks = re.split(r"(?m)^@@[^\n]*\n", d)[1:]   # also bare "@@" headers (aider-style udiff)
    old = lambda h: "\n".join(l[1:].rstrip() for l in h.splitlines()  # noqa: E731
                               if (l[:1] in (" ", "-") or l == "") and not l.startswith("```")).strip("\n")
    if hunks and all(old(h) in src for h in hunks):
        return "ok_lenient"
    return "no_apply"


RANK = ["ok", "ok_fuzz", "ok_lenient", "no_apply"]


def apply_diff(files, text):
    """files: {path: source}; a multi-file diff is split per ---/+++ section and each applied to its file."""
    d = main_block(text, ("diff", "patch"))
    if "@@" not in d:
        d = "\n".join(b for t, b in blocks(text) if "@@" in b) or text
        if "@@" not in d:
            return "no_blocks"
    if len(files) == 1:
        return _apply_one(next(iter(files.values())), d)
    worst = "ok"
    for sec in re.split(r"(?m)^(?=--- )", d):
        if "@@" not in sec:
            continue
        m = re.search(r"(?m)^\+\+\+ (?:b/)?(\S+)", sec)
        target = next((src for p, src in files.items() if m and (p.endswith(m.group(1)) or m.group(1).endswith(p))), None)
        cls = _apply_one(target, sec) if target is not None else min(
            (_apply_one(src, sec) for src in files.values()), key=RANK.index)
        worst = max(worst, cls, key=RANK.index)
    return worst


def py_ok(src):
    try:
        ast.parse(textwrap.dedent(src))
        return True
    except SyntaxError:
        return False


def check_edit(row, g):
    c = row["check"]
    text = g["content"] or ""
    if not text.strip():
        return "empty", None
    files = c["original"] if isinstance(c["original"], dict) else {c.get("path") or "file": c["original"]}
    if c["format"] == "search_replace":
        new, cls = apply_sr(files, text)
        if cls == "ok" and c["lang"] == "python" and not all(py_ok(v) for v in new.values()):
            cls = "syntax_error"
        return cls, None
    if c["format"] == "diff":
        return apply_diff(files, text), None
    if not blocks(text):
        return "no_blocks", None
    if c["lang"] == "python":
        pys = [b for t, b in blocks(text) if t in ("python", "py", "")] or [main_block(text)]
        return ("ok" if all(py_ok(b) for b in pys) else "syntax_error"), None
    return "format_ok", None


def check_code(row, g):
    lang = row["check"]["lang"]
    text = g["content"] or ""
    tags = {"python": ("python", "py"), "javascript": ("javascript", "js"), "bash": ("bash", "sh", "shell")}.get(lang, (lang,))
    if not blocks(text):
        return ("no_code" if len(text) < 40 else "format_ok"), None
    b = main_block(text, tags)
    if lang == "python":
        return ("ok" if py_ok(b) else "syntax_error"), None
    if lang == "javascript":
        with tempfile.NamedTemporaryFile("w", suffix=".mjs" if re.search(r"^\s*(import|export) ", b, re.M) else ".js",
                                         delete=False) as f:
            f.write(b)
        try:
            r = subprocess.run(["node", "--check", f.name], capture_output=True, timeout=20)
        finally:
            os.unlink(f.name)
        return ("ok" if r.returncode == 0 else "syntax_error"), None
    return "format_ok", None


def check_markdown(row, g):
    lines = (g["content"] or "").splitlines()
    tables, cur = [], []
    for l in lines + [""]:
        if l.strip().startswith("|"):
            cur.append(l.strip())
        elif cur:
            tables.append(cur)
            cur = []
    tables = [t for t in tables if len(t) >= 2 and re.match(r"\|?:?-{2,}", t[1].replace(" ", ""))]
    if not tables:
        return "no_table", None
    cells = lambda l: len(re.split(r"(?<!\\)\|", l.strip().strip("|")))  # noqa: E731
    ragged = sum(1 for t in tables if len({cells(l) for l in t}) > 1)
    return ("ok" if not ragged else "ragged"), {"tables": len(tables)}


CHECK = {"json": check_json, "tool": check_tool, "config": check_config, "edit": check_edit, "code": check_code,
         "markdown": check_markdown}


def junk(g):
    if g.get("error") or g.get("junk"):
        return g.get("junk") or "error"
    if not (g.get("content") or "").strip() and not g.get("tool_calls"):
        return "empty_answer"
    return None


def validate(row: dict, g: dict) -> tuple[str, object, bool]:
    """Validity class, detail and keep flag of generation ``g`` for prompt ``row``."""

    j = junk(g)
    if j:
        return "junk:" + j, None, False
    kind = row["check"]["kind"]
    try:
        cls, detail = CHECK[kind](row, g)
    except Exception as e:  # noqa: BLE001
        cls, detail = "check_error", [repr(e)[:160]]
    drop = DROP.get(kind, set()) | ({"schema_mismatch"} if row["check"].get("mode") == "strict" else set())
    return cls, detail, cls not in drop
