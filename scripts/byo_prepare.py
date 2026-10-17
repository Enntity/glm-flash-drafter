#!/usr/bin/env python3
"""Bring your own data: chats, prompt lists or GLM-5.3-Flash logs -> a checked capture index for ``drafter-join``.

Stages (rerun the same command to resume; finished work is never redone):

1. ``validate``  read ``--input`` files (chat JSONL, request/response logs, prompt lists), enforce the teacher rules,
   scrub (``--scrub report|redact|drop``), dedupe; writes ``prompts.jsonl`` and ``reused/generations.jsonl``.
2. ``generate``  GLM-5.3-Flash completions for prompts that have none (scripts/api_generate.py, your endpoint or key).
3. ``items``     render and tokenize with GLM's own chat template (scripts/prepare_items.py).
4. ``capture``   hidden states from the Atlas capture build (scripts/capture.py), FP8 alongside (enntity_drafter.fp8).
5. ``check``     token alignment per row and GLM argmax agreement on a sample (scripts/check_capture.py).
6. ``index``     the capture index ``drafter-join --mode byo`` reads (enntity_drafter.capture_io index).

Then it prints the exact ``drafter-join`` command.  ``--dry-run`` validates and prints the plan, the commands and a
disk estimate without writing anything; ``--until STAGE`` stops after a stage.  Guide: docs/byo-data.md.

    python scripts/byo_prepare.py --input chats.jsonl --work data/byo --teacher /models/GLM-5.3-Flash \\
        --teacher-url http://127.0.0.1:8000/v1/chat/completions --capture-dir /capture \\
        --before-capture './start-atlas-capture.sh' --after-capture './stop-atlas.sh'
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterator

REPO = Path(__file__).resolve().parents[1]
STAGES = ("validate", "generate", "items", "capture", "check", "index")
WINDOW = 2048                                    # rows kept before the generated span (docs/capture.md)
FP8_BYTES, BF16_BYTES = 24_600, 49_152           # stored bytes per token
GLM_MODEL = re.compile(r"glm[-_ .]?5\.3[-_ .]?flash", re.I)
FORBIDDEN_MODEL = re.compile(r"dflash|incoai", re.I)   # incoai/GLM-5.3-Flash-DFlash2 is CC BY-NC-ND
FORBIDDEN_PROVIDER = re.compile(r"^z\.?ai$", re.I)     # Z.AI's terms bar training competing models
ROLES = {"system", "user", "assistant", "tool"}
KEEP_FIELDS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")

# --------------------------------------------------------------------------
# scrub: a regex backstop for secrets and contact data (not a PII detector; see docs/byo-data.md)
# --------------------------------------------------------------------------

EXAMPLE_DOMAIN = re.compile(r"(^|\.)(example\.(com|org|net)|[^.]+\.(test|invalid|example|localhost)|localhost)$", re.I)
SAFE_IP = re.compile(r"^(127\.|0\.0\.0\.0$|192\.0\.2\.|198\.51\.100\.|203\.0\.113\.|255\.255\.255\.)")


def _luhn(text: str) -> bool:
    digits = [int(c) for c in text if c.isdigit()]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(reversed(digits)))
    return 13 <= len(digits) <= 19 and total % 10 == 0


#: (kind, pattern, group to replace (0 = whole match), keep(match) -> False when the hit is harmless)
PATTERNS = [
    ("private_key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"), 0, None),
    ("api_key", re.compile(r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_\w{30,}|"
                           r"xox[abprs]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}|"
                           r"glpat-[A-Za-z0-9_-]{20,}|(?:AKIA|ASIA)[0-9A-Z]{16})\b"), 0, None),
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"), 0, None),
    ("bearer", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/-]{20,}=*)"), 1, None),
    ("url_credentials", re.compile(r"\b[a-z][a-z0-9+.-]*://([^/\s:@]+:[^/\s@]+)@"), 1, None),
    ("secret_value", re.compile(r"(?i)\b(?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\b"
                                r"[\"']?\s*[:=]\s*[\"']([^\s\"']{8,})[\"']"), 1, None),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,})\b"), 0,
     lambda m: not EXAMPLE_DOMAIN.search(m.group(1))),
    ("ipv4", re.compile(r"(?<![\d.])(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)(?![\d.])"), 0,
     lambda m: not SAFE_IP.match(m.group(0))),
    ("phone", re.compile(r"(?<![\w+])(?:\+\d{1,3}[ .-]?\(?\d{1,4}\)?(?:[ .-]?\d{2,4}){2,4}|"
                         r"\(\d{3}\) ?\d{3}-\d{4})(?!\w)"), 0, None),
    ("card_number", re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"), 0, lambda m: _luhn(m.group(0))),
    ("us_ssn", re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"), 0, None),
]


def scrub_text(text: str, redact: bool) -> tuple[str, collections.Counter]:
    """Find (and with ``redact`` replace by ``[REDACTED:<kind>]``) secrets and contact data in ``text``."""

    hits: collections.Counter = collections.Counter()
    for kind, pattern, group, keep in PATTERNS:
        def replace(match: re.Match, kind=kind, group=group, keep=keep) -> str:
            if keep is not None and not keep(match):
                return match.group(0)
            hits[kind] += 1
            if not redact:
                return match.group(0)
            start, end = match.span(group)
            whole = match.group(0)
            return whole[: start - match.start()] + f"[REDACTED:{kind}]" + whole[end - match.start():]
        text = pattern.sub(replace, text)
    return text, hits


def scrub_messages(messages: list[dict], redact: bool) -> tuple[list[dict], collections.Counter]:
    total: collections.Counter = collections.Counter()
    out = []
    for message in messages:
        message = dict(message)
        for key in ("content", "reasoning_content"):
            if isinstance(message.get(key), str):
                message[key], hits = scrub_text(message[key], redact)
                total += hits
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = json.loads(json.dumps(call))
                fn = call.get("function", {})
                args = fn.get("arguments")
                text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                new, hits = scrub_text(text, redact)
                if hits and redact:
                    fn["arguments"] = new if isinstance(args, str) else json.loads(new)
                total += hits
                calls.append(call)
            message["tool_calls"] = calls
        out.append(message)
    return out, total


# --------------------------------------------------------------------------
# validate: input records -> prompt rows + reused completions
# --------------------------------------------------------------------------

class ByoError(RuntimeError):
    """An actionable error: what is wrong and how to fix it."""

    def __init__(self, message: str, fix: str = ""):
        super().__init__(message)
        self.fix = fix


def say(message: str) -> None:
    print(f"[byo] {message}", flush=True)


def read_inputs(paths: list[Path]) -> Iterator[tuple[str, Any]]:
    """(origin, record): ``.txt`` = one prompt per non-empty line; ``.json`` = a list; anything else = JSONL."""

    for path in paths:
        if not path.is_file():
            raise ByoError(f"{path}: no such file", "check --input")
        if path.suffix == ".txt":
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if line.strip():
                    yield f"{path.name}:{number}", {"prompt": line.strip()}
            continue
        if path.suffix == ".json":
            data = json.loads(path.read_text())
            for number, record in enumerate(data if isinstance(data, list) else [data], 1):
                yield f"{path.name}[{number}]", record
            continue
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                yield f"{path.name}:{number}", json.loads(line)
            except json.JSONDecodeError as error:
                yield f"{path.name}:{number}", ByoError(f"not JSON ({error.msg} at column {error.colno})")


def _text(content: Any) -> str:
    if content is None or isinstance(content, str):
        return content or ""
    if isinstance(content, list):   # OpenAI content parts
        if any(not isinstance(p, dict) or p.get("type") not in ("text", "input_text") for p in content):
            raise ByoError("only text content parts are supported")
        return "".join(p.get("text", "") for p in content)
    raise ByoError(f"content must be a string or a list of text parts, not {type(content).__name__}")


def _message(raw: Any) -> dict:
    if not isinstance(raw, dict) or raw.get("role") not in ROLES | {"developer"}:
        raise ByoError(f"message without a valid role ({sorted(ROLES)}): {str(raw)[:80]}")
    message = {"role": "system" if raw["role"] == "developer" else raw["role"], "content": _text(raw.get("content"))}
    reasoning = raw.get("reasoning_content") or raw.get("reasoning")
    if isinstance(reasoning, str) and reasoning:
        message["reasoning_content"] = reasoning
    if message["role"] == "assistant" and not message.get("reasoning_content") and "</think>" in message["content"]:
        head, _, tail = message["content"].partition("</think>")   # a server without a reasoning parser
        message["reasoning_content"], message["content"] = head.replace("<think>", "", 1).strip("\n"), tail.lstrip("\n")
    if raw.get("tool_calls"):
        calls = raw["tool_calls"]
        if not isinstance(calls, list) or any(not isinstance(c, dict) or "name" not in (c.get("function") or {})
                                              for c in calls):
            raise ByoError("tool_calls must be OpenAI-style [{id, type, function: {name, arguments}}]")
        message["tool_calls"] = calls
    for key in ("tool_call_id", "name"):
        if raw.get(key):
            message[key] = raw[key]
    if raw.get("finish_reason"):
        message["finish_reason"] = raw["finish_reason"]
    return message


def _record(obj: Any) -> tuple[list[dict], dict, str | None, str | None]:
    """(messages, request fields, teacher model, provider) of one input record."""

    if isinstance(obj, str):
        obj = {"prompt": obj}
    if not isinstance(obj, dict):
        raise ByoError("expected a JSON object")
    if isinstance(obj.get("request"), dict):            # request/response log of an OpenAI-compatible proxy
        meta, response = dict(obj["request"]), obj.get("response") or {}
        meta.update({k: obj[k] for k in ("id", "category", "source") if k in obj})
        messages = list(meta.get("messages") or [])
        choice = (response.get("choices") or [{}])[0]
        if choice.get("message"):
            messages.append({**choice["message"], "role": "assistant", "finish_reason": choice.get("finish_reason")})
        model, provider = response.get("model") or meta.get("model"), response.get("provider")
    elif "messages" in obj:
        meta, messages = obj, list(obj["messages"] or [])
        model, provider = obj.get("teacher") or obj.get("model"), obj.get("provider")
    elif "prompt" in obj:
        meta, model, provider = obj, None, None
        messages = ([{"role": "system", "content": obj["system"]}] if obj.get("system") else []) + [
            {"role": "user", "content": obj["prompt"]}]
    else:
        raise ByoError("unrecognized record: expected 'messages', 'request'/'response' or 'prompt'")
    if not messages:
        raise ByoError("no messages")
    return [_message(m) for m in messages], meta, model, provider


def teacher_verdict(model: str | None, provider: str | None, opts: argparse.Namespace) -> str | None:
    """None when assistant text in this record may be used as GLM-5.3-Flash teacher text, else the reason not."""

    if provider and FORBIDDEN_PROVIDER.match(str(provider).strip()):
        return f"served by {provider}, whose terms bar training competing models"
    if model:
        if FORBIDDEN_MODEL.search(str(model)):
            return f"model {model!r} is a non-commercial (CC BY-NC-ND) checkpoint"
        if not GLM_MODEL.search(str(model)):
            return f"model {model!r} is not GLM-5.3-Flash"
        return None
    return None if opts.completions_are_glm else "unattested"


def _slug(text: Any, default: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text or "")).strip("-.")[:40]
    return slug or default


def _category(row: dict) -> str:
    if row.get("tools") or any(m["role"] == "tool" or m.get("tool_calls") for m in row["messages"]):
        return "agentic"
    return "structured" if row.get("response_format") else "chat"


def make_row(messages: list[dict], meta: dict, opts: argparse.Namespace, origin: str) -> dict:
    effort = meta.get("reasoning_effort")
    if effort is None and isinstance(meta.get("reasoning"), dict):
        effort = meta["reasoning"].get("effort")
    effort = effort or opts.effort
    if effort is not None and not re.fullmatch(r"[a-z]{1,16}", str(effort)):
        raise ByoError(f"reasoning_effort {effort!r} is not a template effort name (for example high)")
    row = {"messages": [{k: m[k] for k in KEEP_FIELDS if k in m} for m in messages]}
    for key in ("tools", "tool_choice", "response_format"):
        if meta.get(key):
            row[key] = meta[key]
    if effort:
        row["reasoning_effort"] = effort
    sha = hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    row.update(id=f"{_slug(meta.get('id'), 'byo')}-{sha[:12]}", prompt_sha=sha, source=_slug(meta.get("source"), "byo"),
               category=_slug(meta.get("category"), "") or _category(row), split="train",
               temperature=meta["temperature"] if isinstance(meta.get("temperature"), (int, float)) else 1.0,
               origin=origin)
    return row


def validate(opts: argparse.Namespace) -> dict:
    """All inputs -> {rows, reused, report}; raises ByoError listing the first problems (unless --skip-bad)."""

    rows: dict[str, dict] = {}
    reused: dict[str, list[dict]] = collections.defaultdict(list)
    seen_completions: set[str] = set()
    problems: list[str] = []
    scrub_hits: collections.Counter = collections.Counter()
    report = collections.Counter()
    for origin, obj in read_inputs(opts.input):
        report["records"] += 1
        try:
            if isinstance(obj, ByoError):
                raise obj
            messages, meta, model, provider = _record(obj)
            if opts.scrub != "off":
                messages, hits = scrub_messages(messages, opts.scrub == "redact")
                if hits:
                    scrub_hits += hits
                    report["records_with_scrub_hits"] += 1
                    if opts.scrub == "drop":
                        report["dropped_by_scrub"] += 1
                        continue
            assistant = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
            verdict = teacher_verdict(model, provider, opts) if assistant else None
            if verdict is not None:
                if not opts.regenerate:
                    fix = ("pass --completions-are-glm if this assistant text is GLM-5.3-Flash output, or --regenerate"
                           if verdict == "unattested" else "use --regenerate to replace it with GLM-5.3-Flash output")
                    raise ByoError(f"assistant text {verdict}; {fix}")
                messages, assistant = messages[: assistant[0]], []   # nothing from that model is kept, even context
                report["regenerated_records"] += 1
            if opts.turns == "all":
                targets = assistant
            else:
                targets = assistant[-1:] if messages and messages[-1]["role"] == "assistant" else []
            new_rows, new_replies = {}, []   # a record is taken whole or not at all
            for t in targets or [None]:
                prompt = messages if t is None else messages[:t]
                if not prompt or prompt[-1]["role"] not in ("user", "tool"):
                    raise ByoError("a prompt must end with a user or tool message")
                row = make_row(prompt, meta, opts, origin)
                new_rows.setdefault(row["id"], row)
                if t is None:
                    continue
                reply = messages[t]
                if not reply.get("reasoning_content") and not opts.allow_missing_reasoning:
                    raise ByoError("assistant turn without its thinking trace (GLM-5.3-Flash always thinks)",
                                   "log reasoning_content / reasoning, or --regenerate, or --allow-missing-reasoning")
                key = hashlib.sha256(json.dumps([row["id"], reply.get("reasoning_content"), reply["content"],
                                                 reply.get("tool_calls")], sort_keys=True).encode()).hexdigest()
                finish = reply.get("finish_reason") or ("tool_calls" if reply.get("tool_calls") else "stop")
                new_replies.append((key, row["id"], {
                    "reasoning": reply.get("reasoning_content", ""), "content": reply["content"],
                    "tool_calls": reply.get("tool_calls"), "finish": finish,
                    "teacher": str(model or "attested:glm-5.3-flash"), "origin": origin}))
            added = [rid for rid in new_rows if rid not in rows]
            for rid in added:
                rows[rid] = new_rows[rid]
            for key, rid, reply in new_replies:
                if key not in seen_completions:
                    seen_completions.add(key)
                    reused[rid].append(reply)
                    added.append(key)
            report["duplicate_records"] += not added
        except ByoError as error:
            report["bad_records"] += 1
            problems.append(f"{origin}: {error}" + (f" (fix: {error.fix})" if error.fix else ""))
    if problems and not opts.skip_bad:
        listed = "\n  ".join(problems[:10]) + ("\n  ..." if len(problems) > 10 else "")
        raise ByoError(f"{len(problems)} input record(s) failed validation:\n  {listed}",
                       "fix the records, or pass --skip-bad to leave them out")
    completions, seq = [], 0
    for row_id, replies in reused.items():
        row = rows[row_id]
        for k, reply in enumerate(replies):
            completions.append({"id": f"{row_id}#s{k}", "prompt_id": row_id, "sample": k, "seq": seq,
                                "source": row["source"], "category": row["category"], **reply})
            seq += 1
    report.update(prompts=len(rows), reused_completions=len(completions),
                  need_generation=sum(1 for r in rows if r not in reused))
    return {"rows": list(rows.values()), "reused": completions, "report": dict(report),
            "scrub": dict(scrub_hits), "problems": problems}


# --------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------

def run(cmd: list[str], log: Path, echo: bool = False) -> None:
    """Run a repo script; output to ``log`` (and progress lines to the console with ``echo``, up to a JSON
    summary); a failure names the log."""

    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"$ {shlex.join(cmd)}\n")
        stream.flush()
        proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        tail: collections.deque = collections.deque(maxlen=15)
        with proc.stdout:
            for line in proc.stdout:
                stream.write(line)
                tail.append(line.rstrip())
                echo = echo and not line.startswith("{")
                if echo:
                    print("    " + line.rstrip(), flush=True)
        if proc.wait():
            label = Path(cmd[0]).name if cmd[0] != sys.executable else cmd[2] if cmd[1] == "-m" else Path(cmd[1]).name
            raise ByoError(f"{label} failed (exit {proc.returncode}):\n    " + "\n    ".join(tail),
                           f"read {log}, fix, and rerun the same command to resume")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.is_file() else []


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    tmp.replace(path)


def good_generations(path: Path) -> dict[str, int]:
    """Good samples per prompt id in an api_generate output (last attempt of a sample wins)."""

    last = {r["id"]: r for r in jsonl(path)}
    good: collections.Counter = collections.Counter()
    for record in last.values():
        if not record.get("error") and not record.get("junk"):
            good[record["prompt_id"]] += 1
    return dict(good)


class Pipeline:
    def __init__(self, opts: argparse.Namespace):
        self.o, self.w, self.py = opts, opts.work.resolve(), sys.executable
        self.log = self.w / "logs" / "byo.log"

    # ------------------------------------------------------------------ paths
    def p(self, *parts: str) -> Path:
        return self.w.joinpath(*parts)

    # ------------------------------------------------------------------ commands
    def gen_cmd(self, prompts: Path, sources: list[str]) -> list[str]:
        o = self.o
        cmd = [self.py, "scripts/api_generate.py", "--prompts", str(prompts), "--out", str(self.p("gen")),
               "--sources", ",".join(sources) or "byo", "--samples", str(o.samples), "--minutes", str(o.gen_minutes),
               "--spend-cap", str(o.spend_cap), "--max-tokens", str(o.max_tokens)]
        if o.teacher_url:
            cmd += ["--url", o.teacher_url, "--model", o.teacher_model]
        return cmd + ["--key-env", o.teacher_key_env, *o.gen_arg]

    def items_cmd(self, dry: bool = False) -> list[str]:
        gens = [p for p in (self.p("gen", "generations.jsonl"), self.p("reused", "generations.jsonl"))
                if dry or p.is_file()]
        return [self.py, "scripts/prepare_items.py", "--prompts", str(self.p("prompts.jsonl")),
                *[a for g in gens for a in ("--generations", str(g))], "--tokenizer-dir", str(self.tokenizer_dir()),
                "--out", str(self.p("items.jsonl"))]

    def capture_cmd(self) -> list[str]:
        o = self.o
        return [self.py, "scripts/capture.py", "--server", o.atlas_server, "--model", o.atlas_model, "--items",
                str(self.p("items-todo.jsonl")), *[a for d in o.capture_dir for a in ("--capture-dir", str(d))],
                "--dest", str(self.p("rows")), "--out", str(self.p("capture-logs")), "--minutes",
                str(o.capture_minutes), *o.capture_arg]

    def fp8_cmd(self) -> list[str]:
        return [self.py, "-m", "enntity_drafter.fp8", "watch", "--root", str(self.p("rows")), "--log",
                str(self.p("fp8-log.jsonl")), "--stop-file", str(self.p("fp8.stop"))]

    def check_cmd(self) -> list[str]:
        cmd = [self.py, "scripts/check_capture.py", "--rows", str(self.p("check-sample")), "--records",
               str(self.p("capture-logs", "capture-records.jsonl")), "--out", str(self.p("check.json"))]
        return cmd if self.o.skip_agreement else cmd + ["--model", str(self.need_teacher())]

    def index_cmd(self) -> list[str]:
        return [self.py, "-m", "enntity_drafter.capture_io", "index", "--data-root", str(self.w), "--fast", "rows",
                "--meta", str(self.p("capture-logs", "capture-records.jsonl")), "--out", str(self.p("index"))]

    def join_cmds(self) -> list[str]:
        flags = ["--teacher", str(self.need_teacher().resolve()), "--mode", "byo", "--byo-capture-index",
                 str(self.p("index", "index.json")), "--byo-capture-root", str(self.w)]
        flags += ["--tag", self.o.tag] if self.o.tag else []
        return [shlex.join(["drafter-join", verb, *flags]) for verb in ("setup", "run")]

    def need_teacher(self) -> Path:
        if self.o.teacher is None:
            raise ByoError("this stage needs --teacher (your GLM-5.3-Flash checkpoint directory)")
        return self.o.teacher

    def tokenizer_dir(self) -> Path:
        directory = self.o.tokenizer_dir or self.need_teacher()
        missing = [n for n in ("tokenizer.json", "chat_template.jinja") if not (directory / n).is_file()]
        if missing:
            raise ByoError(f"{directory} lacks {', '.join(missing)}", "point --tokenizer-dir (or --teacher) at your "
                                                                      "GLM-5.3-Flash checkpoint directory")
        return directory

    # ------------------------------------------------------------------ stages
    def validate(self, write: bool) -> dict:
        result = validate(self.o)
        r = collections.Counter(result["report"])
        notes = [f"{r[k]} {label}" for k, label in (("duplicate_records", "duplicate records dropped"),
                                                     ("bad_records", "bad records skipped"),
                                                     ("regenerated_records", "records cut for --regenerate")) if r[k]]
        say(f"validate: {r['records']} records -> {r['prompts']} prompts, {r['reused_completions']} reused "
            f"GLM-5.3-Flash completions, {r['need_generation']} prompts without one"
            + "".join(f"; {n}" for n in notes))
        if result["scrub"]:
            hits = ", ".join(f"{k} {v}" for k, v in sorted(result["scrub"].items()))
            action = {"report": "not changed (review, then rerun with --scrub redact or --scrub drop)",
                      "redact": "redacted", "drop": f"dropped {r['dropped_by_scrub']} records"}
            say(f"scrub: {hits} in {r['records_with_scrub_hits']} records, {action[self.o.scrub]}")
        elif self.o.scrub != "off":
            say("scrub: no secrets or contact data found by the regex backstop")
        if not result["rows"]:
            raise ByoError("no usable prompts in the input")
        if write:
            write_jsonl(self.p("prompts.jsonl"), result["rows"])
            write_jsonl(self.p("reused", "generations.jsonl"), result["reused"])
            report = {k: result[k] for k in ("report", "scrub", "problems")}
            self.p("validate-report.json").write_text(json.dumps(report, indent=1))
        return result

    def generate(self, rows: list[dict], reused: list[dict], dry: bool) -> None:
        have = set(good_generations(self.p("gen", "generations.jsonl"))) | {c["prompt_id"] for c in reused}
        todo = [r for r in rows if r["id"] not in have]
        if not todo:
            say("generate: every prompt has a completion; nothing to generate")
            return
        o = self.o
        if not o.teacher_url and not os.environ.get(o.teacher_key_env):
            raise ByoError(f"{len(todo)} prompts need GLM-5.3-Flash completions but no endpoint is configured",
                           f"pass --teacher-url (your OpenAI-compatible GLM-5.3-Flash server) or set "
                           f"{o.teacher_key_env} for OpenRouter")
        cmd = self.gen_cmd(self.p("gen-prompts.jsonl"), sorted({r["source"] for r in todo}))
        say(f"generate: {len(todo)} prompts x{o.samples} -> {o.teacher_url or 'OpenRouter (Z.AI excluded)'}"
            f" (spend cap ${o.spend_cap:g} per run)")
        if dry:
            print("    " + shlex.join(cmd))
            return
        write_jsonl(self.p("gen-prompts.jsonl"), todo)
        run(cmd, self.log, echo=True)
        good = good_generations(self.p("gen", "generations.jsonl"))
        done = sum(1 for r in todo if r["id"] in good)
        records = [r for r in {g["id"]: g for g in jsonl(self.p("gen", "generations.jsonl"))}.values()
                   if not r.get("error") and not r.get("junk")]
        thinking = sum(1 for r in records if (r.get("reasoning") or "").strip())
        say(f"generate: {done}/{len(todo)} prompts now have completions" +
            (f"; {len(todo) - done} failed or did not fit the window (see {self.p('gen', 'gen-summary.json')}); "
             "rerun to retry them" if done < len(todo) else ""))
        if records and thinking < len(records) / 2:
            raise ByoError(f"only {thinking} of {len(records)} generated completions carry a thinking trace",
                           "GLM-5.3-Flash always thinks: start your server with its GLM reasoning parser so replies "
                           "return reasoning_content (or reasoning), delete the work dir's gen/ and rerun")

    def items(self, dry: bool) -> list[dict]:
        cmd = self.items_cmd(dry)
        if dry:
            say("items: render + tokenize with the GLM chat template")
            print("    " + shlex.join(cmd))
            return []
        run(cmd, self.log)
        items = jsonl(self.p("items.jsonl"))
        validity = collections.Counter(v["validity"] for v in jsonl(self.p("items-validity.jsonl")) if not v["keep"])
        tokens = sum(i["prompt_tokens"] + i["completion_tokens"] for i in items)
        say(f"items: {len(items)} sequences, {tokens:,} tokens (locally rendered)" +
            (f"; dropped {dict(validity)} (see {self.p('items-validity.jsonl')})" if validity else ""))
        if not items:
            raise ByoError("no capture items were produced", f"read {self.p('items-validity.jsonl')}")
        return items

    def captured(self) -> set[str]:
        records = jsonl(self.p("capture-logs", "capture-records.jsonl"))
        return {r["name"] for r in records if "error" not in r
                and (self.p("rows", r["name"]).is_dir() or self.p("rejected", r["name"]).is_dir())}

    def capture(self, items: list[dict], dry: bool) -> None:
        o = self.o
        if dry:
            say(f"capture: Atlas capture build at {o.atlas_server} (docs/capture.md), FP8 conversion alongside")
            for cmd in (o.before_capture, self.capture_cmd(), self.fp8_cmd(), o.after_capture):
                if cmd:
                    print("    " + (cmd if isinstance(cmd, str) else shlex.join(cmd)))
            return
        self.absorb_moves()
        done = self.captured()
        todo = [i for i in items if i["name"] not in done]
        if not todo:
            say(f"capture: all {len(items)} sequences already captured")
            return
        if not o.capture_dir:
            raise ByoError("capture needs --capture-dir (each rank's ATLAS_HIDDEN_CAPTURE_DIR as seen from here)")
        for directory in o.capture_dir:
            if not directory.is_dir():
                raise ByoError(f"capture dir {directory} does not exist", "create it and start Atlas with "
                                                                          "ATLAS_HIDDEN_CAPTURE_DIR pointing at it")
        write_jsonl(self.p("items-todo.jsonl"), todo)
        say(f"capture: {len(todo)} of {len(items)} sequences to capture")
        stop = self.p("fp8.stop")
        stop.unlink(missing_ok=True)
        self.p("rows").mkdir(parents=True, exist_ok=True)
        watcher = None
        try:
            if o.before_capture:
                run(shlex.split(o.before_capture), self.log)
            self.wait_server()
            watcher = subprocess.Popen(self.fp8_cmd(), cwd=REPO, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            run(self.capture_cmd(), self.log, echo=True)
        finally:
            try:
                if o.after_capture:
                    run(shlex.split(o.after_capture), self.log)
            finally:
                stop.touch()
                if watcher is not None:
                    watcher.wait()
                self.absorb_moves()

    def absorb_moves(self) -> None:
        """Append the collector's per-row results (capture.py rewrites move-log.json every run) to alignment.jsonl."""

        moves = self.p("capture-logs", "move-log.json")
        if moves.is_file():
            with self.p("capture-logs", "alignment.jsonl").open("a") as stream:
                for name, move in json.loads(moves.read_text()).items():
                    stream.write(json.dumps({"name": name, **move}) + "\n")
            moves.unlink()

    def wait_server(self) -> None:
        deadline = time.time() + (self.o.server_wait if self.o.before_capture else min(self.o.server_wait, 15))
        url = self.o.atlas_server.rstrip("/") + "/v1/models"
        while True:
            try:
                with urllib.request.urlopen(url, timeout=10):
                    return
            except OSError as error:
                if time.time() >= deadline:
                    raise ByoError(f"Atlas server not reachable at {url} ({error})",
                                   "start the capture build (docs/capture.md) or pass --before-capture") from error
                time.sleep(5)

    def check(self, dry: bool) -> dict:
        if dry:
            say("check: token alignment (every row) + GLM argmax agreement (sampled rows)")
            print("    " + shlex.join(self.check_cmd()))
            return {}
        self.absorb_moves()
        self.p("fp8.stop").touch()
        run(self.fp8_cmd(), self.log)   # convert rows an interrupted capture left in BF16
        moves = {m["name"]: m for m in jsonl(self.p("capture-logs", "alignment.jsonl"))}   # last wins
        names = {p.name for d in ("rows", "rejected") if self.p(d).is_dir() for p in self.p(d).iterdir() if p.is_dir()}
        rejected: collections.Counter = collections.Counter()
        for name in sorted(n for n in names if not n.startswith("gate")):
            move = moves.get(name, {"ok": False, "err": "unverified"})
            if move.get("ok") and not move.get("ids_mismatch"):
                continue
            reason = "token_ids_differ" if move.get("ids_mismatch") else str(move.get("err"))
            reason = reason if reason in ("coverage", "unverified", "token_ids_differ") else "writer_error"
            if self.p("rows", name).is_dir():
                self.p("rejected").mkdir(exist_ok=True)
                shutil.move(str(self.p("rows", name)), self.p("rejected", name))
            rejected[reason] += 1
        dirs = sorted(p.name for p in self.p("rows").iterdir() if p.is_dir()) if self.p("rows").is_dir() else []
        rows, gates = [n for n in dirs if not n.startswith("gate")], [n for n in dirs if n.startswith("gate")]
        if not rows:
            raise ByoError("no captured rows to check", "run the capture stage (see the capture log)")
        sample = self.p("check-sample")
        shutil.rmtree(sample, ignore_errors=True)
        sample.mkdir()
        for name in sorted(random.Random(0).sample(rows, min(len(rows), self.o.check_rows))) + gates:
            (sample / name).symlink_to(self.p("rows", name), target_is_directory=True)
        run(self.check_cmd(), self.log)
        verdict = judge(json.loads(self.p("check.json").read_text()), rejected, len(rows), self.o.min_agreement)
        self.p("check-verdict.json").write_text(json.dumps(verdict, indent=1))
        for line in verdict["lines"]:
            say("check: " + line)
        if verdict["status"] == "fail":
            raise ByoError("the data failed the capture check; nothing to train on yet", verdict["fix"])
        return verdict

    def index(self, items: list[dict], dry: bool) -> dict:
        if dry:
            say("index: the capture index drafter-join reads")
            print("    " + shlex.join(self.index_cmd()))
            return {}
        names = {i["name"] for i in items}
        stale = [p for p in self.p("rows").iterdir() if p.is_dir() and p.name not in names
                 and not p.name.startswith("gate")]
        for path in stale:   # rows of prompts no longer in the input
            self.p("stale").mkdir(exist_ok=True)
            shutil.move(str(path), self.p("stale", path.name))
        if stale:
            say(f"index: moved {len(stale)} rows whose prompts are no longer in the input to {self.p('stale')}")
        run(self.index_cmd(), self.log)
        index = json.loads(self.p("index", "index.json").read_text())
        train = index["summary"].get("train", {})
        if not train.get("rows"):
            raise ByoError(f"the index has no trainable rows (skipped: {index['skipped']})",
                           f"read {self.p('index', 'index.json')} skipped_rows")
        say(f"index: {int(train['rows'])} rows, {int(train['stored_tokens']):,} stored tokens, "
            f"{int(train['eligible_anchors']):,} eligible anchors, {train['gb']:.2f} GB"
            + (f"; skipped {index['skipped']}" if index["skipped"] else ""))
        return index

    def estimate(self, rows: list[dict], reused: list[dict]) -> None:
        """Dry run: stored tokens and disk, from a local render when the tokenizer is available."""

        count = None
        try:
            sys.path.insert(0, str(REPO))
            from enntity_drafter import glm_render

            glm_render.set_tokenizer_dir(self.tokenizer_dir())
            count = glm_render.count
        except (ByoError, ImportError, OSError):
            pass

        def tokens(text: str) -> int:
            return count(text) if count else len(text) // 4

        prompt = {r["id"]: tokens(json.dumps(r["messages"], ensure_ascii=False)) for r in rows}
        stored = sum(min(prompt[c["prompt_id"]], WINDOW) + tokens((c["reasoning"] or "") + (c["content"] or ""))
                     for c in reused)
        new = [r for r in rows if r["id"] not in {c["prompt_id"] for c in reused}]
        stored += sum(min(prompt[r["id"]], WINDOW) + self.o.assume_completion_tokens for r in new) * self.o.samples
        sequences = len(reused) + len(new) * self.o.samples
        say(f"estimate ({'GLM tokenizer' if count else 'about 4 characters per token'}; "
            f"{self.o.assume_completion_tokens:,} tokens per completion still to generate): {sequences} sequences, "
            f"{stored:,} stored tokens -> {stored * FP8_BYTES / 1e9:.2f} GB FP8 on disk "
            f"(rows are BF16, {BF16_BYTES:,} B/token, until converted)")


def judge(check: dict, rejected: collections.Counter, rows: int, min_agreement: float) -> dict:
    """Verdict on check_capture output: alignment must be exact; agreement says whether the text is GLM's."""

    lines, status, fix = [], "ok", ""
    checked = check.get("rows", [])
    tiles = sum(bool(r.get("tiles")) for r in checked)
    lines.append(f"alignment: {rows} rows kept, {sum(rejected.values())} rejected {dict(rejected) or ''}".rstrip()
                 + f"; sampled {len(checked)}: {tiles}/{len(checked)} tile [from, T) with the capture records")
    if tiles < len(checked):
        status, fix = "fail", "rows do not match their capture records: delete rows/ and capture-logs/ and recapture"
    for size, gate in check.get("gates", {}).items():   # capture.py --gates: repeat and causal-prefix checks
        failed = [k for k, v in gate.items() if not v]
        lines.append(f"gates {size}: " + (f"FAILED {failed}" if failed else "repeat, causal prefix, suffix: pass"))
        if failed:
            status, fix = "fail", "the engine is not deterministic or not causal here: check the capture settings " \
                                  "against docs/capture.md (prefix cache off, serving-equivalent flags)"
    agreement = check.get("agreement")
    suspects = sorted((r["agree"], r["name"]) for r in checked if r.get("generated", 0) >= 128 and r["agree"] < 0.6)
    if agreement is None:
        lines.append("agreement: SKIPPED (--skip-agreement); the data is not verified as GLM-5.3-Flash text")
        status = "warn" if status == "ok" else status
    else:
        positions = sum(r.get("generated", 0) for r in checked)
        lines.append(f"agreement: {agreement:.2%} over {positions:,} generated positions "
                     "(genuine GLM-5.3-Flash text at T=1: about 83-86%)")
        if agreement < min_agreement:
            status = "fail"
            fix = (f"agreement {agreement:.1%} < {min_agreement:.0%}: this text is very likely not GLM-5.3-Flash "
                   "output "
                   "(another model, edited text, or a wrong chat template/effort). Regenerate it with GLM-5.3-Flash "
                   "(--regenerate) or remove those inputs; see docs/byo-data.md, troubleshooting")
        elif agreement < 0.80 and status == "ok":
            status = "warn"
            lines.append("agreement is below the usual range: check the inputs' origin and reasoning effort")
    if suspects:
        lines.append(f"{len(suspects)} sampled rows below 60% agreement (worst first): "
                     + ", ".join(f"{n} {a:.0%}" for a, n in suspects[:5]))
    return {"status": status, "agreement": agreement, "rejected": dict(rejected), "suspects": suspects[:50],
            "lines": lines, "fix": fix}


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, action="append", required=True,
                    help="chat JSONL, request/response log JSONL, {'prompt'} JSONL, .txt prompt list, .json list")
    ap.add_argument("--work", type=Path, required=True, help="working directory (all outputs; resumable)")
    ap.add_argument("--teacher", type=Path, help="your GLM-5.3-Flash checkpoint directory (template, agreement)")
    ap.add_argument("--tokenizer-dir", type=Path, help="tokenizer.json + chat_template.jinja (default: --teacher)")
    ap.add_argument("--dry-run", action="store_true", help="validate and print the plan; write nothing")
    ap.add_argument("--until", choices=STAGES, default="index", help="stop after this stage")
    g = ap.add_argument_group("input rules")
    g.add_argument("--turns", choices=["last", "all"], default="last",
                   help="assistant turns of a conversation used as completions")
    g.add_argument("--completions-are-glm", action="store_true",
                   help="I attest the assistant text in my inputs is GLM-5.3-Flash output (needed when no model field)")
    g.add_argument("--regenerate", action="store_true",
                   help="drop assistant text that is not attested GLM-5.3-Flash output and generate it fresh")
    g.add_argument("--allow-missing-reasoning", action="store_true", help="accept GLM turns logged without thinking")
    g.add_argument("--effort", default=None, help="reasoning effort for rows that carry none (template names)")
    g.add_argument("--scrub", choices=["report", "redact", "drop", "off"], default="report")
    g.add_argument("--skip-bad", action="store_true", help="leave out records that fail validation")
    g = ap.add_argument_group("generation (prompts without completions)")
    g.add_argument("--teacher-url", default=None, help="your OpenAI-compatible GLM-5.3-Flash chat completions URL")
    g.add_argument("--teacher-model", default="glm-5.3-flash")
    g.add_argument("--teacher-key-env", default="OPENROUTER_API_KEY")
    g.add_argument("--samples", type=int, default=1)
    g.add_argument("--gen-minutes", type=float, default=60)
    g.add_argument("--spend-cap", type=float, default=10.0)
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--gen-arg", action="append", default=[], help="extra scripts/api_generate.py argument")
    g = ap.add_argument_group("capture (Atlas capture build, docs/capture.md)")
    g.add_argument("--atlas-server", default="http://127.0.0.1:8893")
    g.add_argument("--atlas-model", default="glm-5.3-flash-atlas")
    g.add_argument("--capture-dir", type=Path, action="append", default=[], help="per rank, as seen from here")
    g.add_argument("--capture-minutes", type=float, default=240)
    g.add_argument("--before-capture", default=None, help="command that starts the Atlas capture server")
    g.add_argument("--after-capture", default=None, help="command that stops it (frees memory for training)")
    g.add_argument("--server-wait", type=float, default=900, help="seconds to wait for the server to answer")
    g.add_argument("--capture-arg", action="append", default=[], help="extra scripts/capture.py argument")
    g = ap.add_argument_group("check")
    g.add_argument("--check-rows", type=int, default=200, help="rows sampled for the agreement check")
    g.add_argument("--min-agreement", type=float, default=0.78)
    g.add_argument("--skip-agreement", action="store_true", help="skip the GLM agreement check (not recommended)")
    g.add_argument("--tag", default=None, help="opaque label passed to drafter-join (shown to the operator)")
    g.add_argument("--assume-completion-tokens", type=int, default=3000, help="dry-run estimate only")
    return ap


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    for option in ("gen_arg", "capture_arg"):   # accept "--capture-arg='--limit 5'" as well as one token per flag
        setattr(opts, option, [part for value in getattr(opts, option) for part in shlex.split(value)])
    pipe, dry = Pipeline(opts), opts.dry_run
    last = STAGES.index(opts.until)

    def stage(index: int, call, *args):
        if index > last:
            return []
        try:
            return call(*args, dry)
        except ByoError as error:
            if not dry:
                raise
            say(f"{STAGES[index]} would stop: {error}" + (f" (fix: {error.fix})" if error.fix else ""))
            return []

    try:
        result = pipe.validate(write=not dry)
        if dry:
            pipe.estimate(result["rows"], result["reused"])
        stage(1, pipe.generate, result["rows"], result["reused"])
        items = stage(2, pipe.items)
        stage(3, pipe.capture, items)
        stage(4, pipe.check)
        if last >= 5:
            stage(5, pipe.index, items)
            if not dry:
                commands = pipe.join_cmds()
                pipe.p("byo-ready.json").write_text(json.dumps({"commands": commands}, indent=1))
                say("ready. First time on this site, then every round:")
                for command in commands:
                    print("    " + command)
        if dry:
            say("dry run: nothing was written")
    except ByoError as error:
        print(f"[byo] error: {error}", file=sys.stderr)
        if error.fix:
            print(f"[byo] fix: {error.fix}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
