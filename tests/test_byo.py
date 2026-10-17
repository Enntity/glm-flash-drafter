"""CPU tests for scripts/byo_prepare.py: input formats, teacher rules, scrub, verdicts, and the whole chain against
a stand-in GLM endpoint and a stand-in Atlas capture server on loopback (no network, no GPU).

    OMP_NUM_THREADS=1 python -m unittest discover -s tests -p test_byo.py -v
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import shlex
import tempfile
import threading
import sys
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import REPO  # noqa: E402

import torch  # noqa: E402

from test_capture_io import write_fast  # noqa: E402

spec = importlib.util.spec_from_file_location("byo_prepare", REPO / "scripts" / "byo_prepare.py")
byo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(byo)

GEN_PROMPT = [154828, 154841]   # what Atlas's /tokenize ends a generation prompt with (scripts/capture.py)
HIDDEN = 16
TEMPLATE = """[gMASK]<sop>
{%- if tools -%}<|system|>
{% for t in tools %}{{ t | tojson }}
{% endfor %}{%- endif -%}
{%- for m in messages -%}
{%- if m.role == 'assistant' -%}<|assistant|><think>{{ m.reasoning_content or '' }}</think>{{ m.content }}
{%- for tc in m.tool_calls or [] -%}<tool_call>{{ tc.function.name }}{{ tc.function.arguments | tojson }}</tool_call>
{%- endfor -%}
{%- elif m.role == 'tool' -%}<|observation|>{{ m.content }}
{%- else -%}<|{{ m.role }}|>{{ m.content }}
{%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}<|assistant|><think>{%- endif -%}"""


def tiny_glm_dir(root: Path) -> Path:
    """A stand-in for the checkpoint's tokenizer.json + chat_template.jinja (byte-level, GLM's special tokens)."""

    from tokenizers import Tokenizer, models, pre_tokenizers, decoders

    vocab = {c: i for i, c in enumerate(pre_tokenizers.ByteLevel.alphabet())}
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tok.add_special_tokens(["[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
                            "<think>", "</think>", "<tool_call>", "</tool_call>"])
    root.mkdir(parents=True, exist_ok=True)
    tok.save(str(root / "tokenizer.json"))
    (root / "chat_template.jinja").write_text(TEMPLATE)
    return root


def opts(**kw) -> argparse.Namespace:
    base = byo.build_parser().parse_args(["--input", "x", "--work", "w"])
    for key, value in kw.items():
        setattr(base, key, value)
    return base


def write(path: Path, records: list) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def chat(prompt: str, reply: str | None = None, reasoning: str = "Let me think.", **extra) -> dict:
    messages = [{"role": "user", "content": prompt}]
    if reply is not None:
        messages.append({"role": "assistant", "content": reply, "reasoning_content": reasoning})
    return {"messages": messages, **extra}


class Server:
    """A loopback HTTP server running ``handle(path, body) -> reply`` in a thread."""

    def __init__(self, handle):
        calls = self.calls = []

        class Handler(BaseHTTPRequestHandler):
            def _reply(self, payload):
                data = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):  # noqa: N802
                self._reply({"data": [{"id": "stand-in"}]})

            def do_POST(self):  # noqa: N802
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                calls.append((self.path, body))
                self._reply(handle(self.path, body))

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def glm_endpoint(path: str, body: dict) -> dict:
    question = body["messages"][-1]["content"]
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": f"Answer to {question}.", "reasoning_content": f"Thinking about {question}."}}],
        "usage": {"completion_tokens": 12}}


class AtlasStandIn:
    """/tokenize and a capture prefill: writes the armed request's [from, T) segments like the engine (one rank)."""

    def __init__(self, capture_dir: Path, corrupt: set[str] = frozenset()):
        self.capture_dir, self.corrupt = capture_dir, corrupt

    def __call__(self, path: str, body: dict) -> dict:
        if path == "/tokenize":
            text = json.dumps(body["messages"])
            return {"tokens": [10 + (ord(c) % 500) for c in text[:40]] + GEN_PROMPT}
        ids = body["prompt"]
        enable = self.capture_dir / "enable"
        if enable.is_file():
            arm = json.loads(enable.read_text())
            torch.manual_seed(len(ids))
            stored = list(ids)
            if arm["tag"] in self.corrupt:
                stored[-1] += 1
            write_fast(self.capture_dir / "chunks" / arm["tag"], torch.randn(6, len(ids), HIDDEN), stored,
                       arm["from"], 16, sp=False)
        return {"choices": [{"text": "x"}]}


class InputTests(unittest.TestCase):
    def test_formats_normalize_to_prompt_rows_and_reused_completions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tools = [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}]
            chats = write(root / "chats.jsonl", [
                chat("What is 2+2?", "4", model="glm-5.3-flash", reasoning_effort="high"),
                {"messages": [{"role": "user", "content": [{"type": "text", "text": "Find order 7"}]},
                              {"role": "assistant", "content": "", "reasoning_content": "Use the tool.",
                               "tool_calls": [{"id": "c1", "type": "function",
                                               "function": {"name": "lookup", "arguments": "{\"id\": 7}"}}]},
                              {"role": "tool", "tool_call_id": "c1", "content": "{\"status\": \"shipped\"}"},
                              {"role": "assistant", "content": "It shipped.", "reasoning": "Report it."}],
                 "tools": tools, "model": "z-ai/glm-5.3-flash"},
                chat("What is 2+2?", "4", model="glm-5.3-flash", reasoning_effort="high"),   # exact duplicate
            ])
            prompts = root / "prompts.txt"
            prompts.write_text("Write a haiku about rain.\n\nSummarize the plot of a heist film.\n")
            logs = write(root / "proxy.jsonl", [{"request": {"model": "glm-5.3-flash", "messages": [
                {"role": "system", "content": "Be brief."}, {"role": "user", "content": "Name a prime."}]},
                "response": {"model": "glm-5.3-flash", "provider": "SomeHost", "choices": [{
                    "finish_reason": "length", "message": {"content": "<think>Two is prime.</think>2"}}]}},
                {"prompt": "Translate 'hello' to French.", "system": "You translate."}])
            out = byo.validate(opts(input=[chats, prompts, logs]))
            report = out["report"]
            self.assertEqual((report["records"], report["prompts"], report["reused_completions"],
                              report["need_generation"], report["duplicate_records"]), (7, 6, 3, 3, 1))
            by_id = {r["id"]: r for r in out["rows"]}
            agentic = next(r for r in out["rows"] if r.get("tools"))
            self.assertEqual((agentic["category"], [m["role"] for m in agentic["messages"]]),
                             ("agentic", ["user", "assistant", "tool"]))
            self.assertEqual(agentic["messages"][0]["content"], "Find order 7")
            log_reply = next(c for c in out["reused"] if by_id[c["prompt_id"]]["messages"][0]["role"] == "system")
            self.assertEqual((log_reply["reasoning"], log_reply["content"], log_reply["finish"]),
                             ("Two is prime.", "2", "length"))   # a log without a reasoning parser is split
            self.assertTrue(all(r["id"].startswith("byo-") and r["split"] == "train" for r in out["rows"]))
            self.assertEqual(next(r for r in out["rows"] if "2+2" in r["messages"][0]["content"])["reasoning_effort"],
                             "high")
            # --turns all: every GLM turn of a conversation is a completion (the tool call too)
            out = byo.validate(opts(input=[chats], turns="all"))
            self.assertEqual(out["report"]["reused_completions"], 3)
            self.assertTrue(any(c["tool_calls"] for c in out["reused"]))

    def test_teacher_rules(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cases = {
                "other": chat("Q1", "A", model="some-other-model"),
                "nc": chat("Q2", "A", model="incoai/GLM-5.3-Flash-DFlash2"),
                "zai": {"request": {"messages": [{"role": "user", "content": "Q3"}]}, "response": {
                    "model": "z-ai/glm-5.3-flash", "provider": "Z.AI", "choices": [{"message": {
                        "content": "A", "reasoning_content": "r"}}]}},
                "unattested": chat("Q4", "A"),
                "no_thinking": chat("Q5", "A", reasoning="", model="glm-5.3-flash"),
            }
            expect = {"other": "not GLM-5.3-Flash", "nc": "non-commercial", "zai": "Z.AI",
                      "unattested": "--completions-are-glm", "no_thinking": "thinking trace"}
            for name, record in cases.items():
                path = write(root / f"{name}.jsonl", [record])
                with self.assertRaises(byo.ByoError) as caught:
                    byo.validate(opts(input=[path]))
                self.assertIn(expect[name], str(caught.exception), name)
            # attestation covers unlabelled text, never a foreign model
            self.assertEqual(byo.validate(opts(input=[root / "unattested.jsonl"], completions_are_glm=True))[
                "report"]["reused_completions"], 1)
            with self.assertRaises(byo.ByoError):
                byo.validate(opts(input=[root / "other.jsonl"], completions_are_glm=True))
            # --regenerate: foreign text is cut, even as context, and the prompt goes to generation
            multi = write(root / "multi.jsonl", [{"model": "some-other-model", "messages": [
                {"role": "user", "content": "First"}, {"role": "assistant", "content": "foreign"},
                {"role": "user", "content": "Second"}]}])
            out = byo.validate(opts(input=[root / "other.jsonl", multi], regenerate=True))
            self.assertEqual((out["report"]["reused_completions"], out["report"]["need_generation"],
                              out["report"]["regenerated_records"]), (0, 2, 2))
            self.assertNotIn("foreign", json.dumps(out["rows"]))
            self.assertEqual(byo.validate(opts(input=[root / "no_thinking.jsonl"], allow_missing_reasoning=True))[
                "report"]["reused_completions"], 1)
            # --skip-bad keeps the good records
            bad = root / "mixed.jsonl"
            bad.write_text(json.dumps(chat("Fine prompt")) + "\n{not json\n" + json.dumps({"x": 1}) + "\n")
            with self.assertRaises(byo.ByoError) as caught:
                byo.validate(opts(input=[bad]))
            self.assertIn("2 input record(s)", str(caught.exception))
            self.assertIn("mixed.jsonl:2", str(caught.exception))
            self.assertEqual(byo.validate(opts(input=[bad], skip_bad=True))["report"]["prompts"], 1)


class ScrubTests(unittest.TestCase):
    def test_finds_and_redacts_secrets_and_contact_data(self):
        key = "sk-" + "a1B2" * 8
        aws = "AKIA" + "Q" * 16
        ip = ".".join(["10", "20", "30", "40"])
        text = (f"mail me at jane.roe@corp-mail.org or test@example.com; key {key}; aws {aws}; host {ip}; "
                f"local 127.0.0.1; card 4111 1111 1111 1111; not a card 4111 1111 1111 1112; "
                f"password = \"hunter2hunter2\"; db postgres://admin:pw123456@db/x")
        found, hits = byo.scrub_text(text, redact=False)
        self.assertEqual(found, text)
        self.assertEqual(dict(hits), {"email": 1, "api_key": 2, "ipv4": 1, "card_number": 1, "secret_value": 1,
                                      "url_credentials": 1})
        clean, _ = byo.scrub_text(text, redact=True)
        for secret in (key, aws, ip, "jane.roe@corp-mail.org", "hunter2hunter2", "4111 1111 1111 1111", "pw123456"):
            self.assertNotIn(secret, clean)
        for kept in ("test@example.com", "127.0.0.1", "4111 1111 1111 1112", "postgres://", "@db/x"):
            self.assertIn(kept, clean)
        self.assertIn("[REDACTED:api_key]", clean)

    def test_modes_in_validate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write(Path(tmp) / "in.jsonl", [chat("Reach me at jane.roe@corp-mail.org"), chat("Clean prompt")])
            report = byo.validate(opts(input=[path]))
            self.assertEqual((report["scrub"], report["report"]["prompts"]), ({"email": 1}, 2))
            self.assertIn("jane.roe", json.dumps(report["rows"]))
            redacted = byo.validate(opts(input=[path], scrub="redact"))
            self.assertNotIn("jane.roe", json.dumps(redacted["rows"]))
            self.assertEqual(byo.validate(opts(input=[path], scrub="drop"))["report"]["prompts"], 1)


class VerdictTests(unittest.TestCase):
    def rows(self, agree: float, n: int = 4, tiles: bool = True) -> list[dict]:
        return [{"name": f"r{i}", "tiles": tiles, "generated": 500, "agree": agree} for i in range(n)]

    def test_agreement_and_alignment_gate(self):
        ok = byo.judge({"rows": self.rows(0.84), "agreement": 0.84}, {}, 4, 0.78)
        self.assertEqual(ok["status"], "ok")
        self.assertIn("84.00%", " ".join(ok["lines"]))
        warn = byo.judge({"rows": self.rows(0.79), "agreement": 0.79}, {}, 4, 0.78)
        self.assertEqual(warn["status"], "warn")
        fail = byo.judge({"rows": self.rows(0.55), "agreement": 0.55}, {}, 4, 0.78)
        self.assertEqual(fail["status"], "fail")
        self.assertIn("not GLM-5.3-Flash output", fail["fix"])
        self.assertEqual(len(fail["suspects"]), 4)
        skipped = byo.judge({"rows": self.rows(0.0)}, {}, 4, 0.78)
        self.assertEqual(skipped["status"], "warn")
        self.assertIn("SKIPPED", " ".join(skipped["lines"]))
        self.assertEqual(byo.judge({"rows": self.rows(0.85, tiles=False), "agreement": 0.85}, {}, 4, 0.78)["status"],
                         "fail")
        gates = {"32": {"repeat_exact": True, "causal_prefix_exact": False, "mutated_suffix_differs": True}}
        self.assertEqual(byo.judge({"rows": self.rows(0.85), "agreement": 0.85, "gates": gates}, {}, 4, 0.78)[
            "status"], "fail")


class PipelineTests(unittest.TestCase):
    def test_raw_data_to_index_resumable_with_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            glm_dir = tiny_glm_dir(root / "glm")
            capture_dir = root / "capture"
            (capture_dir / "chunks").mkdir(parents=True)
            inputs = write(root / "chats.jsonl", [chat(f"Question {i}?") for i in range(5)] + [
                chat("Reused question?", "Reused answer.", model="glm-5.3-flash")])
            work = root / "work"
            atlas = AtlasStandIn(capture_dir)
            glm, engine = Server(glm_endpoint), Server(atlas)
            try:
                argv = ["--input", str(inputs), "--work", str(work), "--teacher", str(glm_dir),
                        "--teacher-url", glm.url + "/v1/chat/completions", "--effort", "high",
                        "--atlas-server", engine.url, "--capture-dir", str(capture_dir), "--skip-agreement",
                        "--gen-arg=--minutes 1 --drain 1"]

                def main(*extra: str) -> tuple[int, str]:
                    stream = io.StringIO()
                    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                        code = byo.main(argv + list(extra))
                    return code, stream.getvalue()

                code, out = main("--dry-run")
                self.assertEqual(code, 0, out)
                self.assertFalse(work.exists(), "a dry run writes nothing")
                self.assertIn("5 prompts without one", out)
                self.assertIn("GB FP8 on disk", out)
                self.assertIn("scripts/capture.py", out)
                self.assertEqual((glm.calls, engine.calls), ([], []))

                first = next(r for r in byo.validate(opts(input=[inputs], effort="high"))["rows"]
                             if "Question 0" in r["messages"][0]["content"])
                atlas.corrupt = {first["id"] + "_s0"}   # capture tag = generation id with '#' -> '_
                hooks = [f"--{when}-capture=" + shlex.join([sys.executable, "-c", f"open({str(root / when)!r}, 'a')"])
                         for when in ("before", "after")]
                code, out = main("--capture-arg=--limit 3", *hooks)   # a partial capture, as if the window ran out
                self.assertEqual(code, 0, out)
                self.assertTrue((root / "before").exists() and (root / "after").exists())
                self.assertEqual(len(glm.calls), 5)
                self.assertTrue(all(c[1]["chat_template_kwargs"] == {"reasoning_effort": "high"} for c in glm.calls))
                prefills = [c for c in engine.calls if c[0] == "/v1/completions"]
                self.assertEqual(len(prefills), 3)
                code, out = main()   # resume: no new generation, only the remaining captures
                self.assertEqual(code, 0, out)
                self.assertEqual(len(glm.calls), 5)
                self.assertEqual(len([c for c in engine.calls if c[0] == "/v1/completions"]), 6)
                self.assertIn("token_ids_differ", out)
                self.assertTrue((work / "rejected" / next(iter(atlas.corrupt))).is_dir())
                index = json.loads((work / "index" / "index.json").read_text())
                self.assertEqual(len(index["rows"]), 5)
                self.assertTrue(all(r["dtype"] == "fp8_e4m3" and r["P"] == 42 for r in index["rows"]))
                ready = json.loads((work / "byo-ready.json").read_text())["commands"]
                self.assertEqual(ready[1].split()[:2], ["drafter-join", "run"])
                self.assertIn(f"--byo-capture-index {work.resolve() / 'index' / 'index.json'}", ready[1])
                self.assertIn(f"--byo-capture-root {work.resolve()}", ready[1])
                self.assertIn("drafter-join setup", out)
                calls = len(engine.calls)
                code, out = main()   # everything done: a rerun only re-checks and re-indexes
                self.assertEqual((code, len(glm.calls), len(engine.calls)), (0, 5, calls))
                self.assertIn("nothing to generate", out)
                self.assertIn("already captured", out)
            finally:
                glm.close()
                engine.close()

    def test_clear_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            inputs = write(root / "in.jsonl", [chat("Needs a completion")])
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
                code = byo.main(["--input", str(inputs), "--work", str(root / "w"), "--until", "generate",
                                 "--teacher-key-env", "BYO_TEST_UNSET_KEY"])
            self.assertEqual(code, 2)
            self.assertIn("no endpoint is configured", stream.getvalue())
            self.assertIn("fix: pass --teacher-url", stream.getvalue())


if __name__ == "__main__":
    unittest.main()
