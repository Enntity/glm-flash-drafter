"""HTTP front of the coordinator: a public router (dashboard, public JSON, site and validator APIs, local-storage
blobs) and a separate admin router that only ever binds to loopback.  Stdlib ``http.server``; put TLS in front
(a reverse proxy) for anything beyond a laptop.
"""

from __future__ import annotations

import ipaddress
import json
import re
import shutil
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import parse_qs, urlsplit

from .service import ApiError, Service
from .storage import LocalStorage

STATIC = Path(__file__).resolve().parent / "static"
PAGES = {"": "index.html", "/": "index.html", "/how": "how.html", "/join": "join.html"}
TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
         ".css": "text/css; charset=utf-8", ".svg": "image/svg+xml", ".json": "application/json"}
MAX_JSON = 64 * 1024
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY",
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                               "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
}


class RateLimiter:
    """Token buckets per (class, key), in memory only (no IPs are stored)."""

    LIMITS = {"redeem": (20, 1 / 60), "public": (120, 10.0), "site": (60, 2.0), "upload": (20, 20 / 60),
              "heartbeat": (10, 1 / 20), "validator": (600, 50.0)}

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock, self.buckets, self.lock = clock, {}, threading.Lock()

    def allow(self, kind: str, key: str) -> bool:
        burst, rate = self.LIMITS[kind]
        now = self.clock()
        with self.lock:
            tokens, last = self.buckets.get((kind, key), (burst, now))
            tokens = min(burst, tokens + (now - last) * rate)
            if tokens < 1:
                self.buckets[(kind, key)] = (tokens, now)
                return False
            self.buckets[(kind, key)] = (tokens - 1, now)
            if len(self.buckets) > 100_000:
                self.buckets.clear()
            return True


def _route(pattern: str) -> re.Pattern:
    return re.compile("^" + re.sub(r"<(\w+)>", r"(?P<\1>[A-Za-z0-9_.-]+)", pattern) + "$")


class Handler(BaseHTTPRequestHandler):
    server_version = "drafter-coordinator"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    app: "App"

    def log_message(self, fmt: str, *args: Any) -> None:   # paths only: no query strings, no blob tokens
        path = urlsplit(self.path).path
        path = re.sub(r"/blob/[^/]+$", "/blob/<signed>", path)
        print(f"{self.command} {path} {args[1] if len(args) > 1 else ''}", flush=True)

    def _send(self, status: int, payload: Any = None, headers: dict[str, str] | None = None,
              content_type: str = "application/json") -> None:
        body = b"" if payload is None else (payload if isinstance(payload, bytes) else
                                            json.dumps(payload, separators=(",", ":")).encode())
        if not self.consumed:   # an unread request body must never be parsed as the next request (smuggling)
            self.close_connection = True
            headers = {**(headers or {}), "Connection": "close"}
        self.send_response(status)
        for key, value in {**SECURITY_HEADERS, **(headers or {})}.items():
            self.send_header(key, value)
        if payload is not None and "Content-Length" not in (headers or {}):
            self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)) if "Content-Length" not in (headers or {})
                         else headers["Content-Length"])
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_JSON:
            raise ApiError(413, "too_large", "request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        self.consumed = True
        try:
            body = json.loads(raw or b"{}")
        except ValueError as error:
            raise ApiError(400, "bad_json", "body must be JSON") from error
        if not isinstance(body, dict):
            raise ApiError(400, "bad_json", "body must be a JSON object")
        return body

    def _client_key(self) -> str:
        """The peer address; behind a configured proxy, the last hop that proxy saw (the rightmost
        X-Forwarded-For entry), which a client cannot forge."""

        peer = self.client_address[0]
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded and peer in self.app.trusted_proxies:
            return forwarded.split(",")[-1].strip()
        return peer

    def _dispatch(self) -> None:
        length = self.headers.get("Content-Length")
        self.consumed = not length or length.strip() == "0"
        if self.headers.get("Transfer-Encoding"):   # chunked bodies are not supported: never guess their end
            self.consumed = False
            return self._send(411, {"error": "length_required", "message": "send a Content-Length body"})
        try:
            self.app.handle(self)
        except ApiError as error:
            self._send(error.status, {"error": error.code, "message": error.message})
        except ValueError:   # a malformed number in a path or query
            self._send(400, {"error": "bad_request", "message": "malformed request"})
        except Exception as error:  # noqa: BLE001 - never leak internals
            print(f"internal error: {error!r}", flush=True)
            self._send(500, {"error": "internal", "message": "internal error"})

    do_GET = do_POST = do_PUT = do_HEAD = do_DELETE = _dispatch


class App:
    """Route table over a :class:`Service`; ``admin`` selects the loopback admin router."""

    def __init__(self, service: Service, base_path: str = "/drafter", admin: bool = False):
        self.service, self.admin = service, admin
        self.trusted_proxies = {p.strip() for p in service.config.trusted_proxies.split(",") if p.strip()}
        self.base = "/" + base_path.strip("/") if base_path.strip("/") else ""
        self.limiter = RateLimiter()
        s = service
        if admin:
            self.routes = [
                ("POST", "/admin/invites", None, lambda h, b, m: s.issue_invite(b.get("label", ""), b.get("hours"))),
                ("GET", "/admin/invites", None, lambda h, b, m: s.list_invites()),
                ("POST", "/admin/invites/<id>/revoke", None, lambda h, b, m: s.revoke_invite(int(m["id"]))),
                ("GET", "/admin/sites", None, lambda h, b, m: s.list_sites()),
                ("POST", "/admin/sites/<id>/revoke", None, lambda h, b, m: s.revoke_site(m["id"])),
                ("POST", "/admin/validators", None, lambda h, b, m: s.issue_validator(b.get("label", "validator"))),
                ("POST", "/admin/shards", None, lambda h, b, m: s.add_shards(b.get("recipes", []))),
                ("POST", "/admin/settings/<id>", None, lambda h, b, m: s.set_setting(m["id"], b.get("value"))),
                ("GET", "/admin/settings/<id>", None, lambda h, b, m: s.get_setting(m["id"])),
                ("POST", "/admin/deltas/<id>/release", None,
                 lambda h, b, m: s.release_held(m["id"], b.get("decision"), float(b.get("weight_factor", 1.0)),
                                                bool(b.get("publish")))),
                ("GET", "/admin/deltas", None, lambda h, b, m: s.list_deltas(b.get("state"))),
                ("GET", "/admin/attention", None, lambda h, b, m: s.attention()),
                ("GET", "/admin/audit", None, lambda h, b, m: s.audit_log(int(b.get("limit", 200)))),
            ]
        else:
            api, v = self.base + "/api/v1", self.base + "/api/v1/validator"
            self.routes = [
                ("GET", self.base + "/api/public/dashboard", "public", lambda h, b, m: s.dashboard()),
                ("GET", api + "/terms", "public", lambda h, b, m: s.terms()),
                ("POST", api + "/redeem", "redeem",
                 lambda h, b, m: s.redeem(b.get("code", ""), b.get("display_name"), b.get("accept_terms", ""))),
                ("GET", api + "/requirements", "site", lambda h, b, m: s.requirements(h.site)),
                ("GET", api + "/global", "site", lambda h, b, m: s.version_url(_int(h.query.get("version")))),
                ("POST", api + "/shards/claim", "site", lambda h, b, m: {"shard": s.claim_shard(h.site)}),
                ("POST", api + "/shards/<id>/release", "site", lambda h, b, m: s.release_shard(h.site, m["id"])),
                ("POST", api + "/heartbeat", "heartbeat", lambda h, b, m: s.heartbeat(h.site, b)),
                ("POST", api + "/deltas", "upload", lambda h, b, m: s.init_delta(h.site, b)),
                ("POST", api + "/uploads/<id>/refresh", "upload",
                 lambda h, b, m: s.refresh_upload(h.site, m["id"], b.get("parts") or [])),
                ("POST", api + "/deltas/<id>/complete", "upload",
                 lambda h, b, m: s.complete_delta(h.site, m["id"], b.get("parts") or [])),
                ("GET", api + "/deltas/<id>", "site", lambda h, b, m: s.delta_status(h.site, m["id"])),
                ("GET", api + "/me", "site", lambda h, b, m: s.me(h.site)),
                ("POST", api + "/me", "site", lambda h, b, m: s.set_display_name(h.site, b.get("display_name"))),
                ("POST", v + "/hello", "validator", lambda h, b, m: s.validator_hello(h.site, b.get("version", ""))),
                ("GET", v + "/pending", "validator", lambda h, b, m: s.pending()),
                ("POST", v + "/deltas/<id>/claim", "validator", lambda h, b, m: s.claim_delta(h.site, m["id"])),
                ("POST", v + "/deltas/<id>/verdict", "validator", lambda h, b, m: s.verdict(h.site, m["id"], b)),
                ("GET", v + "/mergeable", "validator", lambda h, b, m: s.mergeable()),
                ("GET", v + "/deltas/<id>", "validator", lambda h, b, m: s.delta_url(h.site, m["id"])),
                ("GET", v + "/versions/<id>", "validator", lambda h, b, m: s.version_url(int(m["id"]))),
                ("POST", v + "/versions/<id>/upload", "validator",
                 lambda h, b, m: s.init_version_upload(h.site, int(m["id"]), b.get("size"), b.get("sha256", ""),
                                                       b.get("dtype", ""))),
                ("POST", v + "/versions/<id>/publish", "validator",
                 lambda h, b, m: s.publish_version(h.site, int(m["id"]), b)),
                ("POST", v + "/uploads/<id>/refresh", "validator",
                 lambda h, b, m: s.refresh_upload(h.site, m["id"], b.get("parts") or [])),
            ]
        self.compiled = [(method, _route(path), kind, fn) for method, path, kind, fn in self.routes]

    def handle(self, h: Handler) -> None:
        parts = urlsplit(h.path)
        path, h.query = parts.path, {k: v[0] for k, v in parse_qs(parts.query).items()}
        if self.admin:   # loopback only, and still authenticated (and no DNS-rebinding Host)
            host = (h.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
            if host not in ("127.0.0.1", "localhost", "::1"):
                raise ApiError(403, "bad_host", "admin requests must address the loopback interface")
            auth = h.headers.get("Authorization", "")
            self.service.check_admin(auth[7:].strip() if auth.startswith("Bearer ") else None)
        if not self.admin:
            if self.base and path == self.base and h.command in ("GET", "HEAD"):   # relative links need the slash
                return h._send(301, b"", {"Location": self.base + "/"}, "text/plain")
            if path.startswith(self.base + "/blob/") and isinstance(self.service.storage, LocalStorage):
                return self._blob(h, path[len(self.base) + 6:])
            page = PAGES.get(path[len(self.base):]) if path.startswith(self.base) or not self.base else None
            if h.command in ("GET", "HEAD") and (page or path.startswith(self.base + "/static/")):
                return self._static(h, page or path[len(self.base) + 8:])
        for method, pattern, kind, fn in self.compiled:
            match = pattern.match(path)
            if match and method == h.command:
                h.site = None
                if kind in ("site", "upload", "heartbeat", "validator"):
                    auth = h.headers.get("Authorization", "")
                    token = auth[7:].strip() if auth.startswith("Bearer ") else None
                    h.site = self.service.authenticate(token, "validator" if kind == "validator" else "site")
                    key = h.site["public_id"]
                else:
                    key = h._client_key()
                if not self.admin and not self.limiter.allow(kind, key):
                    raise ApiError(429, "rate_limited", "too many requests; slow down and retry")
                body = h._json() if h.command in ("POST", "PUT") else dict(h.query)
                result = fn(h, body, match.groupdict())
                return h._send(200, {"ok": True} if result is None else result)
        raise ApiError(404, "not_found", "no such route")

    def _static(self, h: Handler, name: str) -> None:
        file = (STATIC / name).resolve()
        if STATIC not in file.parents or not file.is_file():
            raise ApiError(404, "not_found", "no such page")
        h._send(200, file.read_bytes(), {"Cache-Control": "no-cache"}, TYPES.get(file.suffix, "text/plain"))

    def _blob(self, h: Handler, token: str) -> None:
        length = int(h.headers["Content-Length"]) if h.headers.get("Content-Length") else None
        status, headers, payload = self.service.storage.handle_blob(h.command, token, h.rfile, length)
        h.consumed = h.consumed or (h.command == "PUT" and status == 200)
        if isinstance(payload, Path):
            h.send_response(status)
            h.send_header("Content-Type", "application/octet-stream")
            for key, value in headers.items():
                h.send_header(key, value)
            h.end_headers()
            if h.command == "GET":
                with payload.open("rb") as stream:
                    shutil.copyfileobj(stream, h.wfile, 8 * 2**20)
            return
        h._send(status, payload, headers, "text/plain")


def _int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    if not str(value).isdigit():
        raise ApiError(400, "bad_request", "expected a non-negative integer")
    return int(value)


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    if app.admin and not ipaddress.ip_address(host).is_loopback:
        raise ValueError("the admin API only binds to a loopback address")
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


def serve_in_thread(app: App, host: str = "127.0.0.1", port: int = 0) -> tuple[ThreadingHTTPServer, str]:
    """Start a server on a background thread (tests, embedding); returns it and its base URL."""

    server = make_server(app, host, port)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://{host}:{server.server_address[1]}"


__all__ = ["App", "HTTPStatus", "RateLimiter", "make_server", "serve_in_thread"]
