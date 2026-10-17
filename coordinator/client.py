"""Stdlib client for the coordinator API and for signed-URL transfers (used by the site client and the validator).

* :class:`ApiClient`: JSON calls with a bearer token; retries connection errors, 5xx and 429 with exponential
  backoff; 4xx answers raise :class:`ApiError` carrying the server's code and actionable message.
* :func:`upload_file`: resumable upload of a file along an upload plan (``parts`` or ``gcs-resumable``, see
  :mod:`.storage`); progress (ETags, GCS session) lives in a caller-persisted ``state`` dict, so an interrupted
  upload continues where it stopped, even after a restart.
* :func:`download`: streamed download with size and SHA-256 verification (temp file, then rename).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

CHUNK = 8 * 2**20


class ApiError(RuntimeError):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.status, self.code, self.message = status, code, message


def _backoff(attempt: int, base: float = 1.0, cap: float = 60.0) -> float:
    return min(cap, base * 2 ** attempt)


class ApiClient:
    def __init__(self, base_url: str, token: str | None = None, timeout: float = 60, retries: int = 6,
                 sleep: Callable[[float], None] = time.sleep, backoff_base: float = 1.0):
        self.base_url, self.token, self.timeout = base_url.rstrip("/"), token, timeout
        self.retries, self.sleep, self.backoff_base = retries, sleep, backoff_base

    def call(self, method: str, path: str, body: dict[str, Any] | None = None, auth: bool = True) -> Any:
        data = None if body is None else json.dumps(body).encode()
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if auth and self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            request = urllib.request.Request(self.base_url + path, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read() or b"null")
            except urllib.error.HTTPError as error:
                payload = error.read()
                try:
                    info = json.loads(payload)
                except ValueError:
                    info = {"error": f"http_{error.code}", "message": payload[:200].decode(errors="replace")}
                if error.code in (429, 500, 502, 503, 504) and attempt < self.retries:
                    last = ApiError(error.code, info.get("error", ""), info.get("message", ""))
                    self.sleep(_backoff(attempt, self.backoff_base))
                    continue
                raise ApiError(error.code, info.get("error", f"http_{error.code}"), info.get("message", "")) from None
            except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as error:
                last = error
                if attempt < self.retries:
                    self.sleep(_backoff(attempt, self.backoff_base))
                    continue
        raise ApiError(0, "unreachable", f"coordinator unreachable: {last!r}")


def _put(url: str, data: bytes, headers: dict[str, str] | None = None, timeout: float = 600
         ) -> tuple[int, dict[str, str]]:
    request = urllib.request.Request(url, data=data, method="PUT",
                                     headers={"Content-Length": str(len(data)), **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers)


def _read(path: Path, offset: int, length: int) -> bytes:
    with path.open("rb") as stream:
        stream.seek(offset)
        return stream.read(length)


def _gcs_next(headers: dict[str, str]) -> int:
    """Next byte to send after a GCS 308: ``Range: bytes=0-N`` means N+1 bytes are stored; no Range header means
    none are."""

    stored = headers.get("Range") or headers.get("range")
    return int(stored.rsplit("-", 1)[1]) + 1 if stored else 0


def upload_file(path: str | Path, plan: dict[str, Any], refresh: Callable[[list[int]], dict[str, Any]],
                state: dict[str, Any], save: Callable[[], None] = lambda: None, retries: int = 6,
                sleep: Callable[[float], None] = time.sleep, before_part: Callable[[int], None] | None = None,
                progress: Callable[[int, int], None] | None = None) -> list[dict[str, Any]]:
    """Upload ``path``; returns the parts list to report on completion (empty for GCS).  ``state`` keeps
    ``etags`` (parts) or ``session``/``done`` (GCS) and must be persisted by ``save`` for restarts."""

    path, size = Path(path), Path(path).stat().st_size
    if plan["protocol"] == "parts":
        etags: dict[str, str] = state.setdefault("etags", {})
        part_size, urls = plan["part_size"], {p["n"]: p for p in plan.get("parts", [])}
        count = -(-size // part_size)
        for n in range(1, count + 1):
            if str(n) in etags:
                continue
            length = min(part_size, size - (n - 1) * part_size)
            for attempt in range(retries + 1):
                if n not in urls or urls[n].get("length") != length:
                    urls.update({p["n"]: p for p in refresh([n]).get("parts", [])})
                    if n not in urls:
                        raise ApiError(0, "no_url", f"no upload URL for part {n}")
                if before_part is not None:
                    before_part(n)
                try:
                    status, headers = _put(urls[n]["url"], _read(path, (n - 1) * part_size, length))
                except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
                    status, headers = 0, {}
                if status == 200 and headers.get("ETag"):
                    etags[str(n)] = headers["ETag"]
                    save()
                    break
                if status in (400, 403):   # expired or rejected URL: ask for a fresh one
                    urls.pop(n, None)
                if attempt == retries:
                    raise ApiError(status, "upload_failed", f"part {n} failed after {retries + 1} attempts")
                sleep(_backoff(attempt))
            if progress:
                progress(min(n * part_size, size), size)
        return [{"n": int(n), "etag": e} for n, e in sorted(etags.items(), key=lambda kv: int(kv[0]))]
    if plan["protocol"] != "gcs-resumable":
        raise ValueError(f"unknown upload protocol {plan['protocol']!r}")
    chunk = plan["chunk_size"]
    attempt = 0   # consecutive failures: reset whenever a chunk is acknowledged, so a long upload survives blips
    while True:
        try:
            if state.get("done"):
                return []
            if not state.get("session"):
                current = plan if plan.get("init_url") else refresh([])
                request = urllib.request.Request(current["init_url"], data=b"", method="POST",
                                                 headers={**current["init_headers"], "Content-Length": "0"})
                with urllib.request.urlopen(request, timeout=60) as response:
                    state["session"] = response.headers["Location"]
                save()
            status, headers = _put(state["session"], b"", {"Content-Range": f"bytes */{size}"})
            if status in (200, 201):
                state["done"] = True
                save()
                return []
            if status == 404 or status == 410:   # session gone: start over
                state.pop("session", None)
                plan = {}
                raise ApiError(status, "upload_failed", "upload session expired")
            offset = _gcs_next(headers) if status == 308 else 0
            while offset < size:
                if before_part is not None:
                    before_part(offset // chunk + 1)
                data = _read(path, offset, chunk)
                end = offset + len(data) - 1
                status, headers = _put(state["session"], data, {"Content-Range": f"bytes {offset}-{end}/{size}"})
                if status in (200, 201):
                    state["done"] = True
                    save()
                    return []
                if status != 308:
                    raise ApiError(status, "upload_failed", f"chunk at {offset} answered HTTP {status}")
                offset = _gcs_next(headers)
                attempt = 0
                if progress:
                    progress(offset, size)
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ApiError):
            if attempt == retries:
                raise
            sleep(_backoff(attempt))
            attempt += 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, dest: str | Path, sha256: str | None = None, size: int | None = None, retries: int = 6,
             sleep: Callable[[float], None] = time.sleep, timeout: float = 600) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".part")
    for attempt in range(retries + 1):
        try:
            digest, written = hashlib.sha256(), 0
            with urllib.request.urlopen(url, timeout=timeout) as response, temporary.open("wb") as out:
                for chunk in iter(lambda: response.read(CHUNK), b""):
                    digest.update(chunk)
                    out.write(chunk)
                    written += len(chunk)
            if size is not None and written != size:
                raise ApiError(0, "download_size", f"got {written} bytes, expected {size}")
            if sha256 is not None and digest.hexdigest() != sha256:
                raise ApiError(0, "download_hash", "SHA-256 mismatch")
            os.replace(temporary, dest)
            return dest
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError, ApiError):
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                raise
            sleep(_backoff(attempt))
    raise AssertionError("unreachable")
