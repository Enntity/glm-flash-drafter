"""Object storage for weights and deltas: the coordinator only signs URLs; bytes go straight between the site or
validator and the store.

Backends behind one interface (``plan_upload`` / ``refresh_upload`` / ``complete_upload`` / ``download_url`` /
``head`` / ``delete``):

* ``local``: a directory, with URLs signed by the coordinator (HMAC) and served by the coordinator itself
  (``/blob/<token>``).  Tests and development.
* ``s3``: any S3-compatible store (Hetzner Object Storage, ...): presigned multipart upload, every part's
  Content-Length is part of its signature (so the size cap holds), presigned GET.  AWS Signature V4, stdlib only.
* ``gcs``: Google Cloud Storage XML API with V4 signed URLs (service-account RSA key; needs the ``cryptography``
  package): a signed resumable-session initiation whose ``x-goog-content-length-range`` pins the exact size, then
  chunked PUTs to the session; signed GET.

Upload plans returned to clients:

* ``{"protocol": "parts", "part_size", "parts": [{"n", "length", "url"}]}``: PUT each part (any order, retry
  freely), keep each response's ETag, then report them to the coordinator, which completes the upload.
* ``{"protocol": "gcs-resumable", "init_url", "init_headers", "chunk_size"}``: POST ``init_url`` once with
  ``init_headers`` (the ``Location`` response header is the session URL), PUT chunks with ``Content-Range``,
  query progress with ``Content-Range: bytes */<size>``.

Abandoned uploads are aborted by ``abort_upload`` (the coordinator's maintenance).  GCS resumable sessions cannot be
cancelled without their URL, which only the uploader holds; they lapse on their own, and the signed initiation is
short-lived and pinned to "object must not exist" (``x-goog-if-generation-match: 0``), so nothing is overwritten.

Content: sizes are checked here after completion; SHA-256 of every file is declared by the uploader and checked by
whoever downloads it (validator for deltas, sites for globals).
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, BinaryIO

PART_SIZE = 64 * 2**20          # S3/local part size and GCS chunk size (a multiple of 256 KiB)
MAX_PARTS = 10_000
_KEY = re.compile(r"^[A-Za-z0-9._/-]{1,256}$")


class StorageError(RuntimeError):
    pass


def check_key(key: str) -> str:
    if not _KEY.match(key) or ".." in key or key.startswith("/"):
        raise StorageError(f"bad object key {key!r}")
    return key


def part_lengths(size: int, part_size: int) -> list[int]:
    if size <= 0:
        raise StorageError("empty upload")
    count = -(-size // part_size)
    if count > MAX_PARTS:
        raise StorageError("upload has too many parts")
    return [min(part_size, size - i * part_size) for i in range(count)]


def _quote(text: str, safe: str = "-_.~") -> str:
    return urllib.parse.quote(text, safe=safe)


def _request(method: str, url: str, headers: dict[str, str] | None = None, body: bytes = b"",
             timeout: float = 60) -> tuple[int, dict[str, str], bytes]:
    req = urllib.request.Request(url, data=body if method in ("POST", "PUT") else None, method=method,
                                 headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


# --------------------------------------------------------------------------
# local directory (tests, development)
# --------------------------------------------------------------------------

class LocalStorage:
    """Objects under ``root``; URLs are ``<public_url>/blob/<payload>.<hmac>`` handled by :meth:`handle_blob`."""

    name = "local"

    def __init__(self, root: str | Path, public_url: str, secret: bytes | None = None, part_size: int = PART_SIZE,
                 min_free_bytes: int = 512 * 2**20):
        self.root, self.public_url, self.min_free_bytes = Path(root), public_url.rstrip("/"), min_free_bytes
        self.secret = secret or secrets.token_bytes(32)
        self.part_size = part_size
        (self.root / ".uploads").mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / check_key(key)

    def _sign(self, claims: dict[str, Any]) -> str:
        payload = base64.urlsafe_b64encode(json.dumps(claims, sort_keys=True).encode()).decode().rstrip("=")
        mac = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        return f"{self.public_url}/blob/{payload}.{mac}"

    def verify(self, token: str) -> dict[str, Any]:
        payload, _, mac = token.rpartition(".")
        expected = hmac.new(self.secret, payload.encode(), hashlib.sha256).hexdigest()
        if not payload or not hmac.compare_digest(mac, expected):
            raise StorageError("bad signature")
        claims = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        if claims["exp"] < time.time():
            raise StorageError("expired URL")
        return claims

    def plan_upload(self, key: str, size: int, expires_s: int) -> dict[str, Any]:
        upload_id = secrets.token_hex(12)
        (self.root / ".uploads" / upload_id).mkdir(parents=True)
        lengths = part_lengths(size, self.part_size)
        return {"protocol": "parts", "backend": {"upload_id": upload_id}, "part_size": self.part_size,
                "parts": self._part_urls(key, upload_id, lengths, range(1, len(lengths) + 1), expires_s)}

    def _part_urls(self, key, upload_id, lengths, numbers, expires_s):
        exp = time.time() + expires_s
        return [{"n": n, "length": lengths[n - 1],
                 "url": self._sign({"op": "put", "key": key, "upload": upload_id, "n": n, "len": lengths[n - 1],
                                    "exp": exp})} for n in numbers]

    def refresh_upload(self, key: str, size: int, backend: dict[str, Any], numbers: list[int],
                       expires_s: int) -> dict[str, Any]:
        lengths = part_lengths(size, self.part_size)
        numbers = [n for n in numbers if 1 <= n <= len(lengths)] or list(range(1, len(lengths) + 1))
        return {"protocol": "parts", "part_size": self.part_size,
                "parts": self._part_urls(key, backend["upload_id"], lengths, numbers, expires_s)}

    def complete_upload(self, key: str, size: int, backend: dict[str, Any], parts: list[dict[str, Any]]
                        ) -> dict[str, Any]:
        lengths = part_lengths(size, self.part_size)
        folder = self.root / ".uploads" / backend["upload_id"]
        if not folder.is_dir():   # already completed (idempotent)
            if self.head(key) == size:
                return {"size": size}
            raise StorageError("unknown upload")
        given = {int(p["n"]): str(p.get("etag", "")).strip('"') for p in parts}
        for n, length in enumerate(lengths, 1):
            part = folder / f"part-{n:05d}"
            if not part.is_file() or part.stat().st_size != length:
                raise StorageError(f"part {n} missing or incomplete")
            if given.get(n) != hashlib.md5(part.read_bytes()).hexdigest():
                raise StorageError(f"part {n}: ETag mismatch")
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        with temporary.open("wb") as out:
            for n in range(1, len(lengths) + 1):
                with (folder / f"part-{n:05d}").open("rb") as stream:
                    shutil.copyfileobj(stream, out, 8 * 2**20)
        os.replace(temporary, target)
        shutil.rmtree(folder)
        return {"size": size}

    def abort_upload(self, key: str, backend: dict[str, Any]) -> None:
        shutil.rmtree(self.root / ".uploads" / backend.get("upload_id", "-"), ignore_errors=True)

    def download_url(self, key: str, expires_s: int) -> str:
        return self._sign({"op": "get", "key": check_key(key), "exp": time.time() + expires_s})

    def head(self, key: str) -> int | None:
        path = self._path(key)
        return path.stat().st_size if path.is_file() else None

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    # served by the coordinator's HTTP router
    def handle_blob(self, method: str, token: str, body: BinaryIO | None, length: int | None
                    ) -> tuple[int, dict[str, str], Any]:
        """``(status, headers, payload)``; payload is bytes or a file path to stream."""

        try:
            claims = self.verify(token)
        except (StorageError, ValueError, KeyError):
            return 403, {}, b"forbidden"
        if method == "PUT" and claims["op"] == "put":
            if length != claims["len"]:
                return 400, {}, b"wrong part length"
            folder = self.root / ".uploads" / claims["upload"]
            if not folder.is_dir():
                return 404, {}, b"no such upload"
            if shutil.disk_usage(self.root).free < length + self.min_free_bytes:
                return 507, {}, b"insufficient storage"
            part = folder / f"part-{claims['n']:05d}"
            temporary = part.with_name(part.name + f".{secrets.token_hex(4)}.tmp")
            digest, left = hashlib.md5(), length
            with temporary.open("wb") as out:
                while left:
                    chunk = body.read(min(left, 2**20))
                    if not chunk:
                        break
                    digest.update(chunk)
                    out.write(chunk)
                    left -= len(chunk)
            if left:
                temporary.unlink(missing_ok=True)
                return 400, {}, b"truncated part"
            os.replace(temporary, part)
            return 200, {"ETag": f'"{digest.hexdigest()}"'}, b""
        if method in ("GET", "HEAD") and claims["op"] == "get":
            path = self._path(claims["key"])
            if not path.is_file():
                return 404, {}, b"not found"
            return 200, {"Content-Length": str(path.stat().st_size)}, path
        return 405, {}, b"method not allowed"


# --------------------------------------------------------------------------
# S3-compatible (AWS Signature V4)
# --------------------------------------------------------------------------

class S3Storage:
    name = "s3"

    def __init__(self, endpoint: str, bucket: str, access_key: str, secret_key: str, region: str = "fsn1",
                 addressing: str = "path", part_size: int = PART_SIZE, clock=time.time):
        parsed = urllib.parse.urlparse(endpoint.rstrip("/"))
        if parsed.scheme not in ("https", "http") or not parsed.netloc:
            raise StorageError("S3 endpoint must be an http(s) URL")
        self.scheme, self.netloc, self.bucket = parsed.scheme, parsed.netloc, bucket
        self.access_key, self.secret_key, self.region = access_key, secret_key, region
        self.addressing, self.part_size, self.clock = addressing, part_size, clock

    def _host_path(self, key: str) -> tuple[str, str]:
        quoted = "/".join(_quote(part) for part in check_key(key).split("/")) if key else ""
        if self.addressing == "virtual":
            return f"{self.bucket}.{self.netloc}", f"/{quoted}"
        return self.netloc, f"/{_quote(self.bucket)}/{quoted}"

    def _signing_key(self, date: str) -> bytes:
        key = hmac.new(("AWS4" + self.secret_key).encode(), date.encode(), hashlib.sha256).digest()
        for part in (self.region, "s3", "aws4_request"):
            key = hmac.new(key, part.encode(), hashlib.sha256).digest()
        return key

    def _signature(self, method, path, query, headers, payload_hash, amz_date):
        date = amz_date[:8]
        names = sorted(headers)
        canonical = "\n".join([
            method, path, "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items())),
            "".join(f"{n}:{headers[n].strip()}\n" for n in names), ";".join(names), payload_hash])
        scope = f"{date}/{self.region}/s3/aws4_request"
        to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope, hashlib.sha256(canonical.encode()).hexdigest()])
        return scope, ";".join(names), hmac.new(self._signing_key(date), to_sign.encode(), hashlib.sha256).hexdigest()

    def presign(self, method: str, key: str, expires_s: int, query: dict[str, str] | None = None,
                headers: dict[str, str] | None = None) -> str:
        host, path = self._host_path(key)
        amz_date = dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        signed = {"host": host, **{k.lower(): v for k, v in (headers or {}).items()}}
        query = dict(query or {})
        query.update({"X-Amz-Algorithm": "AWS4-HMAC-SHA256",
                      "X-Amz-Credential": f"{self.access_key}/{amz_date[:8]}/{self.region}/s3/aws4_request",
                      "X-Amz-Date": amz_date, "X-Amz-Expires": str(int(expires_s)),
                      "X-Amz-SignedHeaders": ";".join(sorted(signed))})
        _, _, signature = self._signature(method, path, query, signed, "UNSIGNED-PAYLOAD", amz_date)
        query["X-Amz-Signature"] = signature
        return f"{self.scheme}://{host}{path}?" + "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items()))

    def _call(self, method: str, key: str, query: dict[str, str] | None = None, body: bytes = b""
              ) -> tuple[int, dict[str, str], bytes]:
        host, path = self._host_path(key)
        amz_date = dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        payload_hash = hashlib.sha256(body).hexdigest()
        headers = {"host": host, "x-amz-content-sha256": payload_hash, "x-amz-date": amz_date}
        scope, names, signature = self._signature(method, path, query or {}, headers, payload_hash, amz_date)
        headers["Authorization"] = (f"AWS4-HMAC-SHA256 Credential={self.access_key}/{scope}, "
                                    f"SignedHeaders={names}, Signature={signature}")
        qs = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted((query or {}).items()))
        return _request(method, f"{self.scheme}://{host}{path}" + (f"?{qs}" if qs else ""),
                        {k: v for k, v in headers.items() if k != "host"}, body)

    def plan_upload(self, key: str, size: int, expires_s: int) -> dict[str, Any]:
        lengths = part_lengths(size, self.part_size)
        status, _, body = self._call("POST", key, {"uploads": ""})
        if status != 200:
            raise StorageError(f"create multipart upload: HTTP {status}")
        upload_id = next(el.text for el in ET.fromstring(body).iter() if el.tag.endswith("UploadId"))
        return {"protocol": "parts", "backend": {"upload_id": upload_id}, "part_size": self.part_size,
                "parts": self._part_urls(key, upload_id, lengths, range(1, len(lengths) + 1), expires_s)}

    def _part_urls(self, key, upload_id, lengths, numbers, expires_s):
        return [{"n": n, "length": lengths[n - 1],
                 "url": self.presign("PUT", key, expires_s, {"partNumber": str(n), "uploadId": upload_id},
                                     {"content-length": str(lengths[n - 1])})} for n in numbers]

    def refresh_upload(self, key, size, backend, numbers, expires_s):
        lengths = part_lengths(size, self.part_size)
        numbers = [n for n in numbers if 1 <= n <= len(lengths)] or list(range(1, len(lengths) + 1))
        return {"protocol": "parts", "part_size": self.part_size,
                "parts": self._part_urls(key, backend["upload_id"], lengths, numbers, expires_s)}

    def complete_upload(self, key, size, backend, parts):
        if self.head(key) == size:   # already completed (a retry after a crash): idempotent
            return {"size": size}
        count = len(part_lengths(size, self.part_size))
        given = {int(p["n"]): str(p["etag"]) for p in parts}
        if sorted(given) != list(range(1, count + 1)):
            raise StorageError("parts list incomplete")
        xml = "<CompleteMultipartUpload>" + "".join(
            f"<Part><PartNumber>{n}</PartNumber><ETag>{given[n]}</ETag></Part>" for n in sorted(given)
        ) + "</CompleteMultipartUpload>"
        status, _, body = self._call("POST", key, {"uploadId": backend["upload_id"]}, xml.encode())
        if status != 200 or b"<Error>" in body:
            raise StorageError(f"complete multipart upload: HTTP {status}")
        stored = self.head(key)
        if stored != size:
            self.delete(key)
            raise StorageError(f"stored size {stored} != declared {size}")
        return {"size": size}

    def abort_upload(self, key, backend):
        if backend.get("upload_id"):
            self._call("DELETE", key, {"uploadId": backend["upload_id"]})

    def download_url(self, key, expires_s):
        return self.presign("GET", key, expires_s)

    def head(self, key):
        status, headers, _ = self._call("HEAD", key)
        return int(headers.get("Content-Length", 0)) if status == 200 else None

    def delete(self, key):
        self._call("DELETE", key)


# --------------------------------------------------------------------------
# Google Cloud Storage (V4 signed URLs, service-account RSA key)
# --------------------------------------------------------------------------

class GCSStorage:
    name = "gcs"

    def __init__(self, credentials: dict[str, Any], bucket: str, endpoint: str = "https://storage.googleapis.com",
                 chunk_size: int = PART_SIZE, clock=time.time):
        from cryptography.hazmat.primitives import serialization   # optional dependency, GCS only

        self.email = credentials["client_email"]
        self.key = serialization.load_pem_private_key(credentials["private_key"].encode(), password=None)
        parsed = urllib.parse.urlparse(endpoint.rstrip("/"))
        self.scheme, self.netloc, self.bucket = parsed.scheme, parsed.netloc, bucket
        self.chunk_size, self.clock = chunk_size, clock

    def sign(self, method: str, key: str, expires_s: int, headers: dict[str, str] | None = None) -> str:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding

        path = f"/{_quote(self.bucket)}/" + "/".join(_quote(p) for p in check_key(key).split("/"))
        stamp = dt.datetime.fromtimestamp(self.clock(), dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        scope = f"{stamp[:8]}/auto/storage/goog4_request"
        signed = {"host": self.netloc, **{k.lower(): v for k, v in (headers or {}).items()}}
        names = ";".join(sorted(signed))
        query = {"X-Goog-Algorithm": "GOOG4-RSA-SHA256", "X-Goog-Credential": f"{self.email}/{scope}",
                 "X-Goog-Date": stamp, "X-Goog-Expires": str(int(expires_s)), "X-Goog-SignedHeaders": names}
        qs = "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(query.items()))
        canonical = "\n".join([method, path, qs, "".join(f"{n}:{signed[n].strip()}\n" for n in sorted(signed)),
                               names, "UNSIGNED-PAYLOAD"])
        to_sign = "\n".join(["GOOG4-RSA-SHA256", stamp, scope, hashlib.sha256(canonical.encode()).hexdigest()])
        signature = self.key.sign(to_sign.encode(), padding.PKCS1v15(), hashes.SHA256()).hex()
        return f"{self.scheme}://{self.netloc}{path}?{qs}&X-Goog-Signature={signature}"

    INIT_EXPIRY_S = 900   # the signed session initiation is short-lived

    def plan_upload(self, key, size, expires_s):
        headers = {"x-goog-resumable": "start", "x-goog-content-length-range": f"{size},{size}",
                   "x-goog-if-generation-match": "0"}   # create only: never overwrite an existing object
        return {"protocol": "gcs-resumable", "backend": {}, "chunk_size": self.chunk_size,
                "init_url": self.sign("POST", key, min(expires_s, self.INIT_EXPIRY_S), headers),
                "init_headers": headers}

    def refresh_upload(self, key, size, backend, numbers, expires_s):
        plan = self.plan_upload(key, size, expires_s)
        plan.pop("backend")
        return plan

    def complete_upload(self, key, size, backend, parts):
        stored = self.head(key)
        if stored != size:
            if stored is not None:
                self.delete(key)
            raise StorageError(f"stored size {stored} != declared {size}")
        return {"size": size}

    def abort_upload(self, key, backend):
        return None   # sessions lapse; see the module docstring

    def download_url(self, key, expires_s):
        return self.sign("GET", key, expires_s)

    def head(self, key):
        status, headers, _ = _request("HEAD", self.sign("HEAD", key, 300))
        return int(headers.get("Content-Length", 0)) if status == 200 else None

    def delete(self, key):
        _request("DELETE", self.sign("DELETE", key, 300))


def from_env(env: dict[str, str], public_url: str) -> Any:
    """``DRAFTER_STORAGE=local|s3|gcs`` and its settings; credentials only from mounted files."""

    def secret(name: str) -> str:
        path = env.get(name)
        if not path:
            raise StorageError(f"{name} (a file path) is required")
        return Path(path).read_text().strip()

    kind = env.get("DRAFTER_STORAGE", "local")
    if kind == "local":
        root = env.get("DRAFTER_LOCAL_STORAGE_DIR", "/data/objects")
        key_file = Path(root) / ".url-signing-key"
        if not key_file.is_file():
            Path(root).mkdir(parents=True, exist_ok=True)
            key_file.write_bytes(secrets.token_bytes(32))
            key_file.chmod(0o600)
        return LocalStorage(root, public_url, key_file.read_bytes())
    if kind == "s3":
        return S3Storage(env["DRAFTER_S3_ENDPOINT"], env["DRAFTER_BUCKET"], secret("DRAFTER_S3_ACCESS_KEY_FILE"),
                         secret("DRAFTER_S3_SECRET_KEY_FILE"), env.get("DRAFTER_S3_REGION", "fsn1"),
                         env.get("DRAFTER_S3_ADDRESSING", "path"))
    if kind == "gcs":
        return GCSStorage(json.loads(secret("DRAFTER_GCS_CREDENTIALS_FILE")), env["DRAFTER_BUCKET"],
                          env.get("DRAFTER_GCS_ENDPOINT", "https://storage.googleapis.com"))
    raise StorageError(f"DRAFTER_STORAGE={kind!r}: expected local, s3 or gcs")
