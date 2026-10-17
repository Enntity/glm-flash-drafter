"""Coordinator control plane: invites, site tokens, shard leases, deltas, versions, heartbeats, audit log.

It never loads weights: bytes move between sites/validator and object storage through signed URLs
(:mod:`.storage`).  State is one SQLite database (WAL), durable across restarts.  Stdlib only.

Delta lifecycle: ``uploading`` -> ``pending`` (received; credited) -> ``validating`` (claimed by the validator;
reclaimed after ``claim_timeout_s``) -> ``accepted`` | ``held`` (operator review) | ``quarantined`` (rejected,
object kept; re-validated automatically when the validator version changes) -> ``merged`` (into a version).
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterator

from . import terms

CLIENT_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DISPLAY = re.compile(r"^[\w][\w .'-]{0,31}$", re.UNICODE)
TAG = re.compile(r"^[\w.:+-]{1,64}$", re.UNICODE)
VERSION_TEXT = re.compile(r"^[\w.+-]{1,40}$")
PHASES = ("idle", "setup", "canary", "claiming", "generating", "capturing", "pulling", "training", "checking",
          "uploading", "waiting", "cleaning", "error")
HEARTBEAT_FIELDS = {   # name: (type, min, max); anything else is refused (no free text, no hostnames)
    "phase": (str, None, None), "round": (int, 0, 10**9), "version": (int, 0, 10**9), "tokens": (float, 0, 1e15),
    "tokens_per_s": (float, 0, 1e9), "gpu_w": (float, 0, 1e5), "mfu": (float, 0, 1), "gpus": (int, 0, 64),
    "step": (int, 0, 10**12), "client": (str, None, None), "shard_id": (str, None, None),
}
DELTA_META = {"tokens": (float, 0, 1e15), "steps": (int, 1, 10**9), "step_start": (int, 0, 10**12),
              "step_end": (int, 1, 10**12), "inner_steps": (int, 1, 10**9), "delta_dtype": (str, None, None),
              "trainer": (str, None, None), "config_sha256": (str, None, None),
              "validator_version": (str, None, None), "client": (str, None, None)}
RECEIVED = ("pending", "validating", "accepted", "held", "quarantined", "merged")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS invites (id INTEGER PRIMARY KEY, code_hash TEXT UNIQUE NOT NULL, label TEXT,
    created REAL, expires REAL, used REAL, site_id INTEGER, revoked REAL);
CREATE TABLE IF NOT EXISTS sites (id INTEGER PRIMARY KEY, public_id TEXT UNIQUE NOT NULL, token_hash TEXT UNIQUE NOT NULL,
    role TEXT NOT NULL, display_name TEXT, label TEXT, created REAL, revoked REAL, terms_version TEXT,
    terms_sha256 TEXT, terms_accepted REAL, canary_key TEXT, last_seen REAL, heartbeat TEXT);
CREATE TABLE IF NOT EXISTS heartbeats (site_id INTEGER, ts REAL, tokens REAL, gpus INTEGER);
CREATE INDEX IF NOT EXISTS heartbeats_ts ON heartbeats (ts);
CREATE TABLE IF NOT EXISTS shards (id TEXT PRIMARY KEY, recipe TEXT NOT NULL, state TEXT NOT NULL, lease_site INTEGER,
    lease_expires REAL, holders TEXT NOT NULL DEFAULT '[]', created REAL);
CREATE TABLE IF NOT EXISTS versions (version INTEGER PRIMARY KEY, object_key TEXT, sha256 TEXT, size INTEGER,
    dtype TEXT, created REAL, evals TEXT, merged TEXT, heldout_id TEXT, validator_version TEXT, deleted REAL);
CREATE TABLE IF NOT EXISTS uploads (id TEXT PRIMARY KEY, kind TEXT, ref TEXT, object_key TEXT, size INTEGER,
    sha256 TEXT, backend TEXT, created REAL, completed REAL);
CREATE TABLE IF NOT EXISTS deltas (id TEXT PRIMARY KEY, client_id TEXT NOT NULL, site_id INTEGER NOT NULL,
    base_version INTEGER, base_sha256 TEXT, mode TEXT, shard_id TEXT, tag TEXT, canary INTEGER, meta TEXT,
    attestation TEXT, object_key TEXT, size INTEGER, sha256 TEXT, upload_id TEXT, state TEXT, reason_code TEXT,
    reason TEXT, verdict TEXT, validator_version TEXT, claimed REAL, created REAL, received REAL, updated REAL,
    merged_into INTEGER, weight REAL, UNIQUE (site_id, client_id));
CREATE INDEX IF NOT EXISTS deltas_state ON deltas (state);
CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY, ts REAL, event TEXT, actor TEXT, subject TEXT, detail TEXT);
"""


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


@dataclasses.dataclass
class Config:
    public_url: str = "http://127.0.0.1:8080/drafter"   # where sites reach the service (invite links, local URLs)
    max_delta_bytes: int = 5 * 2**30
    max_weights_bytes: int = 6 * 2**30
    url_expiry_s: int = 3600
    lease_s: float = 2 * 3600.0
    lease_grace_s: float = 2 * 3600.0
    keep_globals: int = 8              # newest versions always kept (plus every version still referenced)
    pin_heartbeat_s: float = 12 * 3600.0  # a version a live site trains from is kept this long after its heartbeat
    claim_timeout_s: float = 2 * 3600.0
    invite_hours: float = 72.0
    online_s: float = 900.0
    heartbeat_s: int = 120
    merge_canaries: bool = False

    @classmethod
    def from_env(cls, env: dict[str, str]) -> "Config":
        values = {}
        for field in dataclasses.fields(cls):
            raw = env.get("DRAFTER_" + field.name.upper())
            if raw is not None:
                values[field.name] = (raw.lower() in ("1", "true", "yes")) if field.type in (bool, "bool") else (
                    type(field.default)(raw))
        return cls(**values)


def _typed(body: dict[str, Any], spec: dict[str, tuple], what: str) -> dict[str, Any]:
    if not isinstance(body, dict):
        raise ApiError(400, "bad_request", f"{what} must be an object")
    unknown = set(body) - set(spec)
    if unknown:
        raise ApiError(400, "bad_request", f"{what}: unknown fields {sorted(unknown)}")
    out = {}
    for key, value in body.items():
        kind, low, high = spec[key]
        if value is None:
            continue
        if kind is float and isinstance(value, int) and not isinstance(value, bool):
            value = float(value)
        if not isinstance(value, kind) or isinstance(value, bool):
            raise ApiError(400, "bad_request", f"{what}.{key} must be {kind.__name__}")
        if kind is str and not VERSION_TEXT.match(value) and not HEX64.match(value):
            raise ApiError(400, "bad_request", f"{what}.{key}: unexpected characters")
        if low is not None and not low <= value <= high:
            raise ApiError(400, "bad_request", f"{what}.{key} out of range")
        out[key] = value
    return out


class Service:
    def __init__(self, db_path: str | Path, storage, config: Config | None = None, clock=time.time):
        self.db_path, self.storage, self.config, self.now = str(db_path), storage, config or Config(), clock
        self._lock = threading.Lock()
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(SCHEMA)

    # ------------------------------------------------------------------ db helpers
    def _connect(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextlib.contextmanager
    def tx(self) -> Iterator[sqlite3.Connection]:
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        finally:
            db.close()

    def _meta(self, db, key: str, default: Any = None) -> Any:
        row = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(row["value"]) if row else default

    def _set_meta(self, db, key: str, value: Any) -> None:
        db.execute("INSERT INTO meta (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                   (key, json.dumps(value)))

    def _audit(self, db, event: str, actor: str, subject: str = "", **detail: Any) -> None:
        db.execute("INSERT INTO audit (ts, event, actor, subject, detail) VALUES (?, ?, ?, ?, ?)",
                   (self.now(), event, actor, subject, json.dumps(detail, sort_keys=True)))

    @staticmethod
    def _actor(site) -> str:
        return f"{site['role']}:{site['public_id']}"

    def _current(self, db) -> sqlite3.Row | None:
        return db.execute("SELECT * FROM versions ORDER BY version DESC LIMIT 1").fetchone()

    # ------------------------------------------------------------------ auth
    def authenticate(self, token: str | None, role: str) -> sqlite3.Row:
        if not token:
            raise ApiError(401, "auth_required", "missing bearer token")
        with contextlib.closing(self._connect()) as db:
            site = db.execute("SELECT * FROM sites WHERE token_hash=?", (terms.token_hash(token),)).fetchone()
        if site is None or site["revoked"] or site["role"] != role:
            raise ApiError(401, "auth_invalid", "invalid or revoked token")
        return site

    # ------------------------------------------------------------------ admin
    def issue_invite(self, label: str = "", hours: float | None = None) -> dict[str, Any]:
        code = "inv_" + secrets.token_urlsafe(18)
        expires = self.now() + 3600 * (hours if hours is not None else self.config.invite_hours)
        with self.tx() as db:
            cursor = db.execute("INSERT INTO invites (code_hash, label, created, expires) VALUES (?, ?, ?, ?)",
                                (terms.token_hash(code), label[:80], self.now(), expires))
            self._audit(db, "invite.issued", "admin", str(cursor.lastrowid), label=label[:80], expires=expires)
        # The code travels in the URL fragment: browsers and proxies never send it in requests or logs.
        return {"id": cursor.lastrowid, "code": code, "expires": expires,
                "url": f"{self.config.public_url}/join#invite={code}"}

    def revoke_invite(self, invite_id: int) -> None:
        with self.tx() as db:
            if not db.execute("UPDATE invites SET revoked=? WHERE id=? AND revoked IS NULL",
                              (self.now(), invite_id)).rowcount:
                raise ApiError(404, "not_found", "no such active invite")
            self._audit(db, "invite.revoked", "admin", str(invite_id))

    def list_invites(self) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as db:
            return [dict(r) for r in db.execute(
                "SELECT id, label, created, expires, used, revoked, site_id FROM invites ORDER BY id")]

    def issue_validator(self, label: str = "validator") -> dict[str, Any]:
        token = "dval_" + secrets.token_urlsafe(32)
        public_id = secrets.token_hex(6)
        with self.tx() as db:
            db.execute("INSERT INTO sites (public_id, token_hash, role, label, created) VALUES (?, ?, 'validator', ?, ?)",
                       (public_id, terms.token_hash(token), label[:80], self.now()))
            self._audit(db, "validator.issued", "admin", public_id, label=label[:80])
        return {"token": token, "public_id": public_id}

    def revoke_site(self, public_id: str) -> None:
        with self.tx() as db:
            if not db.execute("UPDATE sites SET revoked=? WHERE public_id=? AND revoked IS NULL",
                              (self.now(), public_id)).rowcount:
                raise ApiError(404, "not_found", "no such active site")
            db.execute("UPDATE shards SET state='open', lease_site=NULL WHERE state='leased' AND lease_site="
                       "(SELECT id FROM sites WHERE public_id=?)", (public_id,))
            self._audit(db, "token.revoked", "admin", public_id)

    def list_sites(self) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as db:
            return [dict(r) for r in db.execute(
                "SELECT public_id, role, display_name, label, created, revoked, terms_version, last_seen, canary_key "
                "FROM sites ORDER BY id")]

    def add_shards(self, recipes: list[dict[str, Any]]) -> list[str]:
        with self.tx() as db:
            start = db.execute("SELECT COUNT(*) AS n FROM shards").fetchone()["n"]
            ids = []
            for offset, recipe in enumerate(recipes):
                if not isinstance(recipe, dict) or "builder" not in recipe:
                    raise ApiError(400, "bad_request", "a recipe needs a builder")
                shard_id = f"s{start + offset:06d}"
                db.execute("INSERT INTO shards (id, recipe, state, created) VALUES (?, ?, 'open', ?)",
                           (shard_id, json.dumps(recipe, sort_keys=True), self.now()))
                ids.append(shard_id)
            self._audit(db, "shards.added", "admin", f"{len(ids)}", first=ids[:1], last=ids[-1:])
        return ids

    SETTABLE = ("requirements", "training", "benchmarks", "wanted_notes")

    def set_setting(self, key: str, value: Any, actor: str = "admin") -> None:
        if key not in self.SETTABLE + ("norm_bounds",):
            raise ApiError(400, "bad_request", f"unknown setting {key!r}")
        with self.tx() as db:
            self._set_meta(db, key, value)
            self._audit(db, f"setting.{key}", actor)

    def get_setting(self, key: str) -> Any:
        with contextlib.closing(self._connect()) as db:
            return self._meta(db, key)

    def release_held(self, delta_id: str, decision: str, weight_factor: float = 1.0) -> None:
        if decision not in ("accept", "reject"):
            raise ApiError(400, "bad_request", "decision must be accept or reject")
        with self.tx() as db:
            row = db.execute("SELECT * FROM deltas WHERE id=? AND state='held'", (delta_id,)).fetchone()
            if row is None:
                raise ApiError(404, "not_found", "no such held delta")
            verdict = json.loads(row["verdict"] or "{}") | {"operator": decision, "weight_factor": weight_factor}
            db.execute("UPDATE deltas SET state=?, verdict=?, updated=? WHERE id=?",
                       ("accepted" if decision == "accept" else "quarantined", json.dumps(verdict), self.now(),
                        delta_id))
            self._audit(db, f"delta.operator_{decision}", "admin", delta_id, weight_factor=weight_factor)

    def audit_log(self, limit: int = 200) -> list[dict[str, Any]]:
        with contextlib.closing(self._connect()) as db:
            return [dict(r) | {"detail": json.loads(r["detail"])} for r in db.execute(
                "SELECT * FROM audit ORDER BY id DESC LIMIT ?", (limit,))]

    # ------------------------------------------------------------------ sites
    def terms(self) -> dict[str, Any]:
        return {"version": terms.TERMS_VERSION, "sha256": terms.TERMS_SHA256, "text": terms.TERMS_TEXT,
                "attestation": {"version": terms.ATTESTATION_VERSION, "sha256": terms.ATTESTATION_SHA256,
                                "text": terms.ATTESTATION_TEXT}}

    def redeem(self, code: str, display_name: str | None, accept_terms: str) -> dict[str, Any]:
        if accept_terms != terms.TERMS_SHA256:
            raise ApiError(400, "terms_required", "accept the current participation terms (GET /api/v1/terms)")
        if display_name is not None and not DISPLAY.match(display_name):
            raise ApiError(400, "bad_display_name", "display name: 1-32 letters, digits, spaces, . ' - or _")
        token, public_id = "dsite_" + secrets.token_urlsafe(32), secrets.token_hex(6)
        with self.tx() as db:
            invite = db.execute("SELECT * FROM invites WHERE code_hash=?", (terms.token_hash(str(code)),)).fetchone()
            if invite is None or invite["used"] or invite["revoked"] or invite["expires"] < self.now():
                raise ApiError(403, "invite_invalid", "this invite is unknown, already used, revoked or expired; "
                                                      "ask the operator for a new one")
            cursor = db.execute(
                "INSERT INTO sites (public_id, token_hash, role, display_name, created, terms_version, terms_sha256, "
                "terms_accepted) VALUES (?, ?, 'site', ?, ?, ?, ?, ?)",
                (public_id, terms.token_hash(token), display_name, self.now(), terms.TERMS_VERSION,
                 terms.TERMS_SHA256, self.now()))
            db.execute("UPDATE invites SET used=?, site_id=? WHERE id=?", (self.now(), cursor.lastrowid, invite["id"]))
            self._audit(db, "invite.redeemed", f"site:{public_id}", str(invite["id"]))
            self._audit(db, "site.registered", f"site:{public_id}", public_id, terms_version=terms.TERMS_VERSION,
                        anonymous=display_name is None)
        return {"token": token, "public_id": public_id, "display_name": display_name}

    def set_display_name(self, site, display_name: str | None) -> None:
        if display_name is not None and not DISPLAY.match(display_name):
            raise ApiError(400, "bad_display_name", "display name: 1-32 letters, digits, spaces, . ' - or _")
        with self.tx() as db:
            db.execute("UPDATE sites SET display_name=? WHERE id=?", (display_name, site["id"]))

    def _canary_key(self, db) -> str:
        req = self._meta(db, "requirements", {})
        return f"{req.get('trainer')}|{req.get('config_sha256')}|{self._meta(db, 'validator_version')}"

    def requirements(self, site) -> dict[str, Any]:
        """What a site must match before it trains (pinning), and the current global."""

        with contextlib.closing(self._connect()) as db:
            current = self._current(db)
            req = self._meta(db, "requirements", {})
            fresh = db.execute("SELECT canary_key FROM sites WHERE id=?", (site["id"],)).fetchone()
            return {
                "version": current["version"] if current else None,
                "sha256": current["sha256"] if current else None,
                "size": current["size"] if current else None, "dtype": current["dtype"] if current else None,
                "trainer": req.get("trainer"), "config_sha256": req.get("config_sha256"),
                "client_min": req.get("client_min"), "validator_version": self._meta(db, "validator_version"),
                "canary_required": fresh["canary_key"] != self._canary_key(db),
                "training": self._meta(db, "training", {}), "norm_bounds": self._meta(db, "norm_bounds"),
                "heartbeat_s": self.config.heartbeat_s, "terms_sha256": terms.TERMS_SHA256,
                "attestation_sha256": terms.ATTESTATION_SHA256,
            }

    def version_url(self, version: int | None = None) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            row = (self._current(db) if version is None else
                   db.execute("SELECT * FROM versions WHERE version=?", (version,)).fetchone())
        if row is None or row["deleted"]:
            raise ApiError(404, "no_version", "no such retained global version")
        return {"version": row["version"], "sha256": row["sha256"], "size": row["size"], "dtype": row["dtype"],
                "url": self.storage.download_url(row["object_key"], self.config.url_expiry_s)}

    def _expire_leases(self, db) -> None:
        for row in db.execute("SELECT id, lease_site FROM shards WHERE state='leased' AND lease_expires < ?",
                              (self.now() - self.config.lease_grace_s,)).fetchall():
            db.execute("UPDATE shards SET state='open', lease_site=NULL WHERE id=?", (row["id"],))
            self._audit(db, "shard.lease_expired", "coordinator", row["id"])

    def claim_shard(self, site) -> dict[str, Any] | None:
        with self.tx() as db:
            self._expire_leases(db)
            row = db.execute("SELECT * FROM shards WHERE state='leased' AND lease_site=?", (site["id"],)).fetchone()
            if row is None:
                row = db.execute("SELECT * FROM shards WHERE state='open' ORDER BY id LIMIT 1").fetchone()
                if row is None:
                    return None
                holders = sorted(set(json.loads(row["holders"])) | {site["public_id"]})
                db.execute("UPDATE shards SET state='leased', lease_site=?, lease_expires=?, holders=? WHERE id=?",
                           (site["id"], self.now() + self.config.lease_s, json.dumps(holders), row["id"]))
                self._audit(db, "shard.leased", self._actor(site), row["id"])
            else:
                db.execute("UPDATE shards SET lease_expires=? WHERE id=?", (self.now() + self.config.lease_s, row["id"]))
            return {"id": row["id"], "recipe": json.loads(row["recipe"]),
                    "lease_expires": self.now() + self.config.lease_s}

    def release_shard(self, site, shard_id: str) -> None:
        with self.tx() as db:
            db.execute("UPDATE shards SET state='open', lease_site=NULL WHERE id=? AND state='leased' AND lease_site=?",
                       (shard_id, site["id"]))

    def heartbeat(self, site, body: dict[str, Any]) -> dict[str, Any]:
        beat = _typed(body, HEARTBEAT_FIELDS, "heartbeat")
        if beat.get("phase") not in PHASES:
            raise ApiError(400, "bad_request", f"heartbeat.phase must be one of {PHASES}")
        now = self.now()
        with self.tx() as db:
            if beat.get("shard_id"):   # lease extension while the site works on it
                db.execute("UPDATE shards SET lease_expires=? WHERE id=? AND state='leased' AND lease_site=?",
                           (now + self.config.lease_s, beat["shard_id"], site["id"]))
            db.execute("UPDATE sites SET last_seen=?, heartbeat=? WHERE id=?", (now, json.dumps(beat), site["id"]))
            db.execute("INSERT INTO heartbeats (site_id, ts, tokens, gpus) VALUES (?, ?, ?, ?)",
                       (site["id"], now, beat.get("tokens", 0.0), beat.get("gpus", 0)))
            db.execute("DELETE FROM heartbeats WHERE ts < ?", (now - 8 * 86400,))
        return {"ok": True, "next_s": self.config.heartbeat_s}

    def init_delta(self, site, body: dict[str, Any]) -> dict[str, Any]:
        """Register an upload; every check that can fail happens here, before any bytes move."""

        if not isinstance(body, dict):
            raise ApiError(400, "bad_request", "body must be an object")
        client_id = str(body.get("client_id", ""))
        if not CLIENT_ID.match(client_id):
            raise ApiError(400, "bad_request", "client_id: 8-64 of A-Z a-z 0-9 _ -")
        sha, base_sha = str(body.get("sha256", "")), str(body.get("base_sha256", ""))
        size, base_version = body.get("size"), body.get("base_version")
        if not HEX64.match(sha) or not HEX64.match(base_sha) or not isinstance(size, int) or not isinstance(
                base_version, int):
            raise ApiError(400, "bad_request", "sha256, base_sha256 (hex), size and base_version (int) required")
        mode, shard_id, tag = body.get("mode"), body.get("shard_id"), body.get("tag")
        if mode not in ("recipe", "byo") or (mode == "recipe") != bool(shard_id) or (
                tag is not None and not TAG.match(str(tag))):
            raise ApiError(400, "bad_request", "mode recipe needs shard_id; mode byo takes an optional tag "
                                               "(1-64 of letters, digits, . : + - _)")
        meta = _typed(body.get("meta", {}), DELTA_META, "meta")
        with self.tx() as db:
            existing = db.execute("SELECT * FROM deltas WHERE site_id=? AND client_id=?",
                                  (site["id"], client_id)).fetchone()
            if existing is not None:   # idempotent retry / resume
                if (existing["sha256"], existing["size"]) != (sha, size):
                    raise ApiError(409, "client_id_reused", "this client_id was used for a different file")
                return self._delta_reply(db, existing, resume=True)
            same = db.execute("SELECT * FROM deltas WHERE sha256=? AND state != 'uploading'", (sha,)).fetchone()
            if same is not None:
                if same["site_id"] != site["id"]:
                    raise ApiError(409, "duplicate", "this exact delta was already uploaded by another site")
                return self._delta_reply(db, same, duplicate=True)
            req = self._meta(db, "requirements", {})   # what makes a delta mergeable at all (the validator version
            wanted = {"trainer": req.get("trainer"), "config_sha256": req.get("config_sha256")}   # is not: it re-checks)
            got = {k: meta.get(k) for k in wanted}
            if any(v is not None and got[k] != v for k, v in wanted.items()):
                raise ApiError(409, "version_mismatch",
                               f"this delta was made with {got}; the network requires {wanted}. Run "
                               "`drafter-join update`, then `drafter-join run` (a short canary runs first).")
            base = db.execute("SELECT * FROM versions WHERE version=?", (base_version,)).fetchone()
            if base is None or base["sha256"] != base_sha:
                raise ApiError(409, "base_unknown", "the base version/hash is not a global this coordinator published")
            if base["deleted"]:
                raise ApiError(409, "base_retired", f"global v{base_version} is no longer retained")
            if not 0 < size <= self.config.max_delta_bytes:
                raise ApiError(413, "too_large", f"delta size must be 1..{self.config.max_delta_bytes} bytes")
            if mode == "recipe":
                shard = db.execute("SELECT holders FROM shards WHERE id=?", (shard_id,)).fetchone()
                if shard is None or site["public_id"] not in json.loads(shard["holders"]):
                    raise ApiError(409, "shard_not_leased", "this site never leased that shard")
            delta = {"client_id": client_id, "sha256": sha, "base_sha256": base_sha, "mode": mode,
                     "shard_id": shard_id, "tag": tag}
            if not terms.verify_attestation(site["token_hash"], delta, body.get("attestation")):
                raise ApiError(400, "attestation_invalid", "missing or invalid data-source attestation")
            delta_id = "d_" + secrets.token_hex(8)
            key = f"deltas/{site['public_id']}/{delta_id}.safetensors"
            plan = self.storage.plan_upload(key, size, self.config.url_expiry_s)
            upload_id = "u_" + secrets.token_hex(8)
            db.execute("INSERT INTO uploads (id, kind, ref, object_key, size, sha256, backend, created) VALUES "
                       "(?, 'delta', ?, ?, ?, ?, ?, ?)",
                       (upload_id, delta_id, key, size, sha, json.dumps(plan.pop("backend")), self.now()))
            db.execute(
                "INSERT INTO deltas (id, client_id, site_id, base_version, base_sha256, mode, shard_id, tag, canary, "
                "meta, attestation, object_key, size, sha256, upload_id, state, created, updated) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'uploading', ?, ?)",
                (delta_id, client_id, site["id"], base_version, base_sha, mode, shard_id, tag,
                 int(bool(body.get("canary"))), json.dumps(meta), json.dumps(body.get("attestation")), key, size, sha,
                 upload_id, self.now(), self.now()))
            row = db.execute("SELECT * FROM deltas WHERE id=?", (delta_id,)).fetchone()
            reply = self._delta_reply(db, row)
            reply["upload"] = plan
            return reply

    def _delta_reply(self, db, row, resume: bool = False, duplicate: bool = False) -> dict[str, Any]:
        reply = {"delta_id": row["id"], "state": row["state"], "duplicate": duplicate, "upload_id": row["upload_id"]}
        if resume and row["state"] == "uploading":
            upload = db.execute("SELECT * FROM uploads WHERE id=?", (row["upload_id"],)).fetchone()
            reply["upload"] = self.storage.refresh_upload(upload["object_key"], upload["size"],
                                                          json.loads(upload["backend"]), [],
                                                          self.config.url_expiry_s)
        return reply

    def refresh_upload(self, owner, upload_id: str, numbers: list[int]) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            upload = db.execute("SELECT * FROM uploads WHERE id=? AND completed IS NULL", (upload_id,)).fetchone()
            if upload is None or not self._owns(db, owner, upload):
                raise ApiError(404, "not_found", "no such open upload")
        return self.storage.refresh_upload(upload["object_key"], upload["size"], json.loads(upload["backend"]),
                                           [int(n) for n in numbers][:1000], self.config.url_expiry_s)

    def _owns(self, db, owner, upload) -> bool:
        if upload["kind"] == "delta":
            row = db.execute("SELECT site_id FROM deltas WHERE id=?", (upload["ref"],)).fetchone()
            return row is not None and row["site_id"] == owner["id"]
        return owner["role"] == "validator"

    def complete_delta(self, site, delta_id: str, parts: list[dict[str, Any]]) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            row = db.execute("SELECT * FROM deltas WHERE id=? AND site_id=?", (delta_id, site["id"])).fetchone()
            if row is None:
                raise ApiError(404, "not_found", "no such delta")
            if row["state"] != "uploading":
                return self.delta_status(site, delta_id)
            upload = db.execute("SELECT * FROM uploads WHERE id=?", (row["upload_id"],)).fetchone()
        try:
            self.storage.complete_upload(upload["object_key"], upload["size"], json.loads(upload["backend"]), parts)
        except Exception as error:  # noqa: BLE001 - reported to the client, which can retry parts
            raise ApiError(409, "upload_incomplete", f"upload not complete: {error}") from error
        with self.tx() as db:
            db.execute("UPDATE uploads SET completed=? WHERE id=?", (self.now(), upload["id"]))
            db.execute("UPDATE deltas SET state='pending', received=?, updated=? WHERE id=? AND state='uploading'",
                       (self.now(), self.now(), delta_id))
            self._audit(db, "delta.received", self._actor(site), delta_id, size=row["size"], mode=row["mode"],
                        canary=bool(row["canary"]))
        return self.delta_status(site, delta_id)

    def delta_status(self, site, delta_id: str) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            row = db.execute("SELECT * FROM deltas WHERE id=? AND site_id=?", (delta_id, site["id"])).fetchone()
        if row is None:
            raise ApiError(404, "not_found", "no such delta")
        return {"delta_id": row["id"], "state": row["state"], "reason_code": row["reason_code"],
                "reason": row["reason"], "verdict": json.loads(row["verdict"] or "null"), "weight": row["weight"],
                "merged_into": row["merged_into"], "canary": bool(row["canary"]), "base_version": row["base_version"]}

    def me(self, site) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            rows = db.execute("SELECT id FROM deltas WHERE site_id=? ORDER BY created DESC LIMIT 50",
                              (site["id"],)).fetchall()
            stats = self._site_stats(db).get(site["id"], {})
        return {"public_id": site["public_id"], "display_name": site["display_name"], **stats,
                "deltas": [self.delta_status(site, r["id"]) for r in rows]}

    # ------------------------------------------------------------------ validator
    def validator_hello(self, validator, version: str) -> dict[str, Any]:
        if not VERSION_TEXT.match(str(version)):
            raise ApiError(400, "bad_request", "bad validator version")
        with self.tx() as db:
            previous = self._meta(db, "validator_version")
            if previous != version:
                self._set_meta(db, "validator_version", version)
                requeued = db.execute(
                    "UPDATE deltas SET state='pending', claimed=NULL, updated=? WHERE state IN ('quarantined', 'held') "
                    "AND (validator_version IS NULL OR validator_version != ?)", (self.now(), version)).rowcount
                self._audit(db, "validator.version", self._actor(validator), str(version), previous=previous,
                            requeued=requeued)
        with contextlib.closing(self._connect()) as db:
            current = self._current(db)
        return {"validator_version": version, "requirements": self.get_setting("requirements") or {},
                "current": {"version": current["version"], "sha256": current["sha256"],
                            "created": current["created"]} if current else None}

    def delta_url(self, validator, delta_id: str) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            row = db.execute("SELECT object_key, sha256, size FROM deltas WHERE id=? AND state != 'uploading'",
                             (delta_id,)).fetchone()
        if row is None:
            raise ApiError(404, "not_found", "no such received delta")
        return {"url": self.storage.download_url(row["object_key"], self.config.url_expiry_s),
                "sha256": row["sha256"], "size": row["size"]}

    def pending(self, limit: int = 20) -> list[dict[str, Any]]:
        with self.tx() as db:
            db.execute("UPDATE deltas SET state='pending', claimed=NULL WHERE state='validating' AND claimed < ?",
                       (self.now() - self.config.claim_timeout_s,))
            rows = db.execute("SELECT d.*, s.public_id FROM deltas d JOIN sites s ON s.id=d.site_id "
                              "WHERE d.state='pending' ORDER BY d.canary DESC, d.received LIMIT ?", (limit,)).fetchall()
        return [self._delta_for_validator(r) for r in rows]

    @staticmethod
    def _delta_for_validator(row) -> dict[str, Any]:
        return {"delta_id": row["id"], "site": row["public_id"], "base_version": row["base_version"],
                "base_sha256": row["base_sha256"], "mode": row["mode"], "shard_id": row["shard_id"],
                "tag": row["tag"], "canary": bool(row["canary"]), "meta": json.loads(row["meta"]),
                "size": row["size"], "sha256": row["sha256"], "state": row["state"]}

    def claim_delta(self, validator, delta_id: str) -> dict[str, Any]:
        with self.tx() as db:
            if not db.execute("UPDATE deltas SET state='validating', claimed=?, updated=? WHERE id=? AND state='pending'",
                              (self.now(), self.now(), delta_id)).rowcount:
                raise ApiError(409, "not_pending", "delta is not pending")
            row = db.execute("SELECT d.*, s.public_id FROM deltas d JOIN sites s ON s.id=d.site_id WHERE d.id=?",
                             (delta_id,)).fetchone()
            recipe = None
            if row["shard_id"]:
                shard = db.execute("SELECT recipe FROM shards WHERE id=?", (row["shard_id"],)).fetchone()
                recipe = json.loads(shard["recipe"]) if shard else None
        return {**self._delta_for_validator(row), "recipe": recipe,
                "url": self.storage.download_url(row["object_key"], self.config.url_expiry_s)}

    def verdict(self, validator, delta_id: str, body: dict[str, Any]) -> dict[str, Any]:
        decision = body.get("decision")
        states = {"accept": "accepted", "reduce": "accepted", "hold": "held", "reject": "quarantined"}
        if decision not in states:
            raise ApiError(400, "bad_request", f"decision must be one of {sorted(states)}")
        version = str(body.get("validator_version", ""))
        with self.tx() as db:
            row = db.execute("SELECT * FROM deltas WHERE id=?", (delta_id,)).fetchone()
            if row is None or row["state"] not in ("validating", "pending"):
                raise ApiError(409, "not_validating", "delta is not being validated")
            verdict = {k: body.get(k) for k in ("decision", "metrics", "base_metrics", "checks", "weight_factor",
                                                 "staleness", "message")}
            db.execute("UPDATE deltas SET state=?, reason_code=?, reason=?, verdict=?, validator_version=?, "
                       "updated=? WHERE id=?",
                       (states[decision], body.get("code"), str(body.get("reason", ""))[:2000], json.dumps(verdict),
                        version, self.now(), delta_id))
            if row["canary"] and decision in ("accept", "reduce"):
                db.execute("UPDATE sites SET canary_key=? WHERE id=?", (self._canary_key(db), row["site_id"]))
            self._audit(db, f"delta.{states[decision]}", self._actor(validator), delta_id, code=body.get("code"),
                        canary=bool(row["canary"]))
        return {"state": states[decision]}

    def mergeable(self) -> list[dict[str, Any]]:
        """Accepted, unmerged deltas (canaries only if ``merge_canaries``)."""

        with contextlib.closing(self._connect()) as db:
            rows = db.execute("SELECT d.*, s.public_id FROM deltas d JOIN sites s ON s.id=d.site_id "
                              "WHERE d.state='accepted' AND d.merged_into IS NULL" +
                              ("" if self.config.merge_canaries else " AND d.canary=0") +
                              " ORDER BY d.received").fetchall()
        return [self._delta_for_validator(r) | {"verdict": json.loads(r["verdict"] or "{}"),
                                                "received": r["received"]} for r in rows]

    def init_version_upload(self, validator, version: int, size: int, sha256: str, dtype: str) -> dict[str, Any]:
        if not HEX64.match(str(sha256)) or dtype not in ("fp32", "bf16") or not isinstance(size, int):
            raise ApiError(400, "bad_request", "sha256 (hex), size (int) and dtype fp32|bf16 required")
        if not 0 < size <= self.config.max_weights_bytes:
            raise ApiError(413, "too_large", "weights too large")
        with self.tx() as db:
            current = self._current(db)
            if version != (current["version"] + 1 if current else 0):
                raise ApiError(409, "version_order", "versions are published in order")
            key = f"globals/v{version:06d}-{sha256[:12]}.safetensors"
            plan = self.storage.plan_upload(key, size, self.config.url_expiry_s)
            upload_id = "u_" + secrets.token_hex(8)
            db.execute("INSERT INTO uploads (id, kind, ref, object_key, size, sha256, backend, created) VALUES "
                       "(?, 'version', ?, ?, ?, ?, ?, ?)",
                       (upload_id, str(version), key, size, sha256, json.dumps(plan.pop("backend")), self.now()))
        return {"upload_id": upload_id, "upload": plan}

    def publish_version(self, validator, version: int, body: dict[str, Any]) -> dict[str, Any]:
        with contextlib.closing(self._connect()) as db:
            upload = db.execute("SELECT * FROM uploads WHERE id=? AND kind='version' AND ref=?",
                                (body.get("upload_id"), str(version))).fetchone()
        if upload is None:
            raise ApiError(404, "not_found", "no such version upload")
        if upload["completed"] is None:
            try:
                self.storage.complete_upload(upload["object_key"], upload["size"], json.loads(upload["backend"]),
                                             body.get("parts") or [])
            except Exception as error:  # noqa: BLE001
                raise ApiError(409, "upload_incomplete", f"upload not complete: {error}") from error
        merged = body.get("merged") or []
        with self.tx() as db:
            current = self._current(db)
            if current is not None and current["version"] >= version:
                return {"version": current["version"], "already": True}
            db.execute("UPDATE uploads SET completed=? WHERE id=?", (self.now(), upload["id"]))
            db.execute("INSERT INTO versions (version, object_key, sha256, size, dtype, created, evals, merged, "
                       "heldout_id, validator_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                       (version, upload["object_key"], upload["sha256"], upload["size"], body.get("dtype", "bf16"),
                        self.now(), json.dumps(body.get("evals") or {}), json.dumps(merged),
                        str(body.get("heldout_id") or "")[:80], self._meta(db, "validator_version")))
            for item in merged:
                db.execute("UPDATE deltas SET state='merged', merged_into=?, weight=?, updated=? WHERE id=? AND "
                           "state='accepted'", (version, float(item.get("weight", 0)), self.now(), item["delta_id"]))
                db.execute("UPDATE shards SET state='done', lease_site=NULL WHERE id=(SELECT shard_id FROM deltas "
                           "WHERE id=?)", (item["delta_id"],))
            if body.get("norm_bounds") is not None:
                self._set_meta(db, "norm_bounds", body["norm_bounds"])
            self._audit(db, "version.published", self._actor(validator), str(version), sha256=upload["sha256"],
                        merged=len(merged))
        self.collect_versions()
        return {"version": version}

    def collect_versions(self) -> list[int]:
        """Delete stored globals nobody can still need: keep the newest ``keep_globals``, every base of an unmerged
        or quarantined delta, and every version a site reported training from within ``pin_heartbeat_s``."""

        with self.tx() as db:
            rows = db.execute("SELECT version, object_key FROM versions WHERE deleted IS NULL ORDER BY version DESC"
                              ).fetchall()
            keep = {r["version"] for r in rows[: self.config.keep_globals]}
            keep |= {r["base_version"] for r in db.execute(
                "SELECT DISTINCT base_version FROM deltas WHERE state != 'merged'")}
            for site in db.execute("SELECT heartbeat FROM sites WHERE last_seen > ?",
                                   (self.now() - self.config.pin_heartbeat_s,)):
                version = json.loads(site["heartbeat"] or "{}").get("version")
                if version is not None:
                    keep.add(version)
            retired = [r for r in rows if r["version"] not in keep]
            for row in retired:
                db.execute("UPDATE versions SET deleted=? WHERE version=?", (self.now(), row["version"]))
        for row in retired:
            self.storage.delete(row["object_key"])
        return [r["version"] for r in retired]

    # ------------------------------------------------------------------ public
    def _site_stats(self, db) -> dict[int, dict[str, Any]]:
        stats: dict[int, dict[str, Any]] = {}
        marks = ",".join("?" * len(RECEIVED))
        for row in db.execute(f"SELECT site_id, state, canary, meta FROM deltas WHERE state IN ({marks})", RECEIVED):
            entry = stats.setdefault(row["site_id"], {"tokens_received": 0.0, "deltas_received": 0,
                                                      "deltas_accepted": 0, "deltas_pending": 0,
                                                      "deltas_quarantined": 0})
            entry["tokens_received"] += float(json.loads(row["meta"]).get("tokens") or 0.0)
            entry["deltas_received"] += 1
            entry["deltas_accepted"] += row["state"] in ("accepted", "merged")
            entry["deltas_pending"] += row["state"] in ("pending", "validating", "held")
            entry["deltas_quarantined"] += row["state"] == "quarantined"
        return stats

    def dashboard(self) -> dict[str, Any]:
        """Everything the public dashboard shows; no tokens, ids, labels or admin data."""

        now = self.now()
        with contextlib.closing(self._connect()) as db:
            current = self._current(db)
            stats = self._site_stats(db)
            sites, online = [], 0
            for row in db.execute("SELECT * FROM sites WHERE role='site' AND revoked IS NULL"):
                beat = json.loads(row["heartbeat"] or "{}")
                live = row["last_seen"] is not None and now - row["last_seen"] < self.config.online_s
                online += live
                entry = stats.get(row["id"], {})
                sites.append({"name": row["display_name"] or "anonymous", "anonymous": row["display_name"] is None,
                              "status": beat.get("phase", "idle") if live else "offline",
                              "tokens_per_s": beat.get("tokens_per_s") if live else None,
                              "gpus": beat.get("gpus") if live else None,
                              "tokens_received": entry.get("tokens_received", 0.0),
                              "deltas_received": entry.get("deltas_received", 0),
                              "deltas_accepted": entry.get("deltas_accepted", 0),
                              "deltas_pending": entry.get("deltas_pending", 0)})
            sites.sort(key=lambda s: (-s["tokens_received"], s["name"]))
            day = {}
            for row in db.execute("SELECT site_id, MIN(tokens) AS lo, MAX(tokens) AS hi FROM heartbeats WHERE ts > ? "
                                  "GROUP BY site_id", (now - 86400,)):
                day[row["site_id"]] = max(0.0, (row["hi"] or 0) - (row["lo"] or 0))
            gpu_hours = 0.0
            last: dict[int, float] = {}
            for row in db.execute("SELECT site_id, ts, gpus FROM heartbeats WHERE ts > ? ORDER BY ts", (now - 86400,)):
                if row["site_id"] in last:
                    gpu_hours += (row["gpus"] or 0) * min(row["ts"] - last[row["site_id"]],
                                                          3 * self.config.heartbeat_s) / 3600
                last[row["site_id"]] = row["ts"]
            curve = [{"version": r["version"], "created": r["created"], "heldout_id": r["heldout_id"],
                      "metrics": json.loads(r["evals"] or "{}").get("metrics", {}),
                      "merged": len(json.loads(r["merged"] or "[]"))}
                     for r in db.execute("SELECT * FROM versions ORDER BY version")]
            reasons = {r["reason_code"] or "unspecified": r["n"] for r in db.execute(
                "SELECT reason_code, COUNT(*) AS n FROM deltas WHERE state IN ('quarantined', 'held') "
                "GROUP BY reason_code")}
            latest = curve[-1]["metrics"] if curve else {}
            weakest = sorted(((v, k) for k, v in latest.items() if "." in k), key=lambda x: x[0])[:3]
            return {
                "generated": now,
                "global": {"version": current["version"], "created": current["created"],
                           "sha256": current["sha256"], "dtype": current["dtype"]} if current else None,
                "totals": {"sites_online": online, "sites": len(sites),
                           "tokens_received": sum(s["tokens_received"] for s in sites),
                           "deltas_received": sum(s["deltas_received"] for s in sites),
                           "deltas_accepted": sum(s["deltas_accepted"] for s in sites),
                           "tokens_per_day": sum(day.values()), "gpu_hours_per_day": round(gpu_hours, 2)},
                "sites": sites, "curve": curve,
                "wanted": {"weakest": [{"category": k, "eal": v} for v, k in weakest],
                           "notes": self._meta(db, "wanted_notes", [])},
                "rejections": reasons, "benchmarks": self._meta(db, "benchmarks"),
                "terms": {"version": terms.TERMS_VERSION, "sha256": terms.TERMS_SHA256},
            }


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
