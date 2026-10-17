"""CPU tests for the volunteer coordinator (control plane; no weights are loaded).

    python -m pytest tests/test_coordinator.py -v

* invites: one use, expiry, revocation, terms required; the token is saved 0600 and only its hash is stored;
* auth: every site/validator route needs a bearer token of the right role; admin routes are not on the public
  router and the admin router refuses non-loopback addresses;
* signed transfers with the local backend: a part interrupted mid-chunk is re-sent and the upload resumes; tampered,
  expired or wrong-length requests are refused; an incomplete upload cannot complete;
* checks at upload initiation (before any bytes): version pins, unknown or retired base, size cap, attestation,
  shard holder; idempotent client ids and duplicate files;
* staleness: deltas against older retained globals are accepted; retention keeps every referenced version;
* shard leases expire after the grace period, heartbeats extend them, and work on an expired lease is accepted;
* heartbeats accept only whitelisted counters; the public dashboard shows anonymous sites as anonymous and leaks
  no tokens, hashes, ids or labels; state survives a coordinator restart;
* storage signers: AWS Signature V4 presigning matches AWS's published example; GCS V4 signatures verify.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import stat
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _common  # noqa: E402,F401  (repo on sys.path)

from coordinator import terms  # noqa: E402
from coordinator.client import ApiClient, ApiError, download, upload_file  # noqa: E402
from coordinator.server import App, make_server, serve_in_thread  # noqa: E402
from coordinator.service import Config, Service  # noqa: E402
from coordinator.storage import LocalStorage, S3Storage  # noqa: E402

PINS = {"trainer": "scratch-v1", "config_sha256": "c" * 64}


class Clock:
    def __init__(self, t: float = 1_800_000_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class Net:
    """Coordinator (public + admin routers) on loopback with a local object store and a fake clock."""

    def __init__(self, root: Path, part_size: int = 1024, **config):
        self.root, self.clock = root, Clock()
        self.storage = LocalStorage(root / "objects", "http://placeholder", b"k" * 32, part_size=part_size)
        self.config = Config(**{"require_canary": False, "curve_delay_s": 0.0, **config})
        self.start()

    def start(self) -> None:
        self.service = Service(self.root / "db.sqlite3", self.storage, self.config, clock=self.clock)
        self.app = App(self.service, "/drafter")
        self.public, self.url = serve_in_thread(self.app)
        self.admin, self.admin_url = serve_in_thread(App(self.service, admin=True))
        self.storage.public_url = self.config.public_url = self.url + "/drafter"
        self.base = self.url + "/drafter"
        self.admin_token = self.service.issue_admin_token()

    def stop(self) -> None:
        for server in (self.public, self.admin):
            server.shutdown()
            server.server_close()

    def admin_call(self, method: str, path: str, body: dict | None = None):
        return ApiClient(self.admin_url, self.admin_token, retries=0).call(method, path, body)

    def pass_canary(self, api: ApiClient) -> None:
        """Mark a site as past its canary (tests of later rounds)."""

        with self.service.tx() as db:
            db.execute("UPDATE sites SET canary_key=? WHERE token_hash=?",
                       (self.service._canary_key(db), terms.token_hash(api.token)))

    def register(self, name: str | None = "site", home: Path | None = None) -> ApiClient:
        invite = self.admin_call("POST", "/admin/invites", {"label": f"for {name}"})
        code = invite["url"].split("#invite=")[1]
        reply = ApiClient(self.base, retries=0).call(
            "POST", "/api/v1/redeem", {"code": code, "display_name": name, "accept_terms": terms.TERMS_SHA256},
            auth=False)
        return ApiClient(self.base, reply["token"], retries=0)

    def validator(self) -> ApiClient:
        return ApiClient(self.base, self.admin_call("POST", "/admin/validators", {})["token"], retries=0)

    def publish(self, validator: ApiClient, version: int, data: bytes) -> str:
        path = self.root / f"global-{version}.bin"
        path.write_bytes(data)
        sha = hashlib.sha256(data).hexdigest()
        reply = validator.call("POST", f"/api/v1/validator/versions/{version}/upload",
                               {"size": len(data), "sha256": sha, "dtype": "bf16"})
        parts = upload_file(path, reply["upload"], lambda n: {}, {})
        validator.call("POST", f"/api/v1/validator/versions/{version}/publish",
                       {"upload_id": reply["upload_id"], "parts": parts, "merged": [], "evals": {"metrics": {
                           "heldout": 2.0 + version, "template.prose": 1.5, "template.code": 2.5}}})
        return sha


class Interrupted(Exception):
    pass


def interrupt_at(part: int):
    def hook(n: int) -> None:
        if n == part:
            raise Interrupted(f"killed before part {n}")
    return hook


def delta_body(api: ApiClient, path: Path, base_version: int, base_sha: str, client_id: str, **extra) -> dict:
    data = path.read_bytes()
    body = {"client_id": client_id, "base_version": base_version, "base_sha256": base_sha, "size": len(data),
            "sha256": hashlib.sha256(data).hexdigest(), "mode": extra.pop("mode", "byo"),
            "meta": {"tokens": 1000.0, "steps": 20, **PINS, "validator_version": "1", **extra.pop("meta", {})},
            **extra}
    body["attestation"] = terms.sign_attestation(api.token, {**body, "shard_id": body.get("shard_id"),
                                                             "tag": body.get("tag")})
    return body


def send_delta(api: ApiClient, path: Path, base_version: int, base_sha: str, client_id: str, **extra) -> dict:
    reply = api.call("POST", "/api/v1/deltas", delta_body(api, path, base_version, base_sha, client_id, **extra))
    if reply["state"] == "uploading":
        parts = upload_file(path, reply["upload"], lambda n: api.call(
            "POST", f"/api/v1/uploads/{reply['upload_id']}/refresh", {"parts": n}), {})
        return api.call("POST", f"/api/v1/deltas/{reply['delta_id']}/complete", {"parts": parts})
    return reply


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.net = Net(self.root)
        self.val = self.net.validator()
        self.val.call("POST", "/api/v1/validator/hello", {"version": "1"})
        self.net.admin_call("POST", "/admin/settings/requirements", {"value": PINS})
        self.v0 = self.net.publish(self.val, 0, os.urandom(3000))
        self.blob = self.root / "delta.bin"
        self.blob.write_bytes(os.urandom(3500))   # 4 parts of 1 KiB

    def tearDown(self):
        self.net.stop()
        self.tmp.cleanup()

    def test_invite_one_use_expiry_revocation_and_terms(self):
        from enntity_drafter import join

        invite = self.net.admin_call("POST", "/admin/invites", {"label": "alice", "hours": 1})
        self.assertIn("/drafter/join#invite=inv_", invite["url"])   # the code travels in the fragment
        home = self.root / "home"
        reply = join.redeem(invite["url"], "Alice's lab", True, home)
        config = json.loads((home / "config.json").read_text())
        self.assertEqual(config["token"], reply["token"])
        self.assertEqual(stat.S_IMODE((home / "config.json").stat().st_mode), 0o600)
        db = (self.root / "db.sqlite3").read_bytes()
        self.assertNotIn(reply["token"].encode(), db)
        self.assertNotIn(invite["code"].encode(), db)
        with self.assertRaises(ApiError) as used:
            join.redeem(invite["url"], "again", True, self.root / "home2")
        self.assertEqual((used.exception.status, used.exception.code), (403, "invite_invalid"))
        late = self.net.admin_call("POST", "/admin/invites", {"hours": 1})
        self.net.clock.t += 3601
        with self.assertRaises(ApiError) as expired:
            join.redeem(late["url"], None, True, self.root / "home3")
        self.assertEqual(expired.exception.code, "invite_invalid")
        revoked = self.net.admin_call("POST", "/admin/invites", {})
        self.net.admin_call("POST", f"/admin/invites/{revoked['id']}/revoke")
        with self.assertRaises(ApiError):
            join.redeem(revoked["url"], None, True, self.root / "home4")
        fresh = self.net.admin_call("POST", "/admin/invites", {})
        with self.assertRaises(ApiError) as no_terms:
            ApiClient(self.net.base, retries=0).call("POST", "/api/v1/redeem", {
                "code": fresh["code"], "accept_terms": "0" * 64}, auth=False)
        self.assertEqual(no_terms.exception.code, "terms_required")
        events = [e["event"] for e in self.net.admin_call("GET", "/admin/audit")]
        for event in ("invite.issued", "invite.redeemed", "site.registered", "invite.revoked"):
            self.assertIn(event, events)

    def test_auth_required_and_admin_only_on_loopback(self):
        anonymous = ApiClient(self.net.base, retries=0)
        for method, path in (("GET", "/api/v1/requirements"), ("POST", "/api/v1/deltas"),
                             ("POST", "/api/v1/heartbeat"), ("GET", "/api/v1/validator/pending")):
            with self.assertRaises(ApiError) as missing:
                anonymous.call(method, path, {} if method == "POST" else None)
            self.assertEqual(missing.exception.status, 401, path)
        with self.assertRaises(ApiError) as wrong:
            ApiClient(self.net.base, "dsite_wrong", retries=0).call("GET", "/api/v1/requirements")
        self.assertEqual(wrong.exception.code, "auth_invalid")
        site = self.net.register("bob")
        with self.assertRaises(ApiError) as role:
            site.call("GET", "/api/v1/validator/pending")
        self.assertEqual(role.exception.status, 401)
        with self.assertRaises(ApiError) as hidden:
            anonymous.call("POST", "/admin/invites", {})
        self.assertEqual(hidden.exception.status, 404)
        with self.assertRaises(ApiError):
            ApiClient(self.net.url, retries=0).call("GET", "/drafter/admin/audit")
        with self.assertRaises(ValueError):
            make_server(App(self.net.service, admin=True), "0.0.0.0", 0)
        public_id = site.call("GET", "/api/v1/me")["public_id"]
        self.net.admin_call("POST", f"/admin/sites/{public_id}/revoke")
        with self.assertRaises(ApiError) as revoked:
            site.call("GET", "/api/v1/requirements")
        self.assertEqual(revoked.exception.status, 401)

    def test_signed_upload_resumes_after_interrupted_chunk(self):
        site = self.net.register("carol")
        reply = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-0001"))
        plan = reply["upload"]
        self.assertEqual([p["length"] for p in plan["parts"]], [1024, 1024, 1024, 428])
        state: dict = {}
        with self.assertRaises(Interrupted):
            upload_file(self.blob, plan, lambda n: {"parts": []}, state, before_part=interrupt_at(2))
        self.assertEqual(sorted(state["etags"]), ["1"])
        url = urllib.parse.urlsplit(plan["parts"][1]["url"])   # part 2: the connection drops halfway
        connection = http.client.HTTPConnection(url.hostname, url.port)
        connection.putrequest("PUT", url.path)
        connection.putheader("Content-Length", "1024")
        connection.endheaders()
        connection.send(self.blob.read_bytes()[1024:1536])
        connection.close()
        tampered = plan["parts"][1]["url"][:-4] + "0000"
        request = urllib.request.Request(tampered, data=b"x" * 1024, method="PUT")
        with self.assertRaises(urllib.error.HTTPError) as forbidden:
            urllib.request.urlopen(request)
        self.assertEqual(forbidden.exception.code, 403)
        with self.assertRaises(urllib.error.HTTPError) as short:
            urllib.request.urlopen(urllib.request.Request(plan["parts"][1]["url"], data=b"x" * 10, method="PUT"))
        self.assertEqual(short.exception.code, 400)
        with self.assertRaises(ApiError) as incomplete:
            site.call("POST", f"/api/v1/deltas/{reply['delta_id']}/complete",
                      {"parts": [{"n": 1, "etag": state["etags"]["1"]}]})
        self.assertEqual(incomplete.exception.code, "upload_incomplete")
        again = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-0001"))
        self.assertEqual(again["delta_id"], reply["delta_id"])   # idempotent: same client id resumes
        parts = upload_file(self.blob, again["upload"], lambda n: site.call(
            "POST", f"/api/v1/uploads/{reply['upload_id']}/refresh", {"parts": n}), state)
        done = site.call("POST", f"/api/v1/deltas/{reply['delta_id']}/complete", {"parts": parts})
        self.assertEqual(done["state"], "pending")
        claim = self.val.call("POST", f"/api/v1/validator/deltas/{reply['delta_id']}/claim")
        got = download(claim["url"], self.root / "got.bin", sha256=hashlib.sha256(self.blob.read_bytes()).hexdigest())
        self.assertEqual(got.read_bytes(), self.blob.read_bytes())

    def test_expired_url_is_refused_and_refreshed(self):
        self.net.stop()
        self.net.config.url_expiry_s = 1
        self.net.start()
        site = self.net.register("dan")
        reply = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-0002"))
        time.sleep(1.2)
        with self.assertRaises(urllib.error.HTTPError) as expired:
            urllib.request.urlopen(urllib.request.Request(reply["upload"]["parts"][0]["url"], data=b"x" * 1024,
                                                          method="PUT"))
        self.assertEqual(expired.exception.code, 403)
        self.net.config.url_expiry_s = 600
        parts = upload_file(self.blob, reply["upload"], lambda n: site.call(
            "POST", f"/api/v1/uploads/{reply['upload_id']}/refresh", {"parts": n}), {}, sleep=lambda s: None)
        self.assertEqual(site.call("POST", f"/api/v1/deltas/{reply['delta_id']}/complete", {"parts": parts})["state"],
                         "pending")

    def test_checks_at_initiation_before_any_bytes(self):
        site = self.net.register("erin")
        cases = {
            "version_mismatch": dict(meta={"trainer": "scratch-v0"}),
            "base_unknown": dict(base_sha="f" * 64),
            "shard_not_leased": dict(mode="recipe", shard_id="s000000"),
        }
        self.net.admin_call("POST", "/admin/shards", {"recipes": [{"builder": "synthetic", "rows": [0, 8]}]})
        for code, extra in cases.items():
            base_sha = extra.pop("base_sha", self.v0)
            with self.assertRaises(ApiError) as refused:
                site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, base_sha, f"cid-{code}", **extra))
            self.assertEqual(refused.exception.code, code)
        mismatch = None
        try:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "cid-mismatch", meta={
                "config_sha256": "d" * 64}))
        except ApiError as error:
            mismatch = error
        self.assertIn("drafter-join update", mismatch.message)
        body = delta_body(site, self.blob, 0, self.v0, "cid-forged")
        body["attestation"]["signature"] = "0" * 64
        with self.assertRaises(ApiError) as forged:
            site.call("POST", "/api/v1/deltas", body)
        self.assertEqual(forged.exception.code, "attestation_invalid")
        big = delta_body(site, self.blob, 0, self.v0, "cid-bigsize")
        big["size"] = self.net.config.max_delta_bytes + 1
        big["attestation"] = terms.sign_attestation(site.token, {**big, "shard_id": None, "tag": None})
        with self.assertRaises(ApiError) as too_large:
            site.call("POST", "/api/v1/deltas", big)
        self.assertEqual(too_large.exception.status, 413)
        self.assertEqual(list((self.root / "objects" / ".uploads").iterdir()), [])   # no upload was ever opened

    def test_duplicates_and_idempotent_retries(self):
        site, other = self.net.register("fay"), self.net.register("gus")
        first = send_delta(site, self.blob, 0, self.v0, "client-dup-1")
        retry = send_delta(site, self.blob, 0, self.v0, "client-dup-1")
        self.assertEqual(retry["delta_id"], first["delta_id"])
        copy = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-dup-2"))
        self.assertEqual((copy["delta_id"], copy["duplicate"]), (first["delta_id"], True))
        with self.assertRaises(ApiError) as stolen:
            other.call("POST", "/api/v1/deltas", delta_body(other, self.blob, 0, self.v0, "client-dup-3"))
        self.assertEqual(stolen.exception.code, "duplicate")
        self.assertEqual(site.call("GET", "/api/v1/me")["deltas_received"], 1)

    def test_stale_bases_accepted_while_retained(self):
        self.net.service.config.keep_globals = 2
        site = self.net.register("hal")
        shas = [self.v0] + [self.net.publish(self.val, v, os.urandom(2000 + v)) for v in (1, 2)]
        one_behind = send_delta(site, self.blob, 1, shas[1], "client-stale-1")
        shas.append(self.net.publish(self.val, 3, os.urandom(2003)))
        self.blob.write_bytes(os.urandom(3500))
        two_behind = send_delta(site, self.blob, 1, shas[1], "client-stale-2")   # v1 is kept: a delta needs it
        self.assertEqual((one_behind["state"], two_behind["state"]), ("pending", "pending"))
        self.assertIsNotNone(self.net.storage.head("globals/v000001-" + shas[1][:12] + ".safetensors"))
        self.blob.write_bytes(os.urandom(3500))
        with self.assertRaises(ApiError) as gone:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, shas[0], "client-stale-3"))
        self.assertEqual(gone.exception.code, "base_retired")
        site.call("POST", "/api/v1/heartbeat", {"phase": "training", "version": 2})
        self.net.publish(self.val, 4, os.urandom(1999))
        self.net.publish(self.val, 5, os.urandom(1998))
        self.assertIsNotNone(self.net.storage.head("globals/v000002-" + shas[2][:12] + ".safetensors"))  # live site

    def test_lease_expiry_extension_and_late_work_accepted(self):
        self.net.admin_call("POST", "/admin/shards", {"recipes": [{"builder": "synthetic", "rows": [i, i + 8]}
                                                                  for i in (0, 8)]})
        alice, bob = self.net.register("alice"), self.net.register("bob")
        first = alice.call("POST", "/api/v1/shards/claim")["shard"]
        self.assertEqual(alice.call("POST", "/api/v1/shards/claim")["shard"]["id"], first["id"])   # idempotent
        self.net.clock.t += self.net.config.lease_s - 60
        alice.call("POST", "/api/v1/heartbeat", {"phase": "training", "shard_id": first["id"]})
        self.net.clock.t += self.net.config.lease_s + self.net.config.lease_grace_s - 120
        self.assertNotEqual(bob.call("POST", "/api/v1/shards/claim")["shard"]["id"], first["id"])   # extended
        self.net.clock.t += self.net.config.lease_s + self.net.config.lease_grace_s + 1
        carl = self.net.register("carl")
        self.assertEqual(carl.call("POST", "/api/v1/shards/claim")["shard"]["id"], first["id"])   # expired
        late = send_delta(alice, self.blob, 0, self.v0, "client-late-1", mode="recipe", shard_id=first["id"])
        self.assertEqual(late["state"], "pending")   # never thrown away for timing
        self.assertIn("shard.lease_expired", [e["event"] for e in self.net.admin_call("GET", "/admin/audit")])

    def test_heartbeat_whitelist_and_public_dashboard(self):
        named, anonymous = self.net.register("Lab one"), self.net.register(None)
        for bad in ({"phase": "training", "hostname": "box1"}, {"phase": "dancing"}, {"phase": "training",
                                                                                     "client": "a b/c"}):
            with self.assertRaises(ApiError) as refused:
                named.call("POST", "/api/v1/heartbeat", bad)
            self.assertEqual(refused.exception.status, 400)
        named.call("POST", "/api/v1/heartbeat", {"phase": "training", "tokens": 1000.0, "tokens_per_s": 50.0,
                                                 "gpus": 1, "version": 0, "client": "0.1.0"})
        self.net.clock.t += 600
        named.call("POST", "/api/v1/heartbeat", {"phase": "training", "tokens": 31000.0, "tokens_per_s": 50.0,
                                                 "gpus": 1, "version": 0})
        send_delta(anonymous, self.blob, 0, self.v0, "client-anon-1")
        self.net.admin_call("POST", "/admin/settings/benchmarks", {"value": {"rows": [["code", 4.89, 3.57]]}})
        page = urllib.request.urlopen(self.net.base + "/api/public/dashboard")
        text = page.read().decode()
        data = json.loads(text)
        names = {s["name"]: s for s in data["sites"]}
        self.assertEqual(names["anonymous"]["anonymous"], True)
        self.assertEqual(names["anonymous"]["tokens_received"], 1000.0)   # credited on receipt
        self.assertEqual(names["Lab one"]["status"], "training")
        self.assertEqual(data["totals"]["tokens_per_day"], 30000.0)
        self.assertEqual(data["totals"]["sites_online"], 1)
        self.assertEqual(data["global"]["version"], 0)
        self.assertEqual(data["wanted"]["weakest"][0]["category"], "heldout")   # groups only, no categories
        self.assertEqual(data["curve"][0]["metrics"], {"heldout": 2.0})
        self.assertEqual(data["benchmarks"], {"rows": [["code", 4.89, 3.57]]})
        secrets_ = [named.token, anonymous.token, terms.token_hash(named.token), "for Lab one"]
        secrets_ += [s["public_id"] for s in self.net.service.list_sites()]
        for secret in secrets_:
            self.assertNotIn(secret, text)
        index = urllib.request.urlopen(self.net.base + "/")
        self.assertIn("default-src 'self'", index.headers["Content-Security-Policy"])
        self.assertIn(b"<title>", index.read())
        for page in ("/how", "/join", "/static/app.js", "/static/style.css"):
            self.assertEqual(urllib.request.urlopen(self.net.base + page).status, 200, page)

    def test_state_survives_restart(self):
        site = self.net.register("ivy")
        reply = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-restart"))
        state: dict = {}
        with self.assertRaises(Interrupted):
            upload_file(self.blob, reply["upload"], lambda n: {"parts": []}, state, before_part=interrupt_at(3))
        self.assertEqual(sorted(state["etags"]), ["1", "2"])
        self.net.stop()
        self.net.start()   # same database and object store, new process state
        site = ApiClient(self.net.base, site.token, retries=0)
        again = site.call("POST", "/api/v1/deltas", delta_body(site, self.blob, 0, self.v0, "client-restart"))
        parts = upload_file(self.blob, again["upload"], lambda n: site.call(
            "POST", f"/api/v1/uploads/{again['upload_id']}/refresh", {"parts": n}), state)
        self.assertEqual(site.call("POST", f"/api/v1/deltas/{reply['delta_id']}/complete", {"parts": parts})["state"],
                         "pending")


class ReviewRegressionTests(unittest.TestCase):
    """Regression tests for the security review of the first volunteer-training commit."""

    setUp, tearDown = CoordinatorTests.setUp, CoordinatorTests.tearDown

    def new_blob(self) -> Path:
        self.blob.write_bytes(os.urandom(3500))
        return self.blob

    def accept(self, delta_id: str) -> None:
        self.val.call("POST", f"/api/v1/validator/deltas/{delta_id}/claim")
        self.val.call("POST", f"/api/v1/validator/deltas/{delta_id}/verdict", {
            "decision": "accept", "validator_version": "1", "metrics": {"heldout": 9.87654}})

    def test_canary_is_decided_and_enforced_by_the_server(self):   # review 8
        self.net.service.config.require_canary = True
        site = self.net.register("cara")
        with self.assertRaises(ApiError) as long_first:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, "canary-0001",
                                                            meta={"steps": 500}))
        self.assertEqual(long_first.exception.code, "canary_required")
        first = send_delta(site, self.new_blob(), 0, self.v0, "canary-0002", canary=False)   # flag ignored
        self.assertTrue(site.call("GET", f"/api/v1/deltas/{first['delta_id']}")["canary"])
        with self.assertRaises(ApiError) as second:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, "canary-0003"))
        self.assertEqual(second.exception.code, "canary_pending")
        self.accept(first["delta_id"])
        later = send_delta(site, self.new_blob(), 0, self.v0, "canary-0004", canary=True)   # flag ignored again
        self.assertFalse(site.call("GET", f"/api/v1/deltas/{later['delta_id']}")["canary"])

    def test_upload_quotas_reaper_retention_and_disk_guard(self):   # review 2 (pending limit) and 4
        site = self.net.register("quinn")
        opened = [site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, f"open-{i:04d}"))
                  for i in range(2)]
        with self.assertRaises(ApiError) as uploads:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, "open-0002"))
        self.assertEqual((uploads.exception.status, uploads.exception.code), (429, "too_many_uploads"))
        self.net.clock.t += self.net.config.upload_ttl_s + 1
        self.assertEqual(self.net.service.maintenance()["aborted"], 2)
        self.assertEqual(list((self.root / "objects" / ".uploads").iterdir()), [])
        self.assertEqual(site.call("GET", f"/api/v1/deltas/{opened[0]['delta_id']}")["state"], "abandoned")
        with self.assertRaises(ApiError):
            site.call("POST", f"/api/v1/uploads/{opened[0]['upload_id']}/refresh", {"parts": [1]})
        self.net.service.config.max_open_deltas = 1
        done = send_delta(site, self.new_blob(), 0, self.v0, "open-0003")
        with self.assertRaises(ApiError) as pending:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, "open-0004"))
        self.assertEqual(pending.exception.code, "too_many_pending")
        self.accept(done["delta_id"])
        data = os.urandom(3100)
        (self.root / "g1.bin").write_bytes(data)
        reply = self.val.call("POST", "/api/v1/validator/versions/1/upload",
                              {"size": len(data), "sha256": hashlib.sha256(data).hexdigest(), "dtype": "bf16"})
        self.val.call("POST", "/api/v1/validator/versions/1/publish", {
            "upload_id": reply["upload_id"], "parts": upload_file(self.root / "g1.bin", reply["upload"], None, {}),
            "merged": [{"delta_id": done["delta_id"], "weight": 1.0}]})
        key = f"deltas/{site.call('GET', '/api/v1/me')['public_id']}/{done['delta_id']}.safetensors"
        self.assertIsNotNone(self.net.storage.head(key))
        self.net.clock.t += self.net.config.delta_retention_s + 1
        self.assertEqual(self.net.service.maintenance()["purged"], 1)
        self.assertIsNone(self.net.storage.head(key))
        self.assertEqual(site.call("GET", "/api/v1/me")["deltas_accepted"], 1)   # credit survives the purge
        self.net.service.config.min_free_bytes = 2**62
        with self.assertRaises(ApiError) as low:
            site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 1, "0" * 64, "open-0005"))
        self.assertEqual((low.exception.status, low.exception.code), (503, "low_disk"))
        self.assertTrue(site.call("POST", "/api/v1/heartbeat", {"phase": "idle"})["ok"])   # still answered
        self.net.storage.min_free_bytes = 2**62
        status, _, _ = self.net.storage.handle_blob("PUT", "x.y", None, 10)
        self.assertEqual(status, 403)
        upload = self.net.storage.plan_upload("deltas/x/y.safetensors", 10, 60)
        token = upload["parts"][0]["url"].rsplit("/blob/", 1)[1]
        self.assertEqual(self.net.storage.handle_blob("PUT", token, None, 10)[0], 507)

    def test_revoked_site_work_is_quarantined_and_never_merged(self):   # review 7
        site = self.net.register("rex")
        pending = send_delta(site, self.new_blob(), 0, self.v0, "revoke-0001")
        accepted = send_delta(site, self.new_blob(), 0, self.v0, "revoke-0002")
        self.accept(accepted["delta_id"])
        public_id = site.call("GET", "/api/v1/me")["public_id"]
        self.net.admin_call("POST", f"/admin/sites/{public_id}/revoke")
        self.assertEqual(self.val.call("GET", "/api/v1/validator/pending"), [])
        self.assertEqual(self.val.call("GET", "/api/v1/validator/mergeable"), [])
        self.val.call("POST", "/api/v1/validator/hello", {"version": "2"})   # a version bump does not revive them
        self.assertEqual(self.val.call("GET", "/api/v1/validator/pending"), [])
        with self.net.service.tx() as db:
            states = {r["id"]: (r["state"], r["reason_code"]) for r in db.execute("SELECT * FROM deltas")}
        for delta in (pending, accepted):
            self.assertEqual(states[delta["delta_id"]], ("quarantined", "site_revoked"))

    def test_sites_see_codes_not_scores_and_the_curve_is_coarse_and_delayed(self):   # review 9
        site = self.net.register("orla")
        delta = send_delta(site, self.new_blob(), 0, self.v0, "oracle-0001")
        self.accept(delta["delta_id"])
        status = site.call("GET", f"/api/v1/deltas/{delta['delta_id']}")
        text = json.dumps(site.call("GET", "/api/v1/me"))
        self.assertNotIn("verdict", status)
        self.assertNotIn("9.87", text)
        self.assertEqual((status["decision"], status["reason"]), ("accept", "accepted"))
        self.net.service.config.curve_delay_s = 3600
        self.net.publish(self.val, 1, os.urandom(2001))
        curve = self.net.service.dashboard()["curve"]
        self.assertEqual([p["version"] for p in curve], [])   # nothing younger than an hour
        self.net.clock.t += 3601
        curve = self.net.service.dashboard()["curve"]
        self.assertEqual([p["version"] for p in curve], [0, 1])
        self.assertTrue(all("." not in k and round(v, 2) == v for p in curve for k, v in p["metrics"].items()))

    def test_unread_request_body_closes_the_connection(self):   # review 6
        import socket

        host, port = self.net.url.split("//")[1].split(":")
        smuggled = b"GET /drafter/api/v1/terms HTTP/1.1\r\nHost: x\r\n\r\n"
        request = (b"POST /drafter/api/v1/heartbeat HTTP/1.1\r\nHost: x\r\nContent-Length: "
                   + str(len(smuggled)).encode() + b"\r\n\r\n" + smuggled)
        with socket.create_connection((host, int(port)), timeout=5) as conn:
            conn.sendall(request)
            received = b""
            while chunk := conn.recv(65536):
                received += chunk
        self.assertIn(b"401", received.split(b"\r\n", 1)[0])
        self.assertIn(b"Connection: close", received)
        self.assertEqual(received.count(b"HTTP/1.1 "), 1)   # the body was never served as a second request

    def test_plan_upload_runs_outside_the_write_lock(self):   # review LOW
        import sqlite3

        original, seen = self.net.storage.plan_upload, []

        def probe(*args, **kwargs):
            db = sqlite3.connect(self.root / "db.sqlite3", timeout=0.2, isolation_level=None)
            db.execute("BEGIN IMMEDIATE")   # raises "database is locked" if the coordinator still holds it
            db.execute("ROLLBACK")
            db.close()
            seen.append(True)
            return original(*args, **kwargs)

        self.net.storage.plan_upload = probe
        site = self.net.register("lock")
        site.call("POST", "/api/v1/deltas", delta_body(site, self.new_blob(), 0, self.v0, "lock-0001"))
        self.assertEqual(seen, [True])

    def test_forwarded_for_used_only_from_the_trusted_proxy(self):   # review LOW
        def redeem(forwarded: str) -> int:
            request = urllib.request.Request(self.net.base + "/api/v1/redeem", method="POST", data=json.dumps(
                {"code": "inv_nope", "accept_terms": terms.TERMS_SHA256}).encode(),
                headers={"X-Forwarded-For": forwarded, "Content-Type": "application/json"})
            try:
                urllib.request.urlopen(request)
            except urllib.error.HTTPError as error:
                return error.code
            return 200

        self.net.app.trusted_proxies = {"127.0.0.1"}
        self.assertEqual([redeem("9.9.9.9, 1.1.1.1") for _ in range(6)], [403] * 5 + [429])
        self.assertEqual(redeem("1.1.1.1, 2.2.2.2"), 403)   # the rightmost hop is the client: its own bucket
        self.net.app.trusted_proxies = set()
        self.net.app.limiter.buckets.clear()
        self.assertEqual([redeem(f"10.0.0.{i}") for i in range(6)], [403] * 5 + [429])   # spoofed hops ignored

    def test_admin_api_needs_token_and_loopback_host(self):   # review LOW
        with self.assertRaises(ApiError) as anonymous:
            ApiClient(self.net.admin_url, retries=0).call("GET", "/admin/audit")
        self.assertEqual(anonymous.exception.status, 401)
        request = urllib.request.Request(self.net.admin_url + "/admin/audit",
                                         headers={"Authorization": f"Bearer {self.net.admin_token}",
                                                  "Host": "evil.example"})
        with self.assertRaises(urllib.error.HTTPError) as rebinding:
            urllib.request.urlopen(request)
        self.assertEqual(rebinding.exception.code, 403)

    def test_bad_numbers_are_400_and_the_image_is_pinned(self):   # review LOW
        site = self.net.register("num")
        for path in ("/api/v1/global?version=abc", "/api/v1/global?version=-1"):
            with self.assertRaises(ApiError) as bad:
                site.call("GET", path)
            self.assertEqual(bad.exception.status, 400, path)
        with self.assertRaises(ApiError) as bad:
            self.val.call("GET", "/api/v1/validator/versions/x1")
        self.assertEqual(bad.exception.status, 400)
        dockerfile = (_common.REPO / "coordinator" / "Dockerfile").read_text()
        self.assertRegex(dockerfile, r"(?m)^FROM python:[\w.-]+@sha256:[0-9a-f]{64}$")


class ControlPlaneTests(unittest.TestCase):
    def test_coordinator_never_loads_weights(self):
        import re

        banned = re.compile(r"^\s*(import|from)\s+(torch|safetensors|numpy|enntity_drafter)\b", re.MULTILINE)
        for path in sorted((_common.REPO / "coordinator").glob("*.py")):
            self.assertIsNone(banned.search(path.read_text()), f"{path.name} must stay a stdlib control plane")


class SignerTests(unittest.TestCase):
    def test_s3_presign_matches_aws_example(self):
        # docs.aws.amazon.com/AmazonS3/latest/API/sigv4-query-string-auth.html (example credentials)
        signer = S3Storage("https://s3.amazonaws.com", "examplebucket", "AKIAIOSFODNN7EXAMPLE",
                           "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", region="us-east-1", addressing="virtual",
                           clock=lambda: 1369353600.0)
        url = signer.presign("GET", "test.txt", 86400)
        self.assertTrue(url.startswith("https://examplebucket.s3.amazonaws.com/test.txt?"))
        self.assertIn("X-Amz-Signature=aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404", url)

    def test_gcs_v4_signature_verifies(self):
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding, rsa
        except ImportError:
            self.skipTest("cryptography not installed")
        from coordinator.storage import GCSStorage

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                serialization.NoEncryption()).decode()
        signer = GCSStorage({"client_email": "drafter@example.iam.gserviceaccount.com", "private_key": pem},
                            "drafter-bucket", clock=lambda: 1369353600.0)
        plan = signer.plan_upload("deltas/a/b.safetensors", 1234, 600)
        self.assertEqual(plan["init_headers"]["x-goog-content-length-range"], "1234,1234")
        url = urllib.parse.urlsplit(plan["init_url"])
        query = dict(urllib.parse.parse_qsl(url.query))
        self.assertEqual(query["X-Goog-SignedHeaders"],
                         "host;x-goog-content-length-range;x-goog-if-generation-match;x-goog-resumable")
        self.assertEqual(plan["init_headers"]["x-goog-if-generation-match"], "0")   # create-only: no overwrite
        self.assertLessEqual(int(query["X-Goog-Expires"]), 900)                    # short-lived initiation
        signature = bytes.fromhex(query.pop("X-Goog-Signature"))
        canonical_query = "&".join(f"{urllib.parse.quote(k, safe='-_.~')}={urllib.parse.quote(v, safe='-_.~')}"
                                   for k, v in sorted(query.items()))
        canonical = "\n".join(["POST", url.path, canonical_query, "host:storage.googleapis.com\n"
                               "x-goog-content-length-range:1234,1234\nx-goog-if-generation-match:0\n"
                               "x-goog-resumable:start\n",
                               query["X-Goog-SignedHeaders"], "UNSIGNED-PAYLOAD"])
        to_sign = "\n".join(["GOOG4-RSA-SHA256", "20130524T000000Z", "20130524/auto/storage/goog4_request",
                             hashlib.sha256(canonical.encode()).hexdigest()])
        key.public_key().verify(signature, to_sign.encode(), padding.PKCS1v15(), hashes.SHA256())


if __name__ == "__main__":
    unittest.main()
