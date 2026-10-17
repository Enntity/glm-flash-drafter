"""Coordinator entry point: ``serve`` (public + loopback admin routers) and admin commands that act on the database
directly (inside the container: ``docker compose exec coordinator python -m coordinator <command>``).

Configuration is environment only; credentials come from mounted files (see docs/operators.md):

    DRAFTER_DB=/data/coordinator.sqlite3  DRAFTER_BASE_PATH=/drafter  DRAFTER_PUBLIC_URL=https://<host>/drafter
    DRAFTER_PORT=8080  DRAFTER_ADMIN_PORT=8081  DRAFTER_TRUST_PROXY=1
    DRAFTER_STORAGE=local|s3|gcs  DRAFTER_BUCKET=...  (+ backend settings in coordinator/storage.py)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import storage as st
from .server import App, make_server
from .service import Config, Service


def build_service(env: dict[str, str] | None = None) -> Service:
    env = dict(os.environ if env is None else env)
    config = Config.from_env(env)
    return Service(env.get("DRAFTER_DB", "/data/coordinator.sqlite3"), st.from_env(env, config.public_url), config)


def plan_recipes(args: argparse.Namespace) -> list[dict]:
    """Shard recipes over a deterministic builder's rows: ``rows=[lo, hi)`` per shard."""

    base = {"builder": args.builder, "args": json.loads(args.builder_args), "train_seed": args.train_seed,
            "generation": json.loads(args.generation)}
    return [{**base, "rows": [lo, min(lo + args.rows_per_shard, args.total)]}
            for lo in range(args.first_row, args.total, args.rows_per_shard)]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="python -m coordinator", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve", help="run the public router and the loopback admin router")
    q = sub.add_parser("invite", help="issue a one-use invite (prints the link once)")
    q.add_argument("--label", default="")
    q.add_argument("--hours", type=float, default=None)
    sub.add_parser("invites", help="list invites (codes are never stored)")
    q = sub.add_parser("revoke-invite")
    q.add_argument("id", type=int)
    sub.add_parser("sites")
    q = sub.add_parser("revoke-site", help="revoke a site's token")
    q.add_argument("public_id")
    q = sub.add_parser("validator", help="issue a validator token (prints it once)")
    q.add_argument("--label", default="validator")
    q = sub.add_parser("add-shards", help="add shard recipes over a deterministic prompt builder")
    q.add_argument("--builder", required=True, choices=["templates", "coverage", "synthetic"])
    q.add_argument("--builder-args", default="{}", help="JSON, e.g. '{\"seed\": 7, \"profile\": \"long\"}'")
    q.add_argument("--generation", default="{}", help="JSON generation settings, e.g. '{\"samples\": 2}'")
    q.add_argument("--rows-per-shard", type=int, required=True)
    q.add_argument("--first-row", type=int, default=0)
    q.add_argument("--total", type=int, required=True)
    q.add_argument("--train-seed", type=int, default=20261014)
    q = sub.add_parser("set", help="set requirements | training | benchmarks | wanted_notes from JSON")
    q.add_argument("key", choices=Service.SETTABLE)
    q.add_argument("json", help="JSON text, or @file.json")
    q = sub.add_parser("get")
    q.add_argument("key")
    q = sub.add_parser("release", help="decide a held delta")
    q.add_argument("delta_id")
    q.add_argument("decision", choices=["accept", "reject"])
    q.add_argument("--weight-factor", type=float, default=1.0)
    sub.add_parser("admin-token", help="issue a new admin API token (printed once; replaces the old one)")
    sub.add_parser("maintenance", help="abort abandoned uploads, purge expired delta objects, retire globals")
    q = sub.add_parser("audit")
    q.add_argument("--limit", type=int, default=100)
    args = p.parse_args(argv)
    service = build_service()

    if args.cmd == "serve":
        env = os.environ
        public = make_server(App(service, env.get("DRAFTER_BASE_PATH", "/drafter")),
                             env.get("DRAFTER_HOST", "0.0.0.0"), int(env.get("DRAFTER_PORT", "8080")))
        admin = make_server(App(service, admin=True), "127.0.0.1", int(env.get("DRAFTER_ADMIN_PORT", "8081")))
        token_file = Path(env.get("DRAFTER_ADMIN_TOKEN_FILE", "/data/admin-token"))
        if service.get_setting("admin_token_hash") is None:   # first start: an admin token for the loopback API
            token_file.touch(mode=0o600)
            token_file.write_text(service.issue_admin_token() + "\n")
            print(f"coordinator: admin API token written to {token_file}", flush=True)
        import threading
        import time

        def maintain() -> None:
            while True:
                try:
                    print(f"maintenance: {json.dumps(service.maintenance())}", flush=True)
                except Exception as error:  # noqa: BLE001 - keep serving
                    print(f"maintenance failed: {error!r}", flush=True)
                time.sleep(600)

        threading.Thread(target=admin.serve_forever, daemon=True).start()
        threading.Thread(target=maintain, daemon=True).start()
        print(f"coordinator: public :{public.server_address[1]}, admin 127.0.0.1:{admin.server_address[1]}",
              flush=True)
        public.serve_forever()
        return
    if args.cmd == "invite":
        result = service.issue_invite(args.label, args.hours)
    elif args.cmd == "invites":
        result = service.list_invites()
    elif args.cmd == "revoke-invite":
        result = service.revoke_invite(args.id)
    elif args.cmd == "sites":
        result = service.list_sites()
    elif args.cmd == "revoke-site":
        result = service.revoke_site(args.public_id)
    elif args.cmd == "validator":
        result = service.issue_validator(args.label)
    elif args.cmd == "add-shards":
        result = service.add_shards(plan_recipes(args))
    elif args.cmd == "set":
        text = Path(args.json[1:]).read_text() if args.json.startswith("@") else args.json
        result = service.set_setting(args.key, json.loads(text))
    elif args.cmd == "get":
        result = service.get_setting(args.key)
    elif args.cmd == "admin-token":
        result = {"token": service.issue_admin_token()}
    elif args.cmd == "maintenance":
        result = service.maintenance()
    elif args.cmd == "release":
        result = service.release_held(args.delta_id, args.decision, args.weight_factor)
    else:
        result = service.audit_log(args.limit)
    json.dump({"ok": True} if result is None else result, sys.stdout, indent=1)
    print()


if __name__ == "__main__":
    main()
