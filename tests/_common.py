"""Shared test setup: make the package importable here and in trainer subprocesses."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
TINY = CONFIGS / "tiny.json"
FULL = CONFIGS / "full.json"
PYTHON = sys.executable
TORCHRUN = str(Path(PYTHON).with_name("torchrun"))
#: ``python -m`` / ``torchrun -m`` target of every trainer subprocess.
TRAIN_MODULE = ["-m", "enntity_drafter.train"]



def launcher(ranks: int) -> list[str]:
    """Trainer launcher: plain python for one rank, else torchrun on loopback.

    ``torchrun --standalone`` hangs creating its store when the host name does not resolve."""

    if ranks == 1:
        return [PYTHON]
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return [TORCHRUN, "--nnodes=1", f"--nproc_per_node={ranks}", "--master_addr=127.0.0.1", f"--master_port={port}"]


if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
os.environ["PYTHONPATH"] = os.pathsep.join(p for p in (str(REPO), os.environ.get("PYTHONPATH")) if p)
