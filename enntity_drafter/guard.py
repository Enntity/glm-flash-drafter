"""License guard: no incoai/GLM-5.3-Flash-DFlash2 bytes may enter this trainer.

The incoai DFlash2 checkpoint (and anything derived from it, such as the
local-inference-lab MXFP8 conversion and any fine-tune warm-started from it)
is CC BY-NC-ND 4.0.  The from-scratch trainer may use the DFlash2
*architecture* (Apache-2.0 vLLM/Speculators code) and the frozen GLM-5.3-Flash
target embedding / final norm / lm_head (MIT), nothing else.

Every file this trainer opens for tensors goes through :func:`guarded_load_file`,
:func:`guarded_safe_open` or :func:`guarded_torch_load`.  A load is refused when

* the path, its resolved symlink target, or any parent directory name matches a
  forbidden pattern (``incoai``, ``GLM-5.3-Flash-DFlash2``, the MXFP8
  derivative, the known source revisions);
* a sibling provenance file (``config.json``, ``README.md``,
  ``PROVENANCE.json``, ``hf_quant_config.json``, ``manifest.json``)
  or the safetensors ``__metadata__`` header references one of them;
* a drafter-role file is missing the Enntity scratch provenance stamp written
  by this trainer (whitelist, so an unlabelled copy of a forbidden checkpoint
  cannot slip through under an innocent file name);
* the file's SHA-256 equals a known forbidden artifact hash (checked for
  drafter-role files, which are small enough to hash).

``tests/test_train.py`` also scans the package sources so that no module calls
``load_file`` / ``safe_open`` / ``torch.load`` outside this guard.
"""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path
from typing import Any

#: Lower-cased substrings that must never appear in a loaded path or its metadata.
FORBIDDEN_SUBSTRINGS = (
    "incoai",
    "glm-5.3-flash-dflash2",
    "glm53-dflash2",
    "local-inference-lab",
    "dflash2-mxfp8",
    "base-dc77",  # an earlier staging name of the incoai BF16 body
)
#: Known revisions / artifact hashes of the forbidden checkpoint and derivatives.
FORBIDDEN_IDENTIFIERS = (
    "dc77ff1c99eeb2df044ee3d4f0094eb033fee410",  # incoai revision
    "7d74cdd881ed7e32c31175984a67823127b66cfe",  # incoai revision in bf16-draft-header
    "610aa967a92bfeb97e3d848dcb8693553e8b6a55",  # local-inference-lab MXFP8 revision
    "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b",  # incoai BF16 weights
    "c033e03d47c7d5608596c8fc4e9336a1fe086eb781c08fe031be2bdea1614e58",  # MXFP8 derivative
    "ccd360d08e610d34fd3e363f37fdbc1ee7e2e680422b78227505d73c34dac343",  # an earlier fine-tune warm-started from incoai
)
FORBIDDEN_SHA256 = frozenset(x for x in FORBIDDEN_IDENTIFIERS if len(x) == 64)
SIBLING_PROVENANCE = (
    "config.json",
    "README.md",
    "PROVENANCE.json",
    "hf_quant_config.json",
    "manifest.json",
)
#: Provenance stamp every drafter-role artifact written by this trainer carries.
SCRATCH_STAMP_KEY = "enntity_drafter_provenance"
SCRATCH_LINEAGE_ROOT = "enntity-scratch-init"
ROLES = ("drafter", "target", "features")

_PATTERN = re.compile(
    "|".join(re.escape(x) for x in FORBIDDEN_SUBSTRINGS + FORBIDDEN_IDENTIFIERS)
)


class ForbiddenCheckpointError(RuntimeError):
    """Raised when a load would touch incoai DFlash2 weights or derivatives."""


def _match(text: str) -> str | None:
    found = _PATTERN.search(text.lower())
    return None if found is None else found.group(0)


def safetensors_metadata(path: Path) -> dict[str, Any]:
    """Read only the safetensors JSON header (no tensor bytes)."""

    with Path(path).open("rb") as stream:
        (length,) = struct.unpack("<Q", stream.read(8))
        if length > 100 * 1024 * 1024:
            raise ForbiddenCheckpointError(f"implausible safetensors header in {path}")
        header = json.loads(stream.read(length))
    return header.get("__metadata__", {}) or {}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_path(path: str | Path, role: str) -> Path:
    """Refuse forbidden names, provenance or (drafter role) unstamped files."""

    if role not in ROLES:
        raise ValueError(f"unknown guard role {role!r}")
    raw = Path(path)
    resolved = raw.resolve()
    for candidate in {str(raw.absolute()), str(resolved)}:
        hit = _match(candidate)
        if hit:
            raise ForbiddenCheckpointError(f"refusing {role} load of {path}: path references {hit!r}")
    for sibling in SIBLING_PROVENANCE:
        for directory in {raw.absolute().parent, resolved.parent}:
            meta = directory / sibling
            if meta.is_file() and meta.stat().st_size < 16 * 1024 * 1024:
                hit = _match(meta.read_text(errors="replace"))
                if hit:
                    raise ForbiddenCheckpointError(
                        f"refusing {role} load of {path}: {meta.name} references {hit!r}"
                    )
    if resolved.suffix == ".safetensors" and resolved.is_file():
        metadata = safetensors_metadata(resolved)
        hit = _match(json.dumps(metadata))
        if hit:
            raise ForbiddenCheckpointError(f"refusing {role} load of {path}: header metadata references {hit!r}")
        if role == "drafter":
            require_scratch_stamp(metadata, path)
    if role == "drafter" and resolved.is_file():
        if sha256_file(resolved) in FORBIDDEN_SHA256:
            raise ForbiddenCheckpointError(f"refusing {role} load of {path}: forbidden artifact hash")
    return resolved


def require_scratch_stamp(metadata: dict[str, Any], path: str | Path) -> dict[str, Any]:
    """Drafter tensors must descend from this trainer's fresh initialization."""

    # Accept a safetensors header / config wrapping the stamp, or a bare stamp
    # (PROVENANCE.json beside resume checkpoints and exports).
    stamp = metadata.get(SCRATCH_STAMP_KEY, metadata if "lineage_root" in metadata else None)
    if isinstance(stamp, str):
        stamp = json.loads(stamp)
    if not isinstance(stamp, dict) or stamp.get("lineage_root") != SCRATCH_LINEAGE_ROOT:
        raise ForbiddenCheckpointError(
            f"refusing drafter load of {path}: missing Enntity scratch provenance stamp"
        )
    if stamp.get("warm_start") not in (None, "none"):
        raise ForbiddenCheckpointError(f"refusing drafter load of {path}: warm-started lineage")
    return stamp


def scratch_stamp(**extra: Any) -> dict[str, Any]:
    return {"lineage_root": SCRATCH_LINEAGE_ROOT, "warm_start": "none", **extra}


def guarded_safe_open(path: str | Path, role: str, framework: str = "pt", device: str = "cpu"):
    from safetensors import safe_open

    return safe_open(str(check_path(path, role)), framework=framework, device=device)


def guarded_load_file(path: str | Path, role: str, device: str = "cpu"):
    from safetensors.torch import load_file

    return load_file(str(check_path(path, role)), device=device)


def guarded_torch_load(path: str | Path, role: str):
    """Resume state: a torch file plus a sibling stamped ``PROVENANCE.json``."""

    import torch

    resolved = check_path(path, role)
    stamp_file = resolved.parent / "PROVENANCE.json"
    if not stamp_file.is_file():
        raise ForbiddenCheckpointError(f"refusing {role} load of {path}: no PROVENANCE.json")
    require_scratch_stamp(json.loads(stamp_file.read_text()), path)
    return torch.load(str(resolved), map_location="cpu", weights_only=True)


def check_target_dir(root: str | Path) -> dict[str, Any]:
    """Verify the frozen-tensor source is the GLM target and record its license.

    Refuses a directory that looks like a drafter (``dflash_config`` or a
    DFlash architecture) or whose card declares a non-commercial / no-derivatives
    license.  Returns the receipt fields to log.
    """

    root = Path(root)
    check_path(root / "config.json", "target")
    config = json.loads((root / "config.json").read_text())
    text_config = config.get("text_config", config)
    architectures = config.get("architectures") or []
    if "dflash_config" in config or any("dflash" in a.lower() or "dspark" in a.lower() for a in architectures):
        raise ForbiddenCheckpointError(f"{root} is a drafter checkpoint, not the GLM target")
    license_id = None
    readme = root / "README.md"
    if readme.is_file():
        front = readme.read_text(errors="replace").split("---")
        if len(front) >= 3:
            found = re.search(r"^license:\s*(\S+)", front[1], re.MULTILINE)
            license_id = found.group(1).strip().strip("'\"").lower() if found else None
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
        if license_id is None and (root / name).is_file():
            head = (root / name).read_text(errors="replace")[:4000].lower()
            license_id = "mit" if "mit license" in head else "unparsed:" + name
    if license_id and ("-nc" in license_id or "-nd" in license_id or "noncommercial" in license_id):
        raise ForbiddenCheckpointError(f"target license {license_id!r} forbids this use")
    return {
        "target_root": str(root),
        "architectures": architectures,
        "hidden_size": text_config.get("hidden_size"),
        "vocab_size": text_config.get("vocab_size"),
        "license_on_disk": license_id,
        "config_sha256": sha256_file(root / "config.json"),
    }


__all__ = [
    "ForbiddenCheckpointError",
    "check_path",
    "check_target_dir",
    "guarded_load_file",
    "guarded_safe_open",
    "guarded_torch_load",
    "require_scratch_stamp",
    "scratch_stamp",
    "SCRATCH_STAMP_KEY",
]
