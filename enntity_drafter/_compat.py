"""Import-time compatibility shims.

speculators imports Hugging Face ``datasets`` at module load (its data-pipeline
modules), although this trainer never calls it. Some images ship a ``datasets``
that cannot import (for example without ``pandas``), which would make every
``import speculators`` fail. :func:`ensure_datasets_importable` leaves a working
``datasets`` alone and otherwise installs a small stand-in whose functions fail
loudly if anything ever calls them.
"""

from __future__ import annotations

import sys
import types


def _unavailable(*_args, **_kwargs):
    raise RuntimeError("Hugging Face `datasets` is not importable in this environment; "
                       "the drafter trainer does not use it (stand-in from enntity_drafter._compat)")


def _stub() -> types.ModuleType:
    module = types.ModuleType("datasets")
    module.__doc__ = "Stand-in installed by enntity_drafter._compat: the real package failed to import."
    module.load_from_disk = module.load_dataset = module.concatenate_datasets = _unavailable

    class Dataset:  # noqa: D401 - placeholder type for annotations and isinstance checks
        def __init__(self, *_args, **_kwargs):
            _unavailable()

    module.Dataset = Dataset
    module.ENNTITY_STUB = True
    return module


def ensure_datasets_importable() -> bool:
    """Return True if the real ``datasets`` imports; otherwise install the stand-in and return False."""

    try:
        import datasets  # noqa: F401
        return True
    except Exception:  # noqa: BLE001 - any failure (missing pandas, broken install) gets the stand-in
        for name in [n for n in sys.modules if n == "datasets" or n.startswith("datasets.")]:
            del sys.modules[name]
        sys.modules["datasets"] = _stub()
        return False
