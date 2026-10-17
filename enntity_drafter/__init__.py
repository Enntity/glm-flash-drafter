"""Openly trained DFlash2-style speculative-decoding drafter for GLM-5.3-Flash.

Modules: :mod:`.model` (architecture + scratch init), :mod:`.objectives`,
:mod:`.data` (row store + synthetic fixtures), :mod:`.capture_io` (Atlas
hidden-state capture reader, sampling, FP8 codec), :mod:`.fp8` (in-place FP8
conversion), :mod:`.fastpath` (opt-in performance path), :mod:`.train`,
:mod:`.export` (Atlas BF16 drafter format) and :mod:`.guard` (license guard).
"""

__version__ = "0.1.0"

from ._compat import ensure_datasets_importable as _ensure_datasets_importable

_ensure_datasets_importable()   # speculators imports `datasets` at load; see enntity_drafter._compat
