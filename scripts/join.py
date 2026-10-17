#!/usr/bin/env python3
"""Run a volunteer training site (see JOIN.md).  Same as ``drafter-join`` / ``python -m enntity_drafter.join``.

    python scripts/join.py redeem '<invite link>' --accept-terms --name "<display name>"
    python scripts/join.py setup --teacher /models/GLM-5.3-Flash ...
    python scripts/join.py run --teacher /models/GLM-5.3-Flash ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter.join import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
