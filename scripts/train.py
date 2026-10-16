#!/usr/bin/env python3
"""Train the drafter (see docs/training.md).  Same as ``python -m enntity_drafter.train``.

    torchrun --nproc_per_node=1 scripts/train.py --config configs/full.json --teacher /models/GLM-5.3-Flash \\
        --capture-index data/index/index.json --capture-root data --output runs/r1 ...
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter.train import main  # noqa: E402

if __name__ == "__main__":
    main()
