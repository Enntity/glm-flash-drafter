#!/usr/bin/env python3
"""Build the Enntity-authored synthetic prompt sets (local; sends nothing).

    python scripts/generate_prompts.py templates --out data/prompts/templates
    python scripts/generate_prompts.py coverage --out data/prompts/coverage1 [--no-tokens]
    python scripts/generate_prompts.py coverage --out data/prompts/coverage2 --profile long --prefix cov2 \\
        --exclude data/prompts/coverage1/train.jsonl

``templates``: enntity_drafter.prompts.templates (round 1's 1,592 template prompts).
``coverage``: enntity_drafter.prompts.coverage (round 2b's structured-output prompts); token counts need
``GLM_TOKENIZER_DIR`` unless ``--no-tokens``.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter.prompts import coverage, templates  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("templates", "coverage"):
        raise SystemExit(__doc__)
    command = sys.argv.pop(1)
    if command == "templates":
        templates.main(sys.argv[1:])
    else:
        coverage.main()
