#!/usr/bin/env bash
# Install vllm-project/speculators (Apache-2.0) at the revision this repo is tested against.
#
# A plain `pip install git+https://...` does not work at this revision: in a source build its setup.py points
# its `hs-connectors` dependency at the clone's own hs_connectors/ directory, which pip has already deleted.
# So clone once into third_party/ (git-ignored) and install both packages from the clone.
#
#   bash scripts/install_speculators.sh              # into the active Python environment
#   bash scripts/install_speculators.sh --print-deps # list what it would install, install nothing
set -euo pipefail
REV=824125a36600b5af8fdd1b18286da51ed60f60a8
DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/speculators"
PYTHON="${PYTHON:-python}"
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/vllm-project/speculators.git "$DIR"
fi
if [ "$(git -C "$DIR" rev-parse HEAD)" != "$REV" ]; then
  git -C "$DIR" fetch --quiet origin "$REV" 2>/dev/null || true
  git -C "$DIR" checkout --quiet "$REV"
fi
test "$(git -C "$DIR" rev-parse HEAD)" = "$REV"
# speculators lists torch, torchaudio and torchvision; its code imports neither torchaudio nor torchvision, and pip
# would pull wheels built for another torch (and upgrade torch: 2.13 breaks compiled flex attention on CPU). So both
# packages go in without their dependencies, then speculators' own dependencies minus those three, with the torch
# already installed pinned. Its `transformers>=5.0.0,<5.17.0` range is kept: a newer transformers is downgraded on
# purpose (speculators at this revision is tested below 5.17).
TORCH_PIN="$("$PYTHON" -c 'import torch; print("torch==" + torch.__version__.split("+")[0])' 2>/dev/null || echo 'torch>=2.11,<2.13')"
DEPS="$("$PYTHON" - "$DIR/pyproject.toml" <<'EOF_DEPS'
import re
import sys

block = open(sys.argv[1]).read().split("[tool.speculators.dependencies]", 1)[1]
base = block.split("base = [", 1)[1].split("]", 1)[0]
skip = {"torch", "torchaudio", "torchvision"}
print("\n".join(d for d in re.findall(r'"([^"]+)"', base) if re.split(r"[\s<>=!~;\[]", d, 1)[0] not in skip))
EOF_DEPS
)"
if [ "${1:-}" = "--print-deps" ]; then
  printf '%s\n' "$DEPS" "$TORCH_PIN"
  exit 0
fi
set -f   # the requirement specifiers are passed as words, never globbed
# shellcheck disable=SC2086
"$PYTHON" -m pip install $DEPS "$TORCH_PIN"
set +f
"$PYTHON" -m pip install --no-deps "$DIR/hs_connectors" "$DIR"
"$PYTHON" -c "import speculators; from speculators import DFlash2DraftModel; print('speculators', speculators.__file__)"
