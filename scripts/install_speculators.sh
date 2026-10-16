#!/usr/bin/env bash
# Install vllm-project/speculators (Apache-2.0) at the revision this repo is tested against.
#
# A plain `pip install git+https://...` does not work at this revision: in a source build its setup.py points
# its `hs-connectors` dependency at the clone's own hs_connectors/ directory, which pip has already deleted.
# So clone once into third_party/ (git-ignored) and install both packages from the clone.
#
#   bash scripts/install_speculators.sh            # into the active Python environment
set -euo pipefail
REV=824125a36600b5af8fdd1b18286da51ed60f60a8
DIR="$(cd "$(dirname "$0")/.." && pwd)/third_party/speculators"
PYTHON="${PYTHON:-python}"
if [ ! -d "$DIR/.git" ]; then
  git clone https://github.com/vllm-project/speculators.git "$DIR"
fi
git -C "$DIR" fetch --quiet origin "$REV" 2>/dev/null || true
git -C "$DIR" checkout --quiet "$REV"
test "$(git -C "$DIR" rev-parse HEAD)" = "$REV"
# Keep the torch already installed: speculators' own dependency range would upgrade it, and torch 2.13
# breaks compiled flex attention on CPU. This repo is tested with torch 2.11.
TORCH_PIN="$("$PYTHON" -c 'import torch; print("torch==" + torch.__version__.split("+")[0])' 2>/dev/null || echo 'torch>=2.11,<2.13')"
"$PYTHON" -m pip install "$DIR/hs_connectors" "$DIR" "$TORCH_PIN"
"$PYTHON" -c "import speculators; from speculators import DFlash2DraftModel; print('speculators', speculators.__file__)"
