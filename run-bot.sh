#!/usr/bin/env bash
# Launch the Discord transcription bot with the CUDA libs on the loader path.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE="$DIR/.venv/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:${LD_LIBRARY_PATH:-}"
exec "$DIR/.venv/bin/python" "$DIR/bot.py"
