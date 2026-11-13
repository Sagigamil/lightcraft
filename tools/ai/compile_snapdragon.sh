#!/usr/bin/env bash
# Compile the SAM 3 encoder for the Snapdragon Hexagon NPU (Qualcomm AI Hub).
#
# The real work is in compile_snapdragon.py; this wrapper keeps the historical
# `tools/ai/compile_snapdragon.sh` entry point and validates the environment first.
#
#   ./tools/ai/compile_snapdragon.sh --onnx tools/ai/onnx/sam3_encoder.onnx
#
# Requires an authenticated AI Hub client (`qai-hub configure --api_token ...`) and a
# Python environment with qai-hub.  Set LIGHTCRAFT_PYTHON to use a specific interpreter.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python="${LIGHTCRAFT_PYTHON:-python3}"

if ! "$python" -c "import qai_hub" >/dev/null 2>&1; then
  echo "error: qai-hub is not importable by $python (set LIGHTCRAFT_PYTHON)" >&2
  exit 1
fi

if [ ! -f "$HOME/.qai_hub/client.ini" ]; then
  echo "error: no AI Hub credentials; run: qai-hub configure --api_token <token>" >&2
  exit 1
fi

exec "$python" "$here/compile_snapdragon.py" "$@"
