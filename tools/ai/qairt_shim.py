#!/usr/bin/env python3
"""Run a QAIRT Python tool with the onnx.version shim its converter expects.

QAIRT 2.50's converter reads ``onnx.version.version``; ONNX 1.19+ moved that to
``onnx.__version__``.  This wrapper installs a compatible module before running the tool, so the
conversion works with the current ONNX package instead of downgrading it.

Usage:
    python tools/ai/qairt_shim.py <path-to-qairt-tool> [tool arguments...]
"""

import runpy
import sys
import types

import onnx

if not hasattr(onnx, "version"):
    module = types.ModuleType("onnx.version")
    module.version = onnx.__version__
    sys.modules["onnx.version"] = module
    onnx.version = module

sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
