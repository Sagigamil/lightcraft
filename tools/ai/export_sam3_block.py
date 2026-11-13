#!/usr/bin/env python3
"""Export one SAM 3 ViT block to ONNX and compile/link it for the Snapdragon HTP.

Used to isolate which part of the encoder breaks HTP graph execution (global vs windowed
attention, or cumulative graph size).  The full encoder fails after ~11 s with QNN err 1003;
a single block is the smallest meaningful test.

    python tools/ai/export_sam3_block.py --block 7 --out tools/ai/onnx/block7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--block", type=int, required=True)
    ap.add_argument("--end", type=int, default=None, help="exclusive end block (default: block+1)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=1008)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    import sam3.model_builder as builder
    import sam3.model.vitdet as vitdet

    def addmm_act_fp32(activation, linear, mat1):
        act = activation() if isinstance(activation, type) else activation
        return act(linear(mat1))

    vitdet.addmm_act = addmm_act_fp32
    orig_vit = builder._create_vit_backbone
    builder._create_vit_backbone = lambda **kw: orig_vit(**{**kw, "use_rope_real": True})
    builder.download_ckpt_from_hf = lambda version="sam3": str(args.checkpoint)  # type: ignore[assignment]

    model = builder.build_sam3_image_model(
        device="cpu", checkpoint_path=str(args.checkpoint), load_from_HF=False, enable_inst_interactivity=True
    ).eval()
    model = model.float()
    trunk = model.backbone.vision_backbone.trunk
    end = args.end if args.end is not None else args.block + 1
    if not 0 <= args.block < end <= len(trunk.blocks):
        raise SystemExit(f"blocks {args.block}:{end} out of range 0..{len(trunk.blocks)}")
    blk = nn.Sequential(*trunk.blocks[args.block:end]).eval()
    token_side = args.size // 14

    args.out.mkdir(parents=True, exist_ok=True)
    onnx_path = args.out / f"blocks{args.block}_{end}.onnx"
    dummy = torch.zeros(1, token_side, token_side, 1024, dtype=torch.float32)
    with torch.inference_mode():
        ref = blk(dummy)
        torch.onnx.export(blk, dummy, str(onnx_path), input_names=["x"], output_names=["y"], opset_version=17, dynamo=False)
    print("exported", onnx_path, os.path.getsize(onnx_path), flush=True)

    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"x": dummy.numpy()})[0]
    err = float(np.abs(got - ref.detach().cpu().numpy()).max())
    scale = max(float(ref.abs().max()), 1e-6)
    print(f"onnx check: max abs err {err:.2e} rel {err / scale:.2e}", flush=True)
    if err / scale > 1e-4:
        raise SystemExit("ONNX mismatch")

    if args.skip_compile:
        return 0

    import qai_hub as hub

    model_hub = hub.upload_model(str(onnx_path), name=f"sam3_block{args.block}")
    device = hub.Device("Samsung Galaxy S26")
    cj = hub.submit_compile_job(
        model=model_hub,
        device=device,
        input_specs={"x": ((1, token_side, token_side, 1024), "float32")},
        options="--target_runtime qnn_dlc --compute_unit npu",
    )
    print("compile", cj.job_id, flush=True)
    while not cj.get_status().finished:
        time.sleep(20)
    print("compile", cj.get_status().code, flush=True)
    if not cj.get_status().success:
        cj.download_job_logs(str(args.out / "logs"))
        raise SystemExit(1)
    dlc = cj.get_target_model()
    lj = hub.submit_link_job(models=[dlc], device=device, options="--qnn_options default_graph_htp_optimizations=O=1")
    print("link", lj.job_id, flush=True)
    while not lj.get_status().finished:
        time.sleep(20)
    print("link", lj.get_status().code, flush=True)
    if not lj.get_status().success:
        lj.download_job_logs(str(args.out / "logs_link"))
        raise SystemExit(1)
    ctx = lj.download_target_model(str(args.out / f"blocks{args.block}_{end}_context"))
    print("context", ctx, flush=True)
    json.dump(
        {"block": args.block, "window": int(getattr(blk, "window_size", -1)), "model": model_hub.model_id, "compile": cj.job_id, "link": lj.job_id, "context": str(ctx)},
        open(args.out / "meta.json", "w"),
        indent=2,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
