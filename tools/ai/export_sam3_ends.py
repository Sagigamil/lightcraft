#!/usr/bin/env python3
"""Export the SAM 3 encoder's stem and FPN neck to ONNX and compile/link them for HTP.

Together with two 16-block chunks this completes the partitioned encoder:

    image -> stem -> blocks 0-16 -> blocks 16-32 -> neck -> 4 FPN levels + position encodings

Each graph fits the HTP graph-memory limit that the monolithic 1.0 GB encoder exceeds.

    python tools/ai/export_sam3_ends.py --checkpoint C:/path/sam3.pt --out tools/ai/onnx/ends
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def compile_link(onnx_path: Path, input_name: str, dims: tuple[int, ...], out_dir: Path, tag: str) -> dict:
    import qai_hub as hub

    model = hub.upload_model(str(onnx_path), name=f"sam3_{tag}")
    device = hub.Device("Samsung Galaxy S26")
    cj = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs={input_name: (dims, "float32")},
        options="--target_runtime qnn_dlc --compute_unit npu",
    )
    print(tag, "compile", cj.job_id, flush=True)
    while not cj.get_status().finished:
        time.sleep(20)
    print(tag, "compile", cj.get_status().code, flush=True)
    if not cj.get_status().success:
        cj.download_job_logs(str(out_dir / f"{tag}_logs"))
        raise SystemExit(f"{tag} compile failed")
    lj = hub.submit_link_job(models=[cj.get_target_model()], device=device, options="--qnn_options default_graph_htp_optimizations=O=1")
    print(tag, "link", lj.job_id, flush=True)
    while not lj.get_status().finished:
        time.sleep(20)
    print(tag, "link", lj.get_status().code, flush=True)
    if not lj.get_status().success:
        lj.download_job_logs(str(out_dir / f"{tag}_link_logs"))
        raise SystemExit(f"{tag} link failed")
    ctx = lj.download_target_model(str(out_dir / f"{tag}_context"))
    return {"model": model.model_id, "compile": cj.job_id, "link": lj.job_id, "context": str(ctx)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--skip-stem", action="store_true")
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

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
    vb = model.backbone.vision_backbone
    trunk = vb.trunk

    class Stem(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = trunk

        def forward(self, image: torch.Tensor) -> torch.Tensor:
            x = self.trunk.patch_embed(image)
            h, w = x.shape[1], x.shape[2]
            if self.trunk.pos_embed is not None:
                x = x + vitdet.get_abs_pos(
                    self.trunk.pos_embed,
                    self.trunk.pretrain_use_cls_token,
                    (h, w),
                    self.trunk.retain_cls_token,
                    tiling=self.trunk.tile_abs_pos,
                )
            return self.trunk.ln_pre(x)

    class Neck(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.convs = vb.sam2_convs
            self.pe = vb.position_encoding

        def forward(self, tokens: torch.Tensor):
            x = tokens.permute(0, 3, 1, 2).contiguous()
            outs = []
            poss = []
            for conv in self.convs:
                y = conv(x)
                outs.append(y)
                poss.append(self.pe(y).to(y.dtype))
            return (*outs, *poss)

    args.out.mkdir(parents=True, exist_ok=True)
    meta: dict = {}

    # stem: image -> tokens
    stem = Stem().eval()
    img = torch.zeros(1, 3, 1008, 1008)
    with torch.inference_mode():
        ref = stem(img)
        p = args.out / "stem.onnx"
        torch.onnx.export(stem, img, str(p), input_names=["image"], output_names=["tokens"], opset_version=17, dynamo=False)
    import onnxruntime as ort

    got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"image": img.numpy()})[0]
    err = float(np.abs(got - ref.detach().cpu().numpy()).max() / max(float(ref.abs().max()), 1e-6))
    print("stem", p, os.path.getsize(p), "rel err", err, flush=True)
    if err > 1e-4:
        raise SystemExit("stem mismatch")
    if args.skip_stem and (args.out / "stem_context.bin").is_file():
        meta["stem"] = {"context": str(args.out / "stem_context.bin")}
    else:
        meta["stem"] = compile_link(p, "image", (1, 3, 1008, 1008), args.out, "stem")

    # neck: tokens -> FPN + pos
    neck = Neck().eval()
    tokens = torch.zeros(1, 72, 72, 1024)
    with torch.inference_mode():
        refs = neck(tokens)
        p = args.out / "neck.onnx"
        names = [f"fpn{i}" for i in range(len(refs) // 2)] + [f"pos{i}" for i in range(len(refs) // 2)]
        torch.onnx.export(neck, tokens, str(p), input_names=["tokens"], output_names=names, opset_version=17, dynamo=False)
    got = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).run(None, {"tokens": tokens.numpy()})
    errs = [float(np.abs(a - b.detach().cpu().numpy()).max() / max(float(b.abs().max()), 1e-6)) for a, b in zip(got, refs)]
    print("neck", p, os.path.getsize(p), "rel errs", errs, flush=True)
    if max(errs) > 1e-4:
        raise SystemExit("neck mismatch")
    meta["neck"] = compile_link(p, "tokens", (1, 72, 72, 1024), args.out, "neck")

    json.dump(meta, open(args.out / "meta.json", "w"), indent=2)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
