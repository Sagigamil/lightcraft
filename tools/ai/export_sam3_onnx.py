#!/usr/bin/env python3
"""Export the SAM 3 vision encoder (trunk + SAM 2 neck) to ONNX for QNN/HTP.

The encoder is the expensive half of AI Object Select: it runs once per photo and produces the
feature pyramid the point decoder consumes.  This script exports exactly that graph from Meta's
`sam3.pt`, with fixed 1008x1008 input, and verifies the ONNX result against PyTorch.

Usage (from the repository root, in the ai-hub-models venv):

    python tools/ai/export_sam3_onnx.py \
        --checkpoint C:/path/sam3.pt --out tools/ai/onnx/sam3_encoder.onnx

Requires: torch, onnx, onnxruntime, and Meta's `sam3` package (pip install
git+https://github.com/facebookresearch/sam3.git).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True, help="Meta's sam3.pt")
    ap.add_argument("--out", type=Path, default=Path("tools/ai/onnx/sam3_encoder.onnx"))
    ap.add_argument("--size", type=int, default=1008, help="square input edge (SAM 3 uses 1008)")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    import numpy as np
    import torch
    import torch.nn as nn

    # The official package downloads from HF unless we pin the local checkpoint.
    import sam3.model_builder as builder
    # Meta's fused MLP casts to bf16 unconditionally (it targets CUDA). On CPU (and for an fp32
    # ONNX export) use the equivalent float32 linear + activation.
    import sam3.model.vitdet as _vitdet

    def _addmm_act_fp32(activation, linear, mat1):
        # 'activation' is the module class (e.g. nn.GELU), not an instance.
        act = activation() if isinstance(activation, type) else activation
        return act(linear(mat1))

    _vitdet.addmm_act = _addmm_act_fp32

    # The default ViT uses complex-valued RoPE (view_as_complex), which ONNX cannot express.
    # Build the same weights with the real-arithmetic RoPE so the graph exports.
    _orig_vit = builder._create_vit_backbone
    builder._create_vit_backbone = lambda **kw: _orig_vit(**{**kw, "use_rope_real": True})

    if not args.checkpoint.is_file():
        raise SystemExit(f"{args.checkpoint}: no such checkpoint")
    builder.download_ckpt_from_hf = lambda version="sam3": str(args.checkpoint)  # type: ignore[assignment]

    t = time.time()
    model = builder.build_sam3_image_model(
        device="cpu",
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        enable_inst_interactivity=True,
    ).eval()
    # the builder may leave bf16 modules on CPU; export in fp32 like the Rust engine
    model = model.float()
    print(f"model loaded in {time.time() - t:.1f}s", flush=True)

    vb = model.backbone.vision_backbone

    class Encoder(nn.Module):
        """pixel_values -> SAM 2 FPN levels and their positional encodings (the tracker's input)."""

        def __init__(self, vb: nn.Module) -> None:
            super().__init__()
            self.vb = vb

        def forward(self, image: torch.Tensor):
            _sam3_f, _sam3_p, sam2_f, sam2_p = self.vb(image)
            if sam2_f is None or sam2_p is None:
                raise RuntimeError("the checkpoint has no SAM 2 branch; cannot export the point tracker")
            return (*sam2_f, *sam2_p)

    enc = Encoder(vb).eval()
    with torch.inference_mode():
        dummy = torch.zeros(1, 3, args.size, args.size, dtype=torch.float32)
        t = time.time()
        ref = enc(dummy)
        print(f"reference forward in {time.time() - t:.1f}s: {len(ref)} outputs {[tuple(x.shape) for x in ref]}", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        t = time.time()
        torch.onnx.export(
            enc,
            dummy,
            str(args.out),
            input_names=["image"],
            output_names=[f"fpn{i}" for i in range(len(ref) // 2)] + [f"pos{i}" for i in range(len(ref) // 2)],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
        print(f"exported {args.out} in {time.time() - t:.1f}s ({args.out.stat().st_size / 1e6:.0f} MB)", flush=True)

    # Verify the ONNX graph against the reference on the same dummy input.
    import onnxruntime as ort

    sess = ort.InferenceSession(str(args.out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": dummy.numpy()})
    ok = True
    for i, (a, b) in enumerate(zip(got, ref)):
        a = np.asarray(a, dtype=np.float32)
        b = b.detach().cpu().numpy().astype(np.float32)
        err = float(np.abs(a - b).max())
        scale = max(float(np.abs(b).max()), 1e-6)
        print(f"  output {i}: shape {a.shape}, max abs err {err:.2e} (rel {err / scale:.2e})")
        ok = ok and err / scale < 1e-4
    if not ok:
        raise SystemExit("ONNX output does not match PyTorch")
    print("ONNX verified against PyTorch")
    return 0


if __name__ == "__main__":
    sys.exit(main())
