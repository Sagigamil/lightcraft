#!/usr/bin/env python3
"""Export MobileSAM's prompt decoder with 8 points to ONNX.

Qualcomm's AI Hub MobileSAM artifact ships a decoder with a fixed single point, but LightCraft's
Object mask sends every positive and negative click at once.  This exports the same weights with
8 prompt slots (unused slots are padded with label -1), keeping the encoder and the mask
semantics identical.

The wrapper keeps the artifact's external layouts: image embeddings are NHWC [1,64,64,256] (the
encoder's output) and the mask is NHWC [1,256,256,1].  Everything in between is the original
decoder.

Usage:
    python tools/ai/export_mobilesam_decoder.py --out tools/ai/mobilesam/decoder8.onnx

The MobileSAM checkpoint (mobile_sam.pt, 40.7 MB, Apache-2.0) is downloaded once by
qai_hub_models to ~/.qaihm.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import torch
import torch.nn as nn


class DecoderNHWC(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, image_embeddings, point_coords, point_labels):
        x = image_embeddings.permute(0, 3, 1, 2).contiguous()
        masks, scores = self.decoder(x, point_coords, point_labels)
        return masks.permute(0, 2, 3, 1).contiguous(), scores


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--points", type=int, default=8, help="prompt slots (unused: label -1)")
    args = ap.parse_args()

    from qai_hub_models.models.mobilesam.model import MobileSAMDecoder

    decoder = MobileSAMDecoder.from_pretrained(single_mask_mode=True).eval()
    model = DecoderNHWC(decoder).eval()

    n = args.points
    embeddings = torch.randn(1, 64, 64, 256)
    coords = torch.rand(1, n, 2) * 1024.0
    labels = torch.ones(1, n)
    # exercise padding: the last slots are "no point"
    labels[0, 3:] = -1.0
    with torch.no_grad():
        ref_masks, ref_scores = model(embeddings, coords, labels)

    torch.onnx.export(
        model,
        (embeddings, coords, labels),
        args.out,
        input_names=["image_embeddings", "point_coords", "point_labels"],
        output_names=["masks", "scores"],
        opset_version=17,
        dynamo=False,
    )

    import onnx
    import onnxruntime as ort

    onnx.checker.check_model(args.out)
    session = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    got_masks, got_scores = session.run(
        None,
        {
            "image_embeddings": embeddings.numpy(),
            "point_coords": coords.numpy(),
            "point_labels": labels.numpy(),
        },
    )
    err_m = float(np.abs(got_masks - ref_masks.numpy()).max())
    err_s = float(np.abs(got_scores - ref_scores.numpy()).max())
    print(f"onnx masks {got_masks.shape} max err {err_m:.2e}; scores {got_scores.shape} max err {err_s:.2e}")
    if err_m > 1e-3 or err_s > 1e-3:
        print("ONNX does not match PyTorch", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
