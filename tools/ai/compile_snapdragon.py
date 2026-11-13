#!/usr/bin/env python3
"""Compile the SAM 3 encoder ONNX for the Snapdragon Hexagon NPU (Qualcomm AI Hub).

This is the reproducible compile step of the NPU pipeline. It uploads the verified ONNX encoder,
submits a QNN context-binary compile job for the Snapdragon 8 Elite Gen 5, waits for it and
downloads the compiled context.  It fails loudly when the job cannot be produced.

Usage:
    python tools/ai/compile_snapdragon.py --onnx tools/ai/onnx/sam3_encoder.onnx
    python tools/ai/compile_snapdragon.py --onnx ... --device "Samsung Galaxy S26" --precision fp16

Requires: qai-hub (authenticated with `qai-hub configure --api_token ...`).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# The target used by the AI Hub catalog for the Snapdragon 8 Elite Gen 5.  The Lenovo Y700 is
# SM8850P (same chip family, "canoe" platform); verify the compiled context on the device.
DEFAULT_DEVICE = "Samsung Galaxy S26"
DEFAULT_INPUT = "image:1,3,1008,1008"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--device", default=DEFAULT_DEVICE)
    ap.add_argument("--input", default=DEFAULT_INPUT, help="name:shape,shape,shape")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32", "float16", "float32"])
    ap.add_argument("--target-runtime", default="precompiled_qnn_onnx")
    ap.add_argument("--model-id", default=None, help="reuse an already uploaded AI Hub model")
    ap.add_argument("--link", action="store_true", help="also link the compiled model into a QNN context binary")
    ap.add_argument("--out", type=Path, default=Path("tools/ai/onnx"))
    ap.add_argument("--timeout", type=int, default=7200, help="seconds to wait for the job")
    args = ap.parse_args()

    if not args.onnx.is_file():
        raise SystemExit(f"{args.onnx}: no such ONNX model")

    import qai_hub as hub

    name, shape = args.input.split(":")
    dims = tuple(int(x) for x in shape.split(","))
    precision = "float16" if args.precision in ("fp16", "float16") else "float32"

    if args.model_id:
        model = hub.get_model(args.model_id)
        print(f"reusing uploaded model {model.model_id}", flush=True)
    else:
        print(f"uploading {args.onnx} ({args.onnx.stat().st_size / 1e6:.0f} MB)", flush=True)
        t = time.time()
        model = hub.upload_model(str(args.onnx), name=args.onnx.stem)
        print(f"uploaded as {model.model_id} in {time.time() - t:.0f}s", flush=True)

    device = hub.Device(args.device)
    options = f"--target_runtime {args.target_runtime} --compute_unit npu"
    print(f"submitting compile job: {device.name} / {options}", flush=True)
    job = hub.submit_compile_job(
        model=model,
        device=device,
        input_specs={name: (dims, "float32")},
        options=options,
    )
    print(f"job {job.job_id} submitted", flush=True)

    t = time.time()
    while True:
        status = job.get_status()
        if status.finished:
            break
        if time.time() - t > args.timeout:
            raise SystemExit(f"compile job {job.job_id} timed out after {args.timeout}s")
        print(f"  {status.code} ({time.time() - t:.0f}s)", flush=True)
        time.sleep(30)

    print(f"final status: {status.code}", flush=True)
    if not status.success:
        log = job.download_job_logs(str(args.out / "compile_logs"))
        print(f"logs: {log}", flush=True)
        raise SystemExit(f"compile job {job.job_id} failed")

    args.out.mkdir(parents=True, exist_ok=True)
    target = job.download_target_model(str(args.out / "sam3_encoder_qnn"))
    print(f"downloaded: {target}", flush=True)

    meta = {
        "source": str(args.onnx),
        "device": device.name,
        "target_runtime": args.target_runtime,
        "precision": precision,
        "input": {name: list(dims)},
        "job_id": job.job_id,
        "target_model": str(target),
        "compiled_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if args.link:
        link = hub.submit_link_job(
            models=[target],
            device=device,
            options="--target_runtime qnn_context_binary",
        )
        print(f"link job {link.job_id} submitted", flush=True)
        while not link.get_status().finished:
            time.sleep(30)
        print(f"link status: {link.get_status().code}", flush=True)
        if not link.get_status().success:
            link.download_job_logs(str(args.out / "link_logs"))
            raise SystemExit("link job failed")
        context = link.download_target_model(str(args.out / "sam3_encoder_context"))
        meta["link_job_id"] = link.job_id
        meta["context"] = str(context)
        print(f"context binary: {context}", flush=True)

    (args.out / "sam3_encoder_qnn.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
