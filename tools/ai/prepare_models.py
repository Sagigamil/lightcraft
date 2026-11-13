#!/usr/bin/env python3
"""Prepare the SAM 3 point-prompt checkpoint for Lightcraft AI on Android.

The Android AI flavor embeds only the point-prompt components of SAM 3 (the
vision encoder, ``tracker_neck`` and ``tracker_model``), stored as F16 so the APK
stays below Android's install limits. The model is loaded and computed in F32 by
``lightcraft-segment``.

Sources (in order):

* ``--source-pt PATH``: Meta's original ``sam3.pt`` checkpoint. This is the
  recommended offline path and the one your local download uses. It is converted
  with the verified name/shape mapping in ``tools/ai/sam3_point_mapping.json``.
* ``--source-dir`` with ``model.safetensors``: the official Hugging Face export
  (``facebook/sam3``); its SHA-256 is verified before use.
* ``--download``: fetch the official files from the public mirror.

Usage (from the repository root):

    python tools/ai/prepare_models.py --accept-license --source-pt C:/path/sam3.pt
    python tools/ai/prepare_models.py --accept-license --source-dir models/sam3
    python tools/ai/prepare_models.py --accept-license --download
    python tools/ai/prepare_models.py --check

The weights are Meta's SAM License, not LightCraft's. The output directory
(``android/app/src/ai/assets/sam3``) is gitignored; weights are never committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from pathlib import Path

OFFICIAL_SHA256 = "6d06f0a5f84e435071fe6603e61d0b4cc7b40e0d39d487cfd4d67d8cc11cc14a"
OFFICIAL_BYTES = 3_439_938_512
REPO_ID = "jetjodh/sam3"  # public mirror of the official facebook/sam3 checkpoint
OFFICIAL_FILES = ("model.safetensors", "vocab.json", "merges.txt", "LICENSE")
AUX_FILES = ("vocab.json", "merges.txt", "LICENSE")
LICENSE_NAME = "SAM License (Meta)"
LICENSE_URL = "https://github.com/facebookresearch/sam3/blob/main/LICENSE"
# The point-prompt path in crates/segment reads these prefixes.
POINT_PREFIXES = ("detector_model.vision_encoder.", "tracker_neck.", "tracker_model.")
CHUNK = 64 * 1024 * 1024  # multiple of 4 bytes


def root_dir() -> Path:
    return Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_header(path: Path) -> tuple[int, dict]:
    with path.open("rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise SystemExit(f"{path}: not a safetensors file")
        n = struct.unpack("<Q", raw)[0]
        if n > 100 << 20:
            raise SystemExit(f"{path}: safetensors header is too large ({n} bytes)")
        header = json.loads(f.read(n))
    if not isinstance(header, dict):
        raise SystemExit(f"{path}: bad safetensors header")
    return n, header


def download(source: Path, repo: str) -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("huggingface_hub is required: python -m pip install -r tools/requirements-ai.txt")
    source.mkdir(parents=True, exist_ok=True)
    for name in OFFICIAL_FILES:
        target = source / name
        if target.is_file() and target.stat().st_size > 0:
            print(f"using {target}")
            continue
        print(f"downloading {repo}/{name} ...", flush=True)
        hf_hub_download(repo_id=repo, filename=name, local_dir=str(source))
        print(f"downloaded {target}")


def write_pt_asset(pt: Path, dst: Path, mapping_path: Path) -> dict:
    try:
        import torch
    except ImportError:
        raise SystemExit("PyTorch is required to read sam3.pt: python -m pip install torch (or use --source-dir with model.safetensors)")
    try:
        import numpy  # noqa: F401
    except ImportError:
        raise SystemExit("numpy is required: python -m pip install -r tools/requirements-ai.txt")
    if not mapping_path.is_file():
        raise SystemExit(f"{mapping_path}: missing the verified tensor mapping")
    mapping = json.loads(mapping_path.read_text())
    entries = mapping.get("mapping")
    if not isinstance(entries, dict) or not entries:
        raise SystemExit(f"{mapping_path}: bad mapping (no `mapping` object)")

    print(f"loading {pt} (PyTorch weights-only, memory-mapped) ...", flush=True)
    obj = torch.load(pt, map_location="cpu", mmap=True, weights_only=True)
    if not isinstance(obj, dict) or not obj:
        raise SystemExit(f"{pt}: not a state dict")

    names = sorted(entries)
    header: dict[str, dict] = {}
    offset = 0
    for name in names:
        spec = entries[name]
        shape = spec.get("shape")
        if not isinstance(shape, list) or not all(isinstance(v, int) and v >= 0 for v in shape):
            raise SystemExit(f"{mapping_path}: {name}: missing a valid shape")
        count = 1
        for v in shape:
            count *= v
        size = count * 2  # F16
        header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size
    encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)

    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with tmp.open("wb") as fo:
            fo.write(struct.pack("<Q", len(encoded)))
            fo.write(encoded)
            for i, name in enumerate(names):
                spec = entries[name]
                op = spec.get("op")
                if op == "direct":
                    t = obj[spec["source"]]
                elif op == "split3":
                    t = obj[spec["source"]].chunk(3, dim=0)[int(spec["index"])]
                elif op == "stack":
                    t = torch.cat([obj[s] for s in spec["sources"]], dim=0)
                elif op == "slice":
                    t = obj[spec["source"]]
                    axis = int(spec["axis"])
                    start = int(spec["start"])
                    end = spec.get("end")
                    length = (int(end) if end is not None else t.shape[axis]) - start
                    t = t.narrow(axis, start, length)
                else:
                    raise SystemExit(f"{mapping_path}: {name}: unknown op {op!r}")
                if tuple(t.shape) != tuple(spec["shape"]):
                    raise SystemExit(f"{name}: shape {tuple(t.shape)} != mapped {tuple(spec['shape'])}")
                arr = t.detach().cpu().to(torch.float16).contiguous().numpy().astype("<f2", copy=False)
                data = arr.tobytes()
                got = hashlib.sha256(data).hexdigest()
                if got != spec.get("sha256"):
                    raise SystemExit(f"{name}: tensor hash mismatch (the checkpoint is not the one this mapping was verified against)")
                fo.write(data)
                if i % 100 == 0:
                    print(f"wrote {i}/{len(names)} tensors ...", flush=True)
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("safetensors is required: python -m pip install -r tools/requirements-ai.txt")
    with safe_open(str(dst), framework="np") as f:
        keys = set(f.keys())
        missing = [name for name in names if name not in keys]
        if missing:
            raise SystemExit(f"{dst}: missing tensors after conversion: {missing[:3]}")
        extra = keys.difference(names)
        if extra:
            raise SystemExit(f"{dst}: unexpected tensors after conversion: {sorted(extra)[:3]}")
    return {"tensors": len(names), "bytes": dst.stat().st_size, "sha256": sha256(dst), "mappingSha256": sha256(mapping_path)}


def convert_official(src: Path, dst: Path, point_only: bool) -> dict:
    n, header = read_header(src)
    items = [(name, spec) for name, spec in header.items() if name != "__metadata__"]
    if point_only:
        items = [(name, spec) for name, spec in items if name.startswith(POINT_PREFIXES)]
    if not items:
        raise SystemExit(f"{src}: no tensors selected")
    # Keep the original data order so the new offsets describe the file we write.
    items.sort(key=lambda item: item[1]["data_offsets"][0])

    out_header: dict[str, dict] = {}
    offset = 0
    for name, spec in items:
        if spec.get("dtype") != "F32":
            raise SystemExit(f"{name}: expected F32, got {spec.get('dtype')}")
        shape = spec["shape"]
        start, end = spec["data_offsets"]
        size = end - start
        if size % 4:
            raise SystemExit(f"{name}: F32 tensor size {size} is not a multiple of 4")
        out_size = (size // 4) * 2
        out_header[name] = {"dtype": "F16", "shape": shape, "data_offsets": [offset, offset + out_size]}
        offset += out_size
    meta = header.get("__metadata__")
    if isinstance(meta, dict):
        out_header["__metadata__"] = meta
    encoded = json.dumps(out_header, separators=(",", ":")).encode("utf-8")
    encoded += b" " * ((-len(encoded)) % 8)

    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with src.open("rb") as fi, tmp.open("wb") as fo:
            fo.write(struct.pack("<Q", len(encoded)))
            fo.write(encoded)
            for name, spec in items:
                start, end = spec["data_offsets"]
                remaining = end - start
                fi.seek(8 + n + start)
                while remaining:
                    chunk = min(remaining, CHUNK)
                    buf = fi.read(chunk)
                    if len(buf) != chunk:
                        raise SystemExit(f"{name}: unexpected end of {src}")
                    import numpy as np

                    arr = np.frombuffer(buf, dtype="<f4")
                    fo.write(arr.astype("<f2").tobytes())
                    remaining -= chunk
        os.replace(tmp, dst)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise

    try:
        from safetensors import safe_open
    except ImportError:
        raise SystemExit("safetensors is required: python -m pip install -r tools/requirements-ai.txt")
    with safe_open(str(dst), framework="np") as f:
        keys = list(f.keys())
        missing = [name for name, _ in items if name not in keys]
        if missing:
            raise SystemExit(f"{dst}: missing tensors after conversion: {missing[:3]}")
        for name, spec in items[:3]:
            t = f.get_tensor(name)
            if list(t.shape) != spec["shape"]:
                raise SystemExit(f"{dst}: {name} shape changed")
    return {"tensors": len(items), "bytes": dst.stat().st_size, "sha256": sha256(dst)}


def copy_aux(out: Path, tokenizer_dir: Path | None, source_dir: Path | None) -> None:
    for name in AUX_FILES:
        candidates = []
        if tokenizer_dir is not None:
            candidates.append(tokenizer_dir / name)
        if source_dir is not None:
            candidates.append(source_dir / name)
        found = next((p for p in candidates if p.is_file()), None)
        if found is None:
            where = ", ".join(str(p) for p in candidates) or "no directory configured"
            raise SystemExit(f"missing {name} (looked in {where}); the Android checkpoint needs the SAM 3 tokenizer and licence")
        shutil.copyfile(found, out / name)


def write_manifest(out: Path, args, info: dict, source_kind: str, checkpoint: dict) -> dict:
    dst = out / "model.safetensors"
    files = {}
    for name in ("model.safetensors",) + AUX_FILES:
        p = out / name
        files[name] = {"size": p.stat().st_size, "sha256": sha256(p)}
    manifest = {
        "formatVersion": 1,
        "model": "SAM 3",
        "checkpoint": checkpoint["name"],
        "checkpointFormat": checkpoint["format"],
        "checkpointSource": checkpoint["source"],
        "checkpointSha256": checkpoint["sha256"],
        "checkpointBytes": checkpoint["bytes"],
        "storageDtype": "F16",
        "computeDtype": "F32",
        "components": ["vision_encoder", "tracker_neck", "tracker_model"] + (["text_encoder", "detector"] if args.full else []),
        "textPrompts": bool(args.full),
        "inputSize": [1, 3, 1008, 1008],
        "maskSide": 288,
        "runtime": "candle CPU (ARM64 NEON)",
        "npu": False,
        "tensors": info.get("tensors"),
        "mapping": "tools/ai/sam3_point_mapping.json" if source_kind == "pt" else None,
        "mappingSha256": info.get("mappingSha256"),
        "license": LICENSE_NAME,
        "licenseUrl": LICENSE_URL,
        "files": files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def check(out: Path) -> int:
    manifest_path = out / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing {manifest_path} (run the preparation script first)")
    manifest = json.loads(manifest_path.read_text())
    for name, info in manifest.get("files", {}).items():
        p = out / name
        if not p.is_file():
            raise SystemExit(f"{p}: missing")
        if p.stat().st_size != info["size"]:
            raise SystemExit(f"{p}: expected {info['size']} bytes, got {p.stat().st_size}")
        if sha256(p) != info["sha256"]:
            raise SystemExit(f"{p}: sha256 does not match the manifest (damaged or replaced)")
    print(f"OK: {manifest.get('model')} ({manifest.get('checkpoint')}) in {out}")
    print(f"    model {manifest['files']['model.safetensors']['size'] / 1e9:.2f} GB, {manifest.get('tensors')} tensors, textPrompts={manifest.get('textPrompts')}")
    return 0


def main() -> int:
    root = root_dir()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", type=Path, default=root / "models" / "sam3", help="directory with model.safetensors or sam3.pt (default: models/sam3)")
    ap.add_argument("--source-pt", type=Path, default=None, help="Meta's original sam3.pt checkpoint (recommended offline path)")
    ap.add_argument("--out", type=Path, default=root / "android" / "app" / "src" / "ai" / "assets" / "sam3", help="AI asset directory")
    ap.add_argument("--mapping", type=Path, default=root / "tools" / "ai" / "sam3_point_mapping.json", help="verified sam3.pt -> HF tensor mapping")
    ap.add_argument("--tokenizer-dir", type=Path, default=root / "tools" / "ai" / "tokenizer", help="directory with vocab.json, merges.txt and LICENSE")
    ap.add_argument("--full", action="store_true", help="keep the text-prompt components too (official safetensors only)")
    ap.add_argument("--no-convert", action="store_true", help="copy the F32 checkpoint instead of converting to F16 (official safetensors only)")
    ap.add_argument("--download", action="store_true", help="download missing official files from the checkpoint repo")
    ap.add_argument("--repo", default=REPO_ID, help=f"Hugging Face checkpoint repo (default: {REPO_ID}; use facebook/sam3 with HF_TOKEN for the official gated repo)")
    ap.add_argument("--accept-license", action="store_true", help="confirm you accept the SAM License")
    ap.add_argument("--force", action="store_true", help="rebuild even if the output already exists")
    ap.add_argument("--check", action="store_true", help="only verify an existing output")
    args = ap.parse_args()

    out: Path = args.out
    if args.check:
        return check(out)
    if not args.accept_license:
        print(
            "The SAM 3 weights are released under the SAM License (Meta), not LightCraft's licence.\n"
            f"Read it at {LICENSE_URL} and pass --accept-license if you accept it.",
            file=sys.stderr,
        )
        return 2

    source: Path = args.source_dir
    pt = args.source_pt
    if pt is None and (source / "sam3.pt").is_file():
        pt = source / "sam3.pt"
    official = source / "model.safetensors"
    if pt is None and not official.is_file() and args.download:
        download(source, args.repo)
    if pt is None and not official.is_file():
        raise SystemExit(
            f"no checkpoint found: pass --source-pt /path/sam3.pt, put model.safetensors or sam3.pt in {source}, or use --download"
        )
    if pt is not None and args.full:
        raise SystemExit("--full needs the official model.safetensors; the sam3.pt path exports the point-prompt components only")
    if pt is not None and args.no_convert:
        raise SystemExit("--no-convert needs the official model.safetensors; sam3.pt is converted to F16 for Android")

    out.mkdir(parents=True, exist_ok=True)
    target = out / "model.safetensors"
    if args.force or not target.is_file():
        if pt is not None:
            if not pt.is_file():
                raise SystemExit(f"{pt}: no such file")
            print(f"using local checkpoint {pt} ({pt.stat().st_size / 1e9:.2f} GB)", flush=True)
            checkpoint = {"name": "Meta SAM 3", "format": "PyTorch", "source": str(pt), "sha256": sha256(pt), "bytes": pt.stat().st_size}
            print("converting the point-prompt tensors to F16 (verified against the mapping) ...", flush=True)
            info = write_pt_asset(pt, target, args.mapping)
            source_kind = "pt"
            print(f"wrote {target} ({info['bytes'] / 1e9:.2f} GB, {info['tensors']} tensors)")
        else:
            size = official.stat().st_size
            if size != OFFICIAL_BYTES:
                raise SystemExit(f"{official}: expected {OFFICIAL_BYTES} bytes, got {size}")
            print("verifying the official checkpoint (SHA-256 of 3.4 GB) ...", flush=True)
            digest = sha256(official)
            if digest != OFFICIAL_SHA256:
                raise SystemExit(f"{official}: not the official checkpoint (got {digest})")
            checkpoint = {"name": "facebook/sam3", "format": "safetensors", "source": args.repo, "sha256": digest, "bytes": size}
            if args.no_convert:
                shutil.copyfile(official, target)
                info = {"tensors": "all", "bytes": target.stat().st_size, "sha256": sha256(target)}
            else:
                print("converting the checkpoint to F16 storage ...", flush=True)
                info = convert_official(official, target, point_only=not args.full)
            source_kind = "safetensors"
            print(f"wrote {target} ({info['bytes'] / 1e9:.2f} GB, {info['tensors']} tensors)")
    else:
        info = {"tensors": "existing", "bytes": target.stat().st_size, "sha256": sha256(target)}
        source_kind = "pt" if pt is not None else "safetensors"
        if pt is not None:
            checkpoint = {"name": "Meta SAM 3", "format": "PyTorch", "source": str(pt), "sha256": sha256(pt), "bytes": pt.stat().st_size}
        else:
            digest = sha256(official)
            if digest != OFFICIAL_SHA256:
                raise SystemExit(f"{official}: not the official checkpoint (got {digest})")
            checkpoint = {"name": "facebook/sam3", "format": "safetensors", "source": args.repo, "sha256": digest, "bytes": official.stat().st_size}
        print(f"using existing {target}")

    copy_aux(out, args.tokenizer_dir, source)
    manifest = write_manifest(out, args, info, source_kind, checkpoint)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
