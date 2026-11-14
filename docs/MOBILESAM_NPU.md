# MobileSAM on the Hexagon NPU (investigation)

Status: **works and is much faster than SAM 3 - encoder on the HTP, decoder on the CPU.**
The Qualcomm artifact is a single-point-prompt model; multi-click/exclude needs a decoder
re-export. SAM 3 work is untouched on `feature/npu-sam3-htp`; this branch builds on it.

## 1. Artifact

`C:\Users\sagig\Downloads\mobilesam-qnn_dlc-float\mobilesam-qnn_dlc-float` (Qualcomm AI Hub):

| file | size | runtime |
|---|---|---|
| `encoder.dlc` | 26.8 MB | ViT-T image encoder |
| `decoder.dlc` | 24.2 MB | prompt encoder + mask decoder |
| `metadata.json` | - | QAIRT 2.45.0.260326 |

- Encoder: `image` `[1,1024,1024,3]` float32, NHWC, 0..1 (pixel mean/std are inside the graph);
  output `image_embeddings` `[1,64,64,256]`.
- Decoder: `point_coords` `[1,1,2]`, `point_labels` `[1,1]`, `image_embeddings` `[1,64,64,256]`;
  outputs `masks` `[1,256,256,1]` (logits) and `scores` `[1,1]`.
- The decoder is fixed to **one point** - no negative points, no padding.

## 2. Compiling to QNN

The local x86 HTP compiler fails on the encoder (`node_sub` in HTP prepare, error 1002), so the
context binaries were generated **on the device** with the Android
`qnn-context-binary-generator` + `--dlc_path` (this finalizes on the real HTP):

```sh
adb push encoder.dlc decoder.dlc libQnnModelDlc.so qnn-context-binary-generator ...
adb shell 'LD_LIBRARY_PATH=./lib ADSP_LIBRARY_PATH=./lib \
  ./bin/qnn-context-binary-generator --dlc_path encoder.dlc --backend ./lib/libQnnHtp.so \
  --binary_file encoder_ctx --output_dir .'
```

| context | size | compile time (Y700) |
|---|---|---|
| `encoder_ctx.bin` | 36.1 MB | 64.5 s |
| `decoder_ctx.bin` | 12.4 MB | 2.4 s |

Compilation is a build step; at runtime the contexts are simply loaded.

## 3. Runtime (Lenovo Y700, SM8850P, HTP v81)

Two contexts, one encoder execution per photo, one decoder execution per click
(`tools/ai/mobilesam`):

| stage | backend | execute |
|---|---|---|
| encoder (1024x1024) | HTP | **2 552 ms** |
| decoder (1 point) | HTP | 19 ms - **mask wrong** (see below) |
| decoder (1 point) | CPU (QNN CPU backend) | **54 ms** - correct |

Whole test process wall time with both contexts + one encode + one decode: **3.8 s**
(context loading ~1.2 s). This compares with SAM 3's 31.2 s encode + ~2 s per click.

## 4. Accuracy

- Encoder HTP vs the same DLC on the CPU backend: embeddings cosine **0.99981**, mean abs 1.4e-3.
- **HTP decoder is unusable**: same inputs on the CPU give IoU 0.17-0.40 against the HTP masks,
  and one pitcher click is an empty mask on HTP. This looks like an HTP/fp16 or layout
  miscompilation of the decoder's attention; not pursued because the CPU decoder takes 54 ms.
- Correct pipeline (HTP encoder + CPU decoder), same click as SAM 3 (normalized 0.215, 0.713):
  mask **IoU 0.906** against SAM 3, same object (the white pitcher). Both are visually tight.

## 5. Preprocessing

Follow Qualcomm's `SAMApp`:

- Resize the **longest side** to 1024 (bilinear), pad the short side to 1024x1024 with the SAM
  pixel mean, RGB, range 0..1, layout NHWC.
- Prompt coordinates are in the resized 1024 frame, `(x, y)`; labels 1 = foreground,
  0 = background, -1 = padding.
- Mask logits are 256x256; threshold at **0** and map back to the photo.

## 6. Limits and next steps

- **Single point only.** LightCraft's Object select sends all clicks (positive and negative);
  MobileSAM needs a decoder exported with N points (e.g. 8, unused entries labelled -1). The
  original `MobileSAM` weights + `qai_hub_models.models.mobilesam` wrapper can be re-exported,
  converted to DLC and compiled for the CPU backend (the HTP decoder is inaccurate anyway).
- The AI edition could shrink from **1.88 GB to roughly 100 MB**: 36 MB encoder context,
  ~12 MB decoder context, plus the QNN runtime; the 947 MB SAM 3 checkpoint and 916 MB contexts
  become unnecessary.
- Then integrate as the AI edition's segmentation backend (the UI and `SegMask` stay unchanged),
  with SAM 3 kept as the CPU/desktop path and fallback.
- The 64 s on-device encoder compile is a build-time cost; it can move to AI Hub once an ONNX
  encoder export is produced.
