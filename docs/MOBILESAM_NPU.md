# MobileSAM on the Hexagon NPU

Status: **integrated in the AI edition.** The encoder runs on the HTP, the prompt decoder on the
QNN CPU backend; the decoder is a re-export of the original weights with **8 prompt slots** and
**three mask alternatives**. The AI APK is 116 MB (was 1.88 GB with SAM 3), an encode takes
2.0-2.65 s and a click 60-95 ms.

SAM 3 work is untouched on `feature/npu-sam3-htp`; this branch builds on it.

## 1. Artifact and re-export

Qualcomm AI Hub's `mobilesam-qnn_dlc-float` (`encoder.dlc` 26.8 MB, `decoder.dlc` 24.2 MB,
QAIRT 2.45):

- Encoder: `image` `[1,1024,1024,3]` float32 NHWC 0..1 (mean/std inside); output
  `image_embeddings` `[1,64,64,256]`.
- Their decoder is fixed to **one point** and one mask. `tools/ai/export_mobilesam_decoder.py`
  re-exports the same MobileSAM weights with:
  - `point_coords` `[1,8,2]` and `point_labels` `[1,8]` (unused slots are label -1);
  - `multimask_output=True`: `masks` `[1,256,256,4]` and `scores` `[1,4]`. Output 0 is SAM's
    single-mask token; outputs 1-3 are the three alternatives. The ONNX matches PyTorch to
    1.6e-5 (`tools/ai/qairt_shim.py` works around the QAIRT converter's `onnx.version` import).

## 2. Compiling to QNN

The local x86 HTP compiler fails on the encoder (`node_sub` in HTP prepare, error 1002), so the
encoder context is generated **on the device** with the Android `qnn-context-binary-generator`
and `--dlc_path` (the real HTP finalizes it). The CPU decoder context can be generated on the
host or the device:

```sh
# on the device, from Qualcomm's encoder.dlc
qnn-context-binary-generator --dlc_path encoder.dlc --backend ./lib/libQnnHtp.so \
  --binary_file encoder_ctx --output_dir .
# decoder8m.dlc -> CPU context (host or device)
qnn-context-binary-generator --dlc_path decoder8m.dlc --backend .../QnnCpu.dll \
  --binary_file decoder8m_cpu --output_dir .
```

| context | size | compile time (Y700) |
|---|---|---|
| `encoder_ctx.bin` | 36.1 MB | 64.5 s (HTP) |
| `decoder8m_cpu.bin` | 20.6 MB | ~2 s (CPU) |

## 3. Runtime (Lenovo Y700, SM8850P, HTP v81)

| stage | backend | execute |
|---|---|---|
| encoder (1024x1024) | HTP | **2 500-2 650 ms** |
| decoder (up to 8 points, 4 masks) | CPU (QNN CPU backend) | **62-95 ms** |
| decoder on the HTP | HTP | 19 ms but **inaccurate** (IoU 0.17-0.40 vs CPU, sometimes empty) |

The HTP decoder is not used. The first call extracts 56 MB of contexts (~1 s) and initializes the
two backends (~1.2 s); after that the contexts stay loaded for the process.

## 4. Accuracy

- Encoder HTP vs the CPU backend: embeddings cosine **0.99981**, mean abs 1.4e-3.
- HTP encoder + CPU decoder vs SAM 3 for the same click: mask **IoU 0.906** (same object).
- On the test prompt the four decoder outputs are the single-mask token (empty), one empty
  alternative, the whole object (~10k pixels, score 0.94) and a small part (~500 pixels, score
  1.09). The app defaults to the highest score and offers all three alternatives.

## 5. Preprocessing: letterboxed, never cropped

The engine's render request uses `fit` (the photo keeps its aspect), and `MobileSam::encode`
then resizes the longest side to 1024 and pads the short side to 1024x1024 with SAM's pixel mean
(`[0.485, 0.456, 0.406]` in 0..1). The model therefore sees the whole photo with a neutral
border, never a stretched or cropped version. Prompt coordinates are scaled into the content
region (`x * content_width`, `y * content_height`), and the 256x256 mask is sampled back over
the content into the 288x288 grid LightCraft stores. A unit test
(`letterboxes_a_wide_photo_and_scales_prompts_to_the_content`) checks a 2:1 photo.

## 6. Mask alternatives in the app

- The worker asks for all outputs, drops the single-mask token, builds a `SegMask` per
  alternative (3) and stores them in `Session::mask_options` keyed by (photo, mask, component);
  the highest score is applied first.
- The Object panel shows a **Mask option 1 2 3** row; clicking one runs
  `mask.selectOption { mask, component, index }`, which replaces the component's `seg` (an
  undoable edit) and updates the selected button. The alternatives are session state; the chosen
  mask is what the develop settings persist.
- Detail passes re-run the encoder on the zoomed region and pick the best alternative for the
  patch.

## 7. Integration

- `crates/segment/src/mobilesam.rs`: preprocessing, prompt scaling, `MobileSamRuntime` trait,
  mask mapping.
- `crates/engine`: the worker holds SAM 3 or MobileSAM, returns the alternatives, stores them and
  implements `mask.selectOption`.
- Android: `MobileSam.java` (JNI) + `libmobilesam.so`; `MainActivity` extracts the two contexts
  and runs `mobileSamEncode` / `mobileSamDecode`; the AI edition ships no SAM 3 checkpoint or
  contexts.
- Building the AI edition needs `QNN_SDK_ROOT` (the Qualcomm runtime is not in the repo).

## 8. Limits

- At most 8 prompt points (the model's slots; extra clicks are ignored).
- The alternatives are not persisted: after a reload only the chosen mask remains (re-click to
  regenerate them).
- Hexagon v81 (SM8850P) encoder context; other SoCs need their own compile.
- The HTP decoder inaccuracy is not investigated further (the CPU decoder is fast enough).
