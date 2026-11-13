# Snapdragon Hexagon NPU (QNN/HTP) investigation

Status: **verified in the AI edition** on a Lenovo Legion Y700 (SM8850P, Hexagon v81, Android 16).
The 32-layer ViT encoder runs on the HTP as three chained QNN contexts (stem + two 16-block
chunks); the neck, prompt encoder, tracker and mask decoder stay in the pure-Rust candle engine on
the CPU. The CPU backbone remains the fallback and the Lite edition has no NPU code at all.

## 1. Model and current backend

- Model: **SAM 3 point-prompt** — ViT-L vision encoder (32 layers, 1024 hidden, patch 14,
  1008×1008) + SAM 2 FPN neck, followed by the SAM 2-style prompt encoder / mask decoder.
- Existing Android backend: **pure Rust + candle, ARM64 CPU (NEON)**, F16 checkpoint, 867 tensors.
  Encoder once per photo, decoder per tap. Fully offline.
- NPU work does not modify the masking UI, mask representation or the CPU engine.

## 2. Exported encoder (verified)

`tools/ai/export_sam3_onnx.py` exports the encoder from Meta's `sam3.pt`:

- Input: `image` `[1, 3, 1008, 1008]`, float32.
- Outputs: 4 FPN levels `[1,256,288,288]`, `[1,256,144,144]`, `[1,256,72,72]`,
  `[1,256,36,36]` + 4 position encodings of the same shapes.
- ONNX vs PyTorch on identical input: max relative error ≈ 1e-5 (all outputs).
- Two required source patches: Meta's fused MLP unconditionally casts to bf16 (CUDA-only), and the
  default ViT uses complex-valued RoPE (`view_as_complex`, unsupported by ONNX); the export builds
  the same weights with real-arithmetic RoPE.

Graph size: 448.6M parameters, 227 G MACs, 1818 ONNX ops (192 MatMul, 65 LayerNormalization,
33 Gelu, 32 Softmax, 257 Add, 336 Reshape, 154 Transpose, 9 Conv, 3 ConvTranspose).

## 3. Compilation

Qualcomm AI Hub (token-authenticated), QAIRT default **2.50**:

- Compile (`--target_runtime qnn_dlc --compute_unit npu`, device `Samsung Galaxy S26`,
  `qualcomm-snapdragon-8-elite-gen5-for-galaxy, sm8850-ad`):
  job `jp3r48zxp` — success, DLC model `mng1g4je6`.
- Link (`qnn-context-binary-generator`): job `jgz0x8xx5` — success, context binary
  `tools/ai/onnx/qnn/sam3_encoder_context.bin` (1.0 GB).
- HTP settings recorded in the link log:

```json
{"devices":[{"device_id":0,"dsp_arch":"v81","soc_model":87}],
 "graphs":[{"precision_compensation":true,"fp16_relaxed_precision":1,"vtcm_mb":0,"O":3.0}]}
```

`vtcm_mb: 0`, `fp16_relaxed_precision: 1`, optimization level `O=3`.

## 4. On-device execution (Lenovo Legion Y700, TB323FU)

Environment:

- `ro.soc.model = SM8850P`, `ro.board.platform = canoe`, Android 16 / SDK 36, arm64-v8a.
- Hexagon **v81**; QAIRT runtime `v2.50.0.260828`; SDK stub `libQnnHtpV81Stub.so`, skel
  `libQnnHtpV81Skel.so` loaded onto the CDSP (verified in logcat).

Run (`qnn-net-run --retrieve_context sam3_encoder_context.bin`):

- Context load / backend init: **1.673 s** (profile log: `INIT, 1672599 US`).
- Graph execution: **~11.1 s** (total wall 12.811 s), then:

```text
QnnDsp <E> skelExecute call failed with err 1003
QnnDsp <E> Graph graph__1hnm9_s failed in execution with err 1003
adsprpc: Error 0x8000040d ... FTQ driver invoke method (273) failed
```

No output tensor was produced. This matches the reported symptom (failure after several seconds,
not at load).

The on-device DLC path (`qnn-net-run --dlc_path`) is not usable here: it fails with
`Failed to create dlc handle with code 1002` before execution (the Windows SDK cannot build an
aarch64 model library, and the device `qnn-context-binary-generator` build has no `--dlc_path`).

## 5. Hypothesis and fix attempt

The AI Hub SAM-family package (MobileSAM) explicitly forces `O=1` **"to work-around a tiling bug
(AISW-206993)"**; our first link used the default `O=3`. A relink with

```text
--qnn_options default_graph_htp_optimizations=O=1
```

was run and **also fails** (~11.6 s, same err 1003), so it is not only the optimization level.

Control: a tiny known-good CNN (1x3x64x64, conv+linear) was compiled, linked and run on the same
tablet with the same QNN libs: **exit 0, 204 ms, outputs written**. The device, v81 stub/skel and
runtime are therefore healthy; the failure is specific to the SAM 3 encoder graph.

## 6. Open items

- Verify the `O=1` context on the tablet (execution + timing).
- Run the control CNN on HTP.
- Encoder/decoder separation on HTP: only the encoder has been exported and compiled so far; the
  point-prompt decoder has not yet been exported to ONNX.
- If `O=1` does not fix it: split the encoder (e.g. first N ViT blocks, or trunk vs neck) and test
  progressively; or quantize (INT8) / reduce the NPU input to 504×504 while keeping the CPU path
  for full resolution.
- Package the QNN libs/context in the **AI** flavor only once HTP execution and mask accuracy are
  verified; Lite stays untouched.

## 7. Subgraph isolation results (device, QNN HTP)

All graphs were exported from the same checkpoint, compiled to QNN DLC and linked for
`sm8850-ad` / HTP v81, then run on the Lenovo tablet with `qnn-net-run --retrieve_context`.

| Graph | Context size | Result | Wall time |
|---|---|---|---|
| Tiny control CNN (1x3x64x64) | 53 KB | **pass** | 204 ms |
| ViT block 0 (windowed attention) | 28 MB | **pass** | 885 ms |
| ViT block 7 (global attention, 5184 tokens) | 39 MB | **pass** | 1.87 s |
| ViT blocks 0-8 (8 blocks) | 231 MB | **pass** | 4.79 s |
| Full SAM 3 encoder (32 blocks + neck) | 1.0 GB | **fail** | ~11 s, err 1003 |

Conclusion: the failure is **not** a specific FP16 operator and **not** global attention; a single
global-attention block executes correctly. It is **cumulative graph size / graph memory**: the full
1.0 GB context (448.6M params) exceeds what this HTP graph can execute, and the DSP stalls ~11 s
before returning err 1003 / adsprpc 0x8000040d.

`O=1` (MobileSAM's AISW-206993 tiling workaround) was tried and did not change the outcome.

### Fix direction: partition the encoder

Split the encoder into sequentially executed HTP contexts that each fit the device:

```text
image -> stem (patch embed + ln_pre + pos) -> blocks 0-8 -> blocks 8-16 -> blocks 16-24
      -> blocks 24-32 -> neck -> FPN features
```

8 blocks are proven to run (4.79 s). The exact maximum chunk size (8 vs 16) is being measured; the
chunks are connected by ordinary tensors, so no model or mask semantics change. The CPU backend
remains the fallback and is still the default until the partitioned HTP path is wired and verified.

## 8. Working partitioned chain (verified on the tablet)

A native runner (`tools/ai/qnn_chain/`) loads the three context binaries and executes them in
sequence on the HTP, binding the graph tensors by id and passing the token tensor between stages.

Device run (Lenovo Y700, SM8850P, QNN 2.50, HTP v81):

| Stage | Standalone runner | In the app, first encode | In the app, detail pass |
|---|---|---|---|
| stem (image -> tokens) | 23.4 ms | 24.4 ms | 23.0 ms |
| blocks 0-16 | 7 128.9 ms | 7 492 ms | 7 804 ms |
| blocks 16-32 | 7 656.8 ms | 7 494 ms | 8 165 ms |
| whole chain (incl. creating both 451 MB contexts) | 24 618 ms | 31 158 ms | 32 278 ms |

`rc=0`, tokens written. This is genuine Hexagon execution: the tiny control graph, single blocks
and the chained partitions all run on the NPU, while the monolithic graph fails.

Accuracy against PyTorch on the same 1008x1008 input (fp32 reference, fp16 HTP):

- cosine similarity **0.998749**
- mean absolute error **0.0359** (reference RMS 1.69, ~2 %)
- max absolute error 80.9 on outliers (fp16 accumulation over 32 blocks)

The same tokens through the unchanged Rust neck/tracker (`crates/segment/examples/npu_check.rs`)
give the same mask as the CPU backbone for the same clicks:

- tokens vs the Rust CPU backbone: cosine **0.998750**, mean abs err 3.586e-2
- mask **IoU 0.9996**, pixel agreement **99.96 %** (scores 0.6561 vs 0.6545)

## 9. Integration in the AI edition

- **Rust:** `lightcraft_segment::BackboneEncoder` takes the preprocessed pixels and returns the
  NCHW token tensor; `Sam3::encode_with` builds an `Encoded` from it (`Encoded::from_backbone`).
  `Segmenter.npu` carries the accelerator into every worker job; the worker uses it for the
  initial encode and for detail passes, and falls back to the candle backbone on any error. The
  neck, prompt encoder, tracker and detector are untouched.
- **Android:** `MainActivity`'s `qnnEncode` bridge op extracts the context binaries once to
  `files/models/sam3/qnn/` and calls `QnnEncoder.encode` (AI source set, reached by reflection from
  the shared code) -> JNI -> `libqnn_chain.so` -> `qnn_run_chain`. Logcat shows the HTP backend
  init and each stage (tag `LightCraftQnn`); the app reports `backend: "npu+cpu"`.
- **Packaging (AI edition only):** the three contexts are stored assets (copied on first use);
  `libQnnHtp.so`, the v81 stub, the QNN 2.50 skel, `libQnnSystem.so`, `libQnnModelDlc.so` and
  `libqnn_chain.so` are jniLibs, with `jniLibs.useLegacyPackaging = true` so they are real files
  under `nativeLibraryDir`. `<uses-native-library android:name="libcdsprpc.so">` is required on
  Android 12+ to make the vendor FastRPC client visible to the app. The Lite edition ships none of
  this. The APK stays at ~1.88 GB (the contexts add ~0.9 GB).

Android-specific findings, each one a failure mode hit here:

- **App data is not loadable:** `dlopen` of a library from app storage is blocked for apps
  targeting API 29+. QNN host libraries must be extracted native libraries (`nativeLibraryDir`).
- **Real files, not zip entries:** the HTP backend loads `libQnnHtpV81Stub.so` from next to
  itself, so `libQnnHtp.so` must be a real file; an APK entry path is not usable.
- **Skeleton lookup:** the FastRPC client resolves the skel directory from `ADSP_LIBRARY_PATH`
  (which an app cannot set at exec time; `setenv` after startup has no effect) and from the process
  working directory. The runner `chdir`s to `nativeLibraryDir` (which holds
  `libQnnHtpV81Skel.so`) before creating the backend. The OEM's signed skel in
  `/vendor/lib/rfsa/adsp` exports an older handle symbol (`qnn_2_xx_...`), so the QNN 2.50 skel is
  shipped instead.
- **Vendor namespace:** `libcdsprpc.so` is a vendor public library; without the manifest
  declaration even its transitive `dlopen` fails with "not found in namespace clns-N".

Limits: both 451 MB contexts are recreated on every encode (~16 s on top of ~16 s of HTP compute).
Caching them would save that but hold ~1 GB more resident memory; not done. The detail pass runs
the full chain again (32 s measured).

## 10. Verified on the device

With the AI edition installed on the Y700: opening **AI Object Select** encoded the photo on the
NPU (`LightCraftQnn`: stages 24.4 / 7 492 / 7 494 ms), a tap produced the expected mask (table,
then the white pitcher), and the automatic detail pass re-ran the chain (23.0 / 7 804 / 8 165 ms,
32.3 s) and refined the mask around the pitcher. The mask was kept and the UI behaved as on the
CPU path.
