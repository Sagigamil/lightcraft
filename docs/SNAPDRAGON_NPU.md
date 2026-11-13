# Snapdragon Hexagon NPU (QNN/HTP) investigation

Status: **in progress — NPU inference is not yet verified.** The CPU backend remains the shipping
path and is unchanged.

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

is running. A tiny known-good CNN is also being compiled + linked for the same device as a
control: if it runs on HTP while SAM 3 fails, the failure is model-graph-specific rather than a
device/runtime problem.

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
