# MobileSAM status (2026-11-14)

Branch `feature/mobilesam-npu`. The AI edition runs MobileSAM end to end: encoder on the Hexagon
HTP, decoder on the QNN CPU backend, 8 prompt points, three mask alternatives, letterboxed
1024x1024 input. AI APK: **116.3 MB** (was 1.88 GB with SAM 3).

## Implemented

- **Artifact/re-export:** Qualcomm's `mobilesam-qnn_dlc-float`; our decoder export uses 8 points
  and `multimask_output=True` (masks `[1,256,256,4]`, scores `[1,4]`, ONNX vs PyTorch 1.6e-5).
  The single-mask token is dropped, leaving three alternatives.
- **Contexts:** encoder 36.1 MB (HTP, compiled on the device), decoder 20.6 MB (CPU).
- **Runtime:** encoder 2.5-2.65 s; decoder 62-95 ms; HTP decoder is inaccurate and unused.
- **Engine:** MobileSAM worker path; the click result carries the alternatives; the highest score
  is applied first; `Session::mask_options` + `mask.selectOption`; detail passes re-encode the
  region.
- **UI:** Object panel shows **Mask option 1 2 3**; clicking one replaces the mask (undoable).
- **Letterboxing:** the engine's `fit` render keeps the photo aspect; `MobileSam::encode` resizes
  the longest side to 1024 and pads with SAM's pixel mean; prompts are scaled to the content; a
  unit test covers a 2:1 photo. No crop or stretch.
- **Packaging:** AI edition ships the two contexts, `libmobilesam.so`, HTP/CPU QNN libs and the
  skel; no SAM 3 checkpoint or contexts.

## Verified on the tablet

- In-app logs: `MobileSAM ready (HTP encoder + CPU decoder)`, `encoder 2543.9 ms`,
  `decoder 68.8 ms`, detail pass `encoder 2512.8 ms`.
- The Object panel showed the **Mask option 1 2 3** buttons with option 2 selected after a click.
- Engine tests 373 passed, UI tests 268 passed, segment MobileSAM tests (including letterboxing)
  pass.

## Open

- The device's display/input mapping is currently distorted (touch y scaled by ~0.63 after the
  IME/rotation experiments); the app works but automated taps are unreliable. The two IMEs are
  disabled on the tablet; restart the tablet to restore normal input.
- The default (highest score) alternative can be an empty or small mask for some prompts; the
  selector lets the user pick another. A future improvement is to default to the single-mask
  token for multi-point prompts (SAM's own rule).
- Alternatives are session-only (not persisted); after a reload only the chosen mask remains.
- The accidental Mask 6 (radial) created during testing can be deleted on the tablet.
