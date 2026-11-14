# MobileSAM integration - current status (2026-11-14)

Branch: `feature/mobilesam-npu` (based on `feature/npu-sam3-htp`; SAM 3 work is intact there).
Last commit on the branch: `8ca87c66d` (the MobileSAM investigation). The app integration is
implemented and built, but **not committed yet** and its visual device check is still open.

## What is verified

Artifact: Qualcomm AI Hub `mobilesam-qnn_dlc-float` (`encoder.dlc` 26.8 MB, `decoder.dlc`
24.2 MB, QAIRT 2.45).

| item | result |
|---|---|
| encoder context (HTP) | 36.1 MB, compiled on the device in 64.5 s |
| decoder (original artifact) | fixed to 1 point; HTP decoder inaccurate |
| decoder re-export (ours) | 8 prompt slots, ONNX vs PyTorch max err 1.6e-5, DLC 19.75 MB, CPU context 20.6 MB |
| encoder on HTP | 2.0-2.65 s per photo, embeddings cosine 0.99981 vs the CPU backend |
| decoder on CPU backend | 54-94 ms per prompt |
| HTP decoder | IoU 0.17-0.40 vs CPU masks, one test point empty -> **not used** |
| mask vs SAM 3, same click | **IoU 0.906** (the white pitcher in the still-life test photo) |
| multi-point prompts | 1 point, positive+negative, and two positives all run (62-94 ms) |

## App integration (implemented, built, running)

- `lightcraft-segment`: new `mobilesam` module - MobileSAM preprocessing (resize longest side to
  1024, pad with the SAM pixel mean, NHWC 0..1), prompt scaling, 256 -> 288 mask mapping, and
  the `MobileSamRuntime` trait.
- `lightcraft-engine`: the worker now holds either SAM 3 or MobileSAM (`Model`/`Cached` enums);
  clicks and detail passes use whichever is configured; text prompts report "not available";
  `Segmenter.mobile` selects MobileSAM.
- Android Rust app: `MobileSamQnn` (bridge calls `mobileSamEncode` / `mobileSamDecode`); the AI
  edition sets `segmenter.mobile` and no longer loads the SAM 3 checkpoint.
- Java: `MobileSam.java` (JNI), `MainActivity` extracts the two contexts and runs the ops;
  the SAM 3 `QnnEncoder`/`extractModel`/`qnnEncode` paths were removed.
- Packaging: AI edition ships only the two contexts (assets), `libmobilesam.so`, `libQnnHtp.so`,
  `libQnnHtpV81Stub.so`, `libQnnHtpV81Skel.so`, `libQnnCpu.so`. The 947 MB checkpoint and the
  916 MB SAM 3 contexts are gone. **AI APK: 116.2 MB** (was 1.88 GB).
- On-device log from the installed AI APK:
  `MobileSAM ready (HTP encoder + CPU decoder)`, `encoder 2650.4 ms`, `decoder 70.0 ms`,
  `encoder 2509.9 ms`, `decoder 62.7 ms` - the full JNI -> bridge -> Rust path executes.
- Lite APK was verified before these changes (AI-free); it must be rebuilt/checked again.

## Open items / blockers

1. **Visual device check not captured yet.** The tablet's pre-existing "library could not be
   locked" dialog plus an IME-resized window (`adjustResize`) leave the app rendering in a
   1904 x 1210 area with the touch y scaled by ~0.626; synthetic taps miss the dialog/buttons.
   The app did run the MobileSAM encode/decode in-app, and a screenshot shows the Masking panel
   with a click pin and a mask, but a clean "mask on a person/object" screenshot is still to be
   taken. Disabling both IMEs and restarting removes the dialog; the last screenshot shows the
   Shibuya photo with the keyboard still present and the touch scaling still off.
2. The integration changes are uncommitted (see `git status`). The generated MobileSAM ONNX/DLC/
   context files are untracked and should be gitignored (`tools/ai/mobilesam/*.onnx|*.dlc|*.bin`),
   `manifest.json` should be committed.
3. Docs not yet updated for the integrated AI edition: `docs/MOBILESAM_NPU.md` (integration
   section) and `docs/AI_SEGMENTATION.md` (the Android AI edition now uses MobileSAM; the SAM 3
   description applies to the desktop build).
4. The old app data on the tablet still contains the SAM 3 checkpoint/contexts (~2 GB) under
   `files/models/sam3`; it is unused and can be deleted (app data clear removes the library too).
5. The 8-point limit: clicks beyond 8 are truncated (the first 8 are used).
6. The HTP decoder issue is not investigated further; the CPU decoder is fast and accurate.
7. The disabled IMEs (SwiftKey/LatinIME) should be re-enabled on the tablet.

## Next steps

1. Finish the device check: with the IMEs disabled and the app restarted, open the Masking panel
   (tap ~ (1848, 842) with the current scaled mapping, or fix the window state), tap AI Object
   Select, click a person, capture the mask, and time a click/detail pass.
2. Commit the integration, update the two docs, add the gitignore entries.
3. Rebuild and verify the Lite APK (AI-free), run `cargo test`/clippy for the touched crates.
4. Optionally cache contexts across sessions (they are already cached per process; first use
   extracts 56 MB and compiles nothing).
