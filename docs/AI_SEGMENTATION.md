# AI Object Select on Android (SAM 3 point prompts)

Lightcraft AI adds **AI Object Select** to the existing Masking panel: tap an object in the
photo and get an editable mask for that specific object. It runs entirely offline on the tablet,
using the same mask representation, rendering, adjustments, history and persistence as every
other Lightcraft mask.

Two separately buildable APKs are produced:

| APK | Application ID | AI model | Notes |
|---|---|---|---|
| `Lightcraft-Lite.apk` | `com.lightcraft.android` | none | The existing app; no AI libraries, weights, assets or menu entries. |
| `Lightcraft-AI.apk` | `com.lightcraft.android.ai` | embedded SAM 3 point-prompt checkpoint | AI Object Select; fully offline; no network permission. |

## 1. Selected model and why

**Model:** SAM 3 (Segment Anything with Concepts), using its genuine **point-prompt** pathway:
the SAM 2-style prompt encoder and two-way mask decoder. The image encoder runs once per photo;
each tap runs only the prompt decoder against the cached embedding.

The inference code is the existing `crates/segment` crate: a pure-Rust, candle-based port of the
Apache-2.0 Hugging Face `transformers` SAM 3 implementation (see `NOTICE`), already used by the
desktop app. The Android AI flavor enables it through the `sam` cargo feature; the Lite flavor
does not link it at all.

**Why not Qualcomm AI Hub's SAM 3 export?** I inspected the local `qai_hub_models` checkout at
tag `v0.66.0` (`src/qai_hub_models/models/sam3/model.py`). It exposes exactly two components:

- `SAM3VisionBackbone`: image → three FPN feature maps.
- `SAM3Head`: **tokenized text prompt** + the FPN maps → boxes, scores, masks.

There is no prompt encoder and no interactive decoder, so the published export cannot answer
"which object is under this tap". Substituting a text prompt such as "object" would select every
instance of a concept, not the tapped object, so it is not used. Running QNN would additionally
add C/C++ runtime dependencies to a pure-Rust product and still not provide point prompts.

**Why the checkpoint itself is genuine SAM 3.** Meta's original `sam3.pt` contains the
visual-prompt components (`tracker.sam_prompt_encoder.*`, `tracker.sam_mask_decoder.*`,
`tracker.maskmem_backbone.*`, the SAM 2 FPN `sam2_convs`, and the ViT trunk). The Android export
keeps exactly those (plus the shared ViT and the point-tracker neck) and drops the text encoder
and the DETR detector head. Nothing is substituted: this is SAM 3's own interactive segmenter.

**Runtime/backend:** candle **CPU** on ARM64 (NEON). There is no Hexagon NPU backend: Qualcomm's
SAM 3 export is text-only, and a QNN port of the interactive decoder does not exist here. The app
reports `backend: "cpu"` and never claims NPU acceleration.

Fallbacks considered (not needed): Qualcomm's SAM 2 / MobileSAM exports are point-capable but
would require a QNN/ONNX Runtime + C++ integration and a separate decoder export. SAM 3's own
point path already existed, is tested, and produces the required quality, so it is used.

## 2. Checkpoint source, version and licence

- **Original publisher:** Meta, `facebookresearch/sam3` (SAM License).
- **File:** `sam3.pt`, 3,450,062,241 bytes, SHA-256
  `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`.
  The local copy used for this build is
  `C:\Users\sagig\Projects\ai-hub-models\sam3_run\checkpoint\sam3.pt`, downloaded from the public
  `jetjodh/sam3` mirror (the same mirror the repo's desktop tooling uses). The official
  `facebook/sam3` repository is gated: accept its licence and set `HF_TOKEN` to download from it
  instead.
- **Licence:** the weights are Meta's, under the **SAM License** — not LightCraft's MIT/Apache
  licence. The AI APK embeds `assets/sam3/LICENSE`. Redistributing the AI APK redistributes the
  SAM Materials and must follow that licence. Weights are never committed to Git.

## 3. What the AI APK embeds

`android/app/src/ai/assets/sam3/` (gitignored, produced by the preparation script):

| File | Size | Notes |
|---|---|---|
| `model.safetensors` | 947,288,924 bytes | F16 storage, 867 tensors, SHA-256 `23c1e179a82c6f14f77d90d9bef9e28c192a8d8ddc7ca8ca473e863f6716911b` |
| `vocab.json`, `merges.txt` | 862,328 / 524,619 | SAM 3 tokenizer (used only by the desktop text path; kept so the model directory is complete) |
| `LICENSE` | 7,352 | Meta SAM License |
| `manifest.json` | — | components, shapes, sizes, SHA-256s, runtime, licence |

Components kept: `detector_model.vision_encoder.*`, `tracker_neck.*`, `tracker_model.*`.
Components omitted: the CLIP text encoder and the DETR detector/segmentation head. The Android UI
therefore hides **Describe**; the desktop app keeps the full checkpoint and its text prompts.

The loader (`crates/segment/src/weights.rs`) accepts both F32 and F16 tensors and converts each
tensor to F32 as it is read.

## 4. Reproducible model preparation

Requirements: Python 3.10–3.12 with `numpy`, `safetensors` (and `torch` plus
`huggingface_hub` only for the `.pt` / download paths):
`python -m pip install -r tools/requirements-ai.txt` and, for a local `.pt`, `pip install torch`.

From the repository root, with your local checkpoint:

```sh
python tools/ai/prepare_models.py --accept-license \
  --source-pt C:/Users/sagig/Projects/ai-hub-models/sam3_run/checkpoint/sam3.pt
python tools/ai/prepare_models.py --check
```

Or from the official Hugging Face export (`model.safetensors`, verified by SHA-256):

```sh
python tools/ai/prepare_models.py --accept-license --source-dir models/sam3
python tools/ai/prepare_models.py --accept-license --download      # public mirror
python tools/ai/prepare_models.py --accept-license --download --repo facebook/sam3   # gated, needs HF_TOKEN
```

The script:

1. Loads `sam3.pt` with PyTorch's `weights_only=True` (mapped, no full-model allocation) or
   verifies the official `model.safetensors` SHA-256.
2. Exports the point-prompt tensors to F16 using `tools/ai/sam3_point_mapping.json`.
3. **Verifies every one of the 867 tensors**: each exported tensor's F16 SHA-256 must equal the
   hash recorded in the mapping for that checkpoint, so a wrong or damaged checkpoint fails loudly.
4. Writes `manifest.json` (components, shapes, sizes, hashes, runtime, licence) and copies the
   tokenizer files and licence.
5. `--check` re-verifies every packaged file's size and SHA-256.

`tools/ai/sam3_point_mapping.json` is the conversion spec: for each target tensor it names the
source tensor in `sam3.pt` (`direct`), or the operation that produces it (`split3` for the fused
ViT qkv, `stack` for the four point embeddings, `slice` for the position embeddings without the
class token), plus the target shape and SHA-256. It was produced by aligning every tensor of the
HF-format point-prompt export the engine's reference test was validated against to `sam3.pt` by
shape and SHA-256 of the F16 bits (867/867 matched), and is committed so preparation is
reproducible offline. The generated asset reproduces the reference smoke-test result
(`score 0.8510`, 5,289 positive cells of 82,944) exactly.

Normal Android builds never need a Hugging Face account or a download once the assets exist.

## 5. Model input, output and coordinate mapping

- **Inference input:** 1008 × 1008, the SAM 3 reference resolution. The photo is rendered as
  developed, uncropped and oriented, then stretched to the square input exactly as SAM 3's
  reference preprocessing does. The original photo is never resized or degraded.
- **Model output:** 288 × 288 mask logits. They are quantized to i8, deflated and base64-encoded
  in the existing `SegMask` (roughly 10–30 KB per mask) and stored with the mask in the develop
  settings. Rendering and export sample the logits bilinearly at any output size; no model is
  needed to render, save or reopen a mask.
- **Tap coordinates:** the shared viewer maps the tap through the existing zoom/pan/crop/EXIF
  transform to normalized coordinates of the uncropped, oriented photo. `mask.objectPoint`
  receives `x, y` in 0..1 and rejects points outside the photo. The mask is normalized to the
  same frame, so previews, exports, rotation and cropping stay aligned.
- **Detail pass:** about a second after the last tap, the area around the selection is rendered
  again from a sharper source and segmented at higher effective resolution; the result is stored
  as a `SegMask` patch with its normalized `rect`. At most four patches are kept per component.
- **Inference resolution ≠ export resolution:** the photo stays at full resolution; only the model
  input is 1008 × 1008.

## 6. Lifecycle and integration

- `apps/lightcraft-android` has a `sam` cargo feature that enables
  `lightcraft-engine/sam` → `lightcraft-segment`; the Lite build never enables it.
- In the AI build, `app.rs` sets `session.segmenter.dir` to `<internal data>/models/sam3` and
  `segmenter.embedded = true`, then starts a one-time extraction from the APK assets.
  `MainActivity.extractModel` copies each asset to a `.part` file, checks its SHA-256 against the
  packaged manifest, atomically renames it, and writes a `.ready` marker naming the model hash.
  Later launches skip verified files. Nothing is downloaded and the release APK has no INTERNET
  permission.
- The existing worker thread owns the model and the last encoded photo. The first AI request
  loads the model and encodes the photo; each later tap reuses the cached embedding and runs only
  the prompt decoder. Superseded queued taps for the same selection are skipped; stale results are
  discarded.
- `onTrimMemory` queues an unload (`Segmenter::release_model`), dropping the model and its cache;
  the worker also unloads after 10 minutes idle.
- `segment.model.status` reports `available`, `embedded`, `backend` (`cpu`), `textPrompts`,
  `installed`, `loaded`, `busy`, `analyzing`, `sizeBytes`, licence and download state.
- Android UI: the Object tile is labelled **AI Object Select**; Describe is hidden; a guidance bar
  with **Keep** / **Cancel** and an **Add to selection** switch is shown. A tap replaces the
  selection by default (one tap = one object); Cancel deletes the pending mask, Keep keeps it as a
  normal editable mask. Selection is undoable and persisted like any mask.
- Lite hides the AI entries entirely; the local **Subject** mask remains available.

## 7. Measured performance (development machine, CPU)

`cargo run --release -p lightcraft-segment --example sam3_check -- android/app/src/ai/assets/sam3`
on the Windows x86-64 development machine, using the same F16 checkpoint that ships in the APK:

| Stage | Time |
|---|---|
| Model load | 0.91 s |
| Image encode (1008 × 1008) | 39.8 s |
| Point click (cached embedding) | 2.69 s |

Output was finite and non-empty: score **0.8510**, 5,289 of 82,944 mask cells positive.
Backend: **CPU** (`Device::Cpu`).

**No device measurements were taken in this session**: no ADB device was connected. Tablet
latency and peak memory are therefore unverified here. The encode dominates and runs once per
photo (and once more for the background detail pass); taps reuse it.

## 8. Build the two APKs

```powershell
$env:ANDROID_HOME = "C:\Users\<you>\AppData\Local\Android\Sdk"
$env:JAVA_HOME    = "C:\Program Files\Microsoft\jdk-17.0.21.21-hotspot\"

.\android\build.ps1 -Configuration Release -Edition Both
```

`-Edition Lite` or `-Edition AI` builds one variant. The script:

- builds the Lite cdylib **without** the `sam` feature into `target/android-lite`;
- builds the AI cdylib **with** `--features sam` into `target/android-ai`;
- copies and strips each `liblightcraft_android.so` into its flavor's `jniLibs`;
- requires and size-checks the prepared AI assets before packaging the AI variant;
- packages `assembleLiteRelease` / `assembleAiRelease`;
- writes the deliverables:

```text
android/dist/Lightcraft-Lite.apk
android/dist/Lightcraft-AI.apk
```

Equivalent Gradle-only invocation (after the native libraries are in place):

```powershell
cd android
.\gradlew.bat :app:assembleLiteRelease
.\gradlew.bat :app:assembleAiRelease
```

Install either or both (different application IDs):

```sh
adb install -r android/dist/Lightcraft-Lite.apk
adb install -r android/dist/Lightcraft-AI.apk
```

Release builds are signed with the local **debug** key, as before (production keys are not stored
in this repository).

Current artifacts (rebuilt from the final sources above):

| APK | Size | SHA-256 |
|---|---|---|
| `android/dist/Lightcraft-Lite.apk` | 34,130,084 bytes (32.5 MB) | `6d7b64999bced1328158f3fb38bc62452a5f1f64ec0f6f05f6dc5c2dc0d5c76b` |
| `android/dist/Lightcraft-AI.apk` | 986,571,607 bytes (940.9 MB) | `88f42f790b3c603100ac37e1de80d4bb65d5c37814a1f439528534fd39810dd5` |

### Debugging a device build

For on-device diagnosis, build the AI flavor with the loopback control channel:

```sh
cargo build --release -p lightcraft-android --target aarch64-linux-android --features sam,control
```

and add a *temporary* `android/app/src/ai/AndroidManifest.xml` with the INTERNET permission
(Android refuses even a loopback socket without it; remove the file before shipping). Then
`adb forward tcp:7980 tcp:7980` and speak the JSON-lines protocol from
`docs/control-protocol.md`. Never ship a build with `control` enabled.
### What the built APKs contain (inspected, not inferred from Gradle)

- Lite: one native library, no `assets/sam3`, no AI strings in the `.so`, no permissions.
- AI: `assets/sam3/{model.safetensors (stored uncompressed), vocab.json, merges.txt, LICENSE,
  manifest.json}` plus a native library containing the SAM 3 engine; no permissions.
- The `model.safetensors` extracted from the built AI APK hashes to the manifest's SHA-256.

## 9. Verification performed in this session

Passed:

- `cargo test -p lightcraft-segment` (9 tests, including F16 loading) and the checkpoint smoke
  test above.
- `cargo test -p lightcraft-engine --features sam` — 373 passed, 0 failed.
- `cargo test -p lightcraft-ui-egui` — 268 passed, 0 failed (Lite builds assert that no AI tiles
  exist; AI builds render them).
- `cargo clippy` with `-D warnings` for `lightcraft-segment`, `lightcraft-engine`,
  `lightcraft-ui-egui`, and the Android targets with and without `sam`.
- Both APKs build; signature, package IDs, permissions and contents inspected as above.
- Headless screenshots: the Lite masking panel shows Subject, Sky, Background, Brush, Linear,
  Radial, Luminance, Color; with the `sam` feature it adds Object and Describe.

Device verification (Lenovo Legion Y700, TB323FU, Android 16, arm64; ADB):

- Both APKs installed; the AI build launches with the embedded model already extracted.
- Tapping **AI Object Select**, then a real touch on the white pitcher, produced a mask of that
  pitcher; the translucent overlay appeared, with a pin on the object and the Keep/Cancel bar.
- A second tap on another cup moved the selection (one tap = one object) and the overlay followed.
- First object on a freshly started app: ~70 s from tap to visible overlay (model load + 1008²
  encode + decoder + the background detail pass). A tap made while the previous detail pass was
  still encoding waited ~64 s for the worker; tapping with the embedding cached and the worker
  idle is the decoder cost only.
- While debugging this, one UI bug was found and fixed: the Android Keep/Cancel bar checked a
  develop-settings snapshot taken before the AI tile's click, so on the frame the mask was created
  it believed the mask was already gone and cleared the tap tool - taps then did nothing. It now
  asks the session for the current settings. Verified fixed on the device with the steps above.

Not verified:

- EXIF-rotated/portrait fixtures, boundary objects and zoom/pan taps on the device (the
  transforms are shared with the desktop engine, which covers them with its mask tests).
- NPU acceleration: no NPU backend exists to verify.

## 10. Known limitations

- CPU inference: the image encode takes tens of seconds on a tablet-class CPU; taps after it take
  a couple of seconds. The encode runs once per photo/look.
- The model holds several GB while loaded; it is unloaded on memory pressure and after 10 minutes
  idle, so the next selection pays the load/encode cost again.
- The Android checkpoint is point-only, so **Describe is hidden** on Android.
- Release APKs use the local debug signing key.
- No Hexagon NPU acceleration (see §1).
- The AI APK is large because the weights are embedded uncompressed-as-stored (required for
  offline use).

## 11. Changing the inference resolution

`crates/engine/src/segment/mod.rs` (`INPUT_EDGE`) and `crates/segment/src/lib.rs` (`INPUT`) are
both 1008 for SAM 3; the checkpoint's positional encoding is tied to that shape. Changing it
requires a compatible export and a new reference test. Inference resolution is independent of the
photo and export resolution, so the user's photo is never degraded.

## 12. Updating the model

1. Replace `sam3.pt` (or `model.safetensors`) with the new checkpoint.
2. Regenerate/update `tools/ai/sam3_point_mapping.json` for any changed tensor layout (the script
   fails on a missing, extra, reshaped or mismatched tensor).
3. Run `prepare_models.py --source-pt <file>` then `--check`; inspect `manifest.json`.
4. Rebuild the AI APK and run the smoke test
   (`cargo run -p lightcraft-segment --release --example sam3_check -- <model dir>`).
5. For full numerical parity with the PyTorch reference, run
   `tools/sam3_reference.py` + `cargo test -p lightcraft-segment --release` with
   `LIGHTCRAFT_SAM3_DIR` and `LIGHTCRAFT_SAM3_REF` (needs `transformers`).

To replace the point-prompt model entirely (e.g. MobileSAM or SAM 2), implement the model behind
the `crates/segment` API (`load`, `encode`, `prepare_clicks`, `segment_clicks`) and keep the
engine worker and UI unchanged; the mask representation and editing pipeline are model-agnostic.

## 13. Relevant files

- `crates/segment/src/{lib,weights,tracker,vit,neck,nn,mask}.rs` — SAM 3 point-prompt engine
- `crates/segment/examples/sam3_check.rs` — checkpoint smoke test
- `crates/engine/src/segment/{mod,worker}.rs` — worker, cache, requests, status, unload
- `crates/engine/src/cmd/masks.rs` — `mask.objectPoint` (`reset`), `segment.model.status`
- `crates/ui-egui/src/panels/{masking,detail}.rs`, `crates/ui-egui/src/state.rs` — AI Object
  Select UI, Keep/Cancel bar, single-tap switching
- `apps/lightcraft-android/{Cargo.toml,src/app.rs}` — `sam` feature, embedded model dir, memory
  pressure release
- `android/app/src/main/java/com/lightcraft/android/MainActivity.java` — `extractModel`
- `android/app/build.gradle.kts`, `android/build.ps1`, `android/gradle.properties` — Lite/AI
  flavors, packaging, deliverables
- `tools/ai/prepare_models.py`, `tools/ai/sam3_point_mapping.json`, `tools/requirements-ai.txt`
  — reproducible checkpoint preparation
