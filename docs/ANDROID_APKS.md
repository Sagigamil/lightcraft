# Android builds: three editions

`android/build.ps1` builds one of three editions (default: all three):

| edition | APK | applicationId | AI engine |
|---|---|---|---|
| `Lite` | `dist/Lightcraft-Lite.apk` | `com.lightcraft.android` | none |
| `SAM3` | `dist/Lightcraft-SAM3.apk` | `com.lightcraft.android` | SAM 3 point prompts on the Hexagon HTP |
| `MobileSAM` | `dist/Lightcraft-MobileSAM.apk` | `com.lightcraft.android` | MobileSAM HTP encoder + CPU decoder |

All three use the **same applicationId**, so installing one replaces another (Android treats
them as the same app). They are signed with the same debug key for local sideloading.

```powershell
$env:ANDROID_HOME = '...\Android\Sdk'
$env:JAVA_HOME = '...\jdk-17'
$env:QNN_SDK_ROOT = '...\qairt\2.50.0.260828'   # SAM3 and MobileSAM only
.\android\build.ps1 -Edition All -Configuration Release
# or: -Edition Lite | -Edition SAM3 | -Edition MobileSAM
```

`-Edition Lite` needs neither `QNN_SDK_ROOT` nor any model artifact.

## What is in the repository, and what is not

The source of all three engines is committed. The large/licensed inputs are **never** committed
(they are in `.gitignore`):

| input | where it is expected | needed by |
|---|---|---|
| QAIRT/QNN SDK (`include/QNN`, `lib/aarch64-android`, `lib/hexagon-v81/unsigned`) | `QNN_SDK_ROOT` | SAM3, MobileSAM |
| SAM 3 point-prompt checkpoint + tokenizer | `android/app/src/sam3/assets/sam3/` | SAM3 |
| SAM 3 QNN contexts (`stem`, `blocks0_16`, `blocks16_32`) | `tools/ai/onnx/ends/`, `tools/ai/onnx/blocks0_16/`, `tools/ai/onnx/blocks16_32/` | SAM3 |
| MobileSAM contexts (`encoder_ctx.bin`, `decoder8m_cpu.bin`) | `tools/ai/mobilesam/ctx/`, `tools/ai/mobilesam/` | MobileSAM |

The SAM 3 assets are produced by:

```powershell
python tools/ai/prepare_models.py --accept-license --source-pt <sam3.pt> `
  --out android/app/src/sam3/assets/sam3
```

The MobileSAM and SAM 3 QNN contexts are produced with the scripts in `docs/MOBILESAM_NPU.md`
and `docs/SNAPDRAGON_NPU.md`.

`build.ps1` verifies every artifact's size against its packaged manifest and fails with a clear
message when one is missing, so a partial checkout cannot produce a broken AI APK.

## GitHub CI

- `.github/workflows/android-ci.yml` builds **Lite** on every push to `main` (and manually). It
  needs no SDK or model binaries, so it stays green on a clean checkout.
- `.github/workflows/android-ai.yml` is **manual** (`workflow_dispatch`) and builds SAM3 and/or
  MobileSAM. It downloads a bundle from the `AI_BUNDLE_URL` repository secret and runs
  `build.ps1` on `windows-latest`.

The bundle is a zip extracted at the repository root:

```
qnn-sdk/                                  # the QAIRT SDK root
  include/QNN/...
  lib/aarch64-android/...
  lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so
tools/ai/onnx/ends/stem_context.bin
tools/ai/onnx/blocks0_16/blocks0_16_context.bin
tools/ai/onnx/blocks16_32/blocks16_32_context.bin
tools/ai/mobilesam/ctx/encoder_ctx.bin
tools/ai/mobilesam/decoder8m_cpu.bin
android/app/src/sam3/assets/sam3/{manifest.json,model.safetensors,vocab.json,merges.txt,LICENSE}
```

Only include the parts the requested edition needs. The SAM 3 bundle is roughly 2 GB (0.95 GB
checkpoint + 0.96 GB contexts) plus the QNN SDK; the MobileSAM bundle is roughly 60 MB plus the
QNN SDK.

## Credentials

The Qualcomm AI Hub token lives only in the local `~/.qai_hub/client.ini` (or
`%USERPROFILE%\.qai_hub\client.ini`) and is used only by the offline export/compile scripts.
`client.ini` and `.qai_hub/` are git-ignored. Never add the token to a workflow, a script or a
committed configuration file; use `qai-hub configure` locally instead.
