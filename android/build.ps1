param(
    [ValidateSet('Debug','Release')][string]$Configuration = 'Release',
    [ValidateSet('Both','Lite','AI')][string]$Edition = 'Both',
    [ValidateSet('arm64-v8a','x86_64')][string]$Abi = 'arm64-v8a'
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot
if (-not $env:ANDROID_HOME) { throw 'Set ANDROID_HOME to the Android SDK directory.' }
if (-not $env:JAVA_HOME) { throw 'Set JAVA_HOME to a Java 17 JDK directory.' }
$ndk = Join-Path $env:ANDROID_HOME 'ndk/28.2.13676358'
$bin = Join-Path $ndk 'toolchains/llvm/prebuilt/windows-x86_64/bin'
$target = if ($Abi -eq 'arm64-v8a') { 'aarch64-linux-android' } else { 'x86_64-linux-android' }
$key = 'CARGO_TARGET_' + $target.ToUpper().Replace('-','_') + '_LINKER'
$oldLinker = [Environment]::GetEnvironmentVariable($key, 'Process')
$oldTargetDir = $env:CARGO_TARGET_DIR
[Environment]::SetEnvironmentVariable($key, (Join-Path $bin ($target + '29-clang.cmd')), 'Process')
$profile = $Configuration.ToLower()
$dist = Join-Path $PSScriptRoot 'dist'
New-Item -ItemType Directory -Force $dist | Out-Null

# Native tools (cargo, llvm-strip, gradlew) write warnings to stderr. PowerShell 5.1
# turns those into error records under ErrorActionPreference=Stop; judge them by exit code.
function Invoke-Native([string]$exe, [string[]]$arguments) {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $exe @arguments 2>&1 | ForEach-Object { Write-Output $_ }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previous
    }
    if ($code -ne 0) { throw "$exe failed with exit code $code" }
}

function Build-Edition([string]$edition) {
    $name = if ($edition -eq 'ai') { 'AI' } else { 'Lite' }
    $assemble = 'assemble' + $name + $Configuration
    $targetDir = Join-Path $repo ("target/android-" + $edition)
    $env:CARGO_TARGET_DIR = $targetDir
    $cargoArgs = @('build','--locked','-p','lightcraft-android','--target',$target)
    if ($edition -eq 'ai') { $cargoArgs += @('--features','sam') }
    if ($Configuration -eq 'Release') { $cargoArgs += '--release' }
    Write-Output ("=== Building $name native library ($Configuration, $Abi) ===")
    Invoke-Native 'cargo' $cargoArgs

    $library = Join-Path $targetDir "$target/$profile/liblightcraft_android.so"
    $dest = Join-Path $PSScriptRoot "app/src/$edition/jniLibs/$Abi"
    New-Item -ItemType Directory -Force $dest | Out-Null
    $so = Join-Path $dest 'liblightcraft_android.so'
    Copy-Item -LiteralPath $library -Destination $so -Force
    Invoke-Native (Join-Path $bin 'llvm-strip.exe') @('--strip-debug', $so)

    if ($edition -eq 'ai') {
        # QNN: build the chained-neck runner and stage the Hexagon runtime libs. The SDK is
        # not in the repo (Qualcomm licence); set QNN_SDK_ROOT to the QAIRT SDK directory.
        $qnnSdk = $env:QNN_SDK_ROOT
        if (-not $qnnSdk) {
            throw "Set QNN_SDK_ROOT to the QAIRT/QNN SDK (for example ...\qairt\2.50.0.260828) to build the AI edition's NPU runner."
        }
        $libs = Join-Path $qnnSdk 'lib/aarch64-android'
        Write-Output '=== Building the QNN chain runner ==='
        $qnnBuild = Join-Path $repo 'build/qnn_chain_android'
        $legacyToolchain = Join-Path $ndk 'build/cmake/android.toolchain.cmake'
        Invoke-Native 'cmake' @('-S', (Join-Path $repo 'tools/ai/qnn_chain'), '-B', $qnnBuild,
            "-DCMAKE_TOOLCHAIN_FILE=$legacyToolchain", '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-29',
            "-DQNN_SDK_ROOT=$qnnSdk", '-G', 'Ninja')
        Invoke-Native 'cmake' @('--build', $qnnBuild, '-j', '8')
        Copy-Item -LiteralPath (Join-Path $qnnBuild 'libqnn_chain.so') -Destination (Join-Path $dest 'libqnn_chain.so') -Force
        Invoke-Native (Join-Path $bin 'llvm-strip.exe') @('--strip-debug', (Join-Path $dest 'libqnn_chain.so'))
        # The QNN host libraries and the Hexagon skeleton must be real files in the app's
        # native library directory (Android forbids dlopen from app data), so they are jniLibs
        # and the AI edition extracts native libraries at install time.
        foreach ($lib in @('libQnnHtp.so','libQnnHtpV81Stub.so','libQnnSystem.so','libQnnModelDlc.so')) {
            Copy-Item -LiteralPath (Join-Path $libs $lib) -Destination (Join-Path $dest $lib) -Force
        }
        Copy-Item -LiteralPath (Join-Path $qnnSdk 'lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so') -Destination (Join-Path $dest 'libQnnHtpV81Skel.so') -Force

        # The compiled context binaries (tools/ai/onnx is generated, never committed).
        $qnnAssets = Join-Path $PSScriptRoot 'app/src/ai/assets/qnn'
        New-Item -ItemType Directory -Force $qnnAssets | Out-Null
        Copy-Item -LiteralPath (Join-Path $repo 'tools/ai/qnn_manifest.json') -Destination (Join-Path $qnnAssets 'manifest.json') -Force
        $artifacts = @{
            'stem_context.bin' = Join-Path $repo 'tools/ai/onnx/ends/stem_context.bin'
            'blocks0_16_context.bin' = Join-Path $repo 'tools/ai/onnx/blocks0_16/blocks0_16_context.bin'
            'blocks16_32_context.bin' = Join-Path $repo 'tools/ai/onnx/blocks16_32/blocks16_32_context.bin'
        }
        $qnnManifest = Get-Content -LiteralPath (Join-Path $repo 'tools/ai/qnn_manifest.json') -Raw | ConvertFrom-Json
        foreach ($name in $artifacts.Keys) {
            $source = $artifacts[$name]
            if (-not (Test-Path -LiteralPath $source)) {
                throw "NPU artifact $name is missing: $source (compile the contexts with tools/ai/export_sam3_block.py and compile_snapdragon.py)."
            }
            $target = Join-Path $qnnAssets $name
            if (-not (Test-Path -LiteralPath $target) -or
                (Get-Item -LiteralPath $target).Length -ne (Get-Item -LiteralPath $source).Length) {
                Copy-Item -LiteralPath $source -Destination $target -Force
            }
            $want = $qnnManifest.files.$name.size
            $have = (Get-Item -LiteralPath $target).Length
            if ($have -ne $want) { throw "NPU artifact $name is $have bytes, expected $want." }
        }

        $assets = Join-Path $PSScriptRoot 'app/src/ai/assets/sam3'
        $manifestPath = Join-Path $assets 'manifest.json'
        if (-not (Test-Path -LiteralPath $manifestPath)) {
            throw "AI assets are missing: run 'python tools/ai/prepare_models.py --accept-license --source-pt <sam3.pt>' first."
        }
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
        foreach ($file in @('model.safetensors','vocab.json','merges.txt','LICENSE')) {
            $path = Join-Path $assets $file
            if (-not (Test-Path -LiteralPath $path)) { throw "AI asset missing: $path" }
            $expected = $manifest.files.$file.size
            $actual = (Get-Item -LiteralPath $path).Length
            if ($actual -ne $expected) { throw "AI asset $file is $actual bytes, expected $expected (run prepare_models.py --check)." }
        }
        if ($manifest.textPrompts) {
            Write-Output 'Warning: the embedded checkpoint keeps text prompts; the Android UI still hides Describe in this build.'
        }
    }

    Write-Output ("=== Packaging $name APK ===")
    Invoke-Native (Join-Path $PSScriptRoot 'gradlew.bat') @('-p', $PSScriptRoot, ":app:$assemble", "-PandroidAbi=$Abi", '--console=plain')

    $apk = Join-Path $PSScriptRoot "app/build/outputs/apk/$edition/$profile/app-$edition-$profile.apk"
    if (-not (Test-Path -LiteralPath $apk)) { throw "Expected APK was not produced: $apk" }
    $deliverable = Join-Path $dist ("Lightcraft-" + $(if ($edition -eq 'ai') { 'AI' } else { 'Lite' }) + '.apk')
    Copy-Item -LiteralPath $apk -Destination $deliverable -Force
    Write-Output ("WROTE " + $deliverable + " (" + [math]::Round((Get-Item -LiteralPath $deliverable).Length / 1MB, 1) + " MB)")
}

try {
    $editions = switch ($Edition) {
        'Both' { @('lite','ai') }
        'Lite' { @('lite') }
        'AI' { @('ai') }
    }
    foreach ($edition in $editions) { Build-Edition $edition }
} finally {
    [Environment]::SetEnvironmentVariable($key, $oldLinker, 'Process')
    $env:CARGO_TARGET_DIR = $oldTargetDir
}
