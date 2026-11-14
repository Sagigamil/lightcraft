param(
    [ValidateSet('Debug','Release')][string]$Configuration = 'Release',
    [ValidateSet('All','Lite','SAM3','MobileSAM')][string]$Edition = 'All',
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

# Gradle flavor name -> display name used in the APK file name.
function Flavor([string]$edition) {
    switch ($edition) {
        'lite' { 'Lite' }
        'sam3' { 'Sam3' }
        'mobilesam' { 'Mobilesam' }
    }
}
function Label([string]$edition) {
    switch ($edition) {
        'lite' { 'Lite' }
        'sam3' { 'SAM3' }
        'mobilesam' { 'MobileSAM' }
    }
}

function Copy-QnnLibs([string]$libs, [string[]]$names, [string]$dest) {
    foreach ($lib in $names) {
        Copy-Item -LiteralPath (Join-Path $libs $lib) -Destination (Join-Path $dest $lib) -Force
    }
    Copy-Item -LiteralPath (Join-Path $QnnSdk 'lib/hexagon-v81/unsigned/libQnnHtpV81Skel.so') -Destination (Join-Path $dest 'libQnnHtpV81Skel.so') -Force
}

function Build-Edition([string]$edition) {
    $name = Label $edition
    $assemble = 'assemble' + (Flavor $edition) + $Configuration
    $targetDir = Join-Path $repo ("target/android-" + $edition)
    $env:CARGO_TARGET_DIR = $targetDir
    $cargoArgs = @('build','--locked','-p','lightcraft-android','--target',$target)
    if ($edition -eq 'sam3' -or $edition -eq 'mobilesam') { $cargoArgs += @('--features',$edition) }
    if ($Configuration -eq 'Release') { $cargoArgs += '--release' }
    Write-Output ("=== Building $name native library ($Configuration, $Abi) ===")
    Invoke-Native 'cargo' $cargoArgs

    $library = Join-Path $targetDir "$target/$profile/liblightcraft_android.so"
    $dest = Join-Path $PSScriptRoot "app/src/$edition/jniLibs/$Abi"
    New-Item -ItemType Directory -Force $dest | Out-Null
    $so = Join-Path $dest 'liblightcraft_android.so'
    Copy-Item -LiteralPath $library -Destination $so -Force
    Invoke-Native (Join-Path $bin 'llvm-strip.exe') @('--strip-debug', $so)

    if ($edition -eq 'sam3' -or $edition -eq 'mobilesam') {
        # QNN: the SDK is not in the repo (Qualcomm licence); set QNN_SDK_ROOT to the QAIRT
        # SDK directory. The host libraries and the Hexagon skeleton must be real files in
        # the app's native library directory (Android forbids dlopen from app data), so they
        # are jniLibs; useLegacyPackaging extracts them at install time.
        $script:QnnSdk = $env:QNN_SDK_ROOT
        if (-not $script:QnnSdk) {
            throw "Set QNN_SDK_ROOT to the QAIRT/QNN SDK (for example ...\qairt\2.50.0.260828) to build the AI editions."
        }
        $libs = Join-Path $script:QnnSdk 'lib/aarch64-android'
        $legacyToolchain = Join-Path $ndk 'build/cmake/android.toolchain.cmake'
        $assets = Join-Path $PSScriptRoot "app/src/$edition/assets"
        $qnnAssets = Join-Path $assets 'qnn'
        New-Item -ItemType Directory -Force $qnnAssets | Out-Null

        if ($edition -eq 'sam3') {
            Write-Output '=== Building the SAM 3 QNN chain runner ==='
            $qnnBuild = Join-Path $repo 'build/qnn_chain_android'
            Invoke-Native 'cmake' @('-S', (Join-Path $repo 'tools/ai/qnn_chain'), '-B', $qnnBuild,
                "-DCMAKE_TOOLCHAIN_FILE=$legacyToolchain", '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-29',
                "-DQNN_SDK_ROOT=$script:QnnSdk", '-G', 'Ninja')
            Invoke-Native 'cmake' @('--build', $qnnBuild, '-j', '8')
            Copy-Item -LiteralPath (Join-Path $qnnBuild 'libqnn_chain.so') -Destination (Join-Path $dest 'libqnn_chain.so') -Force
            Invoke-Native (Join-Path $bin 'llvm-strip.exe') @('--strip-debug', (Join-Path $dest 'libqnn_chain.so'))
            Copy-QnnLibs $libs @('libQnnHtp.so','libQnnHtpV81Stub.so','libQnnSystem.so','libQnnModelDlc.so') $dest

            $manifestSource = Join-Path $repo 'tools/ai/qnn_manifest.json'
            Copy-Item -LiteralPath $manifestSource -Destination (Join-Path $qnnAssets 'manifest.json') -Force
            $artifacts = @{
                'stem_context.bin' = Join-Path $repo 'tools/ai/onnx/ends/stem_context.bin'
                'blocks0_16_context.bin' = Join-Path $repo 'tools/ai/onnx/blocks0_16/blocks0_16_context.bin'
                'blocks16_32_context.bin' = Join-Path $repo 'tools/ai/onnx/blocks16_32/blocks16_32_context.bin'
            }
            $qnnManifest = Get-Content -LiteralPath $manifestSource -Raw | ConvertFrom-Json
            foreach ($artifact in $artifacts.Keys) {
                $source = $artifacts[$artifact]
                if (-not (Test-Path -LiteralPath $source)) {
                    throw "NPU artifact $artifact is missing: $source (compile the contexts with tools/ai/export_sam3_block.py and compile_snapdragon.py)."
                }
                $targetFile = Join-Path $qnnAssets $artifact
                Copy-Item -LiteralPath $source -Destination $targetFile -Force
                $want = $qnnManifest.files.$artifact.size
                $have = (Get-Item -LiteralPath $targetFile).Length
                if ($have -ne $want) { throw "NPU artifact $artifact is $have bytes, expected $want." }
            }

            # The point-prompt checkpoint and tokenizer, prepared by tools/ai/prepare_models.py.
            $sam3Assets = Join-Path $assets 'sam3'
            $manifestPath = Join-Path $sam3Assets 'manifest.json'
            if (-not (Test-Path -LiteralPath $manifestPath)) {
                throw "AI assets are missing: run 'python tools/ai/prepare_models.py --accept-license --source-pt <sam3.pt> --out `"$sam3Assets`"' first."
            }
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
            foreach ($file in @('model.safetensors','vocab.json','merges.txt','LICENSE')) {
                $path = Join-Path $sam3Assets $file
                if (-not (Test-Path -LiteralPath $path)) { throw "AI asset missing: $path" }
                $expected = $manifest.files.$file.size
                $actual = (Get-Item -LiteralPath $path).Length
                if ($actual -ne $expected) { throw "AI asset $file is $actual bytes, expected $expected (run prepare_models.py --check)." }
            }
        }

        if ($edition -eq 'mobilesam') {
            Write-Output '=== Building the MobileSAM runner ==='
            $mobileBuild = Join-Path $repo 'build/mobilesam_android'
            Invoke-Native 'cmake' @('-S', (Join-Path $repo 'tools/ai/mobilesam'), '-B', $mobileBuild,
                "-DCMAKE_TOOLCHAIN_FILE=$legacyToolchain", '-DANDROID_ABI=arm64-v8a', '-DANDROID_PLATFORM=android-29',
                "-DQNN_SDK_ROOT=$script:QnnSdk", '-G', 'Ninja')
            Invoke-Native 'cmake' @('--build', $mobileBuild, '-j', '8')
            Copy-Item -LiteralPath (Join-Path $mobileBuild 'libmobilesam.so') -Destination (Join-Path $dest 'libmobilesam.so') -Force
            Invoke-Native (Join-Path $bin 'llvm-strip.exe') @('--strip-debug', (Join-Path $dest 'libmobilesam.so'))
            Copy-QnnLibs $libs @('libQnnHtp.so','libQnnHtpV81Stub.so','libQnnCpu.so') $dest

            $manifestSource = Join-Path $repo 'tools/ai/mobilesam/manifest.json'
            Copy-Item -LiteralPath $manifestSource -Destination (Join-Path $qnnAssets 'manifest.json') -Force
            $artifacts = @{
                'encoder_ctx.bin' = Join-Path $repo 'tools/ai/mobilesam/ctx/encoder_ctx.bin'
                'decoder8m_cpu.bin' = Join-Path $repo 'tools/ai/mobilesam/decoder8m_cpu.bin'
            }
            $mobileManifest = Get-Content -LiteralPath $manifestSource -Raw | ConvertFrom-Json
            foreach ($artifact in $artifacts.Keys) {
                $source = $artifacts[$artifact]
                if (-not (Test-Path -LiteralPath $source)) {
                    throw "MobileSAM artifact $artifact is missing: $source (see docs/MOBILESAM_NPU.md)."
                }
                $targetFile = Join-Path $qnnAssets $artifact
                Copy-Item -LiteralPath $source -Destination $targetFile -Force
                $want = $mobileManifest.files.$artifact.size
                $have = (Get-Item -LiteralPath $targetFile).Length
                if ($have -ne $want) { throw "MobileSAM artifact $artifact is $have bytes, expected $want." }
            }
        }
    }

    Write-Output ("=== Packaging $name APK ===")
    Invoke-Native (Join-Path $PSScriptRoot 'gradlew.bat') @('-p', $PSScriptRoot, ":app:$assemble", "-PandroidAbi=$Abi", '--console=plain')

    $apk = Join-Path $PSScriptRoot "app/build/outputs/apk/$edition/$profile/app-$edition-$profile.apk"
    if (-not (Test-Path -LiteralPath $apk)) { throw "Expected APK was not produced: $apk" }
    $deliverable = Join-Path $dist ("Lightcraft-" + $name + '.apk')
    Copy-Item -LiteralPath $apk -Destination $deliverable -Force
    Write-Output ("WROTE " + $deliverable + " (" + [math]::Round((Get-Item -LiteralPath $deliverable).Length / 1MB, 1) + " MB)")
}

try {
    $editions = switch ($Edition) {
        'All' { @('lite','sam3','mobilesam') }
        'Lite' { @('lite') }
        'SAM3' { @('sam3') }
        'MobileSAM' { @('mobilesam') }
    }
    foreach ($edition in $editions) { Build-Edition $edition }
} finally {
    [Environment]::SetEnvironmentVariable($key, $oldLinker, 'Process')
    $env:CARGO_TARGET_DIR = $oldTargetDir
}