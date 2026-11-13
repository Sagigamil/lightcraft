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
