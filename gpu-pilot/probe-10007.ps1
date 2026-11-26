$ErrorActionPreference = 'Stop'
$pilotRoot = $PSScriptRoot
$projectRoot = Split-Path $pilotRoot -Parent
$pythonExe = 'C:/Users/heyma/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
$sourceRoot = 'C:/Users/heyma/.codex/worktrees/brain-body-feedback/Project Noesis/src'
$env:PYTHONPATH = "$pilotRoot/runtime;$pilotRoot/arrow-runtime;$pilotRoot/src"
$env:PYTHONUTF8 = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NUMBA_CACHE_DIR = "$pilotRoot/outputs/numba-cache-312"
$parentBank = 'C:/Users/heyma/.codex/worktrees/brain-body-feedback/Project Noesis/outputs/birth-curriculum-v31/round_002/train-5/attempt_0026/episode_000.npz'
$candidateBank = "$pilotRoot/outputs/standing-001-train-11000/weights-last.npz"
$protocolPath = "$pilotRoot/outputs/standing-probe-10007.protocol.json"
if (Test-Path -LiteralPath $protocolPath) { throw 'Probe already registered; inspect its process and output before any restart.' }
$pins = @(Get-ChildItem "$pilotRoot/src/noesis_gpu" -Filter '*.py' | Get-FileHash -Algorithm SHA256 | Select-Object Path,Hash)
$protocol = [ordered]@{
    purpose = 'Second matched frozen standing probe on the new body; not held-out validation'
    seed = 10007; seconds = 5; warmup_ms = 50; measured_samples = 990
    learning = $false; motor_exploration = 0; early_fall_cutoff = $false
    backend = 'cpu'; worlds = 1; standing_threshold = 0.9
    support = 'tarsus5-to-floor-only'; model_promoted = $false
    expected_parent_sha256 = '078cfe3ce59964ca655d39287f9f60249dde878dc13c119a7ee01749d8b4afe8'
    expected_candidate_sha256 = 'fb80cfcc3018329209784751570d7ffb1ed33aa9ef48fceaa85588862f7626a8'
    package_source_pins = $pins
    noesis_source_sha256 = 'f52dff5a0d4d3eb14f08c30d6f9be7e646a06adf1361bb2d4b0dde1a9870396b'
}
if ((Get-FileHash $parentBank).Hash.ToLower() -ne $protocol.expected_parent_sha256) { throw 'Parent hash mismatch' }
if ((Get-FileHash $candidateBank).Hash.ToLower() -ne $protocol.expected_candidate_sha256) { throw 'Candidate hash mismatch' }
$protocol | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $protocolPath -Encoding utf8
$receipts = @()
foreach ($probeCase in @(@{name='parent'; bank=$parentBank}, @{name='candidate'; bank=$candidateBank})) {
    $outputDir = "$pilotRoot/outputs/standing-001-$($probeCase.name)-10007"
    Write-Output "Starting frozen $($probeCase.name), seed 10007"
    & $pythonExe -m noesis_gpu.run --backend cpu --worlds 1 --seconds 5 --seed 10007 --wall-seconds 600 --noesis-source $sourceRoot --data "$projectRoot/data" --checkpoint $probeCase.bank --output $outputDir > "$outputDir.stdout.txt" 2>&1
    $nativeExit = $LASTEXITCODE
    $receipts += @{case=$probeCase.name; exit_code=$nativeExit; output=$outputDir}
    $receipts | ConvertTo-Json -Depth 6 | Set-Content "$pilotRoot/outputs/standing-probe-10007-exit.json" -Encoding utf8
    if ($nativeExit -ne 0) { throw "Frozen probe failed: $($probeCase.name), exit $nativeExit" }
    Write-Output "Completed frozen $($probeCase.name), seed 10007"
}
foreach ($pin in $pins) {
    if ((Get-FileHash -LiteralPath $pin.Path).Hash -ne $pin.Hash) { throw "Source changed during probe: $($pin.Path)" }
}
Write-Output 'Both frozen probes complete; source pins unchanged.'
