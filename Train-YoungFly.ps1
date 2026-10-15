param(
    [switch]$Resume,
    [switch]$Video,
    [switch]$NormalizeRewardError,
    [switch]$EarlyReject,
    [ValidateRange(1, 3600)][int]$WallSeconds = 3600,
    [ValidateRange(0.051, 3600)][double]$Seconds = 5,
    [ValidateRange(1, 10000)][int]$Episodes = 3,
    [ValidateRange(0, 4294967295)][long]$Seed = 0,
    [string]$Output = 'outputs/young-fly',
    [string]$ImportWeights,
    [string]$Data,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    $commonGit = (& git rev-parse --path-format=absolute --git-common-dir).Trim()
    $sharedRoot = Split-Path -Parent $commonGit
    if (-not $Data) {
        $Data = Join-Path $PSScriptRoot 'data'
        if (-not (Test-Path -LiteralPath $Data)) { $Data = Join-Path $sharedRoot 'data' }
    }
    if (-not $Python) {
        foreach ($candidate in @(
            (Join-Path $PSScriptRoot '.venv/Scripts/python.exe'),
            (Join-Path $sharedRoot '.venv/Scripts/python.exe'),
            'C:/dev/venvs/noesis/Scripts/python.exe'
        )) {
            if (Test-Path -LiteralPath $candidate) { $Python = $candidate; break }
        }
    }
    if (-not $Python) { throw 'Pass -Python with the Noesis body environment.' }
    if (-not (Test-Path -LiteralPath (Join-Path $Data 'raw/neurons.csv'))) {
        throw 'Pass -Data with the existing read-only connectome data.'
    }
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    $trainingArguments = @('-u', '-m', 'noesis.experiments.train_validate',
        '--data', $Data, '--output', $Output, '--wall-seconds', $WallSeconds,
        '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture), '--episodes', $Episodes, '--seed', $Seed)
    if ($Resume) { $trainingArguments += '--resume' }
    if ($ImportWeights) { $trainingArguments += @('--import-weights', $ImportWeights) }
    if ($Video) { $trainingArguments += '--video' }
    if ($NormalizeRewardError) { $trainingArguments += '--normalize-reward-error' }
    if ($EarlyReject) { $trainingArguments += '--early-reject' }
    & $Python @trainingArguments
    if ($LASTEXITCODE -eq 124) {
        Write-Host 'Wall budget reached. Resume replays the interrupted episode from its start.'
    } elseif ($LASTEXITCODE -ne 0) { throw "Training exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
