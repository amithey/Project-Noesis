param(
    [switch]$Learn = $true,
    [switch]$Video,
    [double]$Seconds = 3,
    [int]$Episodes = 3,
    [double]$LearningRate = 0.02,
    [ValidateSet('habitat', 'empty')][string]$Arena = 'habitat',
    [ValidateSet('corrected', 'raw')][string]$Brain = 'corrected',
    [switch]$NoEnvironmentSenses,
    [string]$Checkpoint,
    [string]$Data,
    [string]$Python
)
$ErrorActionPreference = 'Stop'
$previousPythonPath = $env:PYTHONPATH
Push-Location -LiteralPath $PSScriptRoot
try {
    # Managed worktrees share the original checkout's read-only data and venv.
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
    if (-not $Python -or -not (Test-Path -LiteralPath $Python)) {
        throw 'Python environment missing; pass -Python with the installed Noesis body environment.'
    }
    if (-not (Test-Path -LiteralPath (Join-Path $Data 'raw/neurons.csv'))) {
        throw 'Connectome data missing; pass -Data with the existing data directory.'
    }
    $runName = 'outputs/lab-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    # Each launch gets its own map; concurrent runs never overwrite one another.
    $liveMap = Join-Path $env:TEMP ('noesis_live_' + [guid]::NewGuid().ToString('N') + '.npz')
    Start-Process -FilePath $Python -ArgumentList @(
        '-m', 'noesis.viz.brain_map', '--data', ('"' + $Data + '"'), '--watch', ('"' + $liveMap + '"')
    ) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden
    $flyArguments = @('-u', '-m', 'noesis.experiments.learning_lab', '--viewer',
                      '--data', $Data, '--brain', $Brain, '--body-model', 'muscles',
                      '--brain-map', $liveMap,
                      '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
                      '--learning-rate', $LearningRate.ToString([cultureinfo]::InvariantCulture),
                      '--arena', $Arena,
                      '--episodes', $Episodes, '--output', $runName)
    if ($Learn) { $flyArguments += '--learn' }
    if ($Video) { $flyArguments += '--video' }
    if ($NoEnvironmentSenses) { $flyArguments += '--no-environment-senses' }
    if ($Checkpoint) { $flyArguments += @('--checkpoint', $Checkpoint) }
    Write-Host "Training=$Learn | brain=$Brain | body=muscles | results=$runName"
    Write-Host 'Full-connectome time advances slowly. Watch simulated time, motor Hz, and leg speed.'
    & $Python @flyArguments
    if ($LASTEXITCODE -ne 0) { throw "Fly simulation exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
