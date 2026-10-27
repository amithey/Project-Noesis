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
    $readyMap = $liveMap + '.ready'
    New-Item -ItemType Directory -Path 'outputs' -Force | Out-Null
    $mapLog = Join-Path $PSScriptRoot ($runName + '-brain.log')
    $mapError = Join-Path $PSScriptRoot ($runName + '-brain-error.log')
    $mapProcess = Start-Process -FilePath $Python -ArgumentList @(
        '-u', '-m', 'noesis.viz.brain_map', '--data', ('"' + $Data + '"'),
        '--watch', ('"' + $liveMap + '"'), '--ready-file', ('"' + $readyMap + '"')
    ) -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -PassThru `
      -RedirectStandardOutput $mapLog -RedirectStandardError $mapError
    Write-Host 'Loading the 3-D neuron network window...'
    $mapDeadline = (Get-Date).AddSeconds(60)
    while (-not (Test-Path -LiteralPath $readyMap)) {
        $mapProcess.Refresh()
        if ($mapProcess.HasExited) {
            throw "Brain window failed: $(Get-Content -LiteralPath $mapError -Raw)"
        }
        if ((Get-Date) -gt $mapDeadline) {
            throw "Brain window has not finished opening. See $mapLog and $mapError"
        }
        Start-Sleep -Milliseconds 200
    }
    Write-Host '3-D neuron network window is ready.'
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
