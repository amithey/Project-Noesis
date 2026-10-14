param(
    [switch]$Learn,
    [switch]$Video,
    [double]$Seconds = 3,
    [int]$Episodes = 1,
    [double]$LearningRate = 0.02,
    [ValidateSet('habitat', 'empty')][string]$Arena = 'habitat',
    [switch]$NoEnvironmentSenses,
    [string]$Checkpoint
)
$ErrorActionPreference = 'Stop'
$previousPythonPath = $env:PYTHONPATH
Push-Location $PSScriptRoot
try {
    $pythonPath = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        throw 'Create .venv and install the body dependencies first; see docs/learning-lab.md.'
    }
    $runName = 'outputs/lab-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff')
    $env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
    # Outside OneDrive so the fly process and the map window can share the file.
    $liveMap = Join-Path $env:TEMP 'noesis_live_activity.npz'
    Start-Process -FilePath $pythonPath -ArgumentList @(
        '-m', 'noesis.viz.brain_map', '--watch', $liveMap
    ) -WorkingDirectory $PSScriptRoot
    $flyArguments = @('-m', 'noesis.experiments.learning_lab', '--viewer',
                      '--brain-map', $liveMap,
                      '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
                      '--learning-rate', $LearningRate.ToString([cultureinfo]::InvariantCulture),
                      '--arena', $Arena,
                      '--episodes', $Episodes, '--output', $runName)
    if ($Learn) { $flyArguments += '--learn' }
    if ($Video) { $flyArguments += '--video' }
    if ($NoEnvironmentSenses) { $flyArguments += '--no-environment-senses' }
    if ($Checkpoint) { $flyArguments += @('--checkpoint', $Checkpoint) }
    & $pythonPath @flyArguments
    if ($LASTEXITCODE -ne 0) { throw "Fly simulation exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH = $previousPythonPath
    Pop-Location
}
