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
    $liveMap = Join-Path $env:TEMP ('noesis_live_' + [guid]::NewGuid().ToString('N') + '.npz')
    $readyMap = $liveMap + '.ready'
    New-Item -ItemType Directory -Path 'outputs' -Force | Out-Null
    $mapLog = Join-Path $PSScriptRoot ($runName + '-brain.log')
    $mapError = Join-Path $PSScriptRoot ($runName + '-brain-error.log')
    $mapProcess = Start-Process -FilePath $pythonPath -ArgumentList @(
        '-u', '-m', 'noesis.viz.brain_map',
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
