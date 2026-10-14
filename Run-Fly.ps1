# Opens the live brain map in its own window, then runs the fly with the MuJoCo viewer.
#   .\Run-Fly.ps1                      # corrected brain (BANC scaled to FAFB counts), 5 simulated seconds
#   .\Run-Fly.ps1 -Raw                 # the brain as detected in BANC
#   .\Run-Fly.ps1 -Seconds 10 -Plasticity
# The simulation runs ~50x slower than real time: the brain wakes at 0.5 s simulated time, which
# takes about half a minute; until then the map says the brain is still asleep.
param(
    [double]$Seconds = 5,
    [double]$Birth = 0.5,
    [switch]$Raw,
    [switch]$Plasticity,
    [string]$Data = 'C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data',
    [string]$Python = 'C:\dev\venvs\noesis\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Python environment not found at $Python (see README: the venv lives outside OneDrive)."
}
$previous = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
# outside OneDrive, so syncing never locks the file both processes share
$live = Join-Path $env:TEMP 'noesis_live_activity.npz'
Push-Location -LiteralPath $PSScriptRoot
try {
    $mapArgs = @('-m', 'noesis.viz.brain_map', '--data', "`"$Data`"", '--watch', "`"$live`"")
    Start-Process -FilePath $Python -ArgumentList $mapArgs -WorkingDirectory $PSScriptRoot
    $simArgs = @('-m', 'noesis.experiments.newborn_muscles', '--viewer', '--data', $Data,
                 '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
                 '--birth', $Birth.ToString([cultureinfo]::InvariantCulture),
                 '--brain-map', $live)
    if (-not $Raw) { $simArgs += '--scaled' }
    if ($Plasticity) { $simArgs += '--plasticity' }
    & $Python @simArgs
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previous
}
