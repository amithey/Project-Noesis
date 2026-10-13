# Opens the live brain map in its own window, then runs the fly with the MuJoCo viewer.
#   .\Run-Fly.ps1                      # 5 simulated seconds, leg + balance senses
#   .\Run-Fly.ps1 -Seconds 10 -Plasticity
#   .\Run-Fly.ps1 -Empty            # plain floor instead of the trigger habitat
# The simulation runs ~50x slower than real time: the brain wakes at 0.5 s simulated time, which
# takes one to two minutes of waiting; until then the map says the brain is still asleep.
param(
    [double]$Seconds = 5,
    [double]$Birth = 0.5,
    [switch]$Empty,
    [switch]$Plasticity,
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
    Start-Process -FilePath $Python -ArgumentList @('-m', 'noesis.viz.brain_map', '--watch', "`"$live`"") `
        -WorkingDirectory $PSScriptRoot
    $simArgs = @('-m', 'noesis.experiments.newborn_muscles', '--viewer',
                 '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
                 '--birth', $Birth.ToString([cultureinfo]::InvariantCulture),
                 '--brain-map', $live)
    if (-not $Empty) { $simArgs += @('--arena', 'habitat') }  # food, odours, light, obstacles
    if ($Plasticity) { $simArgs += '--plasticity' }
    & $Python @simArgs
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previous
}
