# Opens the live 3D neuron network on the right, then runs the embodied fly with the MuJoCo viewer.
#   .\Run-Fly.ps1                      # corrected brain (FAFB + MANC), sensory habitat, internal state, 5 s
#   .\Run-Fly.ps1 -Empty               # plain floor instead of the sensory habitat
#   .\Run-Fly.ps1 -Raw                 # the brain as detected in BANC
#   .\Run-Fly.ps1 -Flat2D              # the older 2D schematic instead of the 3D network
#   .\Run-Fly.ps1 -Seconds 10 -Plasticity
# The simulation runs ~50x slower than real time: the brain wakes at 0.5 s simulated time, which
# takes about half a minute; until then the map says the brain is still asleep.
param(
    [double]$Seconds = 5,
    [double]$Birth = 0.5,
    [switch]$Empty,
    [switch]$Raw,
    [switch]$Plasticity,
    [switch]$Flat2D,
    [string]$Geometry = '960x1000+940+20',
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
    if ($Flat2D) {
        $mapArgs = @('-m', 'noesis.viz.brain_map', '--data', "`"$Data`"", '--watch', "`"$live`"",
                     '--geometry', "`"$Geometry`"")
    } else {
        & $Python -c "import vispy"
        if ($LASTEXITCODE -ne 0) {
            throw "vispy is required for the 3D neuron network. Install it in the noesis venv, or re-run with -Flat2D."
        }
        $mapArgs = @('-m', 'noesis.viz.brain_view3d', '--data', "`"$Data`"", '--watch', "`"$live`"",
                     '--geometry', "`"$Geometry`"")
        if (-not $Raw) { $mapArgs += '--scaled' }
    }
    Start-Process -FilePath $Python -ArgumentList $mapArgs -WorkingDirectory $PSScriptRoot
    $simArgs = @('-m', 'noesis.experiments.newborn_muscles', '--viewer', '--data', $Data,
                 '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
                 '--birth', $Birth.ToString([cultureinfo]::InvariantCulture),
                 '--brain-map', $live)
    if (-not $Empty) { $simArgs += @('--arena', 'habitat') }
    if (-not $Raw) { $simArgs += '--scaled' }
    if ($Plasticity) { $simArgs += '--plasticity' }
    & $Python @simArgs
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previous
}
