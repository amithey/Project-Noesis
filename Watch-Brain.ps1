# Live brain map while sugar-taste neurons switch on and off (brain only, no body: near real time).
#   .\Watch-Brain.ps1            # BANC as detected
#   .\Watch-Brain.ps1 -Scaled    # BANC synapses corrected to FAFB counts (needs data\fafb)
# Close the map window and press Ctrl-C here to stop.
param(
    [switch]$Scaled,
    [double]$Seconds = 120,
    [double]$Period = 2,
    [string]$Data = 'C:\Users\heyma\OneDrive\Desktop\מדעי המחשב\פרוייקטים\Project Noesis\data',
    [string]$Python = 'C:\dev\venvs\noesis\Scripts\python.exe'
)
$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $Python)) { throw "Python environment not found at $Python" }
$previous = $env:PYTHONPATH
$env:PYTHONPATH = Join-Path $PSScriptRoot 'src'
$live = Join-Path $env:TEMP 'noesis_live_brain.npz'
Push-Location -LiteralPath $PSScriptRoot
try {
    Start-Process -FilePath $Python -ArgumentList @('-m', 'noesis.viz.brain_map', '--data', "`"$Data`"",
        '--watch', "`"$live`"") -WorkingDirectory $PSScriptRoot
    $demo = @('-m', 'noesis.experiments.live_brain', '--data', $Data, '--brain-map', $live, '--realtime',
              '--seconds', $Seconds.ToString([cultureinfo]::InvariantCulture),
              '--period', $Period.ToString([cultureinfo]::InvariantCulture))
    if ($Scaled) { $demo += '--scaled' }
    & $Python @demo
}
finally {
    Pop-Location
    $env:PYTHONPATH = $previous
}
