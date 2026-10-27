param(
    [ValidateSet('Antigravity', 'GeminiApi')][string]$Client = 'Antigravity',
    [string]$Task = @'
Work as a collaborating agent on Project Noesis. First read README.md,
docs/learning-lab.md and docs/learning-validation.md, then inspect git status.
Determine the next concrete step needed for a reliable embodied simulation.
Inspect outputs/argon-startup-check-20261014 if present and report its actual
completion status. Do not claim learned walking or flight without evidence.
Preserve all existing uncommitted work. Do not read or transmit secrets or raw
connectome datasets. Coordinate changes with the user before editing files
another agent may be using. Propose a bounded task you can own and execute it
once its scope is clear. Use the existing Python environment and PYTHONPATH=src.
'@
)
$ErrorActionPreference = 'Stop'
$cli = if ($Client -eq 'Antigravity') { 'agy' } else { 'gemini' }
if (-not (Get-Command $cli -ErrorAction SilentlyContinue)) {
    throw "Required CLI is missing: $cli"
}
Push-Location -LiteralPath $PSScriptRoot
try {
    if ($Client -eq 'Antigravity') {
        Write-Host 'Use /model to select Gemini 4 Argon before submitting the task below.'
        Write-Host 'If Argon is absent, this account/client has not established access to it.'
        Write-Host $Task
        & agy
    } else {
        & gemini --model gemini-4-argon --prompt-interactive $Task
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Gemini exited with code $LASTEXITCODE. Check authentication and Argon access."
    }
} finally {
    Pop-Location
}
