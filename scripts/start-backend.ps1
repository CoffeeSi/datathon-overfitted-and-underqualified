param(
    [ValidateRange(1, 65535)][int]$Port = 8000,
    [string]$ActionConfig = "configs/action_service_backend_smoke.yaml"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepoRoot
$Python = Join-Path $RepoRoot ".venv/Scripts/python.exe"
if (-not (Test-Path -LiteralPath $Python)) {
    throw "Repository .venv is missing. Install requirements first."
}
if (-not (Test-Path -LiteralPath $ActionConfig)) {
    throw "Action config does not exist: $ActionConfig"
}
# Explicit values override stale variables inherited from an earlier terminal run.
# This is the research integration checkpoint, not a quality-approved deployment.
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:FORTEBANK_ANALYZER_FACTORY = "overfitted_underqualified.backend.local_analyzer:create_analyzer"
$env:FORTEBANK_ACTION_CONFIG = $ActionConfig
$env:FORTEBANK_ENABLE_MOCK_RESULTS = "false"
& $Python scripts/run_backend.py --port $Port --action-config $ActionConfig
exit $LASTEXITCODE
