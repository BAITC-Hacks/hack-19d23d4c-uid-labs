param([string]$Python = "python")
$ErrorActionPreference = "Stop"
Push-Location (Join-Path $PSScriptRoot "..")
try {
    & $Python -m moneygraph demo --out results/demo --size 2248 --stability
    if ($LASTEXITCODE -ne 0) { throw "Demo failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
