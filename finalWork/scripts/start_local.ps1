param(
    [string]$Python = "",
    [ValidateRange(1024, 65535)][int]$Port = 8520
)

$ErrorActionPreference = "Stop"
$projectDirectory = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = Join-Path $projectDirectory ".venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $Python)) {
        throw "Create .venv and install requirements.txt first. See docs/RUN_REAL_AI.md; or pass -Python with your Python executable."
    }
}

Push-Location -LiteralPath $projectDirectory
try {
    & $Python -m moneygraph serve --data data --out results/real --port $Port
    if ($LASTEXITCODE -ne 0) {
        throw "MoneyGraph exited with code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}
