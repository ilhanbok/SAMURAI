# Start the SAMURAI web portal (Windows / Anaconda Powershell).
#
#   .\run_web.ps1
#   $env:SAMURAI_PORT=8080; .\run_web.ps1
#   $env:SAMURAI_BACKEND="stub"; .\run_web.ps1   # UI only, no SAM2
#
# Run from the repository root with venv_SAMURAI activated.

Set-Location $PSScriptRoot

python -c "import flask" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing web dependencies..."
    pip install -r webapp/requirements-web.txt
}

$portalHost = if ($env:SAMURAI_HOST) { $env:SAMURAI_HOST } else { "127.0.0.1" }
$port = if ($env:SAMURAI_PORT) { $env:SAMURAI_PORT } else { "5000" }
Write-Host "SAMURAI portal on http://${portalHost}:${port}"
python -m webapp.app
