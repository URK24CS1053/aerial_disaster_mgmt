# activate-venv.ps1
# PowerShell helper to activate the virtual environment
# Usage: .\activate-venv.ps1
#
# This script activates the virtual environment using the call operator (&)
# to avoid PowerShell module-loading errors

param(
    [string]$Action = "activate"
)

$venvActivatePath = Join-Path -Path $PSScriptRoot -ChildPath ".venv\Scripts\Activate.ps1"

if (-not (Test-Path $venvActivatePath)) {
    Write-Host "[XX] Virtual environment not found at: $venvActivatePath" -ForegroundColor Red
    Write-Host "     To create it, run: python -m venv .venv" -ForegroundColor Yellow
    exit 1
}

if ($Action -eq "activate") {
    Write-Host "[..] Activating virtual environment..." -ForegroundColor Cyan
    & $venvActivatePath
    Write-Host "[OK] Virtual environment activated" -ForegroundColor Green
    Write-Host "     Run 'deactivate' to exit" -ForegroundColor Gray
} else {
    Write-Host "[XX] Unknown action: $Action" -ForegroundColor Red
    exit 1
}
