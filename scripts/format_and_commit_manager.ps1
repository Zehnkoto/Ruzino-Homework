#!/usr/bin/env pwsh
# Format and Commit Manager - PowerShell Launcher
# Format C/C++ files and commit & push changes interactively

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonScript = Join-Path $scriptDir "format_and_commit_manager.py"

Write-Host "Starting Format and Commit Manager..." -ForegroundColor Cyan
Write-Host ""

# Check if Python is available
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not found in PATH" -ForegroundColor Red
    Write-Host "Please install Python or ensure it's in your PATH" -ForegroundColor Yellow
    exit 1
}

# Run the Python script
python $pythonScript

# Preserve exit code
exit $LASTEXITCODE
