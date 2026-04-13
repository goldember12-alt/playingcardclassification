# bootstrap.ps1
# Reusable project bootstrap for Python repos on Windows/PowerShell.
# Keeps venvs, temp files, and pip cache outside OneDrive-backed repos.

param(
    [string]$BasePython = "C:\Program Files\Python312\python.exe",
    [string]$VenvRoot = "$HOME\.venvs",
    [string]$TempRoot = "$HOME\.tmp",
    [string]$PipCacheRoot = "$HOME\.pip-cache",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host ""
    Write-Host "==> $Message" -ForegroundColor Cyan
}

function Ensure-Directory {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        New-Item -ItemType Directory -Force -Path $Path | Out-Null
    }
}

$ProjectName = Split-Path -Leaf (Get-Location)
$VenvPath = Join-Path $VenvRoot $ProjectName
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"

Write-Step "Validating base Python"
if (-not (Test-Path -LiteralPath $BasePython)) {
    throw "Base Python not found at: $BasePython"
}

Write-Step "Ensuring shared directories exist"
Ensure-Directory -Path $VenvRoot
Ensure-Directory -Path $TempRoot
Ensure-Directory -Path $PipCacheRoot

$env:TEMP = $TempRoot
$env:TMP = $TempRoot
$env:PIP_CACHE_DIR = $PipCacheRoot

Write-Step "Session environment configured"
Write-Host "TEMP=$env:TEMP"
Write-Host "TMP=$env:TMP"
Write-Host "PIP_CACHE_DIR=$env:PIP_CACHE_DIR"

Write-Step "Creating virtual environment if needed"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    & $BasePython -m venv $VenvPath
}

Write-Step "Upgrading pip inside project venv"
& $VenvPython -m pip install --upgrade pip

if ((Test-Path -LiteralPath ".\requirements.txt") -and (-not $SkipInstall)) {
    Write-Step "Installing requirements.txt"
    & $VenvPython -m pip install -r .\requirements.txt
}
elseif (-not (Test-Path -LiteralPath ".\requirements.txt")) {
    Write-Step "No requirements.txt found; skipping dependency install"
}
else {
    Write-Step "Skipping dependency install because -SkipInstall was provided"
}

Write-Step "Python verification"
& $VenvPython -c "import sys; print(sys.executable)"

Write-Step "Done"
Write-Host "Project venv: $VenvPath" -ForegroundColor Green
Write-Host "Project python: $VenvPython" -ForegroundColor Green
Write-Host ""
Write-Host "Use this interpreter for future commands in this repo:" -ForegroundColor Yellow
Write-Host "& '$VenvPython' ..." -ForegroundColor Yellow
