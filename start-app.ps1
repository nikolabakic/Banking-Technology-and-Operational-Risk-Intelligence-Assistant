[CmdletBinding()]
param(
    [ValidateRange(1, 600)]
    [int]$ApiStartupTimeoutSeconds = 300
)

$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
$frontendDir = Join-Path $projectRoot "frontend"
$apiScript = Join-Path $projectRoot "scripts\serve_api.py"
$venvPython = Join-Path $projectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $apiScript -PathType Leaf)) {
    throw "API skripta nije pronadjena: $apiScript"
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "package.json") -PathType Leaf)) {
    throw "Frontend package.json nije pronadjen u: $frontendDir"
}

if (Test-Path -LiteralPath $venvPython -PathType Leaf) {
    $python = $venvPython
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCommand) {
        throw "Python nije pronadjen. Kreiraj .venv ili dodaj Python u PATH."
    }
    $python = $pythonCommand.Source
}

$npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
if (-not $npmCommand) {
    throw "npm nije pronadjen. Instaliraj Node.js i pokreni skriptu ponovo."
}

if (-not (Test-Path -LiteralPath (Join-Path $frontendDir "node_modules") -PathType Container)) {
    Write-Host "Instaliram frontend zavisnosti..." -ForegroundColor Yellow
    & $npmCommand.Source install --prefix $frontendDir
    if ($LASTEXITCODE -ne 0) {
        throw "npm install nije uspeo (exit code $LASTEXITCODE)."
    }
}

$processes = @()

try {
    Write-Host "Pokrecem BankScope API na http://127.0.0.1:8000 ..." -ForegroundColor Cyan
    $apiProcess = Start-Process `
        -FilePath $python `
        -ArgumentList "`"$apiScript`"" `
        -WorkingDirectory $projectRoot `
        -PassThru `
        -NoNewWindow
    $processes += $apiProcess

    Write-Host "Cekam da API ucita modele i prijavi da je spreman ..." -ForegroundColor Yellow
    $apiDeadline = (Get-Date).AddSeconds($ApiStartupTimeoutSeconds)
    $apiReady = $false
    while ((Get-Date) -lt $apiDeadline) {
        Start-Sleep -Milliseconds 500
        $apiProcess.Refresh()
        if ($apiProcess.HasExited) {
            throw "BankScope API se ugasio tokom pokretanja (exit code $($apiProcess.ExitCode)). Proveri poruku iznad."
        }

        try {
            $health = Invoke-RestMethod `
                -Uri "http://127.0.0.1:8000/api/health" `
                -Method Get `
                -TimeoutSec 2 `
                -ErrorAction Stop
            if ($health.status -eq "ready") {
                $apiReady = $true
                break
            }
        } catch {
            # API jos uvek ucitava lokalne modele i indekse.
        }
    }

    if (-not $apiReady) {
        throw "BankScope API nije postao spreman za $ApiStartupTimeoutSeconds sekundi."
    }

    Write-Host "Pokrecem frontend na http://localhost:5173 ..." -ForegroundColor Cyan
    $processes += Start-Process `
        -FilePath $npmCommand.Source `
        -ArgumentList @("run", "dev") `
        -WorkingDirectory $frontendDir `
        -PassThru `
        -NoNewWindow

    Write-Host "`nAplikacija je pokrenuta. Pritisni Ctrl+C da ugasis frontend i API." -ForegroundColor Green

    while ($true) {
        Start-Sleep -Milliseconds 500
        foreach ($process in $processes) {
            $process.Refresh()
            if ($process.HasExited) {
                throw "Proces $($process.Id) se neocekivano ugasio (exit code $($process.ExitCode))."
            }
        }
    }
} finally {
    Write-Host "`nGasim aplikaciju..." -ForegroundColor Yellow
    foreach ($process in $processes) {
        if (-not $process.HasExited) {
            taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
        }
    }
}
