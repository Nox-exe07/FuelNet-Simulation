# start.ps1 — PowerShell launcher for Fuel & CO2 Simulation
Write-Host "=== Fuel & CO2 Simulation — http://127.0.0.1:8000 ===" -ForegroundColor Green

$py313 = "C:\Users\dnaga\AppData\Local\Programs\Python\Python313\python.exe"
$py310 = "C:\Users\megha\AppData\Local\Programs\Python\Python310\python.exe"

$py = $null
if (Test-Path $py313) { $py = $py313; Write-Host "Using Python 3.13: $py" }
elseif (Test-Path $py310) { $py = $py310; Write-Host "Using Python 3.10: $py" }
elseif (Get-Command py -ErrorAction SilentlyContinue) {
    try { py -3.13 --version | Out-Null; $py = "py -3.13" } catch { $py = "python" }
}
elseif (Get-Command python -ErrorAction SilentlyContinue) { $py = "python" }
else { Write-Error "No Python found. Install Python 3.13."; exit 1 }

# install deps if needed
if ($py -like "*.exe") {
    & $py -m pip show fastapi | Out-Null
    if ($LASTEXITCODE -ne 0) { & $py -m pip install -r requirements.txt }
    & $py -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
} else {
    Invoke-Expression "$py -m pip show fastapi | Out-Null"
    Invoke-Expression "$py -m pip install -r requirements.txt"
    Invoke-Expression "$py -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload"
}

