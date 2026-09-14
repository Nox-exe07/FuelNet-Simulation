@echo off
REM run.bat — one-click launch for Fuel & CO2 Simulation
REM Tries Python 3.13 then 3.10 then python

echo === Fuel Consumption & CO2 — Starting API on http://127.0.0.1:8000 ===
echo.

where py >nul 2>&1
if %errorlevel%==0 (
  py -3.13 --version >nul 2>&1
  if %errorlevel%==0 (
    echo Using py -3.13 ...
    py -3.13 -m pip show fastapi >nul 2>&1 || py -3.13 -m pip install -r requirements.txt
    py -3.13 -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
    goto :eof
  )
)

python --version >nul 2>&1
if %errorlevel%==0 (
  echo Using python ...
  python -m pip show fastapi >nul 2>&1 || python -m pip install -r requirements.txt
  python -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
  goto :eof
)

echo ERROR: No Python found. Install Python 3.13 and run: py -3.13 -m pip install -r requirements.txt
pause

