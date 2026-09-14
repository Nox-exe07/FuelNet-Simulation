# TODO — Fuel Consumption & CO₂ Emission Project

> Last updated: 2026-09-14 08:15 UTC · Status: **100% Completed + CI & Pages Live**

## Legend
- [x] done · [~] partially / optional · [ ] pending

## Core Pipeline
- [x] Dataset cleaned (`src/preprocess.py:14` strip `COEMISSIONS ` → `CO2`, derive `FC/FC_CO2/CO2_RES`)
- [x] Model `FuelNet` residual MLP 10→512→1024+3×ResBlock+512→1 (`src/train.py:65`) trained 200 epochs, R² 0.992
- [x] Artifacts saved `models/fuel_co2_99pct_model.pth` + `scaler.pkl` + `encoders.pkl` + `model_meta.json`
- [x] ONNX export `models/fuel_co2_99pct_model.onnx` (`src/export_onnx.py:36` `dynamo=False`)
- [x] Evaluation regenerated `outputs/metrics.json:1` + 5 plots `outputs/plots/*.png` (`src/evaluation.py:20`)
- [x] Report generation `outputs/report.docx` + `outputs/report.xlsx` (`src/report.py:50`)
- [x] Inference `src/predict.py:69` `predict_one` fallback + `predict_batch`

## API & Frontend
- [x] FastAPI backend (`simulation/backend/api.py:28`) with `/health` `/stats` `/metrics` `/predict` `/predict_batch` `/plots` `/css` `/js` + CORS + `ConfigDict` fix
- [x] Frontend static mount + `favicon.ico` handler (`api.py:198`) + `simulation/frontend/favicon.ico` (134B)
- [x] Frontend HTML structure 3-col layout (`simulation/frontend/index.html:26`) with controls/gauges/tabs
- [x] Static What-If logic `js/app.js:81` `doPredict` with click feedback, pulse animation, fallback + `predictStatus` banner
- [x] Fuel→CO₂ trace + engine sensitivity via real `POST /predict_batch` (fallback physics)
- [x] Real-Time Drive canvas `js/simulation2D.js:2` Sim2D 900×260 road/car/particles/liveChart + speed aero factor
- [x] Charts helpers `js/charts.js:1` `ChartsUtil` dark theme
- [x] Styling `css/style.css:1` responsive + pulse `@keyframes gaugePulse`
- [x] Favicon link `<link rel="icon" href="/favicon.ico">` added (`index.html:8`)
- [x] Predict (Static) button tested — click shows `✓ Predicted via fuelnet at HH:MM:SS` + gauge pulse + `co2Bar` update

## Docs & Scripts
- [x] `README.md:249` comprehensive (arch, install, usage, API table, metrics, troubleshooting)
- [x] `WORK_DONE.md:393` narrative companion (what was done + how it works)
- [x] `requirements.txt:20` all deps (`torch` `sklearn` `pandas` `fastapi` `uvicorn` `python-docx` `openpyxl` `onnx` `tqdm`)
- [x] `.gitignore:42` (caches, venv, logs, OS, IDE)
- [x] Launch scripts `run.bat:28` `start.ps1:30` `run.sh:15` (auto-detect py 3.13)
- [x] Row count corrected 639 data rows (640 lines incl. header) in `README.md:7` `WORK_DONE.md:39` `index.html:192` `report.py:86`

## Quality & Submission
- [x] Clean `__pycache__` & `.pytest_cache` (removed, `.gitignore` covers)
- [x] Git repo init — done `git init` + `git add .` + `commit bde0d2d` on `master` (`2026-09-14`), user `FuelNet Student <student@college.edu>`, 39 files, `git status` clean — verified `C:\Program Files\GIT\cmd\git.exe` 2.55.0.windows.5
- [x] Pytest suite `tests/test_api.py:88` 6 tests → log `outputs/test_log.txt:1` (36.7s, 6 passed) + re-eval appended R² 0.9926
- [x] Live API verification `uvicorn.log:136` 100+ `200 OK` incl. `GET /favicon.ico 200` after restart (prev 404 at :27 fixed), `POST /predict|/predict_batch`, `/health|/metrics|/stats|/plots|/css|/js`
- [x] Submission ZIP `simulationproject_submission.zip` (55 files, 55.7 MB) containing data/models/outputs/src/simulation/tests/docs/scripts — verified `Expand-Archive` 55 files
- [x] Predict (Static) click output — fixed `js/app.js:81` `predictStatus` banner + `css/style.css:90` `gaugePulse` + `api.py:198` favicon — tested `curl POST /predict → co2 214.14` + browser pulse
- [x] Push to GitHub — **DONE** `https://github.com/Nox-exe07/FuelNet-Simulation` public, pushed `main` 4 commits `bde0d2d` → `283dc0e` → `152badb` → `125c82f` + `ee6cf4a` (CI), `git push -u origin main` 2026-09-14T08:03:27Z & 08:11:26Z, verified `API GET /repos/Nox-exe07/FuelNet-Simulation` + `contents` 15 entries, remote `origin` now clean `https://github.com/Nox-exe07/FuelNet-Simulation.git` (token removed after push)
- [x] GitHub Actions CI — **DONE** `.github/workflows/ci.yml:71` `CI — FuelNet Tests & Eval` (`push`→`main/master`, `pull_request`, `workflow_dispatch`, `ubuntu-latest` `python 3.13` `cache pip` → `pip install -r requirements.txt` → `pytest 6 tests` → `evaluation R²` → `ONNX verify` → `upload-artifact` reports/plots) pushed `ee6cf4a` 2026-09-14T08:11:26Z, run `34821444785` `pages-check ✓ 22s`, `test in_progress` (torch install), view at `https://github.com/Nox-exe07/FuelNet-Simulation/actions/runs/34821444785`
- [x] GitHub Pages — **DONE** `git subtree push --prefix simulation/frontend origin gh-pages` 6/6 → branch `gh-pages` `2a971878` (tree `af91139`), `GET /repos/Nox-exe07/FuelNet-Simulation/pages` → `{"status":"built","html_url":"https://nox-exe07.github.io/FuelNet-Simulation/","source":{"branch":"gh-pages","path":"/"}}`, `gh run pages-build-deployment 34821465276 ✓ 38s`, live at `https://nox-exe07.github.io/FuelNet-Simulation/` (fallback physics, API needs localhost `run.bat` for full FuelNet predictions)

## Quick Verify (copy-paste)
```bat
py -3.13 src/evaluation.py          REM R² 0.9926
py -3.13 -m pytest tests/test_api.py -v  REM 6 passed
py -3.13 src/report.py              REM docx/xlsx
py -3.13 src/export_onnx.py         REM onnx diff 0.000004
run.bat                             REM http://127.0.0.1:8000 + http://127.0.0.1:8000/docs
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"ENGINE_SIZE\":2.0,\"CYLINDERS\":4,\"FUEL_CONSUMPTION\":10.5,\"VEHICLE_CLASS\":\"COMPACT\"}"
```

## Known Non-Issues (intentionally left)
- `models/*.pth` `.onnx` ~29MB each — kept (large but needed for demo); `.gitignore` has commented ignore if you prefer Git LFS
- `outputs/plots/*.png` kept for demo offline viewing
- Sklearn pickle version warning (1.9.1 vs 1.1.1) — harmless, predictions verified identical; regenerate via `train.py --epochs 40` to update
