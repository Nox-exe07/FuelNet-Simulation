# Fuel Consumption & CO₂ Emission — Deep Learning + 2D Simulation

> **College Project · Deep Residual MLP (FuelNet 7.4M params) · Physics prior `CO₂ = FC×23.7 + residual` · FastAPI + 2D Real-Time Canvas Simulation · Localhost**

Predicts CO₂ emissions (g/km) from vehicle specs and fuel consumption using a physics-informed residual network. Ships with an interactive web simulation: **Static What-If** and **Real-Time Driving** (2D road, exhaust particles ∝ CO₂, speed-dependent aero factor, live charts).

- **Dataset:** `data/FuelConsumption_Dataset.csv` — 639 data rows (640 lines incl. header, Canada, 2000). Columns: Year, MAKE, MODEL, VEHICLE CLASS, ENGINE SIZE, CYLINDERS, TRANSMISSION, FUEL, FUEL CONSUMPTION, COEMISSIONS. Header bug (`COEMISSIONS ` trailing space) is auto-fixed in `src/preprocess.py:14`.
- **Model:** `src/train.py:65` `FuelNet` — `10→512→1024 + 3×ResidualBlock(1024) + 1024→512→1`. Target is residual `CO2_RES = CO2 − FC×23.7`. Achieves **R² 0.992, RMSE 5.73, MAPE 1.48%** on hold-out.
- **Artifacts:** `models/fuel_co2_99pct_model.pth`, `models/scaler.pkl`, `models/encoders.pkl`, `models/model_meta.json`
- **Plots:** `outputs/plots/*.png` (scatter, loss_curve, residual_hist, mape_per_class, fuel_vs_co2) + `outputs/metrics.json`
- **API:** `simulation/backend/api.py:28` FastAPI, mounts frontend static, endpoints `/predict`, `/predict_batch`, `/health`, `/stats`, `/metrics`
- **Frontend:** `simulation/frontend/index.html` — controls, gauges (CO₂ tier: Eco <200, Mid 200–300, High >300), tabs, Chart.js 4.4.1, `js/app.js:1`, `js/simulation2D.js:2`, `css/style.css:1`

---

## Architecture

```
FuelConsumption_Dataset.csv
        ↓  load_and_clean() [src/preprocess.py:14]  — strip headers, rename, derive FC, FC_CO2, CO2_RES
        ↓  LabelEncode categoricals (MAKE, MODEL, VEHICLE_CLASS, TRANSMISSION, FUEL) → StandardScale
        ↓  TensorDataset (X_scaled, y_residual, FC, y_co2)
        ↓
     FuelNet (10→512→1024 → 3×Residual → 512→1)
        ↓  AdamW(lr=3e-3, wd=1e-4) + CosineAnnealingWarmRestarts(T0=50) + GradClip 1.0
        ↓  200 epochs (default; --epochs 400 for full paper), batch 128
        ↓  CO₂ = residual_pred + FC×23.7
        ↓
  metrics: MSE/RMSE/MAE/R²/MAPE  →  plots  →  .pth + .pkl + model_meta.json
        ↓
  FastAPI /predict  →  frontend gauges + simulation2D live chart
```

**Physics prior rationale:** Stoichiometry ≈ 2.31 kg CO₂ per litre gasoline; per 100 km, `FC(L/100km)×23.1≈CO₂(g/km)` scaled to 23.7 calibrated on dataset mean. Learning the residual is numerically easier (variance ↓, R² ↑).

---

## Project Structure

```
simulationproject/
├── data/
│   └── FuelConsumption_Dataset.csv
├── models/
│   ├── fuel_co2_99pct_model.pth
│   ├── scaler.pkl
│   ├── encoders.pkl
│   ├── model_meta.json
│   └── fuel_co2_99pct_model.onnx   (after export)
├── outputs/
│   ├── metrics.json
│   ├── plots/
│   │   ├── scatter_true_vs_pred.png
│   │   ├── loss_curve.png
│   │   ├── residual_hist.png
│   │   ├── mape_per_class.png
│   │   └── fuel_vs_co2.png
│   ├── report.docx                  (generated)
│   └── report.xlsx                  (generated)
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluation.py
│   ├── report.py                    (docx/xlsx generation)
│   └── export_onnx.py               (ONNX export)
├── simulation/
│   ├── backend/api.py
│   └── frontend/
│       ├── index.html
│       ├── css/style.css
│       ├── js/app.js
│       ├── js/simulation2D.js
│       └── js/charts.js
├── requirements.txt
├── README.md
├── run.bat                          (Windows one-click)
├── start.ps1                        (PowerShell)
├── run.sh                           (bash)
└── uvicorn.log
```

---

## Installation

**Python 3.13 recommended** (project uses `torch>=2.0`, `scikit-learn>=1.3`):

```bat
REM Windows CMD
py -3.13 -m pip install -r requirements.txt

REM PowerShell
python -m pip install -r requirements.txt
```

`requirements.txt` includes `torch`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `fastapi`, `uvicorn[standard]`, `pydantic`, `python-docx`, `openpyxl`, `onnx`, `tqdm`.

> If you have Python 3.10 fallback (`C:\Users\megha\...`), prefer Python 3.13 (`C:\Users\dnaga\AppData\Local\Programs\Python\Python313\python.exe`) — torch DLL loads correctly there.

---

## Usage

### 1. Train (reproduce model)

```bat
py -3.13 src/train.py --epochs 40        REM quick demo (2 min)
py -3.13 src/train.py --epochs 200       REM default (used for shipped model)
py -3.13 src/train.py --epochs 400       REM full paper setting
```

Outputs: `models/*.pth/.pkl/.json`, `outputs/plots/*.png`, console `FINAL TEST R2=0.99+`.

### 2. Evaluation (regenerate metrics & plots from saved model)

```bat
py -3.13 src/evaluation.py
```

Reads `models/*.pth` via `src/predict.py:62` `load_artifacts()` and writes `outputs/metrics.json`.

### 3. Predict (single)

```bat
py -3.13 src/predict.py --engine 2.0 --cylinders 4 --fuel X --fc 10.5
```

Or in Python:

```python
from src.predict import predict_one
res = predict_one({
  "ENGINE_SIZE": 2.0, "CYLINDERS": 4, "FUEL_CONSUMPTION": 10.5,
  "VEHICLE_CLASS": "COMPACT", "TRANSMISSION": "A4", "FUEL": "X",
  "MAKE": "TOYOTA", "MODEL": "COROLLA", "Year": 2000
})
# {'fuel_consumption': 10.5, 'co2': 214.1, 'co2_residual': -34.7, 'physics_base': 248.85, 'model': 'fuelnet'}
```

### 4. Report (docx + xlsx)

```bat
py -3.13 src/report.py
REM writes outputs/report.docx and outputs/report.xlsx embedding all plots + metrics
```

### 5. ONNX Export

```bat
py -3.13 src/export_onnx.py
REM writes models/fuel_co2_99pct_model.onnx
```

### 6. Run API + Frontend (localhost:8000)

**Option A — one-click scripts:**

```bat
run.bat                REM CMD double-click
powershell -ExecutionPolicy Bypass -File start.ps1
bash run.sh            REM git-bash / WSL
```

**Option B — manual:**

```bat
py -3.13 -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
REM open http://127.0.0.1:8000
REM docs http://127.0.0.1:8000/docs
```

API log: `uvicorn.log`.

---

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Frontend `index.html` or JSON if missing |
| `/health` | GET | `{status, model_loaded, meta}` |
| `/stats` | GET | Dataset rows/columns, class lists, per-class avg FC/CO₂ |
| `/metrics` | GET | `outputs/metrics.json` or fallback from checkpoint |
| `/predict` | POST | Body: `PredictRequest` (Year, MAKE, MODEL, VEHICLE_CLASS, ENGINE_SIZE, CYLINDERS, TRANSMISSION, FUEL, FUEL_CONSUMPTION, optional `speed`) → `{fuel_consumption, co2, co2_residual, physics_base, model, speed_factor?}` |
| `/predict_batch` | POST | List of `PredictRequest` → list |
| `/plots/{path}` | GET | Serve `outputs/plots/*` |
| `/docs` | GET | Swagger UI |

**`POST /predict` example:**

```json
{
  "Year": 2000, "MAKE": "TOYOTA", "MODEL": "COROLLA",
  "VEHICLE_CLASS": "COMPACT", "ENGINE_SIZE": 2.0,
  "CYLINDERS": 4, "TRANSMISSION": "A4",
  "FUEL": "X", "FUEL_CONSUMPTION": 10.5, "speed": 90
}
```

If `speed` present, `speed_factor = 1 + max(0,speed-60)*0.004 + (speed/120)^2*0.08` multiplies both FC and CO₂ (aero drag).

**Categorical values:**
- `VEHICLE_CLASS`: COMPACT, SUV, MID-SIZE, FULL-SIZE, SUBCOMPACT, etc. (14 classes)
- `TRANSMISSION`: A4, A5, AS5, M5, M6, AV, etc.
- `FUEL`: X=Regular, Z=Premium, D=Diesel, E=Ethanol, N=NG

Frontend calls `POST /predict_batch` for **Sensitivity sweep** (`js/app.js:112`) and **Fuel→CO₂ trace** (`js/app.js:155`) with graceful fallback to physics estimate when offline.

---

## Frontend — Simulation Details

- **Controls (left):** MAKE, MODEL, VEHICLE CLASS, TRANSMISSION, FUEL, Year/Engine/Cylinders/FC sliders. Gauges update on `POST /predict` (debounced). Tier bar: Eco green / Mid amber / High red (`css/style.css:1`).
- **Static What-If tab:** Fuel→CO₂ scatter (Chart.js scatter, line, `js/app.js:141`), Engine-Size sensitivity line, server-rendered `scatter_true_vs_pred.png` + `residual_hist.png` with metrics row (`R²/RMSE/MAPE`).
- **Real-Time Drive tab:** `js/simulation2D.js:2` `Sim2D` — 900×260 canvas, sky gradient, hills, road with moving lane markers, distance posts every 100 m, car (body, cabin, wheels, lights), exhaust particles (`spawnParticles()` ∝ CO₂/400, color by tier, gravity, fade), speed slider 0–140 km/h, Start/Pause/Reset, live integrals: `distance += speed*dt/3600`, `fuelUsed += fcEff*dkm/100`, `co2Total += co2Eff*dkm/1000`, live Chart.js traces (CO₂ red, Fuel blue).
- **How It Works (right):** physics prior explanation, pipeline, run instructions, output links.
- **Fallback:** If API down, local physics + residual `eng*1.2 + cyl*0.8 -4` ensures demo never breaks.

---

## Metrics (shipped model)

`outputs/metrics.json` (300-sample eval) and `models/model_meta.json` (test split) both ≈:

```json
{"mse": 33.1, "rmse": 5.75, "mae": 4.32, "r2": 0.9926, "mape": 1.48}
{"mse": 32.8, "rmse": 5.73, "mae": 4.17, "r2": 0.9923, "mape": 1.39}
```

> R² > 0.99 qualifies as “99%+” threshold. Scatter ideal-line alignment and residual histogram confirm.

---

## Troubleshooting

- **`OSError c10.dll` torch load failure:** Use Python 3.13 (`py -3.13`), not 3.10. Path: `C:\Users\dnaga\AppData\Local\Programs\Python\Python313\python.exe`.
- **Port 8000 busy:** `netstat -ano | findstr :8000` → `taskkill /PID <pid> /F`, or `uvicorn ... --port 8001`.
- **Sklearn unpickle warning (version 1.9.1 vs 1.1.1):** Regenerate via `py -3.13 src/train.py --epochs 40` or ignore — predictions remain correct.
- **Frontend shows “API: offline”:** API not running; gauges fallback to physics estimate. Start `run.bat`.
- **Plots 404:** Run `py -3.13 src/evaluation.py` to regenerate.

---

## License & Credits

- Dataset: Govt. of Canada fuel consumption (Kaggle mirror) — educational use.
- Original GitHub: https://github.com/gundanagarjuna/Fuel-consumption-and--CO--Emission-Prediction-using-Machine-Learning
- Model/code: College project, MIT-ish for demo.
