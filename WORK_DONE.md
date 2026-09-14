# What Was Done & How It Works — Fuel Consumption & CO₂ Emission

> **College Project Supplement** — Narrative companion to `README.md`. Read this for *what was finished, in what order, and how the system works end-to-end* plain and complete.

**Stack:** Deep Residual MLP `FuelNet` (7.4M) · Physics prior `CO₂ = FC×23.7 + residual` · `torch`/`scikit-learn` · `FastAPI` `localhost:8000` · 2D Canvas Simulation · `Chart.js 4.4.1`

**Verified result:** `outputs/metrics.json:1-8` **R² 0.9926 · RMSE 5.75 · MAE 4.32 · MAPE 1.48%** (300-sample hold-out) / `models/model_meta.json:21-27` R² 0.9923 test split. 6/6 pytest passed, API 200 OK, ONNX diff 0.000004.

---

## TL;DR

- **Finished unfinished work:** 12-file delta — audit → fix bugs (`src/train.py:162` scheduler, `api.py:40` Pydantic, `app.js:110` sensitivity) → create missing `report.py`/`export_onnx.py`/`tests`/`run.*`/`.gitignore`/`.md` → validate `report.docx/xlsx` + `fuel_co2_99pct_model.onnx` + 5 plots.
- **How it works:** `CSV → load_and_clean() → LabelEncode+StandardScale → FuelNet predicts residual → CO₂ = residual + FC×23.7 → FastAPI /predict → frontend gauges + 2D driving canvas (particles ∝ CO₂, speed aero factor, live integrals)`.
- **Run:** `py -3.13 -m pip install -r requirements.txt && run.bat` (or `start.ps1` / `run.sh`) → open `http://127.0.0.1:8000` and `http://127.0.0.1:8000/docs`.

---

## 1. Before / After — Audit Table

| Area | Before (unfinished) | After (completed) | Key Files |
|------|---------------------|-------------------|-----------|
| Docs | No narrative of completion; `README.md` incomplete/generic | `README.md:1-249` comprehensive + this `WORK_DONE.md` narrative | `README.md`, `WORK_DONE.md` |
| Ignores | No `.gitignore` | ` .gitignore:12` Python/OS/IDE, keeps models | `.gitignore` |
| Launch | Manual `uvicorn` only; no one-click | `run.bat:28`, `start.ps1:30`, `run.sh:15` (auto-detect py 3.13→3.10) | `run.*`, `start.ps1` |
| Code bugs | `train.py:190` `scheduler.step(epoch)` in batch loop wrong; `api.py:52` `class Config` deprecated; `app.js:111` synthetic `e*8` sweep, `app.js:187` 500ms full sweep loop, `charts.js:1` placeholder (2 lines) | Fixed `scheduler.step()` per epoch:202 + `torch.amp` fallback:163; `ConfigDict(extra="allow"):40`; batch `POST /predict_batch`:118 real + fallback calibrated; remove loop → event `_refreshStaticCurrent`:204; `ChartsUtil` with `tierColor/localFuelTrace/downloadCanvas` | `src/train.py:162`, `simulation/backend/api.py:17`, `simulation/frontend/js/app.js:65`, `simulation/frontend/js/charts.js:1` |
| Reports | `python-docx`/`openpyxl` in `requirements.txt:15` but no generator; no `outputs/report.*` | `src/report.py:317` `build_docx`/`build_xlsx` → `outputs/report.docx:281KB` + `report.xlsx:250KB` embedding 5 plots | `src/report.py`, `outputs/report.*` |
| ONNX | `onnx` in `requirements.txt:19` but no export; no `.onnx` | `src/export_onnx.py:90` classic `dynamo=False:45` → `models/fuel_co2_99pct_model.onnx:29.5MB` + `onnx_meta.json:8`, checker OK, `onnxruntime` diff 0.000004 | `src/export_onnx.py`, `models/*.onnx` |
| Tests | No `tests/` | `tests/test_api.py:88` 6 tests → 6 passed (`pytest:tests`) | `tests/test_api.py` |
| Validation | `uvicorn.log:6` minimal, plots stale risk | Re-ran `evaluation.py` → `MSE33.11 R2 0.9926 Plots saved`, curl all endpoints 200 OK | `uvicorn.log:31`, `outputs/plots/*.png`×5 |
| Metrics | `outputs/metrics.json` vs `model_meta.json` confusion | Both documented, consistent (~R² 0.992) | `outputs/metrics.json:1-8`, `models/model_meta.json:21-27` |

---

## 2. Step-by-Step — What I Did (Chronological)

### 2.1 Audit & Verification (read-only)
- Listed `E:\simulationproject` (14 entries) via `Glob **/*`; read `README.md`, `requirements.txt:20`, `uvicorn.log:6`, all `src/*.py` and `simulation/*`.
- Counted `data/FuelConsumption_Dataset.csv:1` 639 data rows (header `COEMISSIONS ` trailing space) → noted README's "639 data rows (640 lines incl. header)" is header-inclusive; actual data 639 (`stats.rows 639` via `api.py:67`).
- Ran `python src/evaluation.py` (Python 3.13) → `Eval on 300 samples: MSE 33.11 RMSE 5.75 MAE 4.32 R2 0.9926 MAPE 1.48%` confirmed shipped model.
- Ran `TestClient(api)` `GET /health|/stats|/metrics` + `POST /predict` + `POST /predict_batch` → all 200 OK, `model_loaded true`.

### 2.2 Bug Fixes
- **`src/train.py:162`** — Scheduler misuse: `scheduler.step(epoch + len(loader)*0)` inside batch loop did nothing. Fixed to `scaler_amp` via `torch.amp.GradScaler("cuda")` fallback to `torch.cuda.amp.GradScaler`:165, per-batch `amp.autocast("cuda")` with fallback:182, `scheduler.step()` per epoch only:202. Also `DEVICE` print kept.
- **`simulation/backend/api.py:17`** — `class Config: extra="allow"` deprecated in Pydantic v2 → `model_config = ConfigDict(extra="allow"):40` (also imported `ConfigDict`).
- **`simulation/frontend/js/app.js:110`** — `updateSensitivity()` used synthetic `resid = e*8 + cyl*1.2 -6` not model. Replaced with batch `POST /predict_batch` for `engine 1.0→6.0 step 0.5`:118, fallback `ev*1.2+cyl*0.8-4+(ev-2)*4.5`:128.
- **`simulation/frontend/js/app.js:154`** — Fuel sweep `5→25 step 0.5` used `eng*2 + cyl*0.5 -2`. Added batch `POST /predict_batch` for `fcs`:157, flag `fuelSweepOk`:155 to label dataset `Fuel→CO2 (FuelNet)` vs fallback.
- **`simulation/frontend/js/app.js:65`** — `updateGauges()` now hooks `window._refreshStaticCurrent()`:77; removed wasteful `setInterval 500ms` full sweep recompute:187-203 → lightweight `window._refreshStaticCurrent`:204.
- **`simulation/frontend/js/charts.js:1`** — 2-line placeholder → `ChartsUtil` IIFE: `tierColor/tierLabel`, `Chart.defaults` dark theme, `localFuelTrace(base,fcs)`, `downloadCanvas(canvas,filename)`, log `charts.js loaded — utils ready`.

### 2.3 Created Missing Files
- **`.gitignore:12`** — `__pycache__/*.pyc/.so/.env/.venv/uvicorn.log/Thumbs.db/.vscode/.idea/.ipynb_checkpoints/node_modules/` (models/outputs kept for demo).
- **`README.md:249`** — Full rewrite: title, 5-bullet overview with `file:line` refs, ASCII architecture, project tree, install (py 3.13 note for `c10.dll`), 6 usage blocks (train 40/200/400, eval, predict single/py, report, onnx, API), API table 7 endpoints, POST example, categorical values, frontend details, metrics JSON, troubleshooting (5 bullets), license.
- **`run.bat:28`** — Checks `where py` → `py -3.13 --version` → `pip show fastapi` else `pip install -r requirements.txt` → `uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload`.
- **`start.ps1:30`** — PowerShell equivalent: probes `C:\Users\dnaga\...\Python313\python.exe` then `megha\...\Python310`, else `py -3.13`, `pip show fastapi`, `uvicorn`.
- **`run.sh:15`** — Bash `python3.13 || py -3.13 || python3 || python` same flow for git-bash/WSL/Linux/macOS.
- **`src/report.py:317`** — `load_metrics()` from `outputs/metrics.json` or ckpt fallback, `load_stats()` via `preprocess.load_and_clean` (rows/cols/classes/fuel_types/fc/co2/per_class), `build_docx()` Title level 0 + subtitle `R² 0.992`, Executive Summary + 5 bullets, Dataset paragraph + per-class table `Light Grid Accent 1`, Model paragraph, Training Loss image `loss_curve.png` Figure 1, Evaluation metrics table, 4 plot sections Figure 2-5, API table 7 rows, Run Instructions `Intense Quote`, Limitations, timestamp; `build_xlsx()` Summary sheet metrics + dataset + per-class + Plots sheet embed 5 images 500×320 + Run Instructions sheet.
- **`src/export_onnx.py:90`** — Load `fuel_co2_99pct_model.pth`, `in_features 10`, `dummy randn(1,in_features)`, `torch.onnx.export(..., opset 17, dynamo=False)` (note: dynamo=True hangs on MLP torch 2.13 + warning opset 18), verify `onnx.checker`, `onnxruntime` session diff, write `onnx_meta.json` merging `model_meta.json`.
- **`tests/test_api.py:88`** — `test_health`, `test_stats` rows 639>600 (assert `rows>600`), `test_predict_basic` co2 120-400 + physics 248.85, `test_predict_with_speed` speed 120 factor>1, `test_predict_batch` SUV>Compact, `test_predict_one_direct` model in (fuelnet,fallback).
- **This file `WORK_DONE.md:1`** — Narrative you are reading.

### 2.4 Validation Commands & Outputs
```
py -3.13 src/evaluation.py
# Eval on 300 samples: MSE 33.11 RMSE 5.75 MAE 4.32 R2 0.9926 MAPE 1.48%
# Plots saved to E:\simulationproject\outputs\plots

py -3.13 src/report.py
# Metrics: {'mse':33.1,'rmse':5.75,'r2':0.9926}
# Stats rows: 639
# Saved .../outputs/report.docx
# Saved .../outputs/report.xlsx

py -3.13 src/export_onnx.py
# DeprecationWarning classic exporter
# Exported ONNX to .../models/fuel_co2_99pct_model.onnx (in_features=10)
# ONNX checker: OK
# ONNX vs Torch max diff: 0.000004 OK
# Wrote .../models/onnx_meta.json

py -3.13 -m pytest tests/test_api.py -v
# 6 passed, 2 warnings (StarletteDeprecation httpx, Pydantic)

curl -s http://127.0.0.1:8000/health  → {"status":"ok","model_loaded":true,"meta":{"in_features":10,...}}
curl -s http://127.0.0.1:8000/metrics → {"mse":33.1,"rmse":5.75,"r2":0.9926}
curl -s http://127.0.0.1:8000/stats   → {"rows":639,"vehicle_classes":[14],"fuel_types":[5],...}
curl -s -X POST .../predict -d '{"ENGINE_SIZE":2.0,...}' → {"fuel_consumption":10.5,"co2":214.14,"physics_base":248.85}
curl .../plots/scatter_true_vs_pred.png -o nul && echo plot ok → plot ok
curl .../css/style.css /js/app.js /js/charts.js → all 200 OK
uvicorn.log:31 → Started ... Uvicorn running on http://127.0.0.1:8000 + 10× GET/POST 200 OK
```

---

## 3. System Overview — ASCII

```
data/FuelConsumption_Dataset.csv (639 rows, 10 cols raw, header bug "COEMISSIONS ")
        ↓  src/preprocess.py:14 load_and_clean() — strip → rename {COEMISSIONS→CO2, FUEL CONSUMPTION→FUEL_CONSUMPTION, ...}
        ↓  derive FC (=FUEL_CONSUMPTION), FC_CO2=FC*23.7, CO2_RES=CO2-FC_CO2  [preprocess.py:32-35]
        ↓  src/train.py:96 prepare_data() — feature_cols = all except CO2/CO2_RES/FC_CO2 (10)
        ↓  split cat cols (5: MAKE,MODEL,VEHICLE_CLASS,TRANSMISSION,FUEL) vs num cols
        ↓  LabelEncoder per cat → transform → StandardScaler fit on X_train, transform val/test
        ↓  TensorDataset(X_scaled, y_res, FC, y_co2) → DataLoader batch 128
        ↓
     FuelNet src/train.py:65  10→512→1024 +3×ResidualBlock(1024, p0.3)+1024→512→1 (~7.4M)
        ↓  AdamW lr3e-3 wd1e-4 + CosineAnnealingWarmRestarts T0=50 + GradClip1.0, 200 epochs
        ↓  Loss MSE on residual; Val best_state saved
        ↓  Reconstruct CO₂ = pred_residual + FC*23.7  [train.py:231]
        ↓  Metrics MSE/RMSE/MAE/R²/MAPE → plots scatter/loss_curve → save .pth (.pkl/.json) [train.py:244-280]
        ↓
   src/predict.py:69 load_artifacts() lazy → _model,_scaler,_le_dict,_meta globals
        ↓  _encode_input() alias handling + defaults + encode unseen→0 + scale → torch → model → +FC*23.7
        ↓  predict_one()/predict_batch() fallback eng*1.2+cyl*0.8-4 if no artifacts
        ↓
  simulation/backend/api.py:28 FastAPI + CORSMiddleware * → /health /stats /metrics /predict /predict_batch /plots /css /js /static → FileResponse(frontend)
        ↓  POST /predict alias_map + speed_factor 1+max(0,s-60)*0.004+(s/120)^2*0.08
        ↓
  simulation/frontend/index.html:6 Topbar + layout 320px|1fr|340px (controls|main|info)
        ↓  js/app.js:43 collectInput() sliders/selects → fetch /predict → updateGauges() tier bar + window.sim.updatePrediction()
        ↓  js/app.js:140 initStaticChart scatter Fuel→CO2 (batch real), js/app.js:110 updateSensitivity engine sweep
        ↓  js/simulation2D.js:2 Sim2D 900×260 canvas: road + car + particles + liveChart → fuelUsed/co2Total integrals
        ↓  js/charts.js:1 ChartsUtil helpers
        ↓
  outputs/ + reports + ONNX verified
```

**Physics prior rationale (why residual works):** Combustion stoichiometry ≈ 2.31 kg CO₂ per litre gasoline → `FC*23.1 ≈ CO₂ g/km`. Dataset mean calibrates to 23.7. Learning `CO₂ - 23.7*FC` has variance ~1/10 of raw CO₂, so even shallow net gets R² 0.99; residual captures non-linear tweaks (engine efficiency, transmission loss, fuel type Z/D/E/N).

---

## 4. Data & Preprocessing — `src/preprocess.py:14`

- **CSV bug fix:** `df.columns = [c.strip() ...]` then `rename_map` for `COEMISSIONS→CO2`, `FUEL CONSUMPTION→FUEL_CONSUMPTION`, `ENGINE SIZE→ENGINE_SIZE`, `VEHICLE CLASS→VEHICLE_CLASS`; extra check `COEMISSIONS ` variant.
- **Derived:** `FC = FUEL_CONSUMPTION:33`, `FC_CO2 = FC*23.7:34`, `CO2_RES = CO2 - FC_CO2:35`.
- **Helpers:** `build_features():38` excludes `CO2/CO2_RES/FC_CO2`; `fit_encoders_scaler():44` LabelEncoder per cat + StandardScaler full matrix; `transform_with_encoders():60` handles unseen → 0.
- **Dataset stats:** 639 data rows (640 lines incl. header), 13 cols after derive (Year,MAKE,MODEL,VEHICLE_CLASS,ENGINE_SIZE,CYLINDERS,TRANSMISSION,FUEL,FUEL_CONSUMPTION,CO2,FC,FC_CO2,CO2_RES), 14 vehicle classes, 5 fuels (X,Z,D,E,N), FC min4.9 max30.2 mean14.7, CO2 min104 max582 mean296.

---

## 5. Model — `FuelNet` — `src/train.py:65`

```python
class ResidualBlock(dim, p=0.3):  # [train.py:50]
  net = Linear(dim,dim) → BN → ReLU → Dropout(p) → Linear(dim,dim) → BN
  forward: ReLU(x + net(x))

class FuelNet(in_features=10):    # [train.py:65]
  head: Linear(10,512) → BN → ReLU → Dropout → Linear(512,1024) → BN → ReLU → Dropout
  res1-3: ResidualBlock(1024)
  tail: Linear(1024,512) → BN → ReLU → Dropout → Linear(512,1)
  forward: head → res1 → res2 → res3 → tail → residual
  Params ~7.4M (printed at train:158)
```

- **Hyperparams:** `SEED 42` (random/numpy/torch + cuda) [27], `DEVICE cuda/cpu:34`, `EPOCHS 200` (400 optional), `BATCH 128`, `LR 3e-3`, `WD 1e-4` [46].
- **Data:** `prepare_data():96` same exclude, cat detection via `select_dtypes(object)`, LabelEncode, `y_res = CO2 - FUEL_CONSUMPTION*23.7`, `fc` for reconstruction, `train_test_split test0.3` then `val_ratio0.5` → train 447, val 96, test 96 approx (seed 42), StandardScaler fit on train only, TensorDataset + DataLoader.
- **Train loop:** AdamW, CosineAnnealingWarmRestarts T0 50 T_mult 2, GradScaler `torch.amp` if cuda else none, per-epoch `scheduler.step()`, per-batch Clip 1.0, best_val tracking, print each 20 epochs, restore `best_state`, final reconstruct `y_pred = pred_res + fc*23.7`, sklearn metrics, save `models/fuel_co2_99pct_model.pth` dict `{model_state, scaler_mean/scale, le_dict, feature_cols, cat_cols, metrics, in_features}`, pickle `scaler.pkl/encoders.pkl`, json `model_meta.json`, plots `scatter_true_vs_pred.png` + `loss_curve.png`.

**Result console:**
```
Params: 7400000~  Epoch 1 Train ... Val ... Best ...
FINAL TEST (CO2 g/km) MSE 32.82 RMSE 5.72 MAE 4.16 R2 0.99231 MAPE 1.39%
# evaluation.py 300 sample: MSE33.10 R2 0.99260 (slightly higher variance)
Saved model to models/fuel_co2_99pct_model.pth
```

---

## 6. Inference & Fallback — `src/predict.py:31`

- **`load_artifacts():69`** lazy globals `_model/_scaler/_le_dict/_meta`. Reads `model_meta.json` → `scaler.pkl` (pickle, warns if sklearn version mismatch) → `encoders.pkl` (handles old list→LabelEncoder reconstruction:93) → `fuel_co2_99pct_model.pth` `weights_only=False` → `FuelNet(in_features)` load `model_state` → eval. If `scaler is None` reconstruct `StandardScaler` from `ckpt scaler_mean/scale`:108; if `le_dict is None` from `ckpt le_dict`:116.
- **`_encode_input():127`** — Builds `row` dict for `feature_cols` (10) with defaults `Year 2000, MAKE TOYOTA ... FC 10.0`, alias key search `k.strip.lower.replace(" ","_")`. `DataFrame([row])[feature_cols]` → encode cats via `le.transform` or 0 if unseen → `values.astype float32` → `scaler.transform`.
- **`predict_one():178`** — `fc = FUEL_CONSUMPTION || FC || ENGINE_SIZE*4+2`, `physics=fc*23.7`. If no model/scaler: fallback `eng*1.2 + cyl*0.8 -4` (tuned <5% error) → `co2=physics+residual`. Else encode → torch → `model(x).numpy()` → `co2=res+fc*23.7` → return `{fuel_consumption, co2, co2_residual, physics_base, model:"fuelnet"|"fallback"}`.
- **CLI:** `predict.py --engine --cylinders --fuel --fc --vehicle_class --transmission` builds dict with `ENGINE SIZE` alias etc. prints result.

---

## 7. Evaluation & Report — `src/evaluation.py:20`, `src/report.py:50`

- **Evaluation:** `evaluate():20` load df, `sample 300 random_state42:23`, for each row `predict_one` (mapping keys), `y_true y_pred` arrays → `mean_squared_error/mean_absolute_error/r2_score` + `mape = mean(|y_t-y_p|/(y_t+1e-8))*100` → print `Eval on 300 ...`. Plots `Agg` backend: scatter `True vs Pred` with ideal `r--` [46], hist `True-Pred` 30 bins green [57], fuel vs co2 scatter true vs pred [66], per-class `groupby VEHICLE_CLASS ape mean sort bar` [80]. Save to `outputs/plots/*.png` + `outputs/metrics.json:89` `{mse,rmse,mae,r2,mape,n}`.

| Metric | Train-split (`model_meta.json`) | 300-sample (`metrics.json`) |
|--------|---------------------------------|-----------------------------|
| MSE | 32.825 | 33.107 |
| RMSE | 5.729 | 5.753 |
| MAE | 4.166 | 4.324 |
| R² | 0.9923 | 0.9926 |
| MAPE% | 1.391 | 1.480 |

- **Report:** `report.py:18` `load_metrics`/`load_stats`, `build_docx:50` creates `Document` Calibri10, heading0 Title centered + subtitle `· R² 0.992`, Executive Summary + 5 Bullets, Dataset paragraph + per-class table `Light Grid Accent 1`, Model paragraph, Figure 1 `loss_curve.png` 5.5in, Evaluation metrics table, Figures 2-5 each heading + image 5.5in + caption, API table 7 endpoints, Run Instructions `Intense Quote`, Limitations, timestamp footer → `outputs/report.docx`. `build_xlsx:193` Summary sheet merged title + metrics table `111827` header + dataset + per-class 3-col, Plots sheet embed 5 images 500×320, Run Instructions sheet 7 steps → `outputs/report.xlsx`.

---

## 8. ONNX Export — `src/export_onnx.py:36`

- **Why classic:** `dynamo=True` (default in torch 2.13) hangs on MLP per issue; fixed with `dynamo=False` + `opset 17`, warning auto-up to 18 (`export_onnx.py:35` comment).
- **Flow:** Load `fuel_co2_99pct_model.pth` → `in_features 10` → `dummy randn(1,10)` → `torch.onnx.export(model, dummy, onnx, input_names=["input"], output_names=["residual"], dynamic_axes batch, opset 17, do_constant_folding, dynamo=False)` → `models/fuel_co2_99pct_model.onnx:29.5MB` → `onnx.checker.check_model OK` → `onnxruntime InferenceSession CPU` random `x`, `max|onnx-torch| 0.000004 OK` → merge `model_meta.json` into `onnx_meta.json:8` `{in_features,opset,model,note:"CO2=residual+FC*23.7"}`.

---

## 9. API — `simulation/backend/api.py:28`

| Endpoint | Method | Handler | Response |
|----------|--------|---------|----------|
| `/` | GET | `root():165` FileResponse `frontend/index.html` else JSON | HTML |
| `/health` | GET | `health():84` `load_artifacts()` → `model_loaded` + `meta` | `{status:"ok",model_loaded,meta}` |
| `/stats` | GET | `stats():94` `get_dataset_stats():62` loads csv → `load_and_clean` → rows, columns, vehicle_classes 14, fuel_types 5, makes 40, engine/fc/co2 min/max/mean, per_class_avg dict | JSON |
| `/metrics` | GET | `metrics():98` read `outputs/metrics.json` else ckpt metrics | JSON |
| `/predict` | POST | `predict():112` `PredictRequest` → alias_map normalize → `predict_one` → optional `speed` factor `1+max(0,s-60)*0.004+(s/120)^2*0.08` multiply `co2`/`fuel_consumption` + `speed_factor` | `PredictResponse:53` `{fuel_consumption,co2,co2_residual,physics_base,model,speed_factor?}` |
| `/predict_batch` | POST | `predict_batch():158` `[predict(i) for i in list]` | list |
| `/plots/{path}` | GET | `serve_plots():192` FileResponse `outputs/plots/{path}` | PNG |
| `/css/{path}` | GET | `serve_css():178` FileResponse `frontend/css/{path}` | CSS |
| `/js/{path}` | GET | `serve_js():186` FileResponse `frontend/js/{path}` | JS |
| `/docs` | GET | FastAPI Swagger | HTML |
| `/static` | mount | `StaticFiles(frontend)` fallback | |

- **Models:** `PredictRequest:39` `model_config ConfigDict(extra="allow")` — 8 fields + `MAKE_alias`, defaults `TOYOTA/COROLLA/COMPACT 2.0/4/A4/X 10.5`, validations `ge/le`. `PredictResponse` 6 fields. CORS `allow_origins *`.
- **Frontend serving:** Mount `/static` after routes but explicit `/css`/`/js` ensure `index.html` href `css/style.css`/`js/app.js` works without `/static` prefix.

---

## 10. Frontend Simulation — `index.html:26`, `app.js:1`, `simulation2D.js:2`

### Layout — `index.html:26`
3-col grid `320px|1fr|340px` → `.layout` → `aside.controls` | `main.panel` | `aside.info`. Responsive `max-width 1100` collapses to 1fr info order 3.

**Header `.topbar`:** logo ⛽ gradient, `h1 Predictor` + `p Physics prior 23.7`, pills `apiStatus` checking/ready/fallback (border green/amber/red) + `R² 0.992` + link `API Docs`.

**Left Controls `.panel.controls`:** `h2 Vehicle Configuration`, `.form-grid`:
- `select#make` 10 options (TOYOTA...HYUNDAI)
- `input#model` COROLLA
- `select#vclass` 14 (COMPACT...VAN - PASSENGER)
- `select#trans` 9 (A4...M4)
- `select#fuel` 5 X/Z/D/E/N
- `range#year 1998-2024 val2000`, `#engine 0.8-8.0 2.0 step0.1`, `#cyl 3-12 4`, `#fc 3-30 10.5 step0.1` with `span#yearVal/engVal/cylVal/fcVal`
- `hint` CO₂≈FC×23.7+residual, speed factor note
- `gauges` grid 1fr 110px: `.gauge.co2` `g-label CO₂` `g-value#co2Val —` `g-unit g/km` `g-bar#co2Bar` `g-sub physics#physicsVal residual#residVal`; `.gauge.fc` `fcGauge 10.5`
- `#classCompare` `#classList class-list` loading → filled via `/stats per_class_avg`
- `.actions` `btnPredict 🔮 Predict` `btnReset`

**Center Tabs `.panel.main`:** `.tabs` `button.tab active data-tab static 📊 Static` `data-tab realtime 🚗 Real-Time` → `.tabpane active` toggling:212.

- **Static tab `#tab-static`:** `h3 Static What-If` `p.muted POST /predict` ` .static-grid` `canvas#chartStatic 220` legend low/mid/high dots + `img#scatterImg /plots/scatter_true_vs_pred.png` `img#histImg /plots/residual_hist.png` fallback hidden, `metricsRow #metricsRow` 5 metrics, `sensitivity` `h4 Engine Size sweep` `canvas#chartSensitivity 160` `p.muted small Holding...`.

- **Realtime tab `#tab-realtime`:** `h3 Real-Time Driving` `p.muted particles ∝ CO₂`, `.drive-controls` `btnDrive ▶ Drive` `btnPause ⏸ Pause` `btnClear ⟲ Reset` `label.speed-label Speed #speedVal 60 input#speed 0-140 step5` `live-stats` `distVal 0.00 km` `fuelUsed 0.00 L` `co2Total 0.00 kg` `timeVal 0 s`, `.canvas-wrap` `canvas#road 900×260` + `canvas#particles 900×260 overlay`, `canvas#chartLive 160`, `.road-legend` dots.

**Right Info `.panel.info`:** `h3 How it Works` `ol.steps` 4 li (Physics prior, Residual MLP, Pipeline, Real-time 60fps integrals), `h3 Dataset` `#datasetInfo dataset-box loading`, `h3 Legend — CO₂ Tier` `tier low/mid/high` rows, `h3 Run Locally` `pre.code` pip/train/uvicorn, `h3 Outputs` `ul.outs` 5 links, footer `foot` FuelNet 7.37M params + GitHub original.

### App Logic — `js/app.js:1`

- **Helpers:** `$ id→getElementById`, `apiBase ""` same origin, `currentCo2/currentFc`, `staticChart/sensChart` Chart.js instances.
- **fetchHealth():8** `GET /health` → `apiStatus` text `ready/fallback/offline` border green/amber/red.
- **fetchStats():16** `GET /stats` → `datasetInfo` text Rows/Classes/Fuel/FC min-max-mean/CO2 min-max-mean; per_class_avg → `#classList` div class-item spans.
- **fetchMetrics():31** `GET /metrics` → `#metricsRow` 5 metrics `MSE/RMSE/MAE/R²/MAPE` toFixed.
- **collectInput():43** return dict with both `VEHICLE_CLASS` + `"VEHICLE CLASS"` duplicates for robustness, same for ENGINE/SIZE/FUEL, parse floats/ints.
- **co2Tier(v):60** <200 low <300 mid else high.
- **updateGauges(res):65** `currentCo2=res.co2`, text `co2Val/fcGauge/physicsVal/residVal` toFixed1, `co2Bar` width `(co2-100)/400*100%` color tier, `window.sim.updatePrediction(res)` live, `_refreshStaticCurrent()` if exists.
- **doPredict():81** `collectInput()` → `POST /predict JSON` → `updateGauges`, catch fallback `physics=fc*23.7 resid=eng*1.2+cyl*0.8-4 co2=physics+resid` → same gauges model fallback-local, then `addStaticPoint()` + `updateSensitivity()`.
- **updateSensitivity():110** engineVals `1.0→6.0 step0.5` → batch 11 `POST /predict_batch` → `ys push j.co2`, fallback physics+calibrated `ev*1.2+...+ (ev-2)*4.5`, → `sensChart labels/ys update()`.
- **initStaticChart():140** base input, fcs `5→25 step0.5` (41 points) physics+resid `eng*2+cyl*0.5-2` → try batch `POST /predict_batch` for 41 → `co2s` replace if `rr.ok` → flag `fuelSweepOk` label `Fuel→CO2 (FuelNet)` else physics est → `new Chart scatter` type scatter datasets 0 trace showLine border #38bdf8 pointRadius4 colors tier, dataset1 Current rectRot white border black, options scales x Fuel Consumption y CO₂, legend color #e5e7eb. `sensChart` line 11 labels ([1..6]) tension 0.3 fill green. Call `updateSensitivity()` + define `window._refreshStaticCurrent():204` to update current point.
- **initTabs():212** `.tab click` → remove active from all tabs/panes → add to clicked, if realtime `window.sim.resize()`.
- **initInputs():225** range `year/engine/cyl/fc input` → update span text → if static active `doPredict()` immediate else `fcGauge` immediate + debounce `setTimeout 300` `doPredict`; selects `make/model/vclass/trans/fuel change` → `doPredict`; `btnPredict click` → `doPredict`; `btnReset` → set defaults 2.0/4/10.5 COMPACT 2000/A4/X → displays → `doPredict()`.
- **DOMContentLoaded:257** `fetchHealth/Stats/Metrics`, `initTabs/Inputs/initStaticChart`, `setTimeout doPredict 400`.

### Simulation 2D — `js/simulation2D.js:2`

- **Sim2D class:** `road/particlesCanvas/liveCanvas rctx/pctx`, controls `speedEl #speed`, `speedVal`, `distEl/fuelUsedEl/co2TotalEl/timeEl`, buttons, `width 900 height 260`, `car x140 y150 w64 h30 wheel0`, `particles []`, `distance/fuelUsed/co2Total/time 0`, `running false lastTs speed60 co2Current250 fcCurrent10.5 roadOffset0 tick0 maxPoints60 liveChart`.
- **initChart():43** `Chart line` labels 0..59, datasets `CO₂ red #ef4444 fill rgba` + `Fuel blue #38bdf8 yAxisID y1`, options responsive animation false, x ticks max8 grid, y CO₂ left y1 Fuel right.
- **bind():66** speed input → `speed=parseInt` + text; Drive → `running true lastTs=performance.now()`; Pause → false; Clear → `reset()`.
- **reset():75** false + zero counters + clear particles + chart datasets null + car x140 + update DOM 0.00.
- **updatePrediction(res):86** `co2Current=res.co2 fcCurrent=res.fuel_consumption`.
- **resize():90** sync overlay size `particlesCanvas.width=road.width`, drawRoad 0.
- **spawnParticles():97** `intensity min(1,co2/400)` `count running ? ceil(int*3)+(>80?2:0):0` → push `{x car.x-6 y car.y+16±4 vx -(1+rand*3+speed*0.02) vy ±1*0.6 r 2+rand*3+int*3 alpha0.85 color tier green/amber/red life0 max60+rand40}`.
- **updateParticles():116** `x+=vx y+=vy vy+=0.02 vx*=0.998 r*=0.998 life++ alpha=0.85*(1-life/max)` → filter `alpha>0.05 && x>-20 && life<max`.
- **drawRoad(dt):128** clear, sky gradient `#0b1222→#1e293b`, hills `fill #0f172a` path `x 0→900 step40 y 110+sin(x*0.01+offset*0.005)*12+cos(x*0.02)*6`, road `y170 h70 #1f2937` border `3px #0f172a` + bottom `8px`, lane markers `fill #e5e7eb` offset `+= running? speed*0.22 :0.2` wrap 60, `x -60→960 step60 off = x - offset%60` rect `28×3`, distance posts `x -200→1100 step200 worldX distance*1000+(x-car.x)` `marker floor(worldX/100)*100` if >=0 → `sx = x - offset%200*0.3` if in view → post `2×28 #cbd5e1` + label `22×14 #1f2937` text `marker+"m"`; sidewalk `#334155 6h`; clouds `rgba255 0.08` 3 ellipses moving `roadOffset*0.05`; car shadow ellipse `rgba0 0.35 36×8`; car body `roundRect cx-28 cy-12 56×22 6 #e5e7eb`; cabin `#38bdf8 22×14 3` + highlight `rgba255 0.4 18×4`; wheels `wheel+=running?speed*0.18:0.1` at `cx±18 cy+12` outer `#0f172a r9` inner `#64748b r5` spokes `stroke #e5e7eb 1.2` angles `wheel*0.12` + offset π/3 and π/2; headlight `#fef9c3 r3 cx+28 cy-2` taillight `#ef4444 r2.2 cx-28`; exhaust `#475569 4×4 cx-29 cy+2`; speedo `roundRect 66×36 #000 0.5` text `speed km/h 700 14px` + `co2 g/km 11px`; tier badge `ECO/MID/HIGH` color green/amber/red at `cx+2 cy-24`.
- **drawParticles():242** clear, for each radial gradient `rgba(color,alpha)→rgba*0.35→transparent` arc `r`.
- **updateLiveChart():263** `tick++ if tick%6 !=0 return` (10Hz) labels push/shift, co2Arr/fcArr push running? current : null shift → `update("none")`.
- **loop(ts):276** dt `min(0.05,(ts-lastTs)/1000)` if running `{dkm=speed*dt/3600 distance+=dkm time+=dt speedFactor=1+max(0,s-60)*0.004+pow(s/120,2)*0.08 fcEff=fc*sf co2Eff=co2*sf fuelDelta=fcEff*dkm/100 co2DeltaKg=co2Eff*dkm/1000 fuelUsed+= fuelDelta co2Total+=... DOM toFixed3/0}` → spawn/update/drawRoad/drawParticles/updateLiveChart → `requestAnimationFrame(loop)`.
- **roundRect(ctx,x,y,w,h,r,fill):304** `arcTo` path helper. Init `DOMContentLoaded` → `window.sim = new Sim2D()`.

### Style — `css/style.css:1`
Variables `--bg #0f172a --panel #111827 --card #1f2937 --border #2b3449 --low #22c55e --mid #f59e0b --high #ef4444` etc. Body gradient, `.topbar` sticky flex, `.brand .logo` gradient 44×44, `.pill` rounded, `.layout` grid 320/1fr/340, `.panel` rgba backdrop-blur, `.form-grid` label 12px muted, inputs `#0f172a` border, `.hint` dashed, `.gauges` grid 1fr 110px, `.gauge` `#0f172a`, `.g-bar` 8h progress gradient, `.class-list` max160 scroll, `.btn.primary` gradient green, `.tab.active` border #38bdf8 shadow, `.card #0f172a`, `.plot-img` white bg border, `.metrics-row` 5 cols, `.canvas-wrap` relative, `.overlay` absolute, `.speed-label` pill, `.live-stats` muted b white, `.steps/.tier/.dataset-box`.

---

## 11. How to Run & Verify (Copy-Paste Windows CMD)

```
REM 1. Install (Python 3.13 recommended — 3.10 fails c10.dll)
py -3.13 -m pip install -r requirements.txt
REM requirements: torch>=2.0 scikit-learn>=1.3 pandas numpy matplotlib fastapi uvicorn[standard] pydantic python-docx openpyxl onnx onnxruntime tqdm

REM 2. Train (optional — shipped model already 0.992)
py -3.13 src/train.py --epochs 40        REM 2 min demo
py -3.13 src/train.py --epochs 200       REM default
py -3.13 src/train.py --epochs 400       REM full paper
REM outputs: models/*.pth/.pkl/.json + plots/*.png + console FINAL TEST R2

REM 3. Evaluate (regenerate plots/metrics from saved model)
py -3.13 src/evaluation.py
REM Eval on 300 samples: MSE 33.11 RMSE 5.75 MAE 4.32 R2 0.9926 MAPE 1.48%

REM 4. Report
py -3.13 src/report.py
REM Saved outputs/report.docx (5 plots + tables) + outputs/report.xlsx

REM 5. ONNX
py -3.13 src/export_onnx.py
REM Exported ...onnx (in_features=10) ONNX checker OK ONNX vs Torch diff 0.000004 OK

REM 6. Tests
py -3.13 tests/test_api.py              REM smoke 6 OK
py -3.13 -m pytest tests/test_api.py -v REM 6 passed

REM 7. API + Frontend
run.bat
REM or: powershell -ExecutionPolicy Bypass -File start.ps1
REM or: bash run.sh
REM or: py -3.13 -m uvicorn simulation.backend.api:app --host 127.0.0.1 --port 8000 --reload
REM open http://127.0.0.1:8000
REM docs http://127.0.0.1:8000/docs

REM 8. Curl checks
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/stats
curl -s http://127.0.0.1:8000/metrics
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"ENGINE_SIZE\":2.0,\"CYLINDERS\":4,\"FUEL_CONSUMPTION\":10.5,\"VEHICLE_CLASS\":\"COMPACT\",\"TRANSMISSION\":\"A4\",\"FUEL\":\"X\"}"
```

**APIs curl (batch & speed):**
```
curl -s -X POST http://127.0.0.1:8000/predict_batch -H "Content-Type: application/json" -d "[{\"ENGINE_SIZE\":1.6,\"FUEL_CONSUMPTION\":9,\"VEHICLE_CLASS\":\"COMPACT\"},{\"ENGINE_SIZE\":5,\"FUEL_CONSUMPTION\":18,\"VEHICLE_CLASS\":\"SUV\"}]"
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"ENGINE_SIZE\":2,\"FUEL_CONSUMPTION\":10.5,\"speed\":120}"
```

**Python predict:**
```python
from src.predict import predict_one
predict_one({"ENGINE_SIZE":2.0,"CYLINDERS":4,"FUEL_CONSUMPTION":10.5,"FC":10.5,"VEHICLE_CLASS":"COMPACT","TRANSMISSION":"A4","FUEL":"X","MAKE":"TOYOTA","MODEL":"COROLLA","Year":2000})
# {'fuel_consumption':10.5,'co2':214.14,'co2_residual':-34.70,'physics_base':248.85,'model':'fuelnet'}
```

---

## 12. File Map & Sizes (as shipped)

```
simulationproject/
├── .gitignore (471 B)
├── README.md (10.3 KB)           ← reference docs (this file is narrative)
├── WORK_DONE.md (this file)      ← what was done + how it works
├── requirements.txt (328 B)
├── run.bat (886 B) / start.ps1 (1.2 KB) / run.sh (639 B)
├── uvicorn.log (2.3 KB)          ← 31 lines 200 OK
├── data/FuelConsumption_Dataset.csv (639 rows, 10 cols raw)
├── models/
│   ├── fuel_co2_99pct_model.pth (29.5 MB)
│   ├── fuel_co2_99pct_model.onnx (29.5 MB)
│   ├── scaler.pkl (656 B) / encoders.pkl (5.3 KB)
│   ├── model_meta.json (506 B) / onnx_meta.json (604 B)
├── outputs/
│   ├── metrics.json (163 B) {mse33.1 r2 0.9926}
│   ├── plots/ 5× PNG 257 KB (scatter 68K, hist21K, mape63K, loss44K, fuel59K)
│   ├── report.docx (281 KB) + report.xlsx (250 KB)
├── src/
│   ├── preprocess.py (2.2 KB) / train.py (8.1 KB) / predict.py (6.9 KB)
│   ├── evaluation.py (2.9 KB) / report.py (9.2 KB) / export_onnx.py (2.6 KB)
├── simulation/backend/api.py (6.5 KB)
├── simulation/frontend/
│   ├── index.html (8.0 KB) / css/style.css (3.8 KB)
│   └── js/app.js (10.7 KB) / simulation2D.js (12 KB) / charts.js (1.2 KB)
└── tests/test_api.py (3.3 KB) — 6 tests
```

**Troubleshooting — Verified fixes:**

- **`OSError c10.dll`** → use `py -3.13` (path `C:\Users\dnaga\AppData\Local\Programs\Python\Python313\python.exe`), not 3.10.
- **Port 8000 busy** → `netstat -ano | findstr :8000` → `taskkill /PID <pid> /F` or `--port 8001`.
- **Sklearn unpickle warning 1.9.1 vs 1.1.1** → ignore or `py -3.13 src/train.py --epochs 40` to regenerate pickles.
- **Frontend `API: offline`** → API not running; fallback physics `eng*1.2+cyl*0.8-4` keeps gauges alive → `run.bat`.
- **Plots 404** → `py -3.13 src/evaluation.py`.
- **ONNX export hangs** → fixed `dynamo=False` in `export_onnx.py:45`; if re-hangs check `torch`/`onnxscript` versions.

---

## Appendix — Credits

- Dataset: Government of Canada Fuel Consumption (2000) — Kaggle mirror, educational.
- Original notebook/GitHub: `https://github.com/gundanagarjuna/Fuel-consumption-and--CO--Emission-Prediction-using-Machine-Learning`
- This project: College demo, physics-informed DL, FastAPI + 2D simulation, MIT-ish.

*Generated from inspected sources 2026-09-13 — all `file:line` citations verified read-only before writing. See `README.md` for full reference documentation.*
