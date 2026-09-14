"""
FastAPI backend for Fuel Consumption & CO2 Simulation
Serves localhost:8000, mounts frontend static files
"""
import pathlib
import sys
import json
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

try:
    from src.predict import predict_one, load_artifacts
except ImportError:
    sys.path.append(str(ROOT / "src"))
    from predict import predict_one, load_artifacts

from src.preprocess import load_and_clean

app = FastAPI(title="Fuel & CO2 Simulation API", version="1.0", description="Predict CO2 from engine/fuel data + simulation support")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Models ----
class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    Year: int = Field(default=2000, ge=1990, le=2030)
    MAKE: str = Field(default="TOYOTA")
    MODEL: str = Field(default="COROLLA")
    VEHICLE_CLASS: str = Field(default="COMPACT", description="e.g., COMPACT, SUV, MID-SIZE")
    ENGINE_SIZE: float = Field(default=2.0, ge=0.5, le=10.0)
    CYLINDERS: int = Field(default=4, ge=2, le=16)
    TRANSMISSION: str = Field(default="A4")
    FUEL: str = Field(default="X", description="X=Regular, Z=Premium, D=Diesel, E=Ethanol, N=NG")
    FUEL_CONSUMPTION: float = Field(default=10.5, ge=3.0, le=30.0, description="L/100km")
    # aliases for convenience
    MAKE_alias: Optional[str] = None

class PredictResponse(BaseModel):
    fuel_consumption: float
    co2: float
    co2_residual: float
    physics_base: float
    model: str
    speed_factor: Optional[float] = None

# ---- Helpers ----
def get_dataset_stats():
    csv = ROOT / "data" / "FuelConsumption_Dataset.csv"
    if not csv.exists():
        return {}
    df = load_and_clean(csv)
    stats = {
        "rows": len(df),
        "columns": df.columns.tolist(),
        "vehicle_classes": sorted(df["VEHICLE_CLASS"].astype(str).unique().tolist()) if "VEHICLE_CLASS" in df else [],
        "fuel_types": sorted(df["FUEL"].astype(str).unique().tolist()) if "FUEL" in df else [],
        "makes": sorted(df["MAKE"].astype(str).unique().tolist())[:40],
        "engine_size": {"min": float(df["ENGINE_SIZE"].min()), "max": float(df["ENGINE_SIZE"].max()), "mean": float(df["ENGINE_SIZE"].mean())},
        "fuel_consumption": {"min": float(df["FUEL_CONSUMPTION"].min()), "max": float(df["FUEL_CONSUMPTION"].max()), "mean": float(df["FUEL_CONSUMPTION"].mean())},
        "co2": {"min": float(df["CO2"].min()), "max": float(df["CO2"].max()), "mean": float(df["CO2"].mean())},
    }
    # per class avg
    if "VEHICLE_CLASS" in df:
        per_class = df.groupby("VEHICLE_CLASS").agg({"FUEL_CONSUMPTION":"mean","CO2":"mean"}).round(2).to_dict(orient="index")
        stats["per_class_avg"] = per_class
    return stats

@app.get("/health")
def health():
    try:
        m, s, le, meta = load_artifacts()
        model_loaded = m is not None
    except Exception as e:
        model_loaded = False
        meta = None
    return {"status":"ok","model_loaded": model_loaded, "meta": meta}

@app.get("/stats")
def stats():
    return get_dataset_stats()

@app.get("/metrics")
def metrics():
    p = ROOT / "outputs" / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    # fallback from model checkpoint
    import pathlib as pl
    ckpt = ROOT / "models" / "fuel_co2_99pct_model.pth"
    if ckpt.exists():
        import torch
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        return ck.get("metrics", {})
    return {"error":"metrics not found, run train.py"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    data = req.model_dump()
    # normalize keys to match training feature names
    # training uses VEHICLE_CLASS, ENGINE_SIZE, etc.
    alias_map = {
        "VEHICLE_CLASS": "VEHICLE_CLASS",
        "VEHICLE CLASS": "VEHICLE_CLASS",
        "ENGINE_SIZE": "ENGINE_SIZE",
        "ENGINE SIZE": "ENGINE_SIZE",
        "FUEL_CONSUMPTION": "FUEL_CONSUMPTION",
        "FUEL CONSUMPTION": "FUEL_CONSUMPTION",
        "FC": "FUEL_CONSUMPTION",
    }
    inp = {}
    for k,v in data.items():
        if k in alias_map:
            inp[alias_map[k]] = v
        elif k.upper().replace(" ","_") in alias_map:
            inp[alias_map[k.upper().replace(" ","_")]] = v
        else:
            inp[k] = v
    # also ensure VEHICLE_CLASS mapping
    if "VEHICLE_CLASS" not in inp and "VEHICLE CLASS" in data:
        inp["VEHICLE_CLASS"] = data["VEHICLE CLASS"]
    # Also add stripped variants for predict_one robust handling
    inp["VEHICLE CLASS"] = inp.get("VEHICLE_CLASS", "COMPACT")
    inp["ENGINE SIZE"] = inp.get("ENGINE_SIZE", 2.0)
    inp["FUEL CONSUMPTION"] = inp.get("FUEL_CONSUMPTION", 10.0)
    inp["FC"] = inp.get("FUEL_CONSUMPTION", 10.0)

    res = predict_one(inp)
    # speed-adjusted factor for simulation (if caller sends speed)
    speed = data.get("speed", None)
    if speed is not None:
        try:
            speed = float(speed)
            # simple physics: FC increases with speed beyond 60km/h due to aero
            factor = 1.0 + max(0, (speed-60))*0.004 + (speed/120)**2*0.08
            res["co2"] = res["co2"] * factor
            res["fuel_consumption"] = res["fuel_consumption"] * factor
            res["speed_factor"] = factor
        except:
            pass
    return res

@app.post("/predict_batch")
def predict_batch(items: list[PredictRequest]):
    return [predict(i) for i in items]

# ---- Frontend static ----
FRONTEND_DIR = pathlib.Path(__file__).parents[1] / "frontend"

@app.get("/")
def root():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message":"Frontend not found, API is running. See /docs"}

# mount static after routes, for /css, /js etc handled via explicit FileResponse fallback
# Use StaticFiles for everything else
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# Also serve frontend files directly for /css/* /js/*
@app.get("/css/{path:path}")
def serve_css(path: str):
    f = FRONTEND_DIR / "css" / path
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error":"not found"})

@app.get("/js/{path:path}")
def serve_js(path: str):
    f = FRONTEND_DIR / "js" / path
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error":"not found"})

@app.get("/plots/{path:path}")
def serve_plots(path: str):
    f = ROOT / "outputs" / "plots" / path
    if f.exists():
        return FileResponse(str(f))
    return JSONResponse(status_code=404, content={"error":"not found"})

@app.get("/favicon.ico")
def favicon():
    f = FRONTEND_DIR / "favicon.ico"
    if f.exists():
        return FileResponse(str(f), media_type="image/x-icon")
    # fallback: 1x1 transparent
    return JSONResponse(status_code=204, content=None)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
