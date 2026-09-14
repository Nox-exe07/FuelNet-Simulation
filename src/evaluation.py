"""evaluation.py — generate plots from saved model or fallback"""
import pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import sys
sys.path.append(str(pathlib.Path(__file__).parent))
from preprocess import load_and_clean
from predict import predict_one

ROOT = pathlib.Path(__file__).parents[1]
DATA = ROOT / "data" / "FuelConsumption_Dataset.csv"
OUT = ROOT / "outputs" / "plots"
OUT.mkdir(parents=True, exist_ok=True)

def evaluate():
    df = load_and_clean(DATA)
    # sample 200 random rows for scatter
    sdf = df.sample(n=min(300,len(df)), random_state=42)
    y_true, y_pred = [], []
    for _, row in sdf.iterrows():
        inp = row.to_dict()
        # ensure keys map
        inp["ENGINE_SIZE"] = row.get("ENGINE_SIZE", row.get("ENGINE SIZE"))
        inp["VEHICLE_CLASS"] = row.get("VEHICLE_CLASS", row.get("VEHICLE CLASS"))
        inp["FUEL_CONSUMPTION"] = row.get("FUEL_CONSUMPTION", row.get("FUEL CONSUMPTION", row.get("FC")))
        try:
            res = predict_one(inp)
            y_true.append(row["CO2"])
            y_pred.append(res["co2"])
        except Exception as e:
            print(e)
            continue
    y_true = np.array(y_true); y_pred=np.array(y_pred)
    mse = mean_squared_error(y_true,y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true,y_pred)
    r2 = r2_score(y_true,y_pred)
    mape = np.mean(np.abs((y_true-y_pred)/(y_true+1e-8)))*100
    print(f"Eval on {len(y_true)} samples: MSE {mse:.2f} RMSE {rmse:.2f} MAE {mae:.2f} R2 {r2:.4f} MAPE {mape:.2f}%")
    # scatter
    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred, alpha=0.6, s=35, edgecolor="k", linewidth=0.3)
    mn,mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    plt.plot([mn,mx],[mn,mx],"r--",label="Ideal")
    plt.xlabel("True CO2 g/km"); plt.ylabel("Predicted CO2 g/km")
    plt.title(f"CO2 Prediction R2={r2:.4f} RMSE={rmse:.2f}")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "scatter_true_vs_pred.png", dpi=150)
    plt.close()
    # residual histogram
    resid = y_true - y_pred
    plt.figure(figsize=(6,4))
    plt.hist(resid, bins=30, color="#4CAF50", edgecolor="k", alpha=0.8)
    plt.xlabel("Residual (True - Pred)"); plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(OUT / "residual_hist.png", dpi=150)
    plt.close()
    # fuel vs co2
    plt.figure(figsize=(6,4))
    fc = sdf["FUEL_CONSUMPTION"] if "FUEL_CONSUMPTION" in sdf else sdf["FC"]
    plt.scatter(fc, y_true, label="True", alpha=0.5, s=30)
    plt.scatter(fc, y_pred, label="Pred", alpha=0.5, s=30)
    plt.xlabel("Fuel Consumption L/100km"); plt.ylabel("CO2 g/km")
    plt.legend(); plt.grid(alpha=0.3); plt.title("Fuel vs CO2")
    plt.tight_layout(); plt.savefig(OUT / "fuel_vs_co2.png", dpi=150)
    plt.close()
    # per class box
    if "VEHICLE_CLASS" in sdf or "VEHICLE CLASS" in sdf:
        col = "VEHICLE_CLASS" if "VEHICLE_CLASS" in sdf else "VEHICLE CLASS"
        # MAPE per class
        sdf2 = sdf.copy()
        sdf2["pred"]=y_pred
        sdf2["ape"]= np.abs(sdf2["CO2"]-sdf2["pred"])/sdf2["CO2"]*100
        per_class = sdf2.groupby(col)["ape"].mean().sort_values()
        plt.figure(figsize=(8,4))
        per_class.plot(kind="bar", color="#2196F3", edgecolor="k")
        plt.ylabel("MAPE %"); plt.title("MAPE per Vehicle Class")
        plt.xticks(rotation=30, ha="right"); plt.grid(axis="y", alpha=0.3)
        plt.tight_layout(); plt.savefig(OUT / "mape_per_class.png", dpi=150)
        plt.close()
    # save metrics json
    import json, pathlib
    (ROOT / "outputs" / "metrics.json").write_text(json.dumps({"mse":float(mse),"rmse":float(rmse),"mae":float(mae),"r2":float(r2),"mape":float(mape),"n":len(y_true)}, indent=2))
    print(f"Plots saved to {OUT}")

if __name__=="__main__":
    evaluate()
