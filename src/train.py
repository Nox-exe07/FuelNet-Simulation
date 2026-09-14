"""
train.py — reproduce FuelNet from code.ipynb
Residual MLP: 10->512->1024 + 3xResidualBlock + 1024->512->1
Target: CO2_RES = CO2 - FC*23.7  (physics prior)
 """
import random
import pathlib
import pickle
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# local
import sys
sys.path.append(str(pathlib.Path(__file__).parent))
from preprocess import load_and_clean

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

DATA_PATH = pathlib.Path(__file__).parents[1] / "data" / "FuelConsumption_Dataset.csv"
MODELS_DIR = pathlib.Path(__file__).parents[1] / "models"
OUTPUTS_DIR = pathlib.Path(__file__).parents[1] / "outputs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUTS_DIR / "plots").mkdir(parents=True, exist_ok=True)

# Hyperparams (reduced epochs for quick demo but can set 400)
EPOCHS = 200  # original 400, use 200 for speed; set --epochs 400 for full
BATCH = 128
LR = 3e-3
WD = 1e-4

class ResidualBlock(nn.Module):
    def __init__(self, dim, p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(x + self.net(x))

class FuelNet(nn.Module):
    def __init__(self, in_features=10, p=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.res1 = ResidualBlock(1024, p)
        self.res2 = ResidualBlock(1024, p)
        self.res3 = ResidualBlock(1024, p)
        self.tail = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(512, 1),
        )
    def forward(self, x):
        x = self.head(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.tail(x)
        return x

def prepare_data(csv_path=DATA_PATH, test_size=0.3, val_ratio=0.5):
    df = load_and_clean(csv_path)
    # feature selection same as notebook: all except CO2, CO2_RES, FC_CO2
    exclude = ["CO2", "CO2_RES", "FC_CO2"]
    feature_cols = [c for c in df.columns if c not in exclude]
    cat_cols = df[feature_cols].select_dtypes(include=["object"]).columns.tolist()
    num_cols = df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    print(f"Features: {feature_cols}  cat:{cat_cols} num:{num_cols}")
    # Encode categoricals
    le_dict = {}
    df_enc = df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        le_dict[c] = le
    X = df_enc[feature_cols]
    y_res = df_enc["CO2_RES"].values if "CO2_RES" in df_enc else df_enc["CO2"].values - df_enc["FUEL_CONSUMPTION"].values*23.7
    y_co2 = df_enc["CO2"].values
    fc = df_enc["FC"].values if "FC" in df_enc else df_enc["FUEL_CONSUMPTION"].values
    # Split preserving indices
    X_train, X_temp, y_res_train, y_res_temp, y_co2_train, y_co2_temp, fc_train, fc_temp, idx_train, idx_temp = train_test_split(
        X, y_res, y_co2, fc, df_enc.index, test_size=test_size, random_state=SEED
    )
    X_val, X_test, y_res_val, y_res_test, y_co2_val, y_co2_test, fc_val, fc_test, idx_val, idx_test = train_test_split(
        X_temp, y_res_temp, y_co2_temp, fc_temp, idx_temp, test_size=val_ratio, random_state=SEED
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train.values)
    X_val_s = scaler.transform(X_val.values)
    X_test_s = scaler.transform(X_test.values)
    # Convert to tensors
    def to_tensor(a): return torch.tensor(a, dtype=torch.float32)
    train_ds = TensorDataset(to_tensor(X_train_s), to_tensor(y_res_train).view(-1,1), to_tensor(fc_train).view(-1,1), to_tensor(y_co2_train).view(-1,1))
    val_ds = TensorDataset(to_tensor(X_val_s), to_tensor(y_res_val).view(-1,1), to_tensor(fc_val).view(-1,1), to_tensor(y_co2_val).view(-1,1))
    test_ds = TensorDataset(to_tensor(X_test_s), to_tensor(y_res_test).view(-1,1), to_tensor(fc_test).view(-1,1), to_tensor(y_co2_test).view(-1,1))
    meta = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "le_classes": {k: v.classes_.tolist() for k,v in le_dict.items()},
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "scaler_var": scaler.var_.tolist() if hasattr(scaler,"var_") else None,
        "n_features": len(feature_cols)
    }
    return (DataLoader(train_ds,batch_size=BATCH,shuffle=True),
            DataLoader(val_ds,batch_size=BATCH),
            DataLoader(test_ds,batch_size=BATCH),
            le_dict, scaler, meta, df, feature_cols, cat_cols)

def train():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--batch", type=int, default=BATCH)
    args = ap.parse_args()
    epochs = args.epochs
    train_loader, val_loader, test_loader, le_dict, scaler, meta, df, feature_cols, cat_cols = prepare_data()
    in_features = meta["n_features"]
    model = FuelNet(in_features=in_features).to(DEVICE)
    print(model)
    # count params
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params}")
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2)
    # torch.amp API (new) with fallback
    scaler_amp = None
    if DEVICE.type == "cuda":
        try:
            scaler_amp = torch.amp.GradScaler("cuda")
        except Exception:
            scaler_amp = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_state = None
    train_losses, val_losses = [], []
    for epoch in range(1, epochs+1):
        model.train()
        tr_loss = 0
        for xb, yb, fcb, _ in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            if scaler_amp:
                try:
                    with torch.amp.autocast("cuda"):
                        pred = model(xb)
                        loss = criterion(pred, yb)
                except Exception:
                    with torch.cuda.amp.autocast():
                        pred = model(xb)
                        loss = criterion(pred, yb)
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            tr_loss += loss.item()*xb.size(0)
        # scheduler step per epoch (correct usage)
        scheduler.step()
        tr_loss/=len(train_loader.dataset)
        # val
        model.eval()
        val_loss=0
        with torch.no_grad():
            for xb, yb, _, _ in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss+=loss.item()*xb.size(0)
        val_loss/=len(val_loader.dataset)
        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k:v.cpu() for k,v in model.state_dict().items()}
        if epoch%20==0 or epoch==1:
            print(f"Epoch {epoch:3d} Train {tr_loss:.2f} Val {val_loss:.2f} Best {best_val:.2f}")
    # restore best
    if best_state:
        model.load_state_dict(best_state)
    # final test: reconstruct CO2
    model.eval()
    co2_true, co2_pred = [], []
    with torch.no_grad():
        for xb, _, fcb, y_co2b in test_loader:
            xb = xb.to(DEVICE)
            pred_res = model(xb).cpu().numpy().flatten()
            fc = fcb.numpy().flatten()
            y_true = y_co2b.numpy().flatten()
            y_pred = pred_res + fc*23.7
            co2_true.extend(y_true)
            co2_pred.extend(y_pred)
    co2_true = np.array(co2_true); co2_pred=np.array(co2_pred)
    mse = mean_squared_error(co2_true, co2_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(co2_true, co2_pred)
    r2 = r2_score(co2_true, co2_pred)
    mape = np.mean(np.abs((co2_true-co2_pred)/(co2_true+1e-8)))*100
    print(f"\nFINAL TEST (CO2 g/km)\n MSE: {mse:.3f}\n RMSE: {rmse:.3f}\n MAE: {mae:.3f}\n R2: {r2:.5f} -> {'99%+ ' if r2>0.99 else ''}\n MAPE: {mape:.2f}%")
    # save
    torch.save({
        "model_state": best_state if best_state else model.state_dict(),
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "le_dict": {k: v.classes_.tolist() for k,v in le_dict.items()},
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "metrics": {"mse": float(mse), "rmse": float(rmse), "mae": float(mae), "r2": float(r2), "mape": float(mape), "best_val_mse": float(best_val)},
        "in_features": in_features
    }, MODELS_DIR / "fuel_co2_99pct_model.pth")
    # also pickle for predict.py
    with open(MODELS_DIR / "scaler.pkl","wb") as f: pickle.dump(scaler,f)
    with open(MODELS_DIR / "encoders.pkl","wb") as f: pickle.dump(le_dict,f)
    with open(MODELS_DIR / "model_meta.json","w") as f:
        json.dump({"feature_cols":feature_cols,"cat_cols":cat_cols,"metrics":{"mse":float(mse),"rmse":float(rmse),"mae":float(mae),"r2":float(r2),"mape":float(mape)},"in_features":in_features},f,indent=2)
    # plots
    plt.figure(figsize=(6,5))
    plt.scatter(co2_true, co2_pred, alpha=0.6, s=40, edgecolor="k", linewidth=0.3)
    mn, mx = min(co2_true.min(), co2_pred.min()), max(co2_true.max(), co2_pred.max())
    plt.plot([mn,mx],[mn,mx],"r--",label="Ideal")
    plt.xlabel("True CO2 g/km")
    plt.ylabel("Predicted CO2 g/km")
    plt.title(f"CO2 Prediction R2={r2:.5f}")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "plots" / "scatter_true_vs_pred.png", dpi=150)
    plt.close()
    # loss curves
    plt.figure()
    plt.plot(train_losses,label="train")
    plt.plot(val_losses,label="val")
    plt.xlabel("Epoch"); plt.ylabel("MSE (residual)")
    plt.legend(); plt.grid(alpha=0.3)
    plt.title("Training Loss")
    plt.savefig(OUTPUTS_DIR / "plots" / "loss_curve.png", dpi=150)
    plt.close()
    print(f"Saved model to {MODELS_DIR / 'fuel_co2_99pct_model.pth'}")
    return mse, r2

if __name__ == "__main__":
    train()
