"""
preprocess.py — clean dataset header bug, encode, scale
Fixes COEMISSIONS  trailing space reported in GitHub CSV
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "FuelConsumption_Dataset.csv"
MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

def load_and_clean(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    # Fix trailing space bug: COEMISSIONS  -> CO2
    df.columns = [c.strip() for c in df.columns]
    rename_map = {}
    if "COEMISSIONS" in df.columns:
        rename_map["COEMISSIONS"] = "CO2"
    if "FUEL CONSUMPTION" in df.columns:
        rename_map["FUEL CONSUMPTION"] = "FUEL_CONSUMPTION"
    if "ENGINE SIZE" in df.columns:
        rename_map["ENGINE SIZE"] = "ENGINE_SIZE"
    if "VEHICLE CLASS" in df.columns:
        rename_map["VEHICLE CLASS"] = "VEHICLE_CLASS"
    df = df.rename(columns=rename_map)
    # Also handle FuelConsumption.csv variant from kagglehub
    if "CO2" not in df.columns and "COEMISSIONS " in df.columns:
        df = df.rename(columns={"COEMISSIONS ": "CO2"})
    # Derive physics features (same as notebook)
    if "FUEL_CONSUMPTION" in df.columns:
        df["FC"] = df["FUEL_CONSUMPTION"]
        df["FC_CO2"] = df["FC"] * 23.7
        df["CO2_RES"] = df["CO2"] - df["FC_CO2"]
    return df

def build_features(df):
    # Use all columns except CO2 targets, keep original for encoders
    exclude = ["CO2", "CO2_RES", "FC_CO2"]
    feature_cols = [c for c in df.columns if c not in exclude]
    return df[feature_cols], df["CO2_RES"] if "CO2_RES" in df.columns else df["CO2"], feature_cols

def fit_encoders_scaler(X_train_df, cat_cols, num_cols):
    le_dict = {}
    X_enc = X_train_df.copy()
    for c in cat_cols:
        le = LabelEncoder()
        X_enc[c] = le.fit_transform(X_enc[c].astype(str))
        le_dict[c] = le
    scaler = StandardScaler()
    scaler.fit(X_enc[num_cols] if num_cols else X_enc)
    # For consistency, scaler on all numeric after encoding
    # But original notebook scaled all after encoding
    # We'll fit on full encoded matrix
    scaler_full = StandardScaler()
    scaler_full.fit(X_enc.values)
    return le_dict, scaler_full

def transform_with_encoders(df, le_dict, scaler, cat_cols, feature_order):
    X = df[feature_order].copy()
    for c in cat_cols:
        le = le_dict[c]
        # handle unseen categories
        def encode_val(v):
            v = str(v)
            if v in le.classes_:
                return le.transform([v])[0]
            else:
                # unknown -> most frequent class index
                return 0
        X[c] = X[c].apply(encode_val)
    arr = scaler.transform(X.values)
    return arr

if __name__ == "__main__":
    df = load_and_clean()
    print(df.head())
    print(df.columns.tolist())
    print(df.describe())
