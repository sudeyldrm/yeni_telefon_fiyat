from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
import os
import pandas as pd
from typing import List, Dict


app = FastAPI(title="Phone Price Range API")

# CORS (Flutter Web için şart)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # prod'da domain yazarsın
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH  = os.path.join(BASE_DIR, "..", "model", "logistic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "..", "model", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# Modelin beklediği feature sırası (CSV’deki sırayla!)
FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi"
]

class PredictRequest(BaseModel):
    battery_power: int
    blue: int
    clock_speed: float
    dual_sim: int
    fc: int
    four_g: int
    int_memory: int
    m_dep: float
    mobile_wt: int
    n_cores: int
    pc: int
    px_height: int
    px_width: int
    ram: int
    sc_h: int
    sc_w: int
    talk_time: int
    three_g: int
    touch_screen: int
    wifi: int

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    x = np.array([[getattr(req, f) for f in FEATURES]], dtype=float)
    x_scaled = scaler.transform(x)
    pred = int(model.predict(x_scaled)[0])
    proba = model.predict_proba(x_scaled)[0].tolist()

    return {
        "price_range": pred,
        "probabilities": proba
    }
# ===============================
# PHONE LIST (UI DATA)
# ===============================

def load_phones_csv() -> List[Dict]:
    df = pd.read_csv(os.path.join(BASE_DIR, "..", "data", "telefonlar.csv"))
    df = df.where(pd.notnull(df), None)

    phones = []
    for idx, row in df.iterrows():
        phones.append({
            "id": idx,
            "brand": row["brand"],
            "model": row["model"],
            "price_tl": int(row["price_tl"]) if row["price_tl"] else None,
            "ram_gb": int(row["ram_gb"]) if row["ram_gb"] else None,
            "storage_gb": int(row["storage_gb"]) if row["storage_gb"] else None,
            "battery_mah": int(row["battery_mah"]) if row["battery_mah"] else None,
            "camera_mp": int(row["camera_mp"]) if row["camera_mp"] else None,
            "screen_inch": float(row["screen_inch"]) if row["screen_inch"] else None,
            "is_5g": int(row["is_5g"]) if row["is_5g"] else None,
            "os": row["os"],
        })
    return phones


@app.get("/phones")
def get_phones():
    return load_phones_csv()


@app.get("/phones/{phone_id}")
def get_phone(phone_id: int):
    phones = load_phones_csv()
    if phone_id < 0 or phone_id >= len(phones):
        return {"error": "Phone not found"}
    return phones[phone_id]
