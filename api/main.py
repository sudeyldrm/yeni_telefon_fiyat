from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
import joblib
import os
import pandas as pd
from typing import List, Dict

app = FastAPI(title="Phone Price Range API")

# ✅ CORS (Flutter Web için)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ✅ Preflight OPTIONS isteğini garanti karşıla (Flutter Web fetch için çok işe yarar)
@app.options("/{path:path}")
def options_handler(path: str):
    return Response(status_code=200)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ===============================
# MODEL (CSV tabanlı YENİ model)
# ===============================
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "price_range_model.pkl")
model = joblib.load(MODEL_PATH)

class PredictRequest(BaseModel):
    ram_gb: int
    storage_gb: int
    battery_mah: int
    camera_mp: int
    screen_inch: float
    is_5g: int
    os: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    X = [{
        "ram_gb": req.ram_gb,
        "storage_gb": req.storage_gb,
        "battery_mah": req.battery_mah,
        "camera_mp": req.camera_mp,
        "screen_inch": req.screen_inch,
        "is_5g": req.is_5g,
        "os": req.os,
    }]

    pred = int(model.predict(X)[0])
    proba = model.predict_proba(X)[0].tolist()

    return {"price_range": pred, "probabilities": proba}

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
