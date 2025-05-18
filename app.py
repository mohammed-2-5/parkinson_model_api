# app.py  ────────────────────────────────────────────────────────────────
"""
FastAPI service
• Parkinson – numeric & drawing
• Wilson’s disease – numeric (uses saved StandardScaler)
"""

# ── ENV / logging tweaks (optional) ─────────────────────────────────────
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"   # CPU only
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # hide TF info banners
# os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" # disable oneDNN if you prefer

# ── Imports ────────────────────────────────────────────────────────────
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
import joblib
import io

from PIL import Image
from google.cloud import storage
from tensorflow.keras.models import load_model

# ── Google Cloud Storage settings ──────────────────────────────────────
BUCKET_NAME = "parkinson_models"
FILES = {
    # Parkinson artefacts
    "model.pkl":          "model.pkl",
    "scaler.pkl":         "scaler.pkl",
    "drawings.keras":     "drawings.keras",
    # Wilson artefacts
    "wilson_model.pkl":   "wilson_model.pkl",
    "wilson_scaler.pkl":  "wilson_scaler.pkl",
}

def download_models_from_gcs() -> None:
    """Download all artefacts once at startup."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    os.makedirs("models", exist_ok=True)

    for filename, blob_name in FILES.items():
        local_path = f"models/{filename}"
        if not os.path.exists(local_path):
            print(f"🔽  Downloading {filename} …")
            bucket.blob(blob_name).download_to_filename(local_path)

# ── FastAPI lifespan: load artefacts into memory ───────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    global pd_model, pd_scaler, cnn, wilson_model, wilson_scaler
    download_models_from_gcs()

    # Parkinson
    pd_model   = joblib.load("models/model.pkl")
    pd_scaler  = joblib.load("models/scaler.pkl")
    cnn        = load_model("models/drawings.keras")

    # Wilson
    wilson_model  = joblib.load("models/wilson_model.pkl")
    wilson_scaler = joblib.load("models/wilson_scaler.pkl")  #  ← now present
    yield

app = FastAPI(lifespan=lifespan)

# ── tiny redirect so visiting "/" opens Swagger UI ─────────────────────
@app.get("/", include_in_schema=False)
def _root():
    return RedirectResponse(url="/docs", status_code=308)

# ═══════════════════════════════════════════════════════════════════════
#                     P  A  R  K  I  N  S  O  N   E N D P O I N T S
# ═══════════════════════════════════════════════════════════════════════

class PDInput(BaseModel):
    UPDRS: float
    FunctionalAssessment: float
    MoCA: float
    Tremor: int
    Rigidity: int
    Bradykinesia: int
    Age: int
    AlcoholConsumption: float
    BMI: float
    SleepQuality: float
    DietQuality: float
    CholesterolTriglycerides: float

@app.post("/predict", tags=["Parkinson – numeric"])
async def predict_parkinson(data: PDInput):
    df = pd.DataFrame([data.dict()])
    df = df.reindex(columns=pd_scaler.feature_names_in_, fill_value=0)
    scaled = pd_scaler.transform(df)
    pred   = int(pd_model.predict(scaled)[0])
    msg    = (
        "The person has Parkinson disease"
        if pred == 1
        else "The person does not have Parkinson disease"
    )
    return {"prediction_class": "parkinson" if pred else "healthy",
            "prediction_value": pred,
            "result": msg}

@app.post("/predict_image", tags=["Parkinson – drawing"])
async def predict_parkinson_image(file: UploadFile = File(...)):
    image = Image.open(io.BytesIO(await file.read())) \
                 .convert("RGB") \
                 .resize((64, 64))
    arr   = np.expand_dims(np.array(image), axis=0)
    value = float(cnn.predict(arr)[0][0])
    label = "healthy" if value < 0.5 else "parkinson"
    return {"prediction_class": label,
            "prediction_value": value}

# ═══════════════════════════════════════════════════════════════════════
#                   W  I  L  S  O  N ’ S   D  I  S  E  A  S  E
# ═══════════════════════════════════════════════════════════════════════

class WilsonInput(BaseModel):
    Age: int
    ATB7B_Gene_Mutation: int = Field(..., alias="ATB7B Gene Mutation")
    Kayser_Fleischer_Rings: int = Field(..., alias="Kayser-Fleischer Rings")
    Copper_in_Blood_Serum: float = Field(..., alias="Copper in Blood Serum")
    Copper_in_Urine: float = Field(..., alias="Copper in Urine")
    Neurological_Symptoms_Score: float = Field(..., alias="Neurological Symptoms Score")
    Ceruloplasmin_Level: float = Field(..., alias="Ceruloplasmin Level")
    AST: float
    ALT: float
    Family_History: int = Field(..., alias="Family History")
    Gamma_Glutamyl_Transferase: float = Field(..., alias="Gamma-Glutamyl Transferase (GGT)")
    Total_Bilirubin: float

    class Config:
        validate_by_name = True    # renamed key in Pydantic v2
        extra = "allow"            # accept the other 11 features the model saw

@app.post("/predict_wilson", tags=["Wilson disease – numeric"])
async def predict_wilson(data: WilsonInput):
    # build DataFrame with ALL expected columns in correct order
    df = pd.DataFrame([data.dict(by_alias=True)])
    df = df.reindex(columns=wilson_scaler.feature_names_in_, fill_value=0)

    scaled = wilson_scaler.transform(df)
    prob   = float(wilson_model.predict(scaled)[0])
    has_disease = int(round(prob))

    return {
        "prediction_class": "wilson" if has_disease else "healthy",
        "prediction_value": prob,
        "result": (
            "The person has Wilson's disease."
            if has_disease
            else "The person does not have Wilson's disease."
        ),
    }
