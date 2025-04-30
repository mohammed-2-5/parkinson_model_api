from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager

import pandas as pd
import numpy as np
from PIL import Image
import joblib
import gdown
import io
import os

from tensorflow.keras.models import load_model

# ⛔️ لمنع استخدام الـ GPU على السيرفرات
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ⬇️ روابط Google Drive لتحميل النماذج
FILES = {
    "model.pkl": "1CWubdXPizuhvGRayPbKj0XwKNVRsO5jY",
    "scaler.pkl": "1ViMGq8moxRtpNE56cV-nyNwdgBMpUwP7",
    "drawings.keras": "1s7_QLcejB6-4DAYIjktkflP_1FwC0V1J"
}

def download_models():
    os.makedirs("models", exist_ok=True)
    for filename, file_id in FILES.items():
        path = f"models/{filename}"
        if not os.path.exists(path):
            print(f"🔽 Downloading {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, path, quiet=False)

# ⬇️ تحميل النماذج تلقائيًا عند بدء التشغيل
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, cnn
    download_models()
    model = joblib.load("models/model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    cnn = load_model("models/drawings.keras")
    yield

# ⬇️ إنشاء تطبيق FastAPI
app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
def redirect_to_docs():
    return RedirectResponse(url="/docs", status_code=308)

# ⬇️ بيانات النموذج الرقمي
class InputData(BaseModel):
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

# ⬇️ API لتنبؤ مرض باركنسون من القيم الرقمية
@app.post("/predict")
async def predict(data: InputData):
    df = pd.DataFrame([data.dict()])
    df = df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    result = "The person has Parkinson disease" if prediction[0] == 1 else "The person does not have Parkinson disease"
    return {"result": result}

# ⬇️ API لتنبؤ المرض من صورة
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((64, 64))
    array = np.expand_dims(np.array(image), axis=0)
    prediction = cnn.predict(array)
    value = float(prediction[0][0])
    label = "healthy" if value < 0.5 else "parkinson"
    return {"prediction_class": label, "prediction_value": value}
