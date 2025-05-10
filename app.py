from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from fastapi.responses import RedirectResponse
from contextlib import asynccontextmanager
from google.cloud import storage
import pandas as pd
import numpy as np
from PIL import Image
import joblib
import io
import os

from tensorflow.keras.models import load_model

# ⛔️ لمنع استخدام الـ GPU على السيرفرات
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# ⬇️ إعدادات Google Cloud Storage
BUCKET_NAME = "parkinson_models"
FILES = {
    "model.pkl": "model.pkl",
    "scaler.pkl": "scaler.pkl",
    "drawings.keras": "drawings.keras"
}

# ⬇️ تحميل النماذج من Google Cloud Storage
def download_models_from_gcs():
    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    os.makedirs("models", exist_ok=True)
    
    for filename, blob_name in FILES.items():
        path = f"models/{filename}"
        if not os.path.exists(path):
            print(f"🔽 Downloading {filename} from GCS...")
            blob = bucket.blob(blob_name)
            blob.download_to_filename(path)

# ⬇️ تحميل النماذج تلقائيًا عند بدء التشغيل
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, cnn
    download_models_from_gcs()
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
