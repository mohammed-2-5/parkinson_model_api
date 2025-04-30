from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ⛔️ لمنع استخدام GPU

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import io
from contextlib import asynccontextmanager
from fastapi.responses import RedirectResponse

# استخدم Lifespan بدلاً من on_event
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, scaler, cnn
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
    cnn = load_model("drawings.keras")
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs", status_code=308)

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

@app.post("/predict")
async def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = "The person has Parkinson disease" if prediction[0] == 1 else "The person does not have Parkinson disease"
    return {"result": result}

@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize((64, 64))
    image_array = np.array(image)
    input_image = np.expand_dims(image_array, axis=0)
    prediction = cnn.predict(input_image)
    prediction_value = float(prediction[0][0])
    predicted_class = "healthy" if prediction_value < 0.5 else "parkinson"
    return {"prediction_class": predicted_class, "prediction_value": prediction_value}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
