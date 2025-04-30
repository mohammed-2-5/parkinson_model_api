from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os  # 🟰 اضف هذا

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from PIL import Image
import io
from fastapi.responses import RedirectResponse

# إنشاء تطبيق FastAPI
app = FastAPI()
model = None
scaler = None
cnn = None

@app.on_event("startup")
def load_models():
    global model, scaler, cnn
    model = joblib.load("model (1).pkl")
    scaler = joblib.load("scaler.pkl")
    cnn = load_model("drawings (1).keras")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs", status_code=308)


# تعريف شكل بيانات الأرقام
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

# API للأرقام
@app.post("/predict")
async def predict(data: InputData):
    input_dict = data.dict()
    input_df = pd.DataFrame([input_dict])
    input_df = input_df.reindex(columns=scaler.feature_names_in_, fill_value=0)
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)

    if prediction[0] == 1:
        result = "The person has Parkinson disease"
    else:
        result = "The person does not have Parkinson disease"

    return {"result": result}

# API للصورة
@app.post("/predict_image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents))

    # تحويل الصورة لـ RGB لو مش بالفعل
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # تغيير الحجم
    image = image.resize((64, 64))

    # تحويل الصورة لمصفوفة numpy
    image_array = np.array(image)

    # ❌ بدون normalization /255

    input_image = np.expand_dims(image_array, axis=0)

    # التوقع
    prediction = cnn.predict(input_image)
    prediction_value = float(prediction[0][0])

    if prediction_value < 0.5:
        predicted_class = "healthy"
    else:
        predicted_class = "parkinson"

    return {
        "prediction_class": predicted_class,
        "prediction_value": prediction_value
    }
# هنا نشغل التطبيق عند الإقلاع

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))