from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Load the pre-trained model using pickle
with open("notebook/xgb_finalModel.pkl", "rb") as f:
    model = pickle.load(f)



# Define expected features (use the same order as training!)
FEATURE_ORDER = [
    "SugarLevel", "Age", "Gender", "Creatinine", "BMI", "Urea",
    "Cholesterol", "LDL", "VLDL", "TG", "HDL", "HBA1C"
]

# Initialize FastAPI app
app = FastAPI(title="GlucoGuard API", version="1.0")

# Define input data structure (features of your model)
class PatientData(BaseModel):
    SugarLevel: float
    Age: int
    Gender: int
    Creatinine: float
    BMI: float
    Urea: float
    Cholesterol: float
    LDL: float
    VLDL: float
    TG: float
    HDL: float
    HBA1C: float

@app.get("/")
def root():
    return {"message": "GlucoGuard API is running!"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        # Convert input to array in the right order
        features = [getattr(data, col) for col in FEATURE_ORDER]
        features = np.array(features).reshape(1, -1)

        # Scale the features using the same scaler used during training
        scaled_features = scaler.transform(features)

        # Predict the class
        pred = model.predict(scaled_features)[0]
        proba = model.predict_proba(scaled_features)[0].tolist()

        return {
            "predicted_class": int(pred),
            "class_probabilities": proba
        }
    except Exception as e:
        return {"error": str(e)}
