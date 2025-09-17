from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pickle

# Load the pre-trained model and scaler
model = joblib.load("notebook/xgb_final01.joblib")  # Updated model file name
scaler = joblib.load("notebook/scaler01.pkl")  # Updated scaler file name

# Initialize FastAPI app
app = FastAPI()

# Define the input data schema (using Pydantic)
class PredictionRequest(BaseModel):
    gender: float
    age: float
    urea: float
    cr: float
    hbA1c: float
    chol: float
    tg: float
    hdl: float
    ldl: float
    vldl: float
    bmi: float

# Define the POST endpoint for making predictions
@app.post("/predict/")
def predict(data: PredictionRequest):
    try:
        # Extract features from the incoming request data
        features = np.array([
            data.gender, data.age, data.urea, data.cr, data.hbA1c,
            data.chol, data.tg, data.hdl, data.ldl, data.vldl, data.bmi
        ]).reshape(1, -1)

        # Scale the features using the loaded scaler
        scaled_features = scaler.transform(features)

        # Get the prediction from the model
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0].tolist()

        # Return the prediction and class probabilities
        return {
            "predicted_class": int(prediction),  # Convert numpy.int64 to native Python int
            "class_probabilities": probabilities  # Convert numpy array to list
        }
    except Exception as e:
        # Return an error message if something goes wrong
        return {"error": str(e)}

# Root endpoint to verify the server is running
@app.get("/")
def root():
    return {"message": "Welcome to the Diabetes Risk Prediction API!"}
