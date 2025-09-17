from fastapi import FastAPI
from fastapi.responses import FileResponse
import pickle
import pandas as pd
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import os
import numpy 

# Load the pre-trained model using pickle
with open("notebook/xgb_finalModel.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI()

# Serve HTML file
@app.get("/")
async def serve_html():
    return FileResponse("templates/index.html")

# Define the input data structure (features of your model)
class PredictionRequest(BaseModel):
    gender:float
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

# Create a function to handle the prediction
def predict(features: pd.DataFrame):
    # Preprocessing (scaling the features using StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)  # Scale the features like during training
    
    # Predict
    predictions = model.predict(X_scaled)
    return predictions

# Define a prediction endpoint
@app.post("/predict/")
async def make_prediction(request: PredictionRequest):
    # Convert the incoming request to a pandas DataFrame
    data = pd.DataFrame([request.dict()])
    
    # Make the prediction
    prediction = predict(data)
    
    # Return the prediction
    return {"prediction": prediction[0]}  # The prediction result (0 or 1, based on your classification)

# Run the app with Uvicorn
# uvicorn main:app --reload
