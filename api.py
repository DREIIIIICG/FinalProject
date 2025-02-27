from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Load the trained model correctly
with open("knn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Debug: Print model type
print(f"Model loaded: {type(model)}")  # Should be a scikit-learn model

# Define input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API!"}

# Prediction endpoint
@app.post("/predict/")
def predict(data: DiabetesInput):
    df = pd.DataFrame([data.dict()])  # Convert input to DataFrame
    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", 
                     "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
    df = df[feature_names]  # Ensure correct column order

    # Check if model is correct before prediction
    if not hasattr(model, "predict"):
        return {"error": "Model not loaded correctly. Please check knn_model.pkl"}

    # Make prediction
    prediction = model.predict(df)[0]
    return {"prediction": int(prediction)}