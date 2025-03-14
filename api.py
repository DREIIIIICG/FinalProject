import mlflow.pyfunc
import logging
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the trained model from MLflow
MODEL_URI = "models:/DiabetesModel/1"  

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info(f"Model loaded successfully from MLflow ({MODEL_URI})!")
except Exception as e:
    logger.error(f"Error loading model from MLflow: {e}")
    model = None  

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
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please check MLflow model registry.")

    # Convert input to DataFrame
    #df = pd.DataFrame([data.dict()])
    df = pd.DataFrame([data.dict()])

    feature_order = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
                 "BMI", "DiabetesPedigreeFunction", "Age"]

    df = df[feature_order] 
   

    # Make prediction
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}