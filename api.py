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

# Corrected MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"  # ✅ Fixed
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load the trained model from MLflow
MODEL_URI = "models:/DiabetesModel/Production"  # Automatically fetch latest production model

try:
    model = mlflow.pyfunc.load_model(MODEL_URI)
    logger.info(f"✅ Model loaded successfully from MLflow ({MODEL_URI})!")
except Exception as e:
    logger.error(f"❌ Error loading model from MLflow: {e}")
    model = None  # Set model to None to avoid crashes

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
    df = pd.DataFrame([data.dict()])

    # Make prediction
    prediction = model.predict(df)[0]

    return {"prediction": int(prediction)}