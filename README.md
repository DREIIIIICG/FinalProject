# Diabetes Prediction with MLflow and FastAPI

This project trains a diabetes prediction model, logs it with MLflow, and serves it via FastAPI.

## Project Structure
```
/diabetes-mlops
│── data/                # Dataset for training
│── notebooks/           # Jupyter notebooks for model training
│── mlflow_tracking.py   # Script to train and log model with MLflow
│── api.py               # FastAPI app to serve the model
│── docker-compose.yaml  # Docker Compose setup for MLflow and FastAPI
│── requirements.txt     # Dependencies
│── README.md            # Project documentation
```

## Prerequisites
- Python 3.x
- Virtual environment (venv)
- MLflow
- FastAPI
- Uvicorn
- Docker & Docker Compose (if running inside containers)

## Setup
### 1. Clone the repository and navigate to the project folder:
```sh
git clone <repo_url>
cd diabetes-mlops
```

### 2. Create and activate a virtual environment:
```sh
python -m venv venv
```
#### On Windows (PowerShell):
```sh
./venv/Scripts/Activate
```
#### On macOS/Linux:
```sh
source venv/bin/activate
```

### 3. Install dependencies:
```sh
pip install -r requirements.txt
```

## Running MLflow Tracking UI
Set the MLflow tracking URI:
```sh
echo %MLFLOW_TRACKING_URI%
```
Run the MLflow UI:
```sh
mlflow ui --backend-store-uri file:///C:/diabetes-mlops/mlruns
```
Open `http://127.0.0.1:5000` in your browser to view the MLflow UI.

## Training and Logging the Model
Run the following script to train and log the model in MLflow:
```sh
python mlflow_tracking.py
```

## Serving the Model with FastAPI
Run the FastAPI server:
```sh
uvicorn api:app --reload
```
This will start the API on `http://127.0.0.1:8000`.

## Docker Deployment
To deploy with Docker Compose, run:
```sh
docker-compose up --build
```

## API Usage
### Predict Diabetes
Send a POST request to `/predict` with patient data:
```sh
curl -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d '{
  "Pregnancies": 2,
  "Glucose": 120,
  "BloodPressure": 70,
  "SkinThickness": 20,
  "Insulin": 80,
  "BMI": 25.5,
  "DiabetesPedigreeFunction": 0.5,
  "Age": 35
}'
```

## Future Improvements
- Enhance model performance with hyperparameter tuning
- Deploy API in a cloud environment
- Implement authentication for the API
