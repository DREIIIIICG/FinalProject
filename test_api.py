from fastapi.testclient import TestClient
from api import app  # Import FastAPI app

client = TestClient(app)  # Initialize test client

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to the Diabetes Prediction API!"}  #Updated test

def test_prediction():
    input_data = {
        "Pregnancies": 2,
        "Glucose": 100.0,
        "BloodPressure": 70.0,
        "SkinThickness": 25.0,
        "Insulin": 30.0,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    assert "prediction" in response.json()

# New test for invalid data
def test_prediction_invalid_data():
    input_data = {
        "Pregnancies": -1,  # Invalid negative value
        "Glucose": "abc",  # Invalid type (string instead of float)
        "BloodPressure": 70.0,
        "SkinThickness": 25.0,
        "Insulin": 30.0,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }
    
    response = client.post("/predict", json=input_data)
    assert response.status_code == 422  # Expecting validation error