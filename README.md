# Diabetes Prediction MLOps Pipeline  

## Project Overview  
This project builds an end-to-end MLOps pipeline for predicting diabetes using machine learning. It includes data preprocessing, model training, experiment tracking with MLflow, model serving via FastAPI, and workflow orchestration using Dagster.  

##  Dataset Overview  
This dataset originates from the **National Institute of Diabetes and Digestive and Kidney Diseases** and aims to predict whether a patient has diabetes based on various diagnostic measurements.  

### 🔹 Dataset Details:  
- **Number of Instances**: 768  
- **Number of Attributes**: 8 features + 1 target variable  
- **Missing Values**: Yes (handled in preprocessing)  
- **Class Distribution**:  
  - `0`: No Diabetes  
  - `1`: Diabetes  

### 🔹 Features:  
| Feature                     | Description |
|-----------------------------|-------------|
| **Pregnancies**             | Number of times pregnant |
| **Glucose**                 | Plasma glucose concentration (2-hour OGTT) |
| **BloodPressure**           | Diastolic blood pressure (mm Hg) |
| **SkinThickness**           | Triceps skin fold thickness (mm) |
| **Insulin**                 | 2-Hour serum insulin (mu U/ml) |
| **BMI**                     | Body mass index (kg/m²) |
| **DiabetesPedigreeFunction**| Diabetes pedigree function |
| **Age**                     | Age in years |
| **Outcome**                 | Target variable (0 = No Diabetes, 1 = Diabetes) |

## Project Structure  
```
diabetes-mlops/
│── data/               # Contains datasets (raw & processed)
│── models/             # Stores trained models
│── notebooks/          # Jupyter Notebooks for analysis & experimentation
│── scripts/            # Python scripts for preprocessing, training, and evaluation
│── tests/              # Unit tests for validation
│── Dockerfile          # Docker configuration
│── docker-compose.yml  # Multi-container setup for API, MLflow, and pipeline
│── requirements.txt    # Dependencies
│── pyproject.toml      # Dagster configuration
│── README.md           # Project documentation
```

## Installation  
1. **Clone the repository:**  
   ```
   git clone https://github.com/DREIIIIICG/FinalProject.git  
   cd FinalProject  
   ```

2. **Set up a virtual environment:**  
   ```
   python -m venv venv  
   source venv/bin/activate  # On macOS/Linux
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**  
   ```
   pip install -r requirements.txt  
   ```

## Data Preprocessing  
- Missing values in critical features (`Glucose`, `BloodPressure`, `SkinThickness`, `BMI`, `Insulin`) are replaced with the mean.
- Features are standardized using `StandardScaler`.

Run preprocessing:  
```bash
python scripts/data_preprocessing.py  
```

## Model Training & Tracking with MLflow  
- Uses `KNeighborsClassifier` from `sklearn.neighbors` for classification.
- Logs training metrics & model artifacts to MLflow.

Start MLflow UI:  
```bash
mlflow ui --backend-store-uri file:///C:/diabetes-mlops/mlruns  
```

Run training:  
```bash
python scripts/train_model.py  
```

## Model Serving with FastAPI  
- Exposes an API for real-time diabetes prediction.  

Start the API:  
```bash
uvicorn api:app --reload  
```

Test the API using `requests` or Postman:  
```python
import requests
response = requests.post("http://127.0.0.1:8000/predict/", json={
    "Pregnancies": 1, "Glucose": 100, "BloodPressure": 80, "SkinThickness": 20,
    "Insulin": 85, "BMI": 30, "DiabetesPedigreeFunction": 0.5, "Age": 25
})
print(response.json())
```

## Orchestration with Dagster  
Dagster automates workflow execution across the pipeline.

Start Dagster UI:  
```bash
dagster dev  
```

Run the pipeline:  
```bash
dagster job execute --job diabetes_pipeline  
```

## Monitoring Drift

File	                        Description
generate_production_data.py	Generates synthetic production data.
monitor_drift.py	            Runs drift analysis and logs results to MLflow.
requirements.txt	            Lists project dependencies.
processed_data.csv	         Reference dataset (baseline).
production_data.csv	         Production dataset (incoming data).
data_drift_report.html	      HTML report showing feature drift.

Start the MLFlow tracking server 
Run monitor_drift.py. This will compare reference data vs production data, compare metrics for the features, save the html report and log drift results to MLFlows

### How Production Data Was Generated:  
- The production dataset is **synthetically generated** to simulate real-world incoming data.
- Each feature follows a **randomized distribution**:
  - `np.random.randint(start, end, size)` → Generates random integers.
  - `np.random.normal(mean, std, size)` → Generates random values from a normal distribution.
  - `np.random.uniform(low, high, size)` → Generates values from a uniform distribution.
  - `np.random.choice([values], size)` → Randomly selects from given values (`Outcome` is either `0` or `1`).

## Unit Testing with Pytest  
Run tests:  
```bash
pytest tests/test_api.py -v  
```

## Deploying with Docker  
1. **Build and run containers:**  
   ```bash
   docker-compose up --build  
   ```
2. **Access the services:**  
   - MLflow UI → `http://127.0.0.1:5000`
   - FastAPI → `http://127.0.0.1:8000`
   - Dagster UI → `http://127.0.0.1:3000`

## Conclusion  
Built an automated MLOps pipeline  
Tracked experiments & versioned models with MLflow  
Deployed model via FastAPI  
Automated workflows using Dagster  