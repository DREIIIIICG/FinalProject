import subprocess
from dagster import asset

@asset
def preprocess_data():
    """Runs the data preprocessing script."""
    result = subprocess.run(["python", "01_data_preprocessing.py"], capture_output=True, text=True)
    if result.returncode == 0:
        return "Data Preprocessing Completed"
    else:
        raise RuntimeError(f"Error in data preprocessing: {result.stderr}")

@asset
def feature_engineering(preprocess_data):
    """Runs the feature engineering script."""
    result = subprocess.run(["python", "02_feature_engineering.py"], capture_output=True, text=True)
    if result.returncode == 0:
        return "Feature Engineering Completed"
    else:
        raise RuntimeError(f"Error in feature engineering: {result.stderr}")

@asset
def train_model(feature_engineering):
    """Runs the model training script."""
    result = subprocess.run(["python", "03_model_training.py"], capture_output=True, text=True)
    if result.returncode == 0:
        return "Model Training Completed"
    else:
        raise RuntimeError(f"Error in model training: {result.stderr}")

@asset
def evaluate_model(train_model):
    """Runs the model evaluation script."""
    result = subprocess.run(["python", "04_model_evaluation.py"], capture_output=True, text=True)
    if result.returncode == 0:
        return "Model Evaluation Completed"
    else:
        raise RuntimeError(f"Error in model evaluation: {result.stderr}")

