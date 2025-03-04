import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Get all registered models
models = client.search_registered_models()
for model in models:
    print(f"Model: {model.name}, Versions: {[v.version for v in model.latest_versions]}")