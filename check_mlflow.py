import mlflow

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print("MLflow Tracking URI:", mlflow.get_tracking_uri())

# Verify if MLflow can retrieve existing experiments
client = mlflow.tracking.MlflowClient()
experiments = client.search_experiments()

print("\n🔍 Existing Experiments:")
for exp in experiments:
    print(f"- {exp.name} (ID: {exp.experiment_id})")
