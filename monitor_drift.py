import pandas as pd
import mlflow
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset

# Load reference dataset
reference_data = pd.read_csv("processed_data.csv")

# Load production dataset 
production_data = pd.read_csv("production_data.csv")

# Run drift report
data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=reference_data, current_data=production_data)

# Save and log drift report
data_drift_report.save_html("data_drift_report.html")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
with mlflow.start_run(run_name="drift_monitoring"):
    mlflow.log_artifact("data_drift_report.html")

print("Drift monitoring complete! Report saved and logged to MLflow.")
