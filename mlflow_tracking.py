import mlflow
import mlflow.sklearn
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Enable autologging
mlflow.sklearn.autolog()

# Load dataset
X = pd.read_csv("scaled_features.csv", skiprows=1, header=None)  
y = pd.read_csv("labels.csv")

# Convert labels to integers
y = y.astype(int)

# Verify dataset shapes
print("Features shape:", X.shape)
print("Labels shape:", y.shape)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Start MLflow experiment
with mlflow.start_run():
    # Train model
    knn = KNeighborsClassifier(n_neighbors=11, metric='euclidean')
    knn.fit(X_train, y_train.values.ravel())

    # Predict
    y_pred = knn.predict(X_test)

    # Compute Metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Log model
    mlflow.sklearn.log_model(knn, "knn_model", registered_model_name="DiabetesModel")

    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("F1 Score:", f1)
    print("Accuracy:", accuracy)
    print("Model and metrics logged with MLflow!")

# Verify registered models
from mlflow.tracking import MlflowClient

client = MlflowClient()
models = client.search_registered_models()

print("\n🔍 Registered Models in MLflow:")
for model in models:
    print(f"Model Name: {model.name}")
    for version in model.latest_versions:
        print(f" - Version: {version.version}, Stage: {version.current_stage}")