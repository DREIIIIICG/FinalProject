import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# Load full dataset
X = pd.read_csv("scaled_features.csv")
y = pd.read_csv("labels.csv")

# Split the dataset (ensure consistency with model_training.py)
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Load trained model
knn = joblib.load("knn_model.pkl")

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate model
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))