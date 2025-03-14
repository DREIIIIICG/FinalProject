import pandas as pd
import numpy as np

# Load reference dataset
reference_data = pd.read_csv("processed_data.csv")

# Generate new production data
num_samples = 100  # sample size
production_data = pd.DataFrame({
    "Pregnancies": np.random.randint(0, 10, num_samples),
    "Glucose": np.random.normal(120, 30, num_samples),
    "BloodPressure": np.random.normal(70, 10, num_samples),
    "SkinThickness": np.random.normal(20, 8, num_samples),
    "Insulin": np.random.normal(80, 40, num_samples),
    "BMI": np.random.normal(30, 5, num_samples),
    "DiabetesPedigreeFunction": np.random.uniform(0.1, 2.5, num_samples),
    "Age": np.random.randint(21, 80, num_samples),
    "Outcome": np.random.choice([0, 1], num_samples)  # Randomly assign Outcome
})

# Save production data
production_data.to_csv("production_data.csv", index=False)
print("Production dataset created successfully!")

