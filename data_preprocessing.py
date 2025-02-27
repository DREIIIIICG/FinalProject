# data_preprocessing.py

import pandas as pd
import numpy as np

def preprocess_data(input_file: str, output_file: str = "processed_data.csv"):
    """Loads and preprocesses the dataset, replacing zero values with mean."""
    
    # Load dataset
    ds = pd.read_csv(input_file)
    print(f"Dataset size: {ds.shape}")

    # Remove zero values in selected columns
    no_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'BMI', 'Insulin']
    for column in no_zero:
        ds[column] = ds[column].replace(0, np.nan)
        mean = int(ds[column].mean(skipna=True))
        ds[column] = ds[column].replace(np.nan, mean)

    # Save processed data
    ds.to_csv(output_file, index=False)
    print("✅ Data Preprocessing Completed")

if __name__ == "__main__":
    preprocess_data("diabetes.csv")