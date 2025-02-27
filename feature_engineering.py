import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load preprocessed data
df = pd.read_csv("processed_data.csv")

# Separate features and target
X = df.iloc[:, :-1]  # All columns except last (features)
y = df.iloc[:, -1]   # Last column (target)

# Standardize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save processed features
pd.DataFrame(X_scaled).to_csv("scaled_features.csv", index=False)
pd.DataFrame(y).to_csv("labels.csv", index=False)