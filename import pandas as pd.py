import pandas as pd

# Load dataset
data = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")

# Separate features and target
X = data.drop("label", axis=1)
y = data["label"]

# Print checks
print("Features shape (X):", X.shape)
print("Target shape (y):", y.shape)
print("First 5 labels:")
print(y.head())

