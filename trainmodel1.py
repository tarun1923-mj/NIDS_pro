import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("Script started")

# Load dataset
data = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")
print("Dataset loaded successfully")

# Keep only numeric features
X = data.drop("label", axis=1).select_dtypes(include=["number"])
y = data["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/ids_model.pkl")

print("Model saved as model/ids_model.pkl")

