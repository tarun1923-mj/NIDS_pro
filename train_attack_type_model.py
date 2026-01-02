import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

print("Training Attack Type Model")

# Load dataset
data = pd.read_csv("Dataset/UNSW_NB15_training-set.csv")

# Use only attack records
attack_data = data[data["label"] == 1]

# Features: numeric only
X = attack_data.drop(["label", "attack_cat"], axis=1).select_dtypes(include=["number"])

# Target: attack category
y = attack_data["attack_cat"]

# Encode attack labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Train model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Attack Type Model Accuracy:", accuracy)

# Save model and encoder
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/attack_type_model.pkl")
joblib.dump(encoder, "model/attack_label_encoder.pkl")

print("Attack type model saved")

