from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained models
binary_model = joblib.load("model/ids_model.pkl")
attack_model = joblib.load("model/attack_type_model.pkl")
attack_encoder = joblib.load("model/attack_label_encoder.pkl")

# Model expects 40 numeric features
TOTAL_FEATURES = 40

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    attack_type = None

    if request.method == "POST":
        try:
            # Read input values from form
            values = list(request.form.values())
            values = [float(v) for v in values]

            # Pad remaining features with zeros
            if len(values) < TOTAL_FEATURES:
                values.extend([0.0] * (TOTAL_FEATURES - len(values)))

            # Convert to numpy array
            input_data = np.array(values).reshape(1, -1)

            # -------------------------------
            # Stage 1: Normal vs Attack
            # DEMO-safe threshold logic
            # -------------------------------
            if max(values) > 1000:
                result = 1   # Force Attack for abnormal values
            else:
                result = binary_model.predict(input_data)[0]

            # -------------------------------
            # Stage 2: Attack Type Prediction
            # -------------------------------
            if result == 0:
                prediction = "Normal Traffic"
            else:
                prediction = "Attack Detected"

                attack_pred = attack_model.predict(input_data)[0]
                attack_type = attack_encoder.inverse_transform(
                    [attack_pred]
                )[0]

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template(
        "index.html",
        prediction=prediction,
        attack_type=attack_type
    )

if __name__ == "__main__":
    app.run(debug=True)

