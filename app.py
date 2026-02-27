from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("model/aqi_model.pkl")

# AQI category helper
def get_aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Poor"
    elif aqi <= 300:
        return "Very Poor"
    else:
        return "Severe"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    pm25 = float(request.form["pm25"])
    pm10 = float(request.form["pm10"])
    no2 = float(request.form["no2"])
    so2 = float(request.form["so2"])
    co = float(request.form["co"])
    o3 = float(request.form["o3"])

    features = np.array([[pm25, pm10, no2, so2, co, o3]])
    prediction = model.predict(features)[0]
    category = get_aqi_category(prediction)

    return render_template(
        "index.html",
        prediction_text=f"Predicted AQI: {prediction:.2f} ({category})"
    )

if __name__ == "__main__":
    app.run()