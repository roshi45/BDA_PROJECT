import os
from flask import Flask, request, jsonify
import joblib
import pandas as pd  # Ensure pandas is imported

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])  # Convert the incoming data into a DataFrame
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    # Get the port from the environment if available, else default to 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
