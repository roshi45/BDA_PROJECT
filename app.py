from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load('heart_disease_model.pkl')

@app.route('/')
def home():
    return "Welcome to the Heart Disease Prediction API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame(data)
    predictions = model.predict(df)
    return jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug=True)
