from flask import Flask, request, jsonify
import joblib


# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load('../best_models/gradient_boosting_fraud_data_best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud based on the input data
    """
    data = request.get_json(force=True)

    # Perform prediction
    try:
        prediction = model.predict(data['features'])
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)