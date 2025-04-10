from flask import Flask, request, jsonify
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained models from the saved files
rf_model = joblib.load('models/rf_model.pkl')  # Load the Random Forest model
svm_model = joblib.load('models/svm_model.pkl')  # Load the SVM model

# Define the prediction endpoint for Random Forest
@app.route('/predict/rf', methods=['POST'])
def predict_rf():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract features from the JSON data
        features = np.array([data['Day_of_Year'], data['Location'], data['Size']]).reshape(1, -1)
        
        # Predict the price using the Random Forest model
        rf_prediction = rf_model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_price': rf_prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Define the prediction endpoint for SVM
@app.route('/predict/svm', methods=['POST'])
def predict_svm():
    try:
        # Get JSON data from the request
        data = request.get_json()
        
        # Extract features from the JSON data
        features = np.array([data['Day_of_Year'], data['Location'], data['Size']]).reshape(1, -1)
        
        # Predict the price using the SVM model
        svm_prediction = svm_model.predict(features)
        
        # Return the prediction as a JSON response
        return jsonify({'predicted_price': svm_prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
