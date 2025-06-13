import os
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# --- Load Model Artifacts ---
# Define paths to the model artifacts
# Ensure these paths are correct relative to app.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')

MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'weather_prediction_model.joblib')
ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'weather_label_encoder.joblib')
FEATURES_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'classifier_feature_names.joblib')

# Load the artifacts
try:
    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    # These are the feature names the model was trained on (e.g., ['temp_min', 'temp_max', 'precipitation', 'wind'])
    expected_features = joblib.load(FEATURES_PATH)
    print(f"Model, Label Encoder, and Feature List loaded successfully from {MODEL_ARTIFACTS_DIR}")
    print(f"Expected features for prediction: {expected_features}")
except FileNotFoundError as e:
    print(f"Error loading model artifacts: {e}")
    print("Please ensure 'weather_prediction_model.joblib', 'weather_label_encoder.joblib', and 'classifier_feature_names.joblib' are in the 'model_artifacts' directory.")
    model = None
    label_encoder = None
    expected_features = None
except Exception as e:
    print(f"An unexpected error occurred during artifact loading: {e}")
    model = None
    label_encoder = None
    expected_features = None


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    error_text = None
    input_data_for_template = {feature: "" for feature in expected_features} if expected_features else {}

    if request.method == 'POST':
        if not model or not label_encoder or not expected_features:
            error_text = "Model artifacts not loaded. Cannot make a prediction."
            return render_template('index.html', prediction_text=prediction_text, error_text=error_text, input_data=input_data_for_template)

        try:
            # Get data from form
            form_data = request.form.to_dict()
            input_data_for_template = form_data.copy() # Store for re-populating form

            # Validate and prepare features
            feature_values = []
            missing_features = []
            type_errors = []

            for feature_name in expected_features:
                if feature_name not in form_data or form_data[feature_name] == '':
                    missing_features.append(feature_name)
                    continue
                try:
                    feature_values.append(float(form_data[feature_name]))
                except ValueError:
                    type_errors.append(f"Feature '{feature_name}' must be a number.")

            if missing_features:
                error_text = f"Missing input for: {', '.join(missing_features)}."
            elif type_errors:
                error_text = " ".join(type_errors)
            else:
                # Create DataFrame for prediction (model expects 2D array)
                input_df = pd.DataFrame([feature_values], columns=expected_features)

                # Make prediction
                encoded_prediction = model.predict(input_df)
                predicted_weather_category = label_encoder.inverse_transform(encoded_prediction)

                prediction_text = f"Predicted Weather: {predicted_weather_category[0]}"

        except Exception as e:
            error_text = f"Error during prediction: {str(e)}"
            print(f"Prediction error: {e}")


    return render_template('index.html', prediction_text=prediction_text, error_text=error_text, input_data=input_data_for_template, expected_features=expected_features)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    if not model or not label_encoder or not expected_features:
        return jsonify({'error': 'Model artifacts not loaded. Cannot make a prediction.'}), 500

    try:
        data = request.get_json(force=True) # Get data posted as JSON

        # Validate and prepare features
        feature_values = []
        missing_features = []
        type_errors = []

        for feature_name in expected_features:
            if feature_name not in data:
                missing_features.append(feature_name)
                continue
            try:
                feature_values.append(float(data[feature_name]))
            except (TypeError, ValueError): # Handles if value is not a number or None
                type_errors.append(f"Feature '{feature_name}' must be a valid number.")


        if missing_features:
            return jsonify({'error': f"Missing features in JSON payload: {', '.join(missing_features)}"}), 400
        if type_errors:
            return jsonify({'error': " ".join(type_errors)}), 400


        # Create DataFrame for prediction
        input_df = pd.DataFrame([feature_values], columns=expected_features)

        # Make prediction
        encoded_prediction = model.predict(input_df)
        predicted_weather_category = label_encoder.inverse_transform(encoded_prediction)

        return jsonify({'predicted_weather': predicted_weather_category[0]})

    except Exception as e:
        print(f"API Prediction error: {e}")
        return jsonify({'error': f'Error processing request: {str(e)}'}), 500


# --- Run the App ---
if __name__ == '__main__':
    # Check if artifacts loaded correctly before trying to run
    if model and label_encoder and expected_features:
        app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000/
    else:
        print("Application cannot start due to missing model artifacts. Please check the 'model_artifacts' directory and error messages.")