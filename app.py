"""
Flask Web Application for Weather Prediction.

This application provides a user interface and an API endpoint to predict
weather conditions based on a pre-trained machine learning model.
It loads model artifacts at startup and serves predictions via a web form
or a JSON API.
"""

import sys
from pathlib import Path
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# --- Configuration & Global Variables ---
BASE_DIR = Path(__file__).resolve().parent
MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

ARTIFACT_PATHS = {
    "model": MODEL_ARTIFACTS_DIR / "weather_prediction_model.joblib",
    "encoder": MODEL_ARTIFACTS_DIR / "weather_label_encoder.joblib",
    "features": MODEL_ARTIFACTS_DIR / "classifier_feature_names.joblib",
}

# Global variables for loaded artifacts, will be populated at startup.
model = None
label_encoder = None
expected_features = None

# --- Helper Functions ---

def load_artifacts():
    """
    Loads machine learning artifacts from disk.

    This function is called once at application startup. If any artifact
    is not found, it prints an error and exits the application.
    """
    global model, label_encoder, expected_features
    print("Loading model artifacts...")
    try:
        model = joblib.load(ARTIFACT_PATHS["model"])
        label_encoder = joblib.load(ARTIFACT_PATHS["encoder"])
        expected_features = joblib.load(ARTIFACT_PATHS["features"])
        print("Artifacts loaded successfully.")
        print(f"Expected features: {expected_features}")
    except FileNotFoundError as e:
        print(f"Error: Artifact not found at {e.filename}.")
        print("Application cannot start without all model artifacts.")
        sys.exit(1)

def validate_and_prepare_features(data_source):
    """Validates and extracts feature values from a dictionary-like source."""
    feature_values = []
    errors = []

    for feature in expected_features:
        value = data_source.get(feature)
        if value is None or value == '':
            errors.append(f"Missing input for: '{feature}'.")
            continue
        try:
            feature_values.append(float(value))
        except (ValueError, TypeError):
            errors.append(f"Feature '{feature}' must be a number.")

    return feature_values, errors

def handle_form_prediction(form_data):
    """Handles the prediction logic for the web form."""
    feature_values, errors = validate_and_prepare_features(form_data)

    if errors:
        return None, " ".join(errors)

    # Create DataFrame for prediction
    input_df = pd.DataFrame([feature_values], columns=expected_features)

    # Make prediction
    encoded_prediction = model.predict(input_df)
    predicted_weather = label_encoder.inverse_transform(encoded_prediction)[0]

    prediction_text = f"Predicted Weather: {predicted_weather}"
    return prediction_text, None

# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Renders the main page and handles form submissions for predictions."""
    prediction_text = None
    error_text = None
    # Keep user's input in the form after submission
    input_data = {feature: "" for feature in expected_features}

    if request.method == 'POST':
        input_data = request.form.to_dict()
        try:
            prediction_text, error_text = handle_form_prediction(request.form)
        except (ValueError, KeyError) as e:
            # Catch specific errors from model prediction or data handling
            print(f"Unhandled prediction error: {e}")
            error_text = "An internal error occurred during prediction. Please check server logs."

    return render_template(
        'index.html',
        prediction_text=prediction_text,
        error_text=error_text,
        input_data=input_data,
        expected_features=expected_features
    )

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """Provides an API endpoint for weather prediction."""
    try:
        data = request.get_json(force=True)
        feature_values, errors = validate_and_prepare_features(data)

        if errors:
            return jsonify({'error': " ".join(errors)}), 400

        # Create DataFrame for prediction
        input_df = pd.DataFrame([feature_values], columns=expected_features)

        # Make prediction
        encoded_prediction = model.predict(input_df)
        predicted_weather = label_encoder.inverse_transform(encoded_prediction)[0]

        return jsonify({'predicted_weather': predicted_weather})

    except (ValueError, KeyError) as e:
        # Catch specific errors from model prediction or data handling
        print(f"API Prediction error: {e}")
        return jsonify({'error': f'Error processing request: {e}'}), 500

# --- Main Execution ---

if __name__ == '__main__':
    load_artifacts()
    app.run(debug=True, port=5000)



# IGNORE THE CODE BELOW

# # --- app.py (Refactored) ---

# import sys
# from pathlib import Path
# from flask import Flask, request, jsonify, render_template
# import joblib
# import pandas as pd

# # Initialize Flask app
# app = Flask(__name__)

# # --- Configuration & Global Variables ---
# # Feedback: Use pathlib for more concise path handling.
# BASE_DIR = Path(__file__).resolve().parent
# MODEL_ARTIFACTS_DIR = BASE_DIR / "model_artifacts"

# # Feedback: Group related filenames into a single data structure.
# ARTIFACT_PATHS = {
#     "model": MODEL_ARTIFACTS_DIR / "weather_prediction_model.joblib",
#     "encoder": MODEL_ARTIFACTS_DIR / "weather_label_encoder.joblib",
#     "features": MODEL_ARTIFACTS_DIR / "classifier_feature_names.joblib",
# }

# # Global variables for loaded artifacts, initialized to None.
# # These will be loaded by _load_artifacts() at startup.
# model = None
# label_encoder = None
# expected_features = None


# # --- Helper Functions ---
# # Feedback: Create a helper function for loading artifacts.
# def _load_artifacts():
#     """
#     Loads machine learning artifacts from disk.
#     This function is called once at application startup.
#     If any artifact is not found, it will print an error and exit.
#     """
#     global model, label_encoder, expected_features
#     print("Loading model artifacts...")
#     try:
#         model = joblib.load(ARTIFACT_PATHS["model"])
#         label_encoder = joblib.load(ARTIFACT_PATHS["encoder"])
#         expected_features = joblib.load(ARTIFACT_PATHS["features"])
#         print("Artifacts loaded successfully.")
#         print(f"Expected features: {expected_features}")
#     except FileNotFoundError as e:
#         # Feedback: Fail startup if model doesn't load.
#         # This is more robust than letting the app run in a broken state.
#         print(f"Error: Artifact not found at {e.filename}.")
#         print("Application cannot start without all model artifacts.")
#         sys.exit(1) # Exit the application with a non-zero status code

# def _validate_and_prepare_features(data_source, source_type='form'):
#     """Validates and extracts feature values from a dictionary-like source."""
#     feature_values = []
#     errors = []
    
#     for feature in expected_features:
#         value = data_source.get(feature)
#         if value is None or value == '':
#             errors.append(f"Missing input for: '{feature}'.")
#             continue
#         try:
#             feature_values.append(float(value))
#         except (ValueError, TypeError):
#             errors.append(f"Feature '{feature}' must be a number.")
    
#     return feature_values, errors

# # Feedback: Break out POST logic into a helper for a cleaner index route.
# def _handle_form_prediction(form_data):
#     """Handles the prediction logic for the web form."""
#     feature_values, errors = _validate_and_prepare_features(form_data)
    
#     if errors:
#         return None, " ".join(errors)

#     # Create DataFrame for prediction
#     input_df = pd.DataFrame([feature_values], columns=expected_features)
    
#     # Make prediction
#     encoded_prediction = model.predict(input_df)
#     predicted_weather = label_encoder.inverse_transform(encoded_prediction)[0]
    
#     prediction_text = f"Predicted Weather: {predicted_weather}"
#     return prediction_text, None


# # --- Flask Routes ---
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction_text = None
#     error_text = None
#     # Keep user's input in the form after submission
#     input_data = {feature: "" for feature in expected_features}

#     if request.method == 'POST':
#         input_data = request.form.to_dict()
#         try:
#             prediction_text, error_text = _handle_form_prediction(request.form)
#         except Exception as e:
#             # Feedback: Remove generic "unexpected error" messages.
#             # A traceback in the log is more useful for developers.
#             print(f"Unhandled prediction error: {e}")
#             error_text = "An internal error occurred. Please check server logs."

#     return render_template(
#         'index.html',
#         prediction_text=prediction_text,
#         error_text=error_text,
#         input_data=input_data,
#         expected_features=expected_features
#     )

# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     try:
#         data = request.get_json(force=True)
#         feature_values, errors = _validate_and_prepare_features(data, source_type='json')

#         if errors:
#             return jsonify({'error': " ".join(errors)}), 400

#         # Create DataFrame for prediction
#         input_df = pd.DataFrame([feature_values], columns=expected_features)

#         # Make prediction
#         encoded_prediction = model.predict(input_df)
#         predicted_weather = label_encoder.inverse_transform(encoded_prediction)[0]

#         return jsonify({'predicted_weather': predicted_weather})

#     except Exception as e:
#         # Feedback: f-string already calls str().
#         print(f"API Prediction error: {e}")
#         return jsonify({'error': f'Error processing request: {e}'}), 500


# # --- Run the App ---
# if __name__ == '__main__':
#     # Feedback: Defer execution of I/O operations until runtime.
#     _load_artifacts()
#     app.run(debug=True, port=5000)



# IGNORE THE CODE BELOW

# import os
# from flask import Flask, request, jsonify, render_template
# import joblib
# import pandas as pd
# import numpy as np

# # Initialize Flask app
# app = Flask(__name__)

# # --- Load Model Artifacts ---
# # Define paths to the model artifacts
# # Ensure these paths are correct relative to app.py
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_ARTIFACTS_DIR = os.path.join(BASE_DIR, 'model_artifacts')

# MODEL_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'weather_prediction_model.joblib')
# ENCODER_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'weather_label_encoder.joblib')
# FEATURES_PATH = os.path.join(MODEL_ARTIFACTS_DIR, 'classifier_feature_names.joblib')

# # Load the artifacts
# try:
#     model = joblib.load(MODEL_PATH)
#     label_encoder = joblib.load(ENCODER_PATH)
#     # These are the feature names the model was trained on (e.g., ['temp_min', 'temp_max', 'precipitation', 'wind'])
#     expected_features = joblib.load(FEATURES_PATH)
#     print(f"Model, Label Encoder, and Feature List loaded successfully from {MODEL_ARTIFACTS_DIR}")
#     print(f"Expected features for prediction: {expected_features}")
# except FileNotFoundError as e:
#     print(f"Error loading model artifacts: {e}")
#     print("Please ensure 'weather_prediction_model.joblib', 'weather_label_encoder.joblib', and 'classifier_feature_names.joblib' are in the 'model_artifacts' directory.")
#     model = None
#     label_encoder = None
#     expected_features = None
# except Exception as e:
#     print(f"An unexpected error occurred during artifact loading: {e}")
#     model = None
#     label_encoder = None
#     expected_features = None


# # --- Flask Routes ---
# @app.route('/', methods=['GET', 'POST'])
# def index():
#     prediction_text = None
#     error_text = None
#     input_data_for_template = {feature: "" for feature in expected_features} if expected_features else {}

#     if request.method == 'POST':
#         if not model or not label_encoder or not expected_features:
#             error_text = "Model artifacts not loaded. Cannot make a prediction."
#             return render_template('index.html', prediction_text=prediction_text, error_text=error_text, input_data=input_data_for_template)

#         try:
#             # Get data from form
#             form_data = request.form.to_dict()
#             input_data_for_template = form_data.copy() # Store for re-populating form

#             # Validate and prepare features
#             feature_values = []
#             missing_features = []
#             type_errors = []

#             for feature_name in expected_features:
#                 if feature_name not in form_data or form_data[feature_name] == '':
#                     missing_features.append(feature_name)
#                     continue
#                 try:
#                     feature_values.append(float(form_data[feature_name]))
#                 except ValueError:
#                     type_errors.append(f"Feature '{feature_name}' must be a number.")

#             if missing_features:
#                 error_text = f"Missing input for: {', '.join(missing_features)}."
#             elif type_errors:
#                 error_text = " ".join(type_errors)
#             else:
#                 # Create DataFrame for prediction (model expects 2D array)
#                 input_df = pd.DataFrame([feature_values], columns=expected_features)

#                 # Make prediction
#                 encoded_prediction = model.predict(input_df)
#                 predicted_weather_category = label_encoder.inverse_transform(encoded_prediction)

#                 prediction_text = f"Predicted Weather: {predicted_weather_category[0]}"

#         except Exception as e:
#             error_text = f"Error during prediction: {str(e)}"
#             print(f"Prediction error: {e}")


#     return render_template('index.html', prediction_text=prediction_text, error_text=error_text, input_data=input_data_for_template, expected_features=expected_features)


# @app.route('/api/predict', methods=['POST'])
# def api_predict():
#     if not model or not label_encoder or not expected_features:
#         return jsonify({'error': 'Model artifacts not loaded. Cannot make a prediction.'}), 500

#     try:
#         data = request.get_json(force=True) # Get data posted as JSON

#         # Validate and prepare features
#         feature_values = []
#         missing_features = []
#         type_errors = []

#         for feature_name in expected_features:
#             if feature_name not in data:
#                 missing_features.append(feature_name)
#                 continue
#             try:
#                 feature_values.append(float(data[feature_name]))
#             except (TypeError, ValueError): # Handles if value is not a number or None
#                 type_errors.append(f"Feature '{feature_name}' must be a valid number.")


#         if missing_features:
#             return jsonify({'error': f"Missing features in JSON payload: {', '.join(missing_features)}"}), 400
#         if type_errors:
#             return jsonify({'error': " ".join(type_errors)}), 400


#         # Create DataFrame for prediction
#         input_df = pd.DataFrame([feature_values], columns=expected_features)

#         # Make prediction
#         encoded_prediction = model.predict(input_df)
#         predicted_weather_category = label_encoder.inverse_transform(encoded_prediction)

#         return jsonify({'predicted_weather': predicted_weather_category[0]})

#     except Exception as e:
#         print(f"API Prediction error: {e}")
#         return jsonify({'error': f'Error processing request: {str(e)}'}), 500


# # --- Run the App ---
# if __name__ == '__main__':
#     # Check if artifacts loaded correctly before trying to run
#     if model and label_encoder and expected_features:
#         app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000/
#     else:
#         print("Application cannot start due to missing model artifacts. Please check the 'model_artifacts' directory and error messages.")