"""
Rainfall Prediction Flask Application - Customized for Your Model
Matches your exact training code and feature set
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'rainfall.pkl'  # Your trained model
SCALER_PATH = 'scale.pkl'     # Your StandardScaler
ENCODER_PATH = 'encoder.pkl'  # Your LabelEncoder
IMPUTER_PATH = 'impter.pkl'   # Your SimpleImputer

# Load trained model and preprocessing objects
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    encoder = pickle.load(open(ENCODER_PATH, 'rb'))
    imputer = pickle.load(open(IMPUTER_PATH, 'rb'))
    print("✓ Model and preprocessing objects loaded successfully")
    print(f"✓ Model type: {type(model).__name__}")
except Exception as e:
    print(f"✗ Error loading files: {e}")
    print("⚠️  Make sure these files are in the same directory:")
    print("   - rainfall.pkl")
    print("   - scale.pkl")
    print("   - encoder.pkl")
    print("   - impter.pkl")
    model = None
    scaler = None
    encoder = None
    imputer = None

# Feature names - EXACT order from your training code (numeric features only)
FEATURE_NAMES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'
]


@app.route('/')
def home():
    """Render the main prediction form"""
    return render_template('index.html', feature_names=FEATURE_NAMES)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from web form
    Matches your exact preprocessing pipeline
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check server logs.",
                                 feature_names=FEATURE_NAMES)
        
        # Extract features from form
        features = []
        form_data = {}
        
        for feature in FEATURE_NAMES:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html',
                                     error=f"Missing value for {feature}",
                                     feature_names=FEATURE_NAMES)
            try:
                float_value = float(value)
                features.append(float_value)
                form_data[feature] = float_value
            except ValueError:
                return render_template('index.html',
                                     error=f"Invalid value for {feature}. Please enter a number.",
                                     feature_names=FEATURE_NAMES)
        
        # Convert to DataFrame (matching your training format)
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Scale features (matching your preprocessing)
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability (if model supports it)
        try:
            prediction_proba = model.predict_proba(features_scaled)[0]
            confidence = max(prediction_proba) * 100
            prob_no_rain = prediction_proba[0] * 100
            prob_rain = prediction_proba[1] * 100
        except:
            # Some models don't support predict_proba
            confidence = None
            prob_no_rain = None
            prob_rain = None
        
        # Decode prediction (0 = No, 1 = Yes)
        prediction_label = "Yes" if prediction == 1 else "No"
        
        # Prepare result
        result = {
            'prediction': f'{prediction_label} - {"Rain Expected Tomorrow! ☔" if prediction == 1 else "No Rain Expected Tomorrow ☀️"}',
            'prediction_class': int(prediction),
            'confidence': round(confidence, 2) if confidence else None,
            'prob_rain': round(prob_rain, 2) if prob_rain else None,
            'prob_no_rain': round(prob_no_rain, 2) if prob_no_rain else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'input_data': form_data
        }
        
        return render_template('index.html',
                             result=result,
                             feature_names=FEATURE_NAMES,
                             form_data=form_data)
    
    except Exception as e:
        return render_template('index.html',
                             error=f"Prediction error: {str(e)}",
                             feature_names=FEATURE_NAMES)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    Accepts JSON input and returns JSON response
    
    Example request:
    {
        "MinTemp": 15.5,
        "MaxTemp": 28.3,
        "Rainfall": 0.0,
        "WindGustSpeed": 44.0,
        "WindSpeed9am": 13.0,
        "WindSpeed3pm": 20.0,
        "Humidity9am": 71.0,
        "Humidity3pm": 48.0,
        "Pressure9am": 1017.6,
        "Pressure3pm": 1015.8,
        "Temp9am": 18.5,
        "Temp3pm": 25.7
    }
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded'
            }), 500
        
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No JSON data provided'
            }), 400
        
        # Extract features
        features = []
        for feature in FEATURE_NAMES:
            if feature not in data:
                return jsonify({
                    'success': False,
                    'error': f'Missing feature: {feature}'
                }), 400
            features.append(float(data[feature]))
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features], columns=FEATURE_NAMES)
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        
        # Get prediction probability
        try:
            prediction_proba = model.predict_proba(features_scaled)[0]
            confidence = max(prediction_proba) * 100
            prob_no_rain = prediction_proba[0] * 100
            prob_rain = prediction_proba[1] * 100
        except:
            confidence = None
            prob_no_rain = None
            prob_rain = None
        
        # Prepare response
        response = {
            'success': True,
            'prediction': {
                'rain_tomorrow': bool(prediction),
                'prediction_label': 'Yes' if prediction == 1 else 'No',
                'confidence_percent': round(confidence, 2) if confidence else None,
                'probability_rain': round(prob_rain, 2) if prob_rain else None,
                'probability_no_rain': round(prob_no_rain, 2) if prob_no_rain else None
            },
            'input_features': data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except ValueError as e:
        return jsonify({
            'success': False,
            'error': f'Invalid input values: {str(e)}'
        }), 400
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'encoder_loaded': encoder is not None,
        'imputer_loaded': imputer is not None,
        'model_type': type(model).__name__ if model else None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Run Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
