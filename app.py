"""
Rainfall Prediction Flask Application - Location-Based
For Agricultural Decision Making in India
"""

from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import json
import os

# Initialize Flask app
app = Flask(__name__)

# Configuration
MODEL_PATH = 'rainfall.pkl'
SCALER_PATH = 'scale.pkl'
LOCATION_ENCODER_PATH = 'location_encoder.pkl'
TARGET_ENCODER_PATH = 'encoder.pkl'
IMPUTER_PATH = 'impter.pkl'

# Load trained model and preprocessing objects
try:
    model = pickle.load(open(MODEL_PATH, 'rb'))
    scaler = pickle.load(open(SCALER_PATH, 'rb'))
    location_encoder = pickle.load(open(LOCATION_ENCODER_PATH, 'rb'))
    target_encoder = pickle.load(open(TARGET_ENCODER_PATH, 'rb'))
    imputer = pickle.load(open(IMPUTER_PATH, 'rb'))
    
    # Get list of locations from encoder
    locations = location_encoder.classes_.tolist()
    locations.sort()  # Alphabetical order
    
    print("=" * 60)
    print("✓ Model and preprocessing objects loaded successfully")
    print(f"✓ Model type: {type(model).__name__}")
    print(f"✓ Available locations: {len(locations)}")
    print(f"✓ Sample locations: {locations[:10]}")
    print("=" * 60)
    
except Exception as e:
    print(f"✗ Error loading files: {e}")
    print("\n⚠️  Make sure these files exist:")
    print("   - rainfall.pkl")
    print("   - scale.pkl")
    print("   - location_encoder.pkl")
    print("   - encoder.pkl")
    print("   - impter.pkl")
    model = None
    scaler = None
    location_encoder = None
    target_encoder = None
    imputer = None
    locations = []

# Weather feature names (in order, excluding location)
WEATHER_FEATURES = [
    'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
    'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
    'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm'
]

# All features including location (order matters!)
ALL_FEATURES = WEATHER_FEATURES + ['Location_Encoded']


@app.route('/')
def home():
    """Render the main prediction form"""
    return render_template('index.html', locations=locations)


@app.route('/predict-simple', methods=['POST'])
def predict_simple():
    """
    Simple prediction using location only
    For farmers who don't have detailed weather data
    Uses average weather patterns for the location
    """
    try:
        if model is None or scaler is None or location_encoder is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check server logs.",
                                 locations=locations)
        
        # Get selected location
        location = request.form.get('Location')
        if not location:
            return render_template('index.html',
                                 error="Please select a location",
                                 locations=locations)
        
        # Encode location
        try:
            location_encoded = location_encoder.transform([location])[0]
        except ValueError:
            return render_template('index.html',
                                 error=f"Location '{location}' not found in training data",
                                 locations=locations)
        
        # Use typical/average weather values for the location
        # These are reasonable defaults based on typical Indian weather
        # You can customize these based on your location data
        typical_weather = {
            'MinTemp': 20.0,
            'MaxTemp': 32.0,
            'Rainfall': 0.5,
            'WindGustSpeed': 30.0,
            'WindSpeed9am': 15.0,
            'WindSpeed3pm': 20.0,
            'Humidity9am': 70.0,
            'Humidity3pm': 50.0,
            'Pressure9am': 1013.0,
            'Pressure3pm': 1011.0,
            'Temp9am': 24.0,
            'Temp3pm': 30.0
        }
        
        # Build features list
        features = []
        for feature in WEATHER_FEATURES:
            features.append(typical_weather[feature])
        
        # Add encoded location
        features.append(location_encoded)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features], columns=ALL_FEATURES)
        
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
        
        # Prepare result
        prediction_label = "Yes" if prediction == 1 else "No"
        result = {
            'prediction': f'{prediction_label} - {"Rain Expected Tomorrow! ☔" if prediction == 1 else "No Rain Expected Tomorrow ☀️"}',
            'prediction_class': int(prediction),
            'confidence': round(confidence, 2) if confidence else None,
            'prob_rain': round(prob_rain, 2) if prob_rain else None,
            'prob_no_rain': round(prob_no_rain, 2) if prob_no_rain else None,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'location': location,
            'input_data': {'Location': location, 'Mode': 'Simple'}
        }
        
        return render_template('index.html',
                             result=result,
                             locations=locations)
    
    except Exception as e:
        import traceback
        print("Simple prediction error:")
        print(traceback.format_exc())
        return render_template('index.html',
                             error=f"Prediction error: {str(e)}",
                             locations=locations)


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle prediction requests from web form
    Includes location encoding for farmer-friendly interface
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None or location_encoder is None:
            return render_template('index.html', 
                                 error="Model not loaded. Please check server logs.",
                                 locations=locations)
        
        # Get selected location
        location = request.form.get('Location')
        if not location:
            return render_template('index.html',
                                 error="Please select a location",
                                 locations=locations)
        
        # Encode location
        try:
            location_encoded = location_encoder.transform([location])[0]
        except ValueError:
            return render_template('index.html',
                                 error=f"Location '{location}' not found in training data",
                                 locations=locations)
        
        # Extract weather features from form
        features = []
        form_data = {'Location': location}
        
        for feature in WEATHER_FEATURES:
            value = request.form.get(feature)
            if value is None or value == '':
                return render_template('index.html',
                                     error=f"Missing value for {feature}",
                                     locations=locations,
                                     form_data=form_data)
            try:
                float_value = float(value)
                features.append(float_value)
                form_data[feature] = float_value
            except ValueError:
                return render_template('index.html',
                                     error=f"Invalid value for {feature}. Please enter a number.",
                                     locations=locations,
                                     form_data=form_data)
        
        # Add encoded location at the end
        features.append(location_encoded)
        
        # Convert to DataFrame with correct feature order
        features_df = pd.DataFrame([features], columns=ALL_FEATURES)
        
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
            # If model doesn't support predict_proba
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
            'location': location,
            'input_data': form_data
        }
        
        return render_template('index.html',
                             result=result,
                             locations=locations,
                             form_data=form_data)
    
    except Exception as e:
        import traceback
        print("Prediction error:")
        print(traceback.format_exc())
        return render_template('index.html',
                             error=f"Prediction error: {str(e)}",
                             locations=locations)


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """
    API endpoint for predictions
    Accepts JSON with location and weather data
    
    Example request:
    {
        "Location": "Mumbai",
        "MinTemp": 22.5,
        "MaxTemp": 32.1,
        "Rainfall": 0.0,
        ...
    }
    """
    try:
        # Check if model is loaded
        if model is None or scaler is None or location_encoder is None:
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
        
        # Get and encode location
        location = data.get('Location')
        if not location:
            return jsonify({
                'success': False,
                'error': 'Location is required',
                'available_locations': locations[:20]
            }), 400
        
        try:
            location_encoded = location_encoder.transform([location])[0]
        except ValueError:
            return jsonify({
                'success': False,
                'error': f'Location "{location}" not found in training data',
                'available_locations': locations[:20]
            }), 400
        
        # Extract weather features
        features = []
        missing_features = []
        
        for feature in WEATHER_FEATURES:
            if feature not in data:
                missing_features.append(feature)
            else:
                try:
                    features.append(float(data[feature]))
                except ValueError:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid value for {feature}'
                    }), 400
        
        if missing_features:
            return jsonify({
                'success': False,
                'error': 'Missing features',
                'missing_features': missing_features
            }), 400
        
        # Add encoded location
        features.append(location_encoded)
        
        # Convert to DataFrame
        features_df = pd.DataFrame([features], columns=ALL_FEATURES)
        
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
                'probability_no_rain': round(prob_no_rain, 2) if prob_no_rain else None,
                'location': location
            },
            'input_features': data,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        import traceback
        print("API prediction error:")
        print(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f'Prediction error: {str(e)}'
        }), 500


@app.route('/locations', methods=['GET'])
def get_locations():
    """Get list of all available locations"""
    return jsonify({
        'success': True,
        'locations': locations,
        'count': len(locations)
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'location_encoder_loaded': location_encoder is not None,
        'locations_count': len(locations),
        'model_type': type(model).__name__ if model else None,
        'timestamp': datetime.now().isoformat()
    })


if __name__ == '__main__':
    # Run Flask app
    # Set debug=False in production
    app.run(debug=True, host='0.0.0.0', port=5000)