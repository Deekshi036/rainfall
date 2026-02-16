"""
API Testing Script - Test Your Rainfall Prediction API
Matches your exact model features
"""

import requests
import json

BASE_URL = 'http://localhost:5000'

def test_health_check():
    """Test the health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    try:
        response = requests.get(f'{BASE_URL}/health')
        print(f"Status Code: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get('model_loaded'):
            print(f"\n✓ Model loaded successfully: {result.get('model_type')}")
        else:
            print("\n✗ Model not loaded!")
        
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("\n❌ Cannot connect to server. Is the Flask app running?")
        return False

def test_prediction(data, test_name):
    """Test the prediction API endpoint"""
    print("\n" + "="*70)
    print(f"Testing: {test_name}")
    print("="*70)
    
    print("\nRequest Data:")
    print(json.dumps(data, indent=2))
    
    try:
        response = requests.post(
            f'{BASE_URL}/api/predict',
            json=data,
            headers={'Content-Type': 'application/json'}
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print("\nResponse:")
        result = response.json()
        print(json.dumps(result, indent=2))
        
        if result.get('success'):
            pred = result.get('prediction', {})
            print(f"\n🎯 Prediction: {pred.get('prediction_label')}")
            print(f"☔ Rain Tomorrow: {pred.get('rain_tomorrow')}")
            if pred.get('confidence_percent'):
                print(f"📊 Confidence: {pred.get('confidence_percent')}%")
        
        return response
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return None

def run_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("RAINFALL PREDICTION API TEST SUITE")
    print("="*70)
    
    # Test 1: Health Check
    health_ok = test_health_check()
    if not health_ok:
        print("\n❌ Health check failed! Make sure:")
        print("   1. Flask app is running: python app.py")
        print("   2. All pickle files exist in the app folder")
        return
    
    # Test 2: Valid prediction - Sample from your data
    test_prediction({
        "MinTemp": 13.4,
        "MaxTemp": 26.9,
        "Rainfall": 0.6,
        "WindGustSpeed": 44.0,
        "WindSpeed9am": 20.0,
        "WindSpeed3pm": 24.0,
        "Humidity9am": 71.0,
        "Humidity3pm": 48.0,
        "Pressure9am": 1007.7,
        "Pressure3pm": 1007.1,
        "Temp9am": 16.9,
        "Temp3pm": 21.8
    }, "Sample Weather Data (Normal Day)")
    
    # Test 3: Rainy conditions
    test_prediction({
        "MinTemp": 18.0,
        "MaxTemp": 24.0,
        "Rainfall": 10.5,
        "WindGustSpeed": 55.0,
        "WindSpeed9am": 28.0,
        "WindSpeed3pm": 32.0,
        "Humidity9am": 92.0,
        "Humidity3pm": 85.0,
        "Pressure9am": 1005.0,
        "Pressure3pm": 1003.5,
        "Temp9am": 19.5,
        "Temp3pm": 22.0
    }, "Heavy Rain Conditions")
    
    # Test 4: Dry conditions
    test_prediction({
        "MinTemp": 15.0,
        "MaxTemp": 32.0,
        "Rainfall": 0.0,
        "WindGustSpeed": 25.0,
        "WindSpeed9am": 10.0,
        "WindSpeed3pm": 15.0,
        "Humidity9am": 45.0,
        "Humidity3pm": 30.0,
        "Pressure9am": 1020.0,
        "Pressure3pm": 1018.5,
        "Temp9am": 20.0,
        "Temp3pm": 28.0
    }, "Dry Weather (No Rain Expected)")
    
    # Test 5: Missing field (error expected)
    test_prediction({
        "MinTemp": 13.4,
        "MaxTemp": 26.9,
        # Missing other fields
    }, "Missing Fields (Error Expected)")
    
    # Test 6: Invalid value type (error expected)
    test_prediction({
        "MinTemp": "invalid",
        "MaxTemp": 26.9,
        "Rainfall": 0.6,
        "WindGustSpeed": 44.0,
        "WindSpeed9am": 20.0,
        "WindSpeed3pm": 24.0,
        "Humidity9am": 71.0,
        "Humidity3pm": 48.0,
        "Pressure9am": 1007.7,
        "Pressure3pm": 1007.1,
        "Temp9am": 16.9,
        "Temp3pm": 21.8
    }, "Invalid Value Type (Error Expected)")
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETED")
    print("="*70)
    print("\n✅ All tests finished!")
    print("\nTo use the web interface, open: http://localhost:5000")

if __name__ == "__main__":
    try:
        run_tests()
    except KeyboardInterrupt:
        print("\n\n⚠️  Tests interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {str(e)}")
