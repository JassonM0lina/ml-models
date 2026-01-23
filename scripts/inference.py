"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle
import numpy as np

def model_fn(model_dir):
    """Load the model from the model directory"""
    # TODO: Add your model-specific imports here if needed
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler

    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_dict = pickle.load(f)

    print("Model loaded successfully")
    print(f"Model type: {type(model_dict['model'])}")
    print(f"Feature columns: {model_dict['feature_columns']}")
    return model_dict


def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Content type: {request_content_type}")
    print(f"Request body: {request_body}")

    if request_content_type == "application/json":
        data = json.loads(request_body)
        # TODO: Parse your expected input format
        # Expected format: {"features": {"bedrooms": 3, "bathrooms": 2, "sqft": 1500, "year_built": 2020, ...}}
        if "features" in data:
            return data["features"]
        else:
            # Handle direct feature input
            return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_dict):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement your prediction logic
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_columns = model_dict['feature_columns']
    
    # Extract features in the correct order - matches inference schema exactly
    expected_features = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lot_size', 'garage_spaces']
    
    features = []
    for col in expected_features:
        if col in input_data:
            features.append(input_data[col])
        else:
            raise ValueError(f"Missing required feature: {col}")
    
    # Convert to numpy array and reshape for single prediction
    features_array = np.array(features).reshape(1, -1)
    
    # Apply scaling if used during training
    if scaler:
        features_array = scaler.transform(features_array)
    
    # Make prediction
    prediction = model.predict(features_array)
    
    # Convert to Python float for JSON serialization
    prediction_value = float(prediction[0])

    print(f"Features: {features}")
    print(f"Prediction: {prediction_value}")
    return prediction_value


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            "predicted_price": predictions,
            "currency": "USD"
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")