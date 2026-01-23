"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle

def model_fn(model_dir):
    """Load the model from the model directory"""
    # TODO: Add your model-specific imports here if needed
    from sklearn.linear_model import LinearRegression
    
    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Content type: {request_content_type}")
    print(f"Request body: {request_body}")

    if request_content_type == "application/json":
        data = json.loads(request_body)
        # TODO: Parse input data
        # Expected format: {"features": {"distance_km": 5.2, "stations": 3, "peak_hours": 1, ...}}
        features = data.get("features", {})
        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement prediction logic here
    # Extract features for metro price prediction
    distance_km = input_data.get("distance_km", 0)
    stations = input_data.get("stations", 0)
    peak_hours = input_data.get("peak_hours", 0)
    day_of_week = input_data.get("day_of_week", 1)
    month = input_data.get("month", 1)
    
    # Create feature array in the same order as training
    # Assuming the model was trained with these features in this order
    features = [
        distance_km,
        stations, 
        peak_hours,
        day_of_week,
        month,
        input_data.get("year", 2024),
        input_data.get("day", 1),
        input_data.get("day_of_year", 1)
    ]
    
    # Add derived features if they were used in training
    if distance_km > 0:
        features.extend([
            input_data.get("price_per_km", 0),  # Will be calculated during training
            input_data.get("price_per_station", 0)  # Will be calculated during training
        ])
    
    # Make prediction
    import numpy as np
    feature_array = np.array(features).reshape(1, -1)
    predictions = model.predict(feature_array)

    # Convert to list for JSON serialization
    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()

    print(f"Predictions: {predictions}")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            "predicted_price": predictions[0] if isinstance(predictions, list) and len(predictions) > 0 else predictions,
            "currency": "COP",
            "predictions": predictions,
            "count": len(predictions) if isinstance(predictions, list) else 1
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")