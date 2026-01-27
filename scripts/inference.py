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
    # Example: import statsmodels.tsa.arima.model

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
        # TODO: Parse your expected input format
        # Example: {"steps": 12} for forecasting
        # Example: {"features": {...}} for regression
        
        # Extract features from the request
        features = data.get("features", {})
        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement your prediction logic
    # Example for time series:
    # steps = input_data.get("steps", 1)
    # predictions = model.forecast(steps=steps)
    #
    # Example for regression:
    # features = input_data.get("features")
    # predictions = model.predict(features)
    
    # Extract features for linear regression prediction
    day_of_year = input_data.get("day_of_year", 180)  # Default to mid-year
    month = input_data.get("month", 6)  # Default to June
    year = input_data.get("year", 2024)  # Default year
    temp_lag1 = input_data.get("temp_lag1", 70.0)  # Default previous day temp
    temp_lag2 = input_data.get("temp_lag2", 68.0)  # Default temp 2 days ago
    
    # FIXED: Create feature array in EXACT same order as training
    # Order: ['day_of_year', 'month', 'year', 'temp_lag1', 'temp_lag2']
    features = np.array([[day_of_year, month, year, temp_lag1, temp_lag2]])
    
    # Generate prediction
    predictions = model.predict(features)

    # Convert to list for JSON serialization
    if hasattr(predictions, 'tolist'):
        predictions = predictions.tolist()

    print(f"Predictions: {predictions}")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            "predictions": predictions,
            "count": len(predictions) if isinstance(predictions, list) else 1
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")