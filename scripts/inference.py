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

    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Linear regression model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Content type: {request_content_type}")
    print(f"Request body: {request_body}")

    if request_content_type == "application/json":
        data = json.loads(request_body)
        # TODO: Parse your expected input format
        # Expected format: {"features": {"feature1": value1, "feature2": value2, ...}}
        # or {"features": {"values": [value1, value2, ...]}}
        
        if "features" in data:
            features = data["features"]
            return features
        else:
            # Fallback: assume the data itself contains the features
            return data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement your prediction logic
    # Handle different input formats
    if "values" in input_data:
        # Format: {"values": [feature1, feature2, ...]}
        features = np.array(input_data["values"]).reshape(1, -1)
    elif isinstance(input_data, dict) and all(isinstance(v, (int, float)) for v in input_data.values()):
        # Format: {"feature1": value1, "feature2": value2, ...}
        features = np.array(list(input_data.values())).reshape(1, -1)
    elif "features" in input_data:
        # Nested features format
        if isinstance(input_data["features"], list):
            features = np.array(input_data["features"]).reshape(1, -1)
        else:
            features = np.array(list(input_data["features"].values())).reshape(1, -1)
    else:
        # Single value or simple format
        if isinstance(input_data, (int, float)):
            features = np.array([[input_data]])
        elif isinstance(input_data, list):
            features = np.array(input_data).reshape(1, -1)
        else:
            # Try to extract numerical values
            values = []
            for key, value in input_data.items():
                if isinstance(value, (int, float)):
                    values.append(value)
            if values:
                features = np.array(values).reshape(1, -1)
            else:
                raise ValueError(f"Unable to parse features from input: {input_data}")
    
    # Make prediction
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