"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle


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
    """
    Parse input data matching inference_schema.json parameters.

    Request format: {"param1": value1, "param2": value2}
    Returns: {"param1": value1, "param2": value2}
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    features = json.loads(request_body)

    # Ensure dict format {param_name: value}
    if not isinstance(features, dict):
        raise ValueError(f"Expected dict with parameter names, got {type(features).__name__}")

    print(f"Parsed parameters: {features}")
    return features


def predict_fn(input_data, model):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement your prediction logic

    predictions = None

    print(f"Predictions: {predictions}")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        #TODO: Modify the output format of the predictions as needed
        
        return json.dumps({
            predictions
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
