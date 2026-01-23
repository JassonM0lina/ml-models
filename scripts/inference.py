"""
Inference Script for ARIMA Model on SageMaker Serverless Endpoint
Dependencies installed via requirements.txt in code/ directory
"""

import os
import json
import pickle


def model_fn(model_dir):
    """Load the ARIMA model from the model directory"""
    # Import statsmodels here to ensure requirements.txt has been installed
    # This is needed for unpickling the ARIMA model
    import statsmodels.tsa.arima.model
    print(f"statsmodels imported successfully")

    model_path = os.path.join(model_dir, "arima_model.pkl")
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
        # Expect {"steps": N} where N is number of forecast steps
        steps = data.get("steps", 1)
        return {"steps": steps}
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """Generate forecast predictions"""
    steps = input_data.get("steps", 1)
    print(f"Forecasting {steps} steps ahead")

    # Generate forecast
    forecast = model.forecast(steps=steps)

    # Convert to list for JSON serialization
    if hasattr(forecast, 'tolist'):
        predictions = forecast.tolist()
    else:
        predictions = list(forecast)

    print(f"Predictions: {predictions}")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            "predictions": predictions,
            "steps": len(predictions)
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
