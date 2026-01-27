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
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer

    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_obj = pickle.load(f)

    print("Model loaded successfully")
    return model_obj


def input_fn(request_body, request_content_type):
    """Parse input data"""
    print(f"Content type: {request_content_type}")
    print(f"Request body: {request_body}")

    if request_content_type == "application/json":
        data = json.loads(request_body)
        # TODO: Parse input data
        # Expected format: {"features": {"year": 2018, "mileage": 45000, "engine_size": 2.0, "fuel_type": "Gasoline", ...}}
        features = data.get("features", {})
        return features
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model_obj):
    """Generate predictions"""
    print(f"Input data: {input_data}")

    # TODO: Implement prediction logic here
    # Extract model components
    model = model_obj['model']
    preprocessor = model_obj['preprocessor']
    feature_cols = model_obj['feature_cols']
    
    # Convert input to DataFrame format expected by preprocessor
    import pandas as pd
    
    # Create DataFrame from input data with expected columns
    input_df = pd.DataFrame([input_data])
    
    # Handle missing columns by filling with defaults
    for col in feature_cols:
        if col not in input_df.columns:
            # Set reasonable defaults based on column name
            if col == 'age':
                input_df[col] = 5  # Default car age
            elif col == 'year':
                input_df[col] = 2019  # Default year
            elif col == 'mileage':
                input_df[col] = 50000  # Default mileage
            elif col == 'engine_size':
                input_df[col] = 2.0  # Default engine size
            elif 'fuel' in col.lower():
                input_df[col] = 'Gasoline'  # Default fuel type
            elif 'transmission' in col.lower():
                input_df[col] = 'Manual'  # Default transmission
            else:
                input_df[col] = 0  # Default numeric value
    
    # Convert year to age if both exist
    if 'year' in input_df.columns and 'age' not in input_df.columns:
        current_year = 2024
        input_df['age'] = current_year - input_df['year']
    
    # Reorder columns to match training data
    input_df = input_df[feature_cols]
    
    # Preprocess the input
    input_processed = preprocessor.transform(input_df)
    
    # Generate prediction
    prediction = model.predict(input_processed)
    
    predictions = prediction.tolist()

    print(f"Predictions: {predictions}")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            "predicted_price": predictions[0] if predictions else 0,
            "predictions": predictions,
            "count": len(predictions) if isinstance(predictions, list) else 1
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")