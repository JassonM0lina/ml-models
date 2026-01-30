"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def model_fn(model_dir):
    """Load the model from the model directory"""
    import statsmodels.tsa.arima.model

    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print("Model loaded successfully")
    return model


def input_fn(request_body, request_content_type):
    """
    Parse input data matching inference_schema.json parameters.

    Request format: {"forecast_days": 30, "start_date": "2024-01-01", "confidence_level": 0.95}
    Returns: {"forecast_days": 30, "start_date": "2024-01-01", "confidence_level": 0.95}
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    features = json.loads(request_body)

    # Ensure dict format {param_name: value}
    if not isinstance(features, dict):
        raise ValueError(f"Expected dict with parameter names, got {type(features).__name__}")

    # Set defaults for optional parameters
    if "forecast_days" not in features:
        features["forecast_days"] = 30
    if "confidence_level" not in features:
        features["confidence_level"] = 0.95
    if "start_date" not in features:
        features["start_date"] = datetime.now().strftime("%Y-%m-%d")

    # Validate inputs
    if not isinstance(features["forecast_days"], int) or features["forecast_days"] <= 0:
        raise ValueError("forecast_days must be a positive integer")
    
    if not (0.5 <= features["confidence_level"] <= 0.99):
        raise ValueError("confidence_level must be between 0.5 and 0.99")

    try:
        datetime.strptime(features["start_date"], "%Y-%m-%d")
    except ValueError:
        raise ValueError("start_date must be in format YYYY-MM-DD")

    print(f"Parsed parameters: {features}")
    return features


def predict_fn(input_data, model):
    """Generate oil price forecasts"""
    print(f"Input data: {input_data}")

    try:
        forecast_days = input_data["forecast_days"]
        start_date = input_data["start_date"]
        confidence_level = input_data["confidence_level"]
        
        print(f"Generating {forecast_days}-day oil price forecast starting from {start_date}")
        
        # Generate point forecasts
        forecast = model.forecast(steps=forecast_days)
        
        # Generate confidence intervals
        forecast_result = model.get_forecast(steps=forecast_days)
        confidence_intervals = forecast_result.conf_int(alpha=1-confidence_level)
        
        # Create date range for the forecast
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        forecast_dates = [
            (start_dt + timedelta(days=i)).strftime("%Y-%m-%d") 
            for i in range(forecast_days)
        ]
        
        # Calculate some summary statistics
        forecast_mean = float(forecast.mean())
        forecast_min = float(forecast.min())
        forecast_max = float(forecast.max())
        forecast_std = float(forecast.std())
        
        # Price trend analysis
        if forecast_days > 1:
            trend_direction = "up" if forecast[-1] > forecast[0] else "down"
            trend_magnitude = abs(forecast[-1] - forecast[0])
        else:
            trend_direction = "neutral"
            trend_magnitude = 0.0
        
        # Volatility estimate (standard deviation of forecasted prices)
        volatility = forecast_std
        
        predictions = {
            "forecast": {
                "dates": forecast_dates,
                "prices": forecast.tolist(),
                "confidence_intervals": {
                    "lower": confidence_intervals.iloc[:, 0].tolist(),
                    "upper": confidence_intervals.iloc[:, 1].tolist(),
                    "confidence_level": confidence_level
                }
            },
            "summary": {
                "forecast_period_days": forecast_days,
                "start_date": start_date,
                "end_date": forecast_dates[-1],
                "mean_price": round(forecast_mean, 2),
                "min_price": round(forecast_min, 2),
                "max_price": round(forecast_max, 2),
                "price_range": round(forecast_max - forecast_min, 2),
                "volatility": round(volatility, 2),
                "trend": {
                    "direction": trend_direction,
                    "magnitude": round(float(trend_magnitude), 2)
                }
            },
            "market_insights": {
                "price_outlook": _get_price_outlook(forecast_mean, forecast_std),
                "volatility_level": _get_volatility_level(volatility),
                "confidence_note": f"Forecasts provided with {int(confidence_level*100)}% confidence intervals"
            },
            "model_info": {
                "model_type": "ARIMA",
                "forecast_horizon": f"{forecast_days} days",
                "generated_at": datetime.now().isoformat()
            }
        }
        
        print(f"Forecast generated successfully: mean=${forecast_mean:.2f}, volatility=${volatility:.2f}")
        return predictions
        
    except Exception as e:
        print(f"Error generating forecast: {e}")
        return {
            "error": f"Failed to generate forecast: {str(e)}",
            "forecast": None,
            "summary": None
        }


def _get_price_outlook(mean_price, volatility):
    """Generate qualitative price outlook based on mean and volatility"""
    if mean_price > 90:
        price_level = "high"
    elif mean_price > 70:
        price_level = "moderate"
    else:
        price_level = "low"
    
    if volatility > 5:
        volatility_desc = "high volatility"
    elif volatility > 2:
        volatility_desc = "moderate volatility"
    else:
        volatility_desc = "low volatility"
    
    return f"Expecting {price_level} oil prices with {volatility_desc}"


def _get_volatility_level(volatility):
    """Categorize volatility level"""
    if volatility > 5:
        return "High"
    elif volatility > 2:
        return "Moderate"
    else:
        return "Low"


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(predictions, indent=2)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")