"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings
warnings.filterwarnings('ignore')


def model_fn(model_dir):
    """Load the model from the model directory"""

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
    # Ensure dict format {param_name: value}
    if not isinstance(features, dict):
        raise ValueError(f"Expected dict with parameter names, got {type(features).__name__}")

    return features


    if not isinstance(features, dict):
        raise ValueError(f"Expected dict with parameter names, got {type(features).__name__}")
    print(f"Input data: {input_data}")

    # Extract parameters
    forecast_steps = input_data.get('forecast_steps', 12)
    start_date = input_data.get('start_date', '2024-01-01')
    confidence_level = input_data.get('confidence_level', 95)
    include_historical = input_data.get('include_historical', False)
    
    # Convert start_date to datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    
    print(f"Generating {forecast_steps} forecasts from {start_date}")
    
    try:
        if isinstance(model, dict):
            if model.get('type') == 'seasonal_naive':
                predictions = generate_seasonal_naive_forecast(
                    model, forecast_steps, start_date, confidence_level, include_historical
                )
            else:
                # This is a SARIMA model dict
                predictions = generate_sarima_forecast(
                    model, forecast_steps, start_date, confidence_level, include_historical
                )
        else:
            # Legacy: assume it's a fitted SARIMA model
            predictions = generate_sarima_forecast_legacy(
                model, forecast_steps, start_date, confidence_level, include_historical
            )
            
    except Exception as e:
        print(f"Forecast generation failed: {str(e)}")
        # Return fallback response
        predictions = {
            'error': str(e),
            'model_type': 'error',
            'forecast_data': []
        }
    
    return predictions


def generate_seasonal_naive_forecast(model, forecast_steps, start_date, confidence_level, include_historical):
    """Generate forecasts using seasonal naive approach"""
    train_data = model['train_data']
    seasonal_periods = model['seasonal_periods']
    metadata = model.get('metadata', {})
    
    # Generate forecasts
    forecasts = []
    dates = []
    
    for i in range(forecast_steps):
        # Get forecast date
        forecast_date = start_date + relativedelta(months=i)
        dates.append(forecast_date)
        
        # Seasonal naive: use same period from previous seasonal cycle
        seasonal_idx = len(train_data) - seasonal_periods + (i % seasonal_periods)
        if seasonal_idx >= 0:
            forecast_value = train_data.iloc[seasonal_idx]
        else:
            forecast_value = train_data.mean()
            
        forecasts.append(forecast_value)
    
    # Simple confidence intervals (±10% of forecast)
    alpha = (100 - confidence_level) / 100
    confidence_factor = 1.96 * 0.1  # Approximate 95% CI
    
    forecast_data = []
    for i, (date, forecast) in enumerate(zip(dates, forecasts)):
        lower_bound = forecast * (1 - confidence_factor)
        upper_bound = forecast * (1 + confidence_factor)
        
        forecast_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'forecast': round(float(forecast), 2),
            'lower_bound': round(float(lower_bound), 2),
            'upper_bound': round(float(upper_bound), 2),
            'period': i + 1
        })
    
    return build_forecast_response('Seasonal Naive', model, forecast_data, confidence_level, include_historical, metadata)


def generate_sarima_forecast(model, forecast_steps, start_date, confidence_level, include_historical):
    """Generate forecasts using SARIMA model"""
    fitted_model = model['fitted_model']
    metadata = model.get('metadata', {})
    
    # Generate forecasts with confidence intervals
    alpha = (100 - confidence_level) / 100
    forecast_result = fitted_model.forecast(steps=forecast_steps, alpha=alpha)
    conf_int = fitted_model.get_forecast(steps=forecast_steps, alpha=alpha).conf_int()
    
    # Create forecast dates
    dates = []
    for i in range(forecast_steps):
        forecast_date = start_date + relativedelta(months=i)
        dates.append(forecast_date)
    
    forecast_data = []
    for i, (date, forecast) in enumerate(zip(dates, forecast_result)):
        forecast_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'forecast': round(float(forecast), 2),
            'lower_bound': round(float(conf_int.iloc[i, 0]), 2),
            'upper_bound': round(float(conf_int.iloc[i, 1]), 2),
            'period': i + 1
        })
    
    return build_forecast_response('SARIMA', model, forecast_data, confidence_level, include_historical, metadata)


def generate_sarima_forecast_legacy(model, forecast_steps, start_date, confidence_level, include_historical):
    """Generate forecasts for legacy SARIMA model format"""
    alpha = (100 - confidence_level) / 100
    forecast_result = model.forecast(steps=forecast_steps, alpha=alpha)
    conf_int = model.get_forecast(steps=forecast_steps, alpha=alpha).conf_int()
    
    dates = []
    for i in range(forecast_steps):
        forecast_date = start_date + relativedelta(months=i)
        dates.append(forecast_date)
    
    forecast_data = []
    for i, (date, forecast) in enumerate(zip(dates, forecast_result)):
        forecast_data.append({
            'date': date.strftime('%Y-%m-%d'),
            'forecast': round(float(forecast), 2),
            'lower_bound': round(float(conf_int.iloc[i, 0]), 2),
            'upper_bound': round(float(conf_int.iloc[i, 1]), 2),
            'period': i + 1
        })
    
    return build_forecast_response('SARIMA', {}, forecast_data, confidence_level, include_historical, {})


def build_forecast_response(model_type, model, forecast_data, confidence_level, include_historical, metadata):
    """Build standardized forecast response"""
    response = {
        'model_type': model_type,
        'model_parameters': model.get('order', 'N/A') if model_type == 'SARIMA' else 'Seasonal Naive',
        'forecast_summary': {
            'num_periods': len(forecast_data),
            'start_date': forecast_data[0]['date'] if forecast_data else None,
            'confidence_level': confidence_level,
            'mean_forecast': round(np.mean([f['forecast'] for f in forecast_data]), 2),
            'std_forecast': round(np.std([f['forecast'] for f in forecast_data]), 2),
            'min_forecast': round(min([f['forecast'] for f in forecast_data]), 2),
            'max_forecast': round(max([f['forecast'] for f in forecast_data]), 2),
            'units': 'thousand_barrels_per_day'
        },
        'forecast_data': forecast_data
    }
    
    # Add historical data if requested
    if include_historical and 'train_data' in model:
        train_data = model['train_data']
        # Get last 24 periods for context
        historical_periods = min(24, len(train_data))
        recent_data = train_data.tail(historical_periods)
        
        historical_data = []
        for date, value in recent_data.items():
            historical_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'value': round(float(value), 2),
                'type': 'historical'
            })
        
        response['historical_data'] = historical_data

    return response


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps({
            **predictions
        })
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
        raise ValueError(f"Unsupported response type: {response_content_type}")
def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        # Return properly formatted JSON response
        return json.dumps(predictions, indent=2)
    else:
