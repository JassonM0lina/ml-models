"""
Inference Script Template for SageMaker Serverless Endpoint
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def model_fn(model_dir):
    """Load the model from the model directory"""
    
    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"Model loaded successfully - Type: {model['model_type']}")
    print(f"Features: {len(model['feature_names'])}")
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
    """Generate gas price predictions for Orlando"""
    print(f"Input data: {input_data}")

    # Extract prediction parameters
    prediction_days = input_data.get('prediction_days', 7)
    crude_oil_price = input_data.get('crude_oil_price', 75.0)
    demand_level = input_data.get('demand_level', 'normal')
    supply_disruption = input_data.get('supply_disruption', False)
    current_temperature = input_data.get('current_temperature', 75.0)
    current_gas_price = input_data.get('current_gas_price', 3.20)
    
    # Convert demand level to numeric
    demand_mapping = {'low': 90, 'normal': 100, 'high': 110, 'very_high': 120}
    demand_index = demand_mapping.get(demand_level, 100)
    
    # Generate predictions for the next N days
    predictions = []
    current_date = datetime.now()
    
    for day in range(prediction_days):
        future_date = current_date + timedelta(days=day+1)
        
        # Create feature vector for prediction
        # Use current gas price as lag feature for first prediction, then use previous prediction
        gas_price_lag1 = current_gas_price if day == 0 else predictions[day-1]['predicted_price']
        gas_price_lag7 = current_gas_price  # Simplified - would need actual historical data
        
        # Seasonal factors
        month = future_date.month
        quarter = (month - 1) // 3 + 1
        day_of_year = future_date.timetuple().tm_yday
        day_of_week = future_date.weekday()
        weekend = 1 if day_of_week >= 5 else 0
        
        # Hurricane season (June-November in Florida)
        hurricane_season = 1 if 6 <= month <= 11 else 0
        
        # Holiday effect (simplified - major holidays)
        holiday_effect = 1 if (month == 7 and future_date.day == 4) or \
                            (month == 11 and 22 <= future_date.day <= 28) or \
                            (month == 12 and 20 <= future_date.day <= 31) else 0
        
        # Temperature estimation (seasonal variation)
        base_temp = 75 + 15 * np.sin(2 * np.pi * day_of_year / 365.25)
        temperature = current_temperature if day == 0 else base_temp
        
        # Create feature array matching training data
        feature_vector = np.array([
            crude_oil_price,           # crude_oil_price
            demand_index,              # demand_index
            1 if supply_disruption else 0,  # supply_disruption
            temperature,               # temperature
            hurricane_season,          # hurricane_season
            day_of_week,              # day_of_week
            weekend,                  # weekend
            holiday_effect,           # holiday_effect
            month,                    # month
            quarter,                  # quarter
            day_of_year,              # day_of_year
            gas_price_lag1,           # gas_price_lag1
            gas_price_lag7            # gas_price_lag7
        ]).reshape(1, -1)
        
        # Apply same preprocessing as training
        if model['model_type'] == 'linear':
            feature_vector_scaled = model['scaler'].transform(feature_vector)
            feature_vector_selected = model['selector'].transform(feature_vector_scaled)
        else:  # random_forest
            feature_vector_selected = model['selector'].transform(feature_vector)
        
        # Make prediction
        predicted_price = model['model'].predict(feature_vector_selected)[0]
        
        # Ensure reasonable bounds
        predicted_price = max(2.5, min(5.0, predicted_price))
        
        # Calculate confidence based on model type and market conditions
        base_confidence = 0.85 if model['model_type'] == 'random_forest' else 0.80
        
        # Reduce confidence for longer forecasts and market uncertainties
        confidence = base_confidence * (0.98 ** day)  # Decay confidence over time
        if supply_disruption:
            confidence *= 0.90  # Less confident during disruptions
        if demand_level in ['high', 'very_high']:
            confidence *= 0.92  # Less confident during high demand
        
        predictions.append({
            'date': future_date.strftime('%Y-%m-%d'),
            'day_offset': day + 1,
            'predicted_price': round(predicted_price, 3),
            'confidence': round(confidence, 2),
            'factors': {
                'crude_oil_impact': round((crude_oil_price - 70) * 0.02, 3),
                'demand_impact': round((demand_index - 100) * 0.01, 3),
                'supply_disruption': supply_disruption,
                'seasonal_factor': 'summer' if 6 <= month <= 8 else 'winter' if month in [12, 1, 2] else 'transitional',
                'weekend': bool(weekend),
                'hurricane_season': bool(hurricane_season)
            }
        })
    
    # Calculate summary statistics
    predicted_prices = [p['predicted_price'] for p in predictions]
    price_trend = 'rising' if predicted_prices[-1] > predicted_prices[0] else 'falling' if predicted_prices[-1] < predicted_prices[0] else 'stable'
    
    avg_price = np.mean(predicted_prices)
    price_range = max(predicted_prices) - min(predicted_prices)
    
    result = {
        'predictions': predictions,
        'summary': {
            'forecast_period_days': prediction_days,
            'average_predicted_price': round(avg_price, 3),
            'price_range': round(price_range, 3),
            'price_trend': price_trend,
            'model_type': model['model_type'],
            'total_features_used': len(model['feature_names']),
            'selected_features': sum(model['selector'].get_support())
        },
        'input_parameters': {
            'crude_oil_price': crude_oil_price,
            'demand_level': demand_level,
            'supply_disruption': supply_disruption,
            'current_temperature': current_temperature,
            'current_gas_price': current_gas_price
        },
        'market_insights': {
            'crude_oil_sensitivity': 'Each $1 change in crude oil affects gas price by ~$0.02',
            'seasonal_impact': 'Summer prices typically 10-20 cents higher',
            'demand_sensitivity': 'High demand periods can add 5-15 cents per gallon',
            'supply_risk': 'Supply disruptions can add 15-30 cents per gallon'
        }
    }

    print(f"Generated {len(predictions)} price predictions")
    print(f"Average predicted price: ${avg_price:.3f}")
    print(f"Price trend: {price_trend}")
    
    return result


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        
        return json.dumps(predictions, indent=2)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")