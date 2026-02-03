"""
Inference Script for Saudi Crude Burn Forecaster
Generates crude burn forecasts based on weather data and historical patterns
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta


def model_fn(model_dir):
    """Load the model package from the model directory"""
    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_package = pickle.load(f)

    print(f"Model loaded successfully: {model_package.get('model_name', 'Unknown')}")
    return model_package


def fetch_current_weather():
    """Fetch current and forecast weather for Saudi cities"""
    cities = {
        'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.30},
        'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.25},
        'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.25},
        'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.20}
    }

    all_weather = []
    current_date = datetime.now()

    for city, info in cities.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': info['lat'],
                'longitude': info['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean',
                'past_days': 60,
                'forecast_days': 16,
                'timezone': 'Asia/Riyadh'
            }

            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()['daily']
                city_df = pd.DataFrame({
                    'date': pd.to_datetime(data['time']),
                    'temp_max': data['temperature_2m_max'],
                    'temp_min': data['temperature_2m_min'],
                    'temp_mean': data['temperature_2m_mean'],
                    'humidity': data['relative_humidity_2m_mean'],
                    'wind_speed': data['wind_speed_10m_mean'],
                    'city': city,
                    'weight': info['weight']
                })
                all_weather.append(city_df)
        except Exception as e:
            print(f"Warning: Could not fetch weather for {city}: {e}")

    if not all_weather:
        raise ValueError("No weather data collected. Check internet connection.")

    combined = pd.concat(all_weather)

    # Aggregate daily weather
    daily_weather = []
    for date in combined['date'].unique():
        date_data = combined[combined['date'] == date]
        weights = date_data['weight'].values
        weighted_avg = lambda x: np.sum(x * weights) / np.sum(weights)

        daily_row = {
            'date': date,
            'temp_max_mean': weighted_avg(date_data['temp_max'].values),
            'temp_max_max': date_data['temp_max'].max(),
            'temp_min_mean': weighted_avg(date_data['temp_min'].values),
            'temp_min_min': date_data['temp_min'].min(),
            'temp_mean_mean': weighted_avg(date_data['temp_mean'].values),
            'temp_mean_std': date_data['temp_mean'].std(),
            'humidity_mean': weighted_avg(date_data['humidity'].values),
            'humidity_min': date_data['humidity'].min(),
            'wind_speed': weighted_avg(date_data['wind_speed'].values),
            'dewpoint': 0,
            'apparent_max_max': date_data['temp_max'].max()
        }
        daily_weather.append(daily_row)

    daily_weather = pd.DataFrame(daily_weather)
    return daily_weather


def prepare_monthly_features(weather_daily, target_year, target_month, burn_history=None):
    """Prepare features for a specific month matching training exactly"""
    month_weather = weather_daily[
        (weather_daily['date'].dt.year == target_year) &
        (weather_daily['date'].dt.month == target_month)
    ]

    if len(month_weather) == 0:
        return None

    days_in_month = pd.Timestamp(target_year, target_month, 1).days_in_month
    days_available = len(month_weather)
    scaling_factor = days_in_month / days_available if days_available > 0 else 1

    features = {}

    # Temperature features
    features['temp_max_mean_mean'] = month_weather['temp_max_mean'].mean()
    features['temp_max_mean_std'] = month_weather['temp_max_mean'].std() if len(month_weather) > 1 else 1
    features['temp_max_max_max'] = month_weather['temp_max_max'].max()
    features['temp_max_max_mean'] = month_weather['temp_max_max'].mean()
    features['temp_mean_mean_mean'] = month_weather['temp_mean_mean'].mean()
    features['temp_mean_mean_max'] = month_weather['temp_mean_mean'].max()
    features['temp_mean_std_mean'] = month_weather['temp_mean_std'].mean()
    features['humidity_mean_mean'] = month_weather['humidity_mean'].mean()
    features['humidity_min_min'] = month_weather['humidity_min'].min()

    # Heat index
    T = month_weather['temp_mean_mean'].mean()
    RH = month_weather['humidity_mean'].mean()
    features['heat_index_mean'] = -8.78 + 1.61*T + 2.34*RH - 0.146*T*RH
    features['heat_index_max'] = features['heat_index_mean'] + 5
    features['apparent_max_max_max'] = month_weather['apparent_max_max'].max()

    # Cooling degree days
    for base in [22, 24, 26, 28]:
        daily_cdd = np.maximum(0, month_weather['temp_mean_mean'] - base)
        features[f'cdd_{base}_sum'] = daily_cdd.sum() * scaling_factor

    # Extreme heat categories
    features['extreme_48_sum'] = (month_weather['temp_max_max'] > 48).sum() * scaling_factor
    features['extreme_46_sum'] = (month_weather['temp_max_max'] > 46).sum() * scaling_factor
    features['extreme_44_sum'] = (month_weather['temp_max_max'] > 44).sum() * scaling_factor
    features['extreme_42_sum'] = (month_weather['temp_max_max'] > 42).sum() * scaling_factor

    # Heat streaks
    hot_days = (month_weather['temp_max_mean'] > 42).astype(int)
    if len(hot_days) > 0:
        streaks = hot_days.groupby((hot_days != hot_days.shift()).cumsum()).sum()
        features['heat_streak_max'] = streaks.max() if len(streaks) > 0 else 0
    else:
        features['heat_streak_max'] = 0

    # Time-based features
    features['months_since_start'] = (target_year - 2009) * 12 + target_month - 1
    features['year_scaled'] = (target_year - 2009) / 10
    features['is_ramadan_summer'] = 0
    features['gas_availability_proxy'] = features['year_scaled'] * (-50)

    # Historical burn features
    if burn_history is not None and len(burn_history) > 0:
        recent_burns = burn_history[-12:] if len(burn_history) >= 12 else burn_history

        if len(recent_burns) > 0:
            features['burn_lag1'] = recent_burns[-1] if len(recent_burns) >= 1 else 500
            features['burn_lag2'] = recent_burns[-2] if len(recent_burns) >= 2 else recent_burns[-1]
            features['burn_lag3'] = recent_burns[-3] if len(recent_burns) >= 3 else recent_burns[-1]
            features['burn_lag12'] = recent_burns[0] if len(recent_burns) >= 12 else recent_burns[-1]
            features['burn_ma2'] = np.mean(recent_burns[-2:]) if len(recent_burns) >= 2 else recent_burns[-1]
            features['burn_ma3'] = np.mean(recent_burns[-3:]) if len(recent_burns) >= 3 else recent_burns[-1]
            features['burn_ma6'] = np.mean(recent_burns[-6:]) if len(recent_burns) >= 6 else np.mean(recent_burns)
            features['burn_yoy'] = (recent_burns[-1] / recent_burns[0] - 1) if len(recent_burns) >= 12 and recent_burns[0] > 0 else 0
    else:
        features['burn_lag1'] = 500
        features['burn_lag2'] = 500
        features['burn_lag3'] = 500
        features['burn_lag12'] = 500
        features['burn_ma2'] = 500
        features['burn_ma3'] = 500
        features['burn_ma6'] = 500
        features['burn_yoy'] = 0

    # Temperature lags (use current as proxy)
    features['temp_lag1'] = features['temp_max_max_max']
    features['temp_lag2'] = features['temp_max_max_max']
    features['temp_lag3'] = features['temp_max_max_max']
    features['temp_lag12'] = features['temp_max_max_max']
    features['temp_ma2'] = features['temp_max_max_max']
    features['temp_ma3'] = features['temp_max_max_max']
    features['temp_ma6'] = features['temp_max_max_max']
    features['temp_yoy'] = 0

    # Summer-specific features
    is_summer = 1 if target_month in [6, 7, 8, 9] else 0
    is_peak_summer = 1 if target_month in [7, 8] else 0

    features['summer_intensity'] = is_summer * features['temp_max_max_max']
    features['peak_heat'] = is_peak_summer * features['extreme_44_sum']
    features['dry_heat'] = features['temp_max_max_max'] * (100 - features['humidity_min_min']) / 100
    features['temp_above_45'] = max(0, features['temp_max_max_max'] - 45) ** 2
    features['temp_above_43'] = max(0, features['temp_max_max_max'] - 43) ** 1.5
    features['cumulative_cdd'] = features['cdd_24_sum'] * 3
    features['temp_regional_spread'] = features['temp_mean_std_mean']

    return features


def apply_temperature_rules(model_prediction, features):
    """Override model with temperature-based rules for extreme conditions"""
    max_temp = features['temp_max_max_max']
    extreme_days = features['extreme_44_sum']
    cdd_24 = features['cdd_24_sum']

    if max_temp >= 48 and extreme_days >= 18:
        min_burn = 750
        max_burn = 850
        cdd_factor = min(1.0, cdd_24 / 350)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "EXTREME HEAT OVERRIDE"
        confidence = "HIGH"

    elif max_temp >= 47 and extreme_days >= 15:
        min_burn = 700
        max_burn = 750
        cdd_factor = min(1.0, cdd_24 / 300)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "VERY HOT OVERRIDE"
        confidence = "HIGH"

    elif max_temp >= 46 and extreme_days >= 10:
        min_burn = 650
        max_burn = 700
        cdd_factor = min(1.0, cdd_24 / 250)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "HOT WEATHER ADJUSTMENT"
        confidence = "MODERATE"

    else:
        return model_prediction, "STANDARD MODEL", "NORMAL"

    # Blend rule with model
    if confidence == "HIGH":
        final_prediction = 0.7 * rule_prediction + 0.3 * model_prediction
    else:
        final_prediction = 0.5 * rule_prediction + 0.5 * model_prediction

    return final_prediction, adjustment_type, confidence


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


def predict_fn(input_data, model_package):
    """Generate crude burn forecast"""
    model = model_package['model']
    feature_cols = model_package['feature_cols']

    # Get parameters from input
    forecast_months = input_data.get('forecast_months', 3)
    target_year = input_data.get('target_year', datetime.now().year)
    target_month = input_data.get('target_month', datetime.now().month)
    use_temperature_override = input_data.get('use_temperature_override', True)
    confidence_level = input_data.get('confidence_level', 95)

    print(f"Generating forecast for {forecast_months} months starting {target_year}-{target_month:02d}")

    # Fetch current weather data
    weather_data = fetch_current_weather()

    # TODO: In production, you might want to load historical burn data from S3
    # For now, we'll use default values for burn history
    burn_history = None  # Could be loaded from S3 or passed as input

    # Generate forecasts
    forecasts = []
    current_date = datetime(target_year, target_month, 1)

    for month_offset in range(forecast_months):
        forecast_date = current_date + timedelta(days=30 * month_offset)
        year = forecast_date.year
        month = forecast_date.month
        month_name = forecast_date.strftime('%B %Y')

        # Prepare features for this month
        features = prepare_monthly_features(weather_data, year, month, burn_history)

        if features is None:
            print(f"Warning: Insufficient weather data for {month_name}")
            continue

        # Create feature dataframe
        X_pred = pd.DataFrame([features])

        # Ensure all required features are present
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0

        X_pred = X_pred[feature_cols]

        # Get base prediction
        base_prediction = model.predict(X_pred.values)[0]

        # Apply temperature-based rules if enabled
        if use_temperature_override:
            final_prediction, adjustment_type, confidence = apply_temperature_rules(base_prediction, features)
        else:
            final_prediction = base_prediction
            adjustment_type = "STANDARD MODEL"
            confidence = "NORMAL"

        # Calculate confidence intervals based on model performance
        is_summer = month in [6, 7, 8, 9]
        model_mae = model_package['final_performance']['summer_mae'] if is_summer else model_package['final_performance']['overall_mae']

        # Adjust confidence interval based on confidence level
        if confidence_level == 90:
            ci_multiplier = 1.645
        elif confidence_level == 95:
            ci_multiplier = 2.0
        elif confidence_level == 99:
            ci_multiplier = 2.576
        else:
            ci_multiplier = 2.0

        lower_bound = final_prediction - ci_multiplier * model_mae
        upper_bound = final_prediction + ci_multiplier * model_mae

        forecasts.append({
            'month': month_name,
            'year': year,
            'month_number': month,
            'prediction_kbd': float(final_prediction),
            'base_prediction_kbd': float(base_prediction),
            'adjustment_type': adjustment_type,
            'confidence': confidence,
            'confidence_interval': {
                'level': confidence_level,
                'lower': float(lower_bound),
                'upper': float(upper_bound)
            },
            'weather_conditions': {
                'max_temp_c': float(features['temp_max_max_max']),
                'extreme_days_44c': int(features['extreme_44_sum']),
                'cooling_degree_days': float(features['cdd_24_sum'])
            }
        })

    return {
        'forecasts': forecasts,
        'model_info': {
            'name': model_package['model_name'],
            'performance': {
                'overall_mae': model_package['final_performance']['overall_mae'],
                'summer_mae': model_package['final_performance']['summer_mae'],
                'summer_mape': model_package['final_performance']['summer_mape']
            }
        },
        'forecast_date': datetime.now().isoformat()
    }


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(predictions, indent=2)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
