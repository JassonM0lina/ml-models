"""
Saudi Arabia Crude Oil Burn Forecaster - Inference Script
SageMaker-compliant inference for generating monthly burn forecasts

This script:
1. Loads the trained XGBoost model package
2. Fetches current/forecast weather from Open-Meteo API
3. Prepares features matching training pipeline
4. Applies temperature-based override rules for extreme heat
5. Generates multi-month forecasts with confidence intervals
"""

import os
import json
import pickle
from datetime import datetime, timedelta
import warnings

import numpy as np
import pandas as pd
import requests

warnings.filterwarnings('ignore')


# City configuration for weather data
CITIES = {
    'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.30},
    'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.25},
    'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.25},
    'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.20}
}

# Equal weights alternative
CITIES_EQUAL = {
    'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.25},
    'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.25},
    'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.25},
    'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.25}
}


def model_fn(model_dir):
    """Load the model package from the model directory"""
    model_path = os.path.join(model_dir, "model.pkl")
    print(f"Loading model from: {model_path}")

    with open(model_path, "rb") as f:
        model_package = pickle.load(f)

    print(f"Model loaded: {model_package.get('model_name', 'Unknown')}")
    print(f"Features: {len(model_package.get('feature_cols', []))}")
    print(f"Performance - Overall MAE: {model_package['final_performance']['overall_mae']:.1f} kbd")

    return model_package


def input_fn(request_body, request_content_type):
    """
    Parse input parameters matching inference_schema.json.

    Expected parameters:
    - forecast_months: Number of months to forecast (1-6)
    - use_weather_forecast: Use API weather forecasts vs historical averages
    - confidence_level: Confidence interval percentage (90, 95, 99)
    - apply_extreme_heat_rules: Enable temperature override rules
    - cities_weight_override: "population_weighted" or "equal_weighted"
    - season_focus: "all_seasons", "summer_optimized", or "winter_optimized"
    """
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    params = json.loads(request_body)

    if not isinstance(params, dict):
        raise ValueError(f"Expected dict with parameter names, got {type(params).__name__}")

    # Apply defaults from schema
    defaults = {
        'forecast_months': 3,
        'use_weather_forecast': True,
        'confidence_level': 95,
        'apply_extreme_heat_rules': True,
        'cities_weight_override': 'population_weighted',
        'season_focus': 'summer_optimized'
    }

    for key, default in defaults.items():
        if key not in params:
            params[key] = default

    print(f"Parsed parameters: {params}")
    return params


def fetch_current_weather(cities, use_forecast=True):
    """Fetch current and forecast weather from Open-Meteo API"""
    all_weather = []

    for city, info in cities.items():
        try:
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                'latitude': info['lat'],
                'longitude': info['lon'],
                'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_mean',
                'past_days': 60,
                'forecast_days': 16 if use_forecast else 0,
                'timezone': 'Asia/Riyadh'
            }

            response = requests.get(url, params=params, timeout=30, verify=False)
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
                print(f"  Weather fetched for {city}: {len(city_df)} days")
        except Exception as e:
            print(f"  Weather fetch failed for {city}: {e}")

    if not all_weather:
        raise ValueError("Failed to fetch weather data from any city")

    combined = pd.concat(all_weather)

    # Calculate weighted daily averages
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
            'temp_mean_std': date_data['temp_mean'].std() if len(date_data) > 1 else 1,
            'humidity_mean': weighted_avg(date_data['humidity'].values),
            'humidity_min': date_data['humidity'].min(),
            'wind_speed': weighted_avg(date_data['wind_speed'].values),
            'dewpoint': 0,
            'apparent_max_max': date_data['temp_max'].max()
        }
        daily_weather.append(daily_row)

    return pd.DataFrame(daily_weather)


def prepare_monthly_features(weather_daily, target_year, target_month, burn_history=None):
    """Prepare features for a specific forecast month"""
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

    # Extreme heat days
    features['extreme_48_sum'] = (month_weather['temp_max_max'] > 48).sum() * scaling_factor
    features['extreme_46_sum'] = (month_weather['temp_max_max'] > 46).sum() * scaling_factor
    features['extreme_44_sum'] = (month_weather['temp_max_max'] > 44).sum() * scaling_factor
    features['extreme_42_sum'] = (month_weather['temp_max_max'] > 42).sum() * scaling_factor

    # Heat streak
    hot_days = (month_weather['temp_max_mean'] > 42).astype(int)
    if len(hot_days) > 0:
        streaks = hot_days.groupby((hot_days != hot_days.shift()).cumsum()).sum()
        features['heat_streak_max'] = streaks.max() if len(streaks) > 0 else 0
    else:
        features['heat_streak_max'] = 0

    # Time features
    features['months_since_start'] = (target_year - 2009) * 12 + target_month - 1
    features['year_scaled'] = (target_year - 2009) / 10
    features['is_ramadan_summer'] = 0
    features['gas_availability_proxy'] = features['year_scaled'] * (-50)

    # Lag features (use burn history if available, otherwise use defaults)
    if burn_history is not None and len(burn_history) > 0:
        recent_burns = burn_history[-12:] if len(burn_history) >= 12 else burn_history

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

    # Temperature lag features (use current as proxy)
    features['temp_lag1'] = features['temp_max_max_max']
    features['temp_lag2'] = features['temp_max_max_max']
    features['temp_lag3'] = features['temp_max_max_max']
    features['temp_lag12'] = features['temp_max_max_max']
    features['temp_ma2'] = features['temp_max_max_max']
    features['temp_ma3'] = features['temp_max_max_max']
    features['temp_ma6'] = features['temp_max_max_max']
    features['temp_yoy'] = 0

    # Interaction features
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
    """Apply temperature-based override rules for extreme heat conditions"""
    max_temp = features['temp_max_max_max']
    extreme_days = features['extreme_44_sum']
    cdd_24 = features['cdd_24_sum']

    # Extreme heat (>48°C with many extreme days)
    if max_temp >= 48 and extreme_days >= 18:
        min_burn = 750
        max_burn = 850
        cdd_factor = min(1.0, cdd_24 / 350)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "EXTREME_HEAT_OVERRIDE"
        confidence = "HIGH"

    # Very hot (>47°C)
    elif max_temp >= 47 and extreme_days >= 15:
        min_burn = 700
        max_burn = 750
        cdd_factor = min(1.0, cdd_24 / 300)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "VERY_HOT_OVERRIDE"
        confidence = "HIGH"

    # Hot (>46°C)
    elif max_temp >= 46 and extreme_days >= 10:
        min_burn = 650
        max_burn = 700
        cdd_factor = min(1.0, cdd_24 / 250)
        rule_prediction = min_burn + (max_burn - min_burn) * cdd_factor
        adjustment_type = "HOT_WEATHER_ADJUSTMENT"
        confidence = "MODERATE"

    else:
        return model_prediction, "STANDARD_MODEL", "NORMAL"

    # Blend rule and model predictions
    if confidence == "HIGH":
        final_prediction = 0.7 * rule_prediction + 0.3 * model_prediction
    else:
        final_prediction = 0.5 * rule_prediction + 0.5 * model_prediction

    return final_prediction, adjustment_type, confidence


def predict_fn(input_data, model_package):
    """Generate multi-month burn forecasts"""
    print(f"Generating forecasts with parameters: {input_data}")

    model = model_package['model']
    feature_cols = model_package['feature_cols']
    performance = model_package['final_performance']

    # Select city weights
    cities = CITIES if input_data['cities_weight_override'] == 'population_weighted' else CITIES_EQUAL

    # Fetch weather data
    print("Fetching weather data...")
    weather = fetch_current_weather(cities, input_data['use_weather_forecast'])

    # Generate forecasts
    current_date = datetime.now()
    forecasts = []
    forecast_months = []
    confidence_lower = []
    confidence_upper = []
    weather_drivers = []
    model_adjustments = []

    # Confidence interval multiplier based on level
    ci_multipliers = {90: 1.645, 95: 1.96, 99: 2.576}
    ci_mult = ci_multipliers.get(input_data['confidence_level'], 1.96)

    # Placeholder for burn history (would come from data in production)
    burn_history = [500, 520, 480, 510, 530, 600, 700, 750, 720, 500, 400, 380]

    for month_offset in range(input_data['forecast_months']):
        target_date = current_date + timedelta(days=30 * month_offset)
        target_year = target_date.year
        target_month = target_date.month
        month_name = target_date.strftime('%B %Y')

        features = prepare_monthly_features(weather, target_year, target_month, burn_history)

        if features is None:
            continue

        # Prepare feature vector
        X_pred = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0
        X_pred = X_pred[feature_cols]

        # Model prediction
        base_prediction = float(model.predict(X_pred.values)[0])

        # Apply temperature rules if enabled
        if input_data['apply_extreme_heat_rules']:
            final_prediction, adjustment_type, confidence = apply_temperature_rules(base_prediction, features)
        else:
            final_prediction = base_prediction
            adjustment_type = "STANDARD_MODEL"
            confidence = "NORMAL"

        # Select appropriate MAE for confidence interval
        is_summer = target_month in [6, 7, 8, 9]
        model_mae = performance['summer_mae'] if is_summer else performance['overall_mae']

        # Store results
        forecasts.append(round(final_prediction, 1))
        forecast_months.append(month_name)
        confidence_lower.append(round(final_prediction - ci_mult * model_mae, 1))
        confidence_upper.append(round(final_prediction + ci_mult * model_mae, 1))

        weather_drivers.append({
            'month': month_name,
            'max_temp_c': round(features['temp_max_max_max'], 1),
            'max_temp_f': round(features['temp_max_max_max'] * 9/5 + 32, 1),
            'cdd_24': round(features['cdd_24_sum'], 0),
            'extreme_days_44c': int(features['extreme_44_sum']),
            'humidity_min': round(features['humidity_min_min'], 1)
        })

        if adjustment_type != "STANDARD_MODEL":
            model_adjustments.append({
                'month': month_name,
                'adjustment_type': adjustment_type,
                'base_prediction': round(base_prediction, 1),
                'final_prediction': round(final_prediction, 1),
                'confidence': confidence
            })

    # Build response
    predictions = {
        'monthly_forecasts': forecasts,
        'confidence_lower': confidence_lower,
        'confidence_upper': confidence_upper,
        'forecast_months': forecast_months,
        'weather_drivers': weather_drivers,
        'model_adjustments': model_adjustments,
        'historical_comparison': {
            'summer_avg_historical': 672,
            'winter_avg_historical': 350
        },
        'model_confidence': {
            'overall_mae': performance['overall_mae'],
            'summer_mae': performance['summer_mae'],
            'summer_mape': performance['summer_mape'],
            'confidence_level': input_data['confidence_level']
        }
    }

    print(f"Generated {len(forecasts)} month forecast(s)")
    return predictions


def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(predictions, indent=2)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
