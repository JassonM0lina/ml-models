import os
import pickle
import pandas as pd
import numpy as np
import json
from io import StringIO, BytesIO
import requests
import warnings
from urllib3.exceptions import InsecureRequestWarning
import xgboost as xgb

# Suppress warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
warnings.filterwarnings('ignore')

# Global variables
model_package = None
burn_history = None
booster = None

# ============================================================
# SAGEMAKER INFERENCE FUNCTIONS
# ============================================================

def load_burn_history():
    """Load historical burn data for lag features"""
    global burn_history
    try:
        history_path = "Saudi Crude Burn History.xlsx"
        df = pd.read_excel(history_path)
        
        if df.shape[1] == 2:
            df.columns = ['date', 'burn_bpd']
        else:
            date_col = [col for col in df.columns if 'date' in col.lower()][0] if any('date' in col.lower() for col in df.columns) else df.columns[0]
            burn_col = [col for col in df.columns if 'saudi' in col.lower() or 'burn' in col.lower()][0] if any('saudi' in col.lower() or 'burn' in col.lower() for col in df.columns) else df.columns[1]
            df = df[[date_col, burn_col]]
            df.columns = ['date', 'burn_bpd']
        
        df['date'] = pd.to_datetime(df['date'])
        df['burn_bpd'] = pd.to_numeric(df['burn_bpd'], errors='coerce')
        df = df.dropna()
        
        if df['burn_bpd'].mean() > 10000:
            df['burn_kbd'] = df['burn_bpd'] / 1000
        else:
            df['burn_kbd'] = df['burn_bpd']
        
        df['month'] = df['date'].dt.month
        burn_history = df
        return df
    except Exception as e:
        print(f"Warning: Could not load burn history: {e}")
        return pd.DataFrame({'burn_kbd': [500], 'month': [7]})

def model_fn(model_dir):
    """
    Load the trained model from the specified directory.
    """
    global model_package, burn_history, booster
    
    print(f"Loading model from local directory: {model_dir}")
    print(f"Files available: {os.listdir(model_dir)}")
    
    try:
        import tempfile
        
        model_path = os.path.join(model_dir, "summer_peak_model_cv.pkl")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        with open(model_path, 'rb') as f:
            try:
                loaded_package = pickle.load(f, encoding='latin1')
            except Exception as e:
                print(f"Initial load failed: {e}. Retrying...")
                f.seek(0)
                loaded_package = pickle.load(f)
        
        # Assign to global
        model_package = loaded_package
        
        print(f"✓ Model loaded: {model_package['model_name']}")
        
        # Validate model package structure
        required_keys = ['model_json', 'model_name', 'feature_cols', 'final_performance']
        for key in required_keys:
            if key not in model_package:
                print(f"✗ Missing required key in model_package")
                raise KeyError(f"Missing required key in model_package: {key}")
        
        # Reconstruct XGBoost model from JSON
        booster = xgb.Booster()
        
        # Handle both string and bytes JSON
        model_json = model_package['model_json']
        if isinstance(model_json, bytes) or isinstance(model_json, bytearray):
            model_json = model_json.decode('utf-8')
        
        print(f"✓ Model JSON type: {type(model_json)}, length: {len(model_json)}")
        
        # Write to temp file and load (binary mode for safety)
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.json', delete=False) as f:
            if isinstance(model_json, str):
                f.write(model_json.encode('utf-8'))
            else:
                f.write(model_json)
            temp_path = f.name
        
        try:
            booster.load_model(temp_path)
            print(f"✓ Booster reconstructed from JSON")
        finally:
            os.unlink(temp_path)

        # Load burn history
        burn_history = load_burn_history()
        print(f"✓ Burn history loaded")
        
        print(f"✓ Model validation passed")
        return model_package
    
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        raise

def input_fn(request_body, request_content_type):
    """
    Deserialize the request body into a dictionary.
    """
    print(f"Request Body: {request_body}")
    print(f"Content Type: {request_content_type}")
    
    if request_content_type == "application/json":
        try:
            input_data = json.loads(request_body)
            
            target_year = input_data.get("target_year")
            target_month = input_data.get("target_month")
            include_weather = input_data.get("include_weather", True)
            
            if target_year is None or target_month is None:
                raise ValueError("Missing required fields: 'target_year' and 'target_month'")
            
            if not (1 <= int(target_month) <= 12):
                raise ValueError("target_month must be between 1 and 12")
            
            if int(target_year) < 2020:
                raise ValueError("target_year must be 2020 or later")
            
            return {
                "target_year": int(target_year),
                "target_month": int(target_month),
                "include_weather": bool(include_weather)
            }
        except Exception as e:
            print(f"Error parsing input: {e}")
            raise ValueError(f"Invalid input format. Error: {str(e)}")
    else:
        raise ValueError("Unsupported content type. Please use 'application/json'.")

def predict_fn(input_data, model):
    """
    Perform prediction using the model.
    """
    global model_package, burn_history, booster
    
    try:
        target_year = input_data["target_year"]
        target_month = input_data["target_month"]
        include_weather = input_data["include_weather"]
        
        print(f"Forecasting for {target_year}-{target_month:02d}")
        print(f"Booster status: {booster is not None}")
        print(f"Model package status: {model_package is not None}")
        
        # Validate booster is loaded
        if booster is None:
            raise RuntimeError("Model booster not loaded. Call model_fn() first.")
        
        if model_package is None:
            raise RuntimeError("Model package not loaded. Call model_fn() first.")
        
        if not include_weather:
            raise ValueError("Weather data is required for forecasting")
        
        weather = fetch_weather_data()
        features = prepare_monthly_features(weather, target_year, target_month)
        
        if features is None:
            raise ValueError(f"Insufficient weather data for {target_year}-{target_month}")
        
        # Create feature dataframe
        X_pred = pd.DataFrame([features])
        feature_cols = model_package['feature_cols']
        
        for col in feature_cols:
            if col not in X_pred.columns:
                X_pred[col] = 0.0
        
        X_pred = X_pred[feature_cols]
        
        # Model prediction using reconstructed booster
        dmatrix = xgb.DMatrix(X_pred.values)
        base_prediction = float(booster.predict(dmatrix)[0])
        
        print(f"Base prediction: {base_prediction:.1f} kbd")
        
        # Apply temperature rules
        final_prediction, adjustment_type, confidence = apply_temperature_rules(base_prediction, features)
        
        # Get historical average
        historical_month_data = burn_history[burn_history['month'] == target_month]['burn_kbd']
        historical_avg = float(historical_month_data.mean()) if len(historical_month_data) > 0 else 500.0
        
        # Calculate bounds
        model_mae = model_package['final_performance']['summer_mae'] \
                   if target_month in [6, 7, 8, 9] \
                   else model_package['final_performance']['overall_mae']
        
        lower_bound = final_prediction - 2 * model_mae
        upper_bound = final_prediction + 2 * model_mae
        
        month_name = pd.Timestamp(target_year, target_month, 1).strftime('%B %Y')
        
        print(f"Final prediction: {final_prediction:.1f} kbd ({adjustment_type})")
        
        return {
            "month": month_name,
            "prediction_kbd": round(final_prediction, 1),
            "base_prediction_kbd": round(base_prediction, 1),
            "adjustment_type": adjustment_type,
            "confidence": confidence,
            "lower_bound_kbd": round(lower_bound, 1),
            "upper_bound_kbd": round(upper_bound, 1),
            "max_temp_celsius": round(float(features['temp_max_max_max']), 1),
            "days_above_44c": int(features['extreme_44_sum']),
            "cdd_24_sum": round(float(features['cdd_24_sum']), 1),
            "historical_avg_kbd": round(historical_avg, 1),
            "vs_historical_pct": round((final_prediction / historical_avg - 1) * 100, 1),
            "weather_summary": {
                "max_temp_f": round(float(features['temp_max_max_max']) * 9/5 + 32, 1),
                "humidity_min_pct": round(float(features['humidity_min_min']), 1),
                "cdd_22": round(float(features['cdd_22_sum']), 1),
                "heat_index_mean": round(float(features['heat_index_mean']), 1)
            }
        }
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise

def output_fn(prediction, response_content_type):
    """
    Serialize the prediction into the desired response format.
    """
    if response_content_type == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError("Unsupported response content type. Please use 'application/json'.")

# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def fetch_weather_data():
    """Fetch current weather from Open-Meteo API"""
    cities = {
        'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.30},
        'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.25},
        'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.25},
        'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.20}
    }
    
    all_weather = []
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
        except:
            pass
    
    if not all_weather:
        raise ValueError("Unable to fetch weather data")
    
    combined = pd.concat(all_weather, ignore_index=True)
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
            'apparent_max_max': date_data['temp_max'].max()
        }
        daily_weather.append(daily_row)
    
    return pd.DataFrame(daily_weather)

def prepare_monthly_features(weather_daily, target_year, target_month):
    """Prepare features for a specific month"""
    global burn_history
    
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
    features['temp_max_mean_mean'] = float(month_weather['temp_max_mean'].mean())
    features['temp_max_mean_std'] = float(month_weather['temp_max_mean'].std()) if len(month_weather) > 1 else 1.0
    features['temp_max_max_max'] = float(month_weather['temp_max_max'].max())
    features['temp_max_max_mean'] = float(month_weather['temp_max_max'].mean())
    features['temp_mean_mean_mean'] = float(month_weather['temp_mean_mean'].mean())
    features['temp_mean_mean_max'] = float(month_weather['temp_mean_mean'].max())
    features['temp_mean_std_mean'] = float(month_weather['temp_mean_std'].mean())
    features['humidity_mean_mean'] = float(month_weather['humidity_mean'].mean())
    features['humidity_min_min'] = float(month_weather['humidity_min'].min())
    
    # Heat index
    T = float(month_weather['temp_mean_mean'].mean())
    RH = float(month_weather['humidity_mean'].mean())
    features['heat_index_mean'] = float(-8.78 + 1.61*T + 2.34*RH - 0.146*T*RH)
    features['heat_index_max'] = float(features['heat_index_mean'] + 5)
    features['apparent_max_max_max'] = float(month_weather['apparent_max_max'].max())
    
    # Cooling degree days
    for base in [22, 24, 26, 28]:
        daily_cdd = np.maximum(0, month_weather['temp_mean_mean'].values - base)
        features[f'cdd_{base}_sum'] = float(daily_cdd.sum() * scaling_factor)
    
    # Extreme heat days
    features['extreme_48_sum'] = float((month_weather['temp_max_max'] > 48).sum() * scaling_factor)
    features['extreme_46_sum'] = float((month_weather['temp_max_max'] > 46).sum() * scaling_factor)
    features['extreme_44_sum'] = float((month_weather['temp_max_max'] > 44).sum() * scaling_factor)
    features['extreme_42_sum'] = float((month_weather['temp_max_max'] > 42).sum() * scaling_factor)
    
    # Heat streak
    hot_days = (month_weather['temp_max_mean'] > 42).astype(int).values
    if len(hot_days) > 0:
        streaks = pd.Series(hot_days).groupby((pd.Series(hot_days) != pd.Series(hot_days).shift()).cumsum()).sum()
        features['heat_streak_max'] = float(streaks.max()) if len(streaks) > 0 else 0.0
    else:
        features['heat_streak_max'] = 0.0
    
    # Time features
    features['months_since_start'] = float((target_year - 2009) * 12 + target_month - 1)
    features['year_scaled'] = float((target_year - 2009) / 10)
    features['is_ramadan_summer'] = 0.0
    features['gas_availability_proxy'] = float(features['year_scaled'] * (-50))
    
    # Burn lags
    if burn_history is not None and len(burn_history) > 0:
        recent_burns = burn_history['burn_kbd'].values[-12:] if len(burn_history) >= 12 else burn_history['burn_kbd'].values
        
        if len(recent_burns) > 0:
            features['burn_lag1'] = float(recent_burns[-1]) if len(recent_burns) >= 1 else 500.0
            features['burn_lag2'] = float(recent_burns[-2]) if len(recent_burns) >= 2 else float(recent_burns[-1])
            features['burn_lag3'] = float(recent_burns[-3]) if len(recent_burns) >= 3 else float(recent_burns[-1])
            features['burn_lag12'] = float(recent_burns[0]) if len(recent_burns) >= 12 else float(recent_burns[-1])
            features['burn_ma2'] = float(np.mean(recent_burns[-2:])) if len(recent_burns) >= 2 else float(recent_burns[-1])
            features['burn_ma3'] = float(np.mean(recent_burns[-3:])) if len(recent_burns) >= 3 else float(recent_burns[-1])
            features['burn_ma6'] = float(np.mean(recent_burns[-6:])) if len(recent_burns) >= 6 else float(np.mean(recent_burns))
            features['burn_yoy'] = float((recent_burns[-1] / recent_burns[0] - 1)) if len(recent_burns) >= 12 and recent_burns[0] > 0 else 0.0
    else:
        features['burn_lag1'] = 500.0
        features['burn_lag2'] = 500.0
        features['burn_lag3'] = 500.0
        features['burn_lag12'] = 500.0
        features['burn_ma2'] = 500.0
        features['burn_ma3'] = 500.0
        features['burn_ma6'] = 500.0
        features['burn_yoy'] = 0.0
    
    # Temperature lags
    features['temp_lag1'] = features['temp_max_max_max']
    features['temp_lag2'] = features['temp_max_max_max']
    features['temp_lag3'] = features['temp_max_max_max']
    features['temp_lag12'] = features['temp_max_max_max']
    features['temp_ma2'] = features['temp_max_max_max']
    features['temp_ma3'] = features['temp_max_max_max']
    features['temp_ma6'] = features['temp_max_max_max']
    features['temp_yoy'] = 0.0
    
    # Non-linear features
    is_summer = 1.0 if target_month in [6, 7, 8, 9] else 0.0
    is_peak_summer = 1.0 if target_month in [7, 8] else 0.0
    
    features['summer_intensity'] = float(is_summer * features['temp_max_max_max'])
    features['peak_heat'] = float(is_peak_summer * features['extreme_44_sum'])
    features['dry_heat'] = float(features['temp_max_max_max'] * (100 - features['humidity_min_min']) / 100)
    features['temp_above_45'] = float(max(0, features['temp_max_max_max'] - 45) ** 2)
    features['temp_above_43'] = float(max(0, features['temp_max_max_max'] - 43) ** 1.5)
    features['cumulative_cdd'] = float(features['cdd_24_sum'] * 3)
    features['temp_regional_spread'] = features['temp_mean_std_mean']
    
    return features

def apply_temperature_rules(model_prediction, features):
    """Apply temperature-based rules for extreme conditions"""
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
        return float(model_prediction), "STANDARD MODEL", "NORMAL"
    
    if confidence == "HIGH":
        final_prediction = 0.7 * rule_prediction + 0.3 * model_prediction
    else:
        final_prediction = 0.5 * rule_prediction + 0.5 * model_prediction
    
    return float(final_prediction), adjustment_type, confidence