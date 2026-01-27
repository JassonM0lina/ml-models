"""
Saudi Crude Burn Forecaster - Model Training Module
Trains XGBoost model for summer peak crude oil burn prediction
"""

import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pickle
import warnings
from pathlib import Path

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

warnings.filterwarnings('ignore')


class SaudiCrudeBurnTrainer:
    """
    Trains XGBoost model to forecast Saudi Arabia crude oil burn rates,
    with focus on summer peak periods.
    
    Required files:
    - Saudi Crude Burn History.xlsx
    """
    
    def __init__(self, working_dir=None):
        """
        Initialize trainer with working directory.
        
        Args:
            working_dir (str, optional): Path to working directory. 
                                        Defaults to current directory.
        """
        if working_dir is None:
            working_dir = os.path.dirname(os.path.abspath(__file__))
        
        self.working_dir = working_dir
        self.historical_burn_path = os.path.join(working_dir, "Saudi Crude Burn History.xlsx")
        
        # Data containers
        self.burn_df = None
        self.weather_daily = None
        self.weather_monthly = None
        self.merged_df = None
        self.feature_cols = None
        
        # Model
        self.model = None
        self.performance_metrics = None
        
        # Validation data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        
        print("="*80)
        print("SAUDI SUMMER PEAK MODEL TRAINING")
        print("="*80)
        print(f"Working directory: {self.working_dir}")
        print("="*80)
    
    def load_burn_data(self):
        """STEP 1: Load historical burn data and identify peaks"""
        print("\n[STEP 1: LOADING DATA & IDENTIFYING PEAKS]")
        print("-"*60)
        
        # Load burn data
        burn_df = pd.read_excel(self.historical_burn_path)
        if burn_df.shape[1] == 2:
            burn_df.columns = ['date', 'burn_bpd']
        else:
            date_col = [col for col in burn_df.columns if 'date' in col.lower()][0] if any('date' in col.lower() for col in burn_df.columns) else burn_df.columns[0]
            burn_col = [col for col in burn_df.columns if 'saudi' in col.lower()][0] if any('saudi' in col.lower() for col in burn_df.columns) else burn_df.columns[1]
            burn_df = burn_df[[date_col, burn_col]]
            burn_df.columns = ['date', 'burn_bpd']
        
        burn_df['date'] = pd.to_datetime(burn_df['date'])
        burn_df['burn_bpd'] = pd.to_numeric(burn_df['burn_bpd'], errors='coerce')
        burn_df = burn_df.dropna()
        
        # Convert to kbd
        if burn_df['burn_bpd'].mean() > 10000:
            burn_df['burn_kbd'] = burn_df['burn_bpd'] / 1000
        else:
            burn_df['burn_kbd'] = burn_df['burn_bpd']
        
        burn_df['year'] = burn_df['date'].dt.year
        burn_df['month'] = burn_df['date'].dt.month
        
        # Identify summer months and peaks
        burn_df['is_summer'] = burn_df['month'].isin([6, 7, 8, 9]).astype(int)
        burn_df['is_peak_summer'] = burn_df['month'].isin([7, 8]).astype(int)
        
        # Calculate percentiles
        summer_burns = burn_df[burn_df['is_summer'] == 1]['burn_kbd']
        p75 = summer_burns.quantile(0.75)
        p90 = summer_burns.quantile(0.90)
        
        print(f"✓ Loaded {len(burn_df)} months")
        print(f"\nSummer burn statistics:")
        print(f"  Mean: {summer_burns.mean():.0f} kbd")
        print(f"  75th percentile: {p75:.0f} kbd")
        print(f"  90th percentile: {p90:.0f} kbd")
        print(f"  Max: {summer_burns.max():.0f} kbd")
        
        # Show outlier months
        outliers = burn_df[(burn_df['is_summer'] == 1) & (burn_df['burn_kbd'] > p90)]
        print(f"\nExtreme summer burns (>{p90:.0f} kbd):")
        for _, row in outliers.iterrows():
            print(f"  {row['date'].strftime('%Y-%m')}: {row['burn_kbd']:.0f} kbd")
        
        self.burn_df = burn_df
    
    def fetch_weather_data(self):
        """STEP 2: Fetch weather data from Open-Meteo API"""
        print("\n[STEP 2: FETCHING WEATHER DATA]")
        print("-"*60)
        
        cities = {
            'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.25, 'type': 'capital'},
            'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.15, 'type': 'coastal'},
            'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.15, 'type': 'industrial'},
            'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.15, 'type': 'inland'},
            'Jubail': {'lat': 27.0174, 'lon': 49.6225, 'weight': 0.15, 'type': 'industrial'},
            'Yanbu': {'lat': 24.0943, 'lon': 38.0493, 'weight': 0.15, 'type': 'industrial'}
        }
        
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        start_date = self.burn_df['date'].min().strftime('%Y-%m-%d')
        end_date = self.burn_df['date'].max().strftime('%Y-%m-%d')
        
        all_weather = []
        
        for city, info in cities.items():
            params = {
                'latitude': info['lat'],
                'longitude': info['lon'],
                'start_date': start_date,
                'end_date': end_date,
                'daily': ['temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean',
                         'relative_humidity_2m_mean', 'wind_speed_10m_mean',
                         'dewpoint_2m_mean', 'apparent_temperature_max'],
                'timezone': 'Asia/Riyadh'
            }
            
            try:
                response = requests.get(base_url, params=params, timeout=30, verify=False)
                if response.status_code == 200:
                    data = response.json()['daily']
                    city_df = pd.DataFrame({
                        'date': pd.to_datetime(data['time']),
                        'temp_max': data['temperature_2m_max'],
                        'temp_min': data['temperature_2m_min'],
                        'temp_mean': data['temperature_2m_mean'],
                        'humidity': data['relative_humidity_2m_mean'],
                        'wind_speed': data['wind_speed_10m_mean'],
                        'dewpoint': data['dewpoint_2m_mean'],
                        'apparent_max': data['apparent_temperature_max'],
                        'city': city,
                        'weight': info['weight'],
                        'city_type': info['type']
                    })
                    all_weather.append(city_df)
                    print(f"  ✓ {city} ({info['type']}): {len(city_df)} days")
            except Exception as e:
                print(f"  ✗ {city}: {e}")
        
        # Combine
        combined = pd.concat(all_weather)
        
        # Calculate both weighted average AND extremes
        daily_weather = combined.groupby('date').agg({
            'temp_max': ['mean', 'max'],
            'temp_min': ['mean', 'min'],
            'temp_mean': ['mean', 'std'],
            'humidity': ['mean', 'min'],
            'wind_speed': 'mean',
            'dewpoint': 'mean',
            'apparent_max': ['mean', 'max']
        }).reset_index()
        
        # Flatten columns
        daily_weather.columns = ['_'.join(col).strip('_') for col in daily_weather.columns]
        
        # Add heat stress indicators
        daily_weather['year'] = daily_weather['date'].dt.year
        daily_weather['month'] = daily_weather['date'].dt.month
        
        # Cooling degree days with multiple bases
        for base in [22, 24, 26, 28]:
            daily_weather[f'cdd_{base}'] = np.maximum(0, daily_weather['temp_mean_mean'] - base)
        
        # Extreme heat categories
        daily_weather['extreme_48'] = (daily_weather['temp_max_max'] > 48).astype(int)
        daily_weather['extreme_46'] = (daily_weather['temp_max_max'] > 46).astype(int)
        daily_weather['extreme_44'] = (daily_weather['temp_max_max'] > 44).astype(int)
        daily_weather['extreme_42'] = (daily_weather['temp_max_max'] > 42).astype(int)
        
        # Heat index approximation
        T = daily_weather['temp_mean_mean']
        RH = daily_weather['humidity_mean']
        daily_weather['heat_index'] = -8.78 + 1.61*T + 2.34*RH - 0.146*T*RH
        
        # Consecutive hot days
        daily_weather['hot_day'] = (daily_weather['temp_max_mean'] > 42).astype(int)
        daily_weather['heat_streak'] = daily_weather.groupby(
            (daily_weather['hot_day'] != daily_weather['hot_day'].shift()).cumsum()
        )['hot_day'].cumsum() * daily_weather['hot_day']
        
        self.weather_daily = daily_weather
    
    def aggregate_weather_monthly(self):
        """STEP 3: Aggregate daily weather to monthly level"""
        print("\n[STEP 3: MONTHLY AGGREGATION WITH EXTREME METRICS]")
        print("-"*60)
        
        # Monthly aggregation preserving extreme information
        weather_monthly = self.weather_daily.groupby(['year', 'month']).agg({
            'temp_max_mean': ['mean', 'std'],
            'temp_max_max': ['max', 'mean'],
            'temp_mean_mean': ['mean', 'max'],
            'temp_mean_std': 'mean',
            'humidity_mean': 'mean',
            'humidity_min': 'min',
            'heat_index': ['mean', 'max'],
            'apparent_max_max': 'max',
            'cdd_22': 'sum',
            'cdd_24': 'sum',
            'cdd_26': 'sum',
            'cdd_28': 'sum',
            'extreme_48': 'sum',
            'extreme_46': 'sum',
            'extreme_44': 'sum',
            'extreme_42': 'sum',
            'heat_streak': 'max'
        }).reset_index()
        
        # Flatten columns
        weather_monthly.columns = ['_'.join(col).strip('_') for col in weather_monthly.columns]
        
        print(f"✓ Created {len(weather_monthly.columns)} weather features")
        
        self.weather_monthly = weather_monthly
    
    def engineer_features(self):
        """STEP 4: Merge data and create advanced features"""
        print("\n[STEP 4: FEATURE ENGINEERING]")
        print("-"*60)
        
        # Merge burn and weather data
        merged_df = pd.merge(
            self.burn_df[['year', 'month', 'burn_kbd', 'is_summer', 'is_peak_summer']],
            self.weather_monthly,
            on=['year', 'month'],
            how='inner'
        )
        
        # Time-based features
        merged_df['months_since_start'] = range(len(merged_df))
        merged_df['year_scaled'] = (merged_df['year'] - merged_df['year'].min()) / 10
        merged_df['is_ramadan_summer'] = 0
        ramadan_summer_years = [2013, 2014, 2015, 2016]
        merged_df.loc[
            (merged_df['year'].isin(ramadan_summer_years)) & 
            (merged_df['is_summer'] == 1), 
            'is_ramadan_summer'
        ] = 1
        merged_df['gas_availability_proxy'] = merged_df['year_scaled'] * (-50)
        
        # Sort by date for lag calculation
        merged_df['year'] = merged_df['year'].astype(int)
        merged_df['month'] = merged_df['month'].astype(int)
        merged_df['date'] = pd.to_datetime(
            merged_df['year'].astype(str) + '-' + 
            merged_df['month'].astype(str).str.zfill(2) + '-01'
        )
        merged_df = merged_df.sort_values('date').reset_index(drop=True)
        
        # Lag features
        lags = [1, 2, 3, 12]
        for lag in lags:
            merged_df[f'burn_lag{lag}'] = merged_df['burn_kbd'].shift(lag)
            merged_df[f'temp_lag{lag}'] = merged_df['temp_max_max_max'].shift(lag)
        
        # Moving averages
        windows = [2, 3, 6]
        for window in windows:
            merged_df[f'burn_ma{window}'] = merged_df['burn_kbd'].rolling(window, min_periods=1).mean()
            merged_df[f'temp_ma{window}'] = merged_df['temp_max_max_max'].rolling(window, min_periods=1).mean()
        
        # Year-over-year changes
        merged_df['burn_yoy'] = merged_df['burn_kbd'] / merged_df['burn_lag12'] - 1
        merged_df['temp_yoy'] = merged_df['temp_max_max_max'] - merged_df['temp_lag12']
        
        # Summer-specific features
        merged_df['summer_intensity'] = merged_df['is_summer'] * merged_df['temp_max_max_max']
        merged_df['peak_heat'] = merged_df['is_peak_summer'] * merged_df['extreme_44_sum']
        merged_df['dry_heat'] = merged_df['temp_max_max_max'] * (100 - merged_df['humidity_min_min']) / 100
        merged_df['temp_above_45'] = np.maximum(0, merged_df['temp_max_max_max'] - 45) ** 2
        merged_df['temp_above_43'] = np.maximum(0, merged_df['temp_max_max_max'] - 43) ** 1.5
        merged_df['cumulative_cdd'] = merged_df['cdd_24_sum'].rolling(3, min_periods=1).sum()
        merged_df['temp_regional_spread'] = merged_df['temp_mean_std_mean']
        
        # Drop NaN from lags
        merged_df = merged_df.dropna()
        
        print(f"✓ Created advanced features")
        print(f"  Final dataset: {len(merged_df)} samples, {len(merged_df.columns)} columns")
        
        self.merged_df = merged_df
    
    def train_model(self):
        """STEP 5: Train XGBoost model"""
        print("\n[STEP 5: TRAINING MODELS]")
        print("-"*60)
        
        # Prepare features
        exclude_cols = ['date', 'year', 'month', 'burn_kbd', 'is_summer', 'is_peak_summer']
        feature_cols = [col for col in self.merged_df.columns if col not in exclude_cols]
        
        X = self.merged_df[feature_cols].values
        y = self.merged_df['burn_kbd'].values
        is_summer = self.merged_df['is_summer'].values
        is_peak = self.merged_df['is_peak_summer'].values
        
        # Use last 12 months as test set for out-of-sample validation
        split_idx = len(X) - 12
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        is_summer_train, is_summer_test = is_summer[:split_idx], is_summer[split_idx:]
        is_peak_train, is_peak_test = is_peak[:split_idx], is_peak[split_idx:]
        
        print(f"Final split: Train={len(X_train)}, Test={len(X_test)} months")
        
        # Create sample weights
        sample_weights = np.ones(len(y_train))
        sample_weights[is_summer_train == 1] = 5
        sample_weights[is_peak_train == 1] = 10
        
        # Train XGBoost model (best performer)
        best_model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.02,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=5,
            random_state=42,
            verbosity=0
        )
        
        best_model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Final predictions
        y_pred = best_model.predict(X_test)
        
        # Save test predictions for later use
        test_predictions = pd.DataFrame({
            'date': self.merged_df.iloc[split_idx:]['date'],
            'actual_kbd': y_test,
            'predicted_kbd': y_pred
        })
        test_predictions.to_csv(os.path.join(self.working_dir, "model_test_predictions.csv"), index=False)
        
        # Calculate metrics
        final_mae = mean_absolute_error(y_test, y_pred)
        summer_mae = mean_absolute_error(y_test[is_summer_test == 1], y_pred[is_summer_test == 1]) if is_summer_test.sum() > 0 else 0
        peak_mae = mean_absolute_error(y_test[is_peak_test == 1], y_pred[is_peak_test == 1]) if is_peak_test.sum() > 0 else 0
        
        print(f"\nFinal Model Performance:")
        print(f"  Overall MAE: {final_mae:.1f} kbd")
        print(f"  Summer MAE: {summer_mae:.1f} kbd")
        print(f"  Peak MAE: {peak_mae:.1f} kbd")
        
        # Store results
        self.model = best_model
        self.feature_cols = feature_cols
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred
        self.is_summer_test = is_summer_test
        
        self.performance_metrics = {
            'overall_mae': final_mae,
            'summer_mae': summer_mae,
            'peak_mae': peak_mae,
            'summer_mape': (np.mean(np.abs(y_pred[is_summer_test == 1] - y_test[is_summer_test == 1]) / y_test[is_summer_test == 1]) * 100) if is_summer_test.sum() > 0 else 0
        }
    
    def save_model(self, model_name="summer_peak_model_cv.pkl"):
        """STEP 6: Save trained model"""
        print("\n[STEP 6: SAVING MODEL]")
        print("-"*60)
        
        model_package = {
            'model': self.model,
            'model_name': 'XGBoost_Peak',
            'feature_cols': self.feature_cols,
            'final_performance': self.performance_metrics
        }
        
        model_path = os.path.join(self.working_dir, model_name)
        with open(model_path, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f"✓ Model saved to {model_path}")
        print("\n✓ Model training complete!")
        print("  You can now run the forecast code to make predictions")
    
    def train(self):
        """Execute full training pipeline"""
        self.load_burn_data()
        self.fetch_weather_data()
        self.aggregate_weather_monthly()
        self.engineer_features()
        self.train_model()
        self.save_model()


if __name__ == "__main__":
    trainer = SaudiCrudeBurnTrainer()
    trainer.train()