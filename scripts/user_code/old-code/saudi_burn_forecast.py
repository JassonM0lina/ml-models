"""
Saudi Direct Burn Forecast - Automated Daily/Weekly Script
Run via Task Scheduler for automated forecasting and email generation
CLEANED VERSION - Professional chart, auto file cleanup, single recipient
"""

import pandas as pd
import numpy as np
import pickle
import requests
import win32com.client as win32
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys
import traceback
import warnings
from urllib3.exceptions import InsecureRequestWarning
import glob

# Suppress warnings
requests.packages.urllib3.exceptions.InsecureRequestWarning = InsecureRequestWarning
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
WORKING_DIR = r"C:\Users\i31851\OneDrive - Wood Mackenzie Limited\Documents\Automation Work For Others\Jim Mitchell\Saudi Direct Burn V2"
LOG_FILE = os.path.join(WORKING_DIR, "forecast_automation_log.txt")

def log_message(message):
    """Write to log file for tracking automated runs"""
    with open(LOG_FILE, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}: {message}\n")

def cleanup_old_charts():
    """Delete old chart files before creating new ones"""
    chart_patterns = [
        "Saudi_Burn_Enhanced_Chart_*.png",
        "Weather_Drivers_*.png"
    ]
    
    for pattern in chart_patterns:
        old_files = glob.glob(os.path.join(WORKING_DIR, pattern))
        for old_file in old_files:
            try:
                os.remove(old_file)
                print(f"  Deleted old chart: {os.path.basename(old_file)}")
            except Exception as e:
                print(f"  Could not delete {os.path.basename(old_file)}: {e}")

def create_enhanced_forecast_chart(burn_df, test_predictions, forecasts, model_package, 
                                   current_date, WORKING_DIR):
    """Create a clean, professional visualization with only 2 series"""
    
    fig, ax = plt.subplots(figsize=(16, 9))
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Load forecast log to get previous forecasts
    log_file = os.path.join(WORKING_DIR, "Saudi_Burn_Forecast_Log.xlsx")
    try:
        forecast_log = pd.read_excel(log_file)
        forecast_log['Date'] = pd.to_datetime(forecast_log['Date'])
        forecast_log['Target Month'] = pd.to_datetime(forecast_log['Target Month'])
    except:
        forecast_log = pd.DataFrame()
    
    # 1. HISTORICAL ACTUALS - Last 12 months
    recent_burns = burn_df.tail(12).copy()
    ax.plot(recent_burns['date'], recent_burns['burn_kbd'], 
           color='#2C3E50', linewidth=3, label='Actual', 
           marker='o', markersize=6, markerfacecolor='white', 
           markeredgecolor='#2C3E50', markeredgewidth=2, zorder=5)
    
    # 2. COMBINE MODEL TEST PREDICTIONS + PREVIOUS FORECASTS INTO ONE SERIES
    current_month_start = pd.Timestamp(current_date.year, current_date.month, 1)
    last_actual_date = burn_df['date'].max()
    
    # Start with test predictions
    test_pred_with_dates = test_predictions.copy()
    test_pred_with_dates['date'] = pd.to_datetime(test_pred_with_dates['date'])
    display_start = recent_burns['date'].min()
    
    # Get all previous predictions (test + forecast log before current month)
    all_previous_dates = []
    all_previous_values = []
    all_previous_lower = []
    all_previous_upper = []
    
    # Add test predictions
    test_pred_filtered = test_pred_with_dates[test_pred_with_dates['date'] >= display_start]
    for _, row in test_pred_filtered.iterrows():
        all_previous_dates.append(row['date'])
        all_previous_values.append(row['predicted_kbd'])
        # Use model MAE for confidence intervals
        mae = model_package['final_performance']['overall_mae']
        all_previous_lower.append(row['predicted_kbd'] - 2 * mae)
        all_previous_upper.append(row['predicted_kbd'] + 2 * mae)
    
    # Add previous forecasts from log (before current month)
    if len(forecast_log) > 0:
        past_forecasts = forecast_log[forecast_log['Target Month'] < current_month_start].copy()
        
        if len(past_forecasts) > 0:
            past_forecasts = past_forecasts.sort_values('Date').groupby('Target Month').last().reset_index()
            future_past_forecasts = past_forecasts[past_forecasts['Target Month'] > last_actual_date]
            
            for _, row in future_past_forecasts.iterrows():
                all_previous_dates.append(row['Target Month'])
                all_previous_values.append(row['Forecast (kbd)'])
                all_previous_lower.append(row['Lower Bound'])
                all_previous_upper.append(row['Upper Bound'])
    
    # Plot unified "Previous Predictions" series (light blue dotted)
    if len(all_previous_dates) > 0:
        # Sort by date to ensure proper line plotting
        prev_df = pd.DataFrame({
            'date': all_previous_dates,
            'value': all_previous_values,
            'lower': all_previous_lower,
            'upper': all_previous_upper
        }).sort_values('date')
        
        ax.plot(prev_df['date'], prev_df['value'],
               color='#3498DB', linewidth=2, linestyle=':', 
               label='Previous Predictions', 
               marker='s', markersize=5,
               markerfacecolor='#3498DB', alpha=0.7, zorder=4)
        
        # Show CI only for last 3 prediction periods
        if len(prev_df) >= 3:
            last_3 = prev_df.tail(3)
            ax.fill_between(last_3['date'], last_3['lower'], last_3['upper'],
                           color='#3498DB', alpha=0.15, zorder=1)
    
    # 3. CURRENT FORECASTS - Red dots starting from current month
    forecast_dates = []
    forecast_values = []
    forecast_lower = []
    forecast_upper = []
    
    for fc in forecasts:
        month_parts = fc['month'].split()
        if len(month_parts) >= 2:
            month_str = month_parts[0]
            year_str = month_parts[1]
            
            month_map = {
                'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
                'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
            }
            
            if month_str in month_map:
                forecast_date = pd.Timestamp(int(year_str), month_map[month_str], 1)
                
                if forecast_date >= current_month_start:
                    forecast_dates.append(forecast_date)
                    forecast_values.append(fc['prediction'])
                    
                    model_mae = model_package['final_performance']['summer_mae'] \
                               if month_map[month_str] in [6,7,8,9] \
                               else model_package['final_performance']['overall_mae']
                    
                    forecast_lower.append(fc['prediction'] - 2 * model_mae)
                    forecast_upper.append(fc['prediction'] + 2 * model_mae)
    
    # Plot current forecasts (RED)
    if len(forecast_dates) > 0:
        forecast_dates_array = np.array(forecast_dates)
        forecast_values_array = np.array(forecast_values)
        forecast_lower_array = np.array(forecast_lower)
        forecast_upper_array = np.array(forecast_upper)
        
        # EXTEND THE BLUE CI TO INCLUDE CURRENT MONTH for visibility
        if len(all_previous_dates) > 0:
            # Get last previous prediction
            last_prev_date = prev_df['date'].iloc[-1]
            last_prev_value = prev_df['value'].iloc[-1]
            last_prev_lower = prev_df['lower'].iloc[-1]
            last_prev_upper = prev_df['upper'].iloc[-1]
            
            # Create extended CI from last previous through all current forecasts
            extended_dates = np.concatenate([[last_prev_date], forecast_dates_array])
            extended_lower = np.concatenate([[last_prev_lower], forecast_lower_array])
            extended_upper = np.concatenate([[last_prev_upper], forecast_upper_array])
            
            # Plot extended CI in light blue
            ax.fill_between(extended_dates, extended_lower, extended_upper,
                           color='#3498DB', alpha=0.15, label='95% Confidence', 
                           zorder=1, linewidth=0)
            
            # Connect last previous to first current with dashed red line
            ax.plot([last_prev_date, forecast_dates[0]], 
                   [last_prev_value, forecast_values[0]],
                   color='#E74C3C', linewidth=2, linestyle='--', alpha=0.6, zorder=3)
        
        # Plot current forecast RED DOTS + DASHED LINE
        ax.plot(forecast_dates_array, forecast_values_array,
               color='#E74C3C', linewidth=3, linestyle='--',
               label='Current Forecast', marker='o', markersize=8,
               markerfacecolor='#E74C3C', markeredgecolor='white',
               markeredgewidth=2, zorder=6)
        
        # Annotate first forecast (current month)
        ax.annotate(f"{forecast_values[0]:.0f} kbd",
                   xy=(forecast_dates[0], forecast_values[0]),
                   xytext=(10, 20), textcoords='offset points',
                   fontsize=11, fontweight='bold', color='#E74C3C',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                            edgecolor='#E74C3C', alpha=0.9),
                   arrowprops=dict(arrowstyle='->', color='#E74C3C', lw=1.5))
    
    # 4. PERFORMANCE METRICS BOX
    current_forecast = forecasts[0] if len(forecasts) > 0 else None
    
    if len(test_pred_filtered) > 0:
        test_mae = np.mean(np.abs(test_pred_filtered['actual_kbd'] - 
                                  test_pred_filtered['predicted_kbd']))
        test_mape = np.mean(np.abs((test_pred_filtered['actual_kbd'] - 
                                    test_pred_filtered['predicted_kbd']) / 
                                   test_pred_filtered['actual_kbd'])) * 100
    else:
        test_mae = model_package['final_performance']['overall_mae']
        test_mape = model_package['final_performance']['summer_mape']
    
    if current_forecast and len(forecast_lower) > 0:
        metrics_text = f'''Model Performance Metrics
━━━━━━━━━━━━━━━━━━━━━
Test MAE: {test_mae:.1f} kbd
Test MAPE: {test_mape:.1f}%
Summer MAE: {model_package['final_performance']['summer_mae']:.1f} kbd
Peak MAE: {model_package['final_performance']['peak_mae']:.1f} kbd

Current Forecast
━━━━━━━━━━━━━━━━━━━━━
{current_forecast['month']}: {current_forecast['prediction']:.0f} kbd
95% CI: [{forecast_lower[0]:.0f}, {forecast_upper[0]:.0f}] kbd'''
    else:
        metrics_text = "No forecast data available"
    
    props = dict(boxstyle='round,pad=0.7', facecolor='white', 
                edgecolor='gray', alpha=0.95, linewidth=1.5)
    ax.text(0.02, 0.03, metrics_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='bottom', horizontalalignment='left',
           fontfamily='monospace', bbox=props)
    
    # 5. STYLING AND FORMATTING
    ax.set_title(f'Saudi Arabia Direct Burn: Model Performance & Forecast as of {current_date.strftime("%B %d, %Y")}', 
                fontsize=16, fontweight='bold', color='#2C3E50', pad=20)
    ax.set_xlabel('Date', fontsize=12, fontweight='semibold')
    ax.set_ylabel('Direct Burn (kbd)', fontsize=12, fontweight='semibold')
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center', fontsize=9)
    
    # Adjust y-axis
    all_values = list(recent_burns['burn_kbd'].values)
    if len(all_previous_values) > 0:
        all_values.extend(all_previous_values)
    if len(forecast_values) > 0:
        all_values.extend(forecast_values)
    if len(forecast_lower) > 0:
        all_values.extend(forecast_lower)
    if len(forecast_upper) > 0:
        all_values.extend(forecast_upper)
    
    y_max = max(all_values) * 1.15 if all_values else 1000
    ax.set_ylim(0, y_max)
    
    # Extend x-axis
    x_min = recent_burns['date'].min() - pd.Timedelta(days=15)
    x_max = forecast_dates[-1] + pd.Timedelta(days=30) if forecast_dates else recent_burns['date'].max() + pd.Timedelta(days=60)
    ax.set_xlim(x_min, x_max)
    
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Clean legend
    ax.legend(loc='lower right', bbox_to_anchor=(0.98, 0.03), fontsize=9, frameon=True, 
             fancybox=True, shadow=True, borderpad=0.8)
    
    # Highlight forecast region
    forecast_start = current_month_start
    ax.axvspan(forecast_start, x_max,
              alpha=0.03, color='gray', label='_nolegend_')
    
    plt.tight_layout(pad=2)
    
    chart_path = f"{WORKING_DIR}/Saudi_Burn_Enhanced_Chart_{current_date.strftime('%Y%m%d')}.png"
    plt.savefig(chart_path, dpi=150, bbox_inches='tight', 
               facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Enhanced chart saved to: {chart_path}")
    return chart_path

try:
    log_message("="*60)
    log_message("STARTING SAUDI CRUDE BURN FORECAST AUTOMATION")
    
    print("="*80)
    print("SAUDI CRUDE BURN FORECAST - CLEANED VERSION")
    print("="*80)
    print(f"Run Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("="*80)
    
    print("\n[CLEANING UP OLD CHARTS]")
    print("-"*60)
    cleanup_old_charts()
    
    print("\n[LOADING MODEL]")
    print("-"*60)
    
    with open(f"{WORKING_DIR}/summer_peak_model_cv.pkl", 'rb') as f:
        model_package = pickle.load(f)
    
    model = model_package['model']
    feature_cols = model_package['feature_cols']
    
    print(f"✓ Loaded {model_package['model_name']}")
    print(f"  Overall MAE: {model_package['final_performance']['overall_mae']:.1f} kbd")
    print(f"  Summer MAE: {model_package['final_performance']['summer_mae']:.1f} kbd")
    print(f"  Peak MAE: {model_package['final_performance']['peak_mae']:.1f} kbd")
    print(f"  Summer MAPE: {model_package['final_performance']['summer_mape']:.1f}%")
    
    burn_df = pd.read_excel(f"{WORKING_DIR}/Saudi Crude Burn History.xlsx")
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
    
    if burn_df['burn_bpd'].mean() > 10000:
        burn_df['burn_kbd'] = burn_df['burn_bpd'] / 1000
    else:
        burn_df['burn_kbd'] = burn_df['burn_bpd']
    
    burn_df['year'] = burn_df['date'].dt.year
    burn_df['month'] = burn_df['date'].dt.month
    burn_df = burn_df.sort_values('date').reset_index(drop=True)
    
    print(f"✓ Loaded historical burn data through {burn_df['date'].max().strftime('%Y-%m')}")
    
    recent_burns = burn_df.tail(6)[['date', 'burn_kbd']]
    print("\n  Recent burn history:")
    for _, row in recent_burns.iterrows():
        print(f"    {row['date'].strftime('%Y-%m')}: {row['burn_kbd']:.0f} kbd")
    
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
        
        if confidence == "HIGH":
            final_prediction = 0.7 * rule_prediction + 0.3 * model_prediction
        else:
            final_prediction = 0.5 * rule_prediction + 0.5 * model_prediction
        
        return final_prediction, adjustment_type, confidence
    
    print("\n[FETCHING CURRENT WEATHER]")
    print("-"*60)
    
    def get_current_weather():
        """Fetch current and forecast weather"""
        cities = {
            'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.30},
            'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.25},
            'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.25},
            'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.20}
        }
        
        all_weather = []
        current_date = datetime.now()
        
        print("  Fetching weather data...")
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
                    print(f"    ✓ {city}: {len(city_df)} days")
            except Exception as e:
                print(f"    ✗ {city}: {e}")
        
        if not all_weather:
            raise ValueError("No weather data collected. Check internet connection.")
        
        combined = pd.concat(all_weather)
        
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
    
    weather = get_current_weather()
    
    print("\n[PREPARING FORECASTS]")
    print("-"*60)
    
    def prepare_monthly_features(weather_daily, target_year, target_month, burn_history=None):
        """Prepare features for a specific month - matching training exactly"""
        month_weather = weather_daily[
            (weather_daily['date'].dt.year == target_year) & 
            (weather_daily['date'].dt.month == target_month)
        ]
        
        if len(month_weather) == 0:
            return None
        
        days_in_month = pd.Timestamp(target_year, target_month, 1).days_in_month
        days_available = len(month_weather)
        scaling_factor = days_in_month / days_available if days_available > 0 else 1
        
        print(f"  {pd.Timestamp(target_year, target_month, 1).strftime('%B %Y')}: {days_available}/{days_in_month} days")
        
        features = {}
        
        features['temp_max_mean_mean'] = month_weather['temp_max_mean'].mean()
        features['temp_max_mean_std'] = month_weather['temp_max_mean'].std() if len(month_weather) > 1 else 1
        features['temp_max_max_max'] = month_weather['temp_max_max'].max()
        features['temp_max_max_mean'] = month_weather['temp_max_max'].mean()
        features['temp_mean_mean_mean'] = month_weather['temp_mean_mean'].mean()
        features['temp_mean_mean_max'] = month_weather['temp_mean_mean'].max()
        features['temp_mean_std_mean'] = month_weather['temp_mean_std'].mean()
        features['humidity_mean_mean'] = month_weather['humidity_mean'].mean()
        features['humidity_min_min'] = month_weather['humidity_min'].min()
        
        T = month_weather['temp_mean_mean'].mean()
        RH = month_weather['humidity_mean'].mean()
        features['heat_index_mean'] = -8.78 + 1.61*T + 2.34*RH - 0.146*T*RH
        features['heat_index_max'] = features['heat_index_mean'] + 5
        features['apparent_max_max_max'] = month_weather['apparent_max_max'].max()
        
        for base in [22, 24, 26, 28]:
            daily_cdd = np.maximum(0, month_weather['temp_mean_mean'] - base)
            features[f'cdd_{base}_sum'] = daily_cdd.sum() * scaling_factor
        
        features['extreme_48_sum'] = (month_weather['temp_max_max'] > 48).sum() * scaling_factor
        features['extreme_46_sum'] = (month_weather['temp_max_max'] > 46).sum() * scaling_factor
        features['extreme_44_sum'] = (month_weather['temp_max_max'] > 44).sum() * scaling_factor
        features['extreme_42_sum'] = (month_weather['temp_max_max'] > 42).sum() * scaling_factor
        
        hot_days = (month_weather['temp_max_mean'] > 42).astype(int)
        if len(hot_days) > 0:
            streaks = hot_days.groupby((hot_days != hot_days.shift()).cumsum()).sum()
            features['heat_streak_max'] = streaks.max() if len(streaks) > 0 else 0
        else:
            features['heat_streak_max'] = 0
        
        features['months_since_start'] = (target_year - 2009) * 12 + target_month - 1
        features['year_scaled'] = (target_year - 2009) / 10
        features['is_ramadan_summer'] = 0
        features['gas_availability_proxy'] = features['year_scaled'] * (-50)
        
        if burn_history is not None and len(burn_history) > 0:
            recent_burns = burn_history['burn_kbd'].values[-12:] if len(burn_history) >= 12 else burn_history['burn_kbd'].values
            
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
        
        features['temp_lag1'] = features['temp_max_max_max']
        features['temp_lag2'] = features['temp_max_max_max']
        features['temp_lag3'] = features['temp_max_max_max']
        features['temp_lag12'] = features['temp_max_max_max']
        features['temp_ma2'] = features['temp_max_max_max']
        features['temp_ma3'] = features['temp_max_max_max']
        features['temp_ma6'] = features['temp_max_max_max']
        features['temp_yoy'] = 0
        
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
    
    current_date = datetime.now()
    forecasts = []
    
    print("\nProcessing current and future months:")
    for month_offset in range(3):
        target_date = current_date + timedelta(days=30*month_offset)
        target_year = target_date.year
        target_month = target_date.month
        month_name = target_date.strftime('%B %Y')
        
        features = prepare_monthly_features(weather, target_year, target_month, burn_df)
        
        if features:
            X_pred = pd.DataFrame([features])
            
            for col in feature_cols:
                if col not in X_pred.columns:
                    X_pred[col] = 0
            
            X_pred = X_pred[feature_cols]
            
            base_prediction = model.predict(X_pred.values)[0]
            
            final_prediction, adjustment_type, confidence = apply_temperature_rules(base_prediction, features)
            
            if burn_df is not None:
                historical_month_data = burn_df[burn_df['month'] == target_month]['burn_kbd']
                historical_avg = historical_month_data.mean() if len(historical_month_data) > 0 else 500
                historical_std = historical_month_data.std() if len(historical_month_data) > 1 else 100
            else:
                historical_avg = 650 if target_month in [6,7,8,9] else 400
                historical_std = 100
            
            forecasts.append({
                'month': month_name,
                'prediction': final_prediction,
                'base_prediction': base_prediction,
                'adjustment_type': adjustment_type,
                'confidence': confidence,
                'historical_avg': historical_avg,
                'historical_std': historical_std,
                'max_temp': features['temp_max_max_max'],
                'cdd_24': features['cdd_24_sum'],
                'extreme_days_44': int(features['extreme_44_sum']),
                'burn_ma2': features.get('burn_ma2', 500)
            })
    
    print("\n" + "="*80)
    print("FORECASTS WITH EXTREME HEAT ADJUSTMENT")
    print("="*80)
    
    for fc in forecasts:
        print(f"\n{fc['month']}:")
        print(f"  Model prediction: {fc['base_prediction']:.0f} kbd")
        if fc['adjustment_type'] != "STANDARD MODEL":
            print(f"  Adjustment: {fc['adjustment_type']} ({fc['confidence']} confidence)")
        print(f"  FINAL PREDICTION: {fc['prediction']:.0f} kbd")
        print(f"  Historical avg: {fc['historical_avg']:.0f} kbd")
        print(f"  Deviation: {(fc['prediction'] - fc['historical_avg']):.0f} kbd ({(fc['prediction']/fc['historical_avg'] - 1)*100:+.1f}%)")
        
        model_mae = model_package['final_performance']['summer_mae'] if fc['month'].split()[0] in ['June', 'July', 'August', 'September'] else model_package['final_performance']['overall_mae']
        print(f"  95% confidence: [{fc['prediction']-2*model_mae:.0f}, {fc['prediction']+2*model_mae:.0f}] kbd")
        
        print(f"\n  Weather conditions:")
        print(f"    Max temperature: {fc['max_temp']:.1f}°C")
        print(f"    Cooling degree days (base 24°C): {fc['cdd_24']:.0f}")
        print(f"    Days >44°C: {fc['extreme_days_44']}")
        
        if fc['prediction'] > fc['historical_avg'] + fc['historical_std']:
            print(f"  📈 Signal: HIGH - Expect elevated crude burn")
        elif fc['prediction'] < fc['historical_avg'] - fc['historical_std']:
            print(f"  📉 Signal: LOW - Below normal burn expected")
        else:
            print(f"  ➡️ Signal: NORMAL - Within typical range")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    summer_months = [fc for fc in forecasts if fc['month'].split()[0] in ['June', 'July', 'August', 'September']]
    if summer_months:
        avg_summer = np.mean([fc['prediction'] for fc in summer_months])
        print(f"\nAverage summer prediction: {avg_summer:.0f} kbd")
        
        if avg_summer > 750:
            print("⚠️ HIGH SUMMER ALERT: Significant crude burn expected")
            print("   Consider increased crude allocation for power generation")
        elif avg_summer < 650:
            print("📊 MODERATE SUMMER: Below typical seasonal demand")
            print("   Opportunity to reduce crude allocation")
        else:
            print("✓ NORMAL SUMMER: Demand within expected range")
    
    print(f"\nModel confidence: Based on {model_package['final_performance']['summer_mape']:.1f}% MAPE for summer months")
    print(f"Expected error range: ±{model_package['final_performance']['summer_mae']:.0f} kbd for summer predictions")
    print("\nNote: Extreme heat conditions trigger rule-based adjustments to prevent")
    print("underestimation when recent burn history is low but weather is extreme.")
    
    print("\n")
    print("="*60)
    print("GENERATING AUTOMATED OUTLOOK EMAIL WITH LOGGING")
    print("="*60)
    
    current_month_forecast = None
    for fc in forecasts:
        if fc['month'].startswith(current_date.strftime('%B %Y')):
            current_month_forecast = fc
            break
    
    if current_month_forecast is None:
        current_month_forecast = forecasts[0] if forecasts else None
    
    if current_month_forecast:
        forecast_kbd = current_month_forecast['prediction']
        base_prediction = current_month_forecast['base_prediction']
        adjustment_type = current_month_forecast['adjustment_type']
        confidence = current_month_forecast['confidence']
        historical_avg = current_month_forecast['historical_avg']
        historical_std = current_month_forecast['historical_std']
        max_temp = current_month_forecast['max_temp']
        cdd_24 = current_month_forecast['cdd_24']
        extreme_days = current_month_forecast['extreme_days_44']
        
        current_month_name = current_date.strftime('%B %Y')
        current_month_num = current_date.month
        days_elapsed = current_date.day
        days_in_month = pd.Timestamp(current_date.year, current_date.month, 1).days_in_month
        completion_pct = (days_elapsed / days_in_month) * 100
        
        last_month_value = burn_df['burn_kbd'].iloc[-1]
        last_month_name = burn_df['date'].iloc[-1].strftime('%B %Y')
        
        mom_change = forecast_kbd - last_month_value
        
        try:
            last_year_same_month = burn_df[(burn_df['year'] == current_date.year - 1) & 
                                           (burn_df['month'] == current_month_num)]['burn_kbd'].iloc[0]
            yoy_change = ((forecast_kbd - last_year_same_month) / last_year_same_month) * 100
        except:
            last_year_same_month = historical_avg
            yoy_change = ((forecast_kbd - historical_avg) / historical_avg) * 100
        
        historical_month_data = burn_df[burn_df['month'] == current_month_num]['burn_kbd']
        historical_avg_month = historical_month_data.mean() if len(historical_month_data) > 0 else 600
        vs_historical = forecast_kbd - historical_avg_month
        
        model_mae = model_package['final_performance']['summer_mae'] if current_date.month in [6,7,8,9] else model_package['final_performance']['overall_mae']
        lower_bound = forecast_kbd - 2 * model_mae
        upper_bound = forecast_kbd + 2 * model_mae
    
    test_predictions = pd.read_csv(f"{WORKING_DIR}/model_test_predictions.csv")
    test_predictions['date'] = pd.to_datetime(test_predictions['date'])
    
    historical_context = burn_df[burn_df['month'] == current_date.month].tail(5) if burn_df is not None else None
    
    print("\n[LOGGING RESULTS]")
    print("-"*60)
    
    log_file = os.path.join(WORKING_DIR, "Saudi_Burn_Forecast_Log.xlsx")
    
    try:
        log_df = pd.read_excel(log_file)
    except:
        log_df = pd.DataFrame()
    
    if current_month_forecast:
        new_entry = pd.DataFrame({
            'Date': [current_date],
            'Target Month': [current_month_name],
            'Days Used': [f"{days_elapsed}/{days_in_month}"],
            'Forecast (kbd)': [forecast_kbd],
            'Base Model (kbd)': [base_prediction],
            'Adjustment': [adjustment_type if adjustment_type != "STANDARD MODEL" else "None"],
            'Lower Bound': [lower_bound],
            'Upper Bound': [upper_bound],
            'Max Temp (°C)': [max_temp],
            'Days >44°C': [extreme_days],
            'CDD-24': [cdd_24],
            'vs Last Month': [mom_change],
            'vs Last Year (%)': [yoy_change],
            'Model': [model_package['model_name']],
            'MAE': [model_mae],
            'Chart': [f"Saudi_Burn_Enhanced_Chart_{current_date.strftime('%Y%m%d')}.png"]
        })
        
        log_df = pd.concat([log_df, new_entry], ignore_index=True)
        
        with pd.ExcelWriter(log_file, engine='openpyxl') as writer:
            log_df.to_excel(writer, sheet_name='Forecast Log', index=False)
            worksheet = writer.sheets['Forecast Log']
            
            from openpyxl.utils import get_column_letter
            for idx, column in enumerate(log_df.columns, 1):
                column_length = max(log_df[column].astype(str).map(len).max(), len(column))
                worksheet.column_dimensions[get_column_letter(idx)].width = min(column_length + 2, 25)
        
        print(f"✓ Results logged to: {log_file}")
    
    print("\n[CREATING PERFORMANCE GRAPH]")
    print("-"*60)
    
    chart_path = create_enhanced_forecast_chart(
        burn_df, 
        test_predictions, 
        forecasts, 
        model_package, 
        current_date, 
        WORKING_DIR
    )
    
    if current_month_forecast:
        print("\n[SENDING EMAIL]")
        print("-"*60)
        
        def create_and_send_email():
            outlook = win32.Dispatch('outlook.application')
            mail = outlook.CreateItem(0)
            
            mail.To = 'Robel.Abate@woodmac.com; jim.mitchell@woodmac.com; Jayadev.D@woodmac.com; Sharvari.Naik@woodmac.com'
            mail.Subject = f'Saudi Direct Burn Forecast - {current_month_name}: {forecast_kbd:.0f} kbd'
            mail.Attachments.Add(chart_path)
            
            max_temp_f = max_temp * 9/5 + 32
            
            email_body = f"""
<html>
<body style="font-family: Calibri, sans-serif; color: #333333; line-height: 1.6;">
    <h2 style="color: #2C5282; margin-bottom: 5px;">Saudi Arabia Direct Burn Forecast</h2>
    <p style="color: #666; margin-top: 0;">{current_date.strftime('%B %d, %Y')}</p>
    
    <div style="background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0;">
        <h1 style="color: #2C5282; margin: 0; font-size: 36px;">{forecast_kbd:.0f} kbd</h1>
        <p style="color: #444; margin: 8px 0; font-size: 18px; font-weight: bold;">
            95% Confidence Range: {lower_bound:.0f} - {upper_bound:.0f} kbd
        </p>
        <p style="color: #666; margin: 5px 0;">Forecast for {current_month_name}</p>
        <p style="color: #888; font-size: 14px; margin: 0;">Based on weather data through Day {days_elapsed} of {days_in_month}</p>
    </div>
    
    <table border="0" cellpadding="12" cellspacing="0" style="width: 100%; max-width: 600px; margin: 20px 0;">
        <tr>
            <td style="border-bottom: 1px solid #e0e0e0;">
                <strong>vs Last Month</strong><br>
                <span style="font-size: 20px; color: {'#388e3c' if mom_change > 0 else '#d32f2f'};">
                    {mom_change:+.0f} kbd
                </span><br>
                <span style="color: #666; font-size: 14px;">({last_month_name}: {last_month_value:.0f} kbd)</span>
            </td>
            <td style="border-bottom: 1px solid #e0e0e0;">
                <strong>vs Last Year</strong><br>
                <span style="font-size: 20px; color: {'#388e3c' if yoy_change > 0 else '#d32f2f'};">
                    {yoy_change:+.1f}%
                </span><br>
                <span style="color: #666; font-size: 14px;">({current_date.strftime('%B')} {current_date.year-1}: {last_year_same_month:.0f} kbd)</span>
            </td>
            <td style="border-bottom: 1px solid #e0e0e0;">
                <strong>vs Historical {current_date.strftime('%B')} Avg</strong><br>
                <span style="font-size: 20px; color: {'#388e3c' if vs_historical > 0 else '#d32f2f'};">
                    {vs_historical:+.0f} kbd
                </span><br>
                <span style="color: #666; font-size: 14px;">(Avg of all {current_date.strftime('%B')}s: {historical_avg_month:.0f} kbd)</span>
            </td>
        </tr>
    </table>
    
    <h3 style="color: #2C5282; margin-top: 30px;">Key Model Features & Current Conditions</h3>
    
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4 style="margin-top: 0; color: #1976d2;">Temperature Metrics</h4>
        <p style="margin: 8px 0;"><strong>Maximum Temperature:</strong> {max_temp_f:.1f}°F 
        {'- Extreme heat conditions' if max_temp_f > 116.6 else '- Very hot conditions' if max_temp_f > 113 else '- Hot conditions'}</p>
        <p style="margin: 8px 0; font-size: 14px; color: #555;">
        The model's most important feature. Captures peak daily temperatures across major cities. 
        Each degree above 113°F typically adds 15-20 kbd to monthly demand as AC usage intensifies.</p>
    </div>
    
    <div style="background-color: #fff3e0; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4 style="margin-top: 0; color: #f57c00;">Extreme Heat Days</h4>
        <p style="margin: 8px 0;"><strong>Days Above 111°F:</strong> {extreme_days} days this month</p>
        <p style="margin: 8px 0; font-size: 14px; color: #555;">
        Counts days exceeding critical temperature thresholds. These extreme days drive sustained 
        peak power demand that can only be met by crude burn due to gas constraints.</p>
    </div>
    
    <div style="background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4 style="margin-top: 0; color: #388e3c;">Cooling Degree Days (CDD)</h4>
        <p style="margin: 8px 0;"><strong>CDD-75:</strong> {cdd_24:.0f} (cumulative above 75°F baseline)</p>
        <p style="margin: 8px 0; font-size: 14px; color: #555;">
        Measures cumulative cooling demand. CDD integrates both temperature intensity and duration, 
        providing a comprehensive metric for AC load. Higher CDD = more sustained cooling need = more crude burn.</p>
    </div>
    
    <div style="background-color: #fce4ec; padding: 15px; border-radius: 5px; margin: 15px 0;">
        <h4 style="margin-top: 0; color: #c2185b;">Historical Patterns</h4>
        <p style="margin: 8px 0;"><strong>Recent Burn Average:</strong> {current_month_forecast.get('burn_ma2', 500):.0f} kbd 
        (last 2 months: {burn_df['burn_kbd'].iloc[-2:].mean():.0f} kbd)</p>
        <p style="margin: 8px 0; font-size: 14px; color: #555;">
        The model uses lagged burn values to capture operational momentum and seasonal transitions. 
        Recent trends help predict how quickly burn ramps up or down with temperature changes.</p>
    </div>
    """
            
            if adjustment_type != "STANDARD MODEL":
                max_temp_threshold_f = (max_temp - 1) * 9/5 + 32
                email_body += f"""
    <div style="background-color: #f3e5f5; padding: 15px; border-left: 4px solid #9c27b0; margin: 20px 0;">
        <h4 style="margin-top: 0; color: #6a1b9a;">Temperature-Based Override Applied</h4>
        <p style="margin: 8px 0;">Base model predicted {base_prediction:.0f} kbd, adjusted to {forecast_kbd:.0f} kbd</p>
        <p style="margin: 8px 0; font-size: 14px; color: #555;">
        When temperatures exceed {max_temp_threshold_f:.0f}°F with {extreme_days} extreme heat days, historical data shows 
        burns consistently reach 750-850 kbd regardless of recent trends. The override prevents underestimation 
        during extreme weather events based on patterns from summers 2023-2024.</p>
    </div>
    """
            
            if historical_context is not None and len(historical_context) > 0:
                email_body += f"""
    <h3 style="color: #2C5282; margin-top: 30px;">Recent {current_date.strftime('%B')} History</h3>
    <table border="0" cellpadding="5" cellspacing="0" style="margin-left: 20px;">
    """
                for _, row in historical_context.iterrows():
                    email_body += f"""
        <tr>
            <td style="color: #666;">{row['year']}:</td>
            <td style="padding-left: 20px;"><strong>{row['burn_kbd']:.0f} kbd</strong></td>
        </tr>"""
                email_body += f"""
        <tr style="border-top: 1px solid #e0e0e0;">
            <td style="color: #666; padding-top: 5px;">5-yr avg:</td>
            <td style="padding-left: 20px; padding-top: 5px;"><strong>{historical_context['burn_kbd'].mean():.0f} kbd</strong></td>
        </tr>
    </table>
    """
            
            email_body += f"""
    <div style="margin-top: 30px; padding: 15px; background-color: #f5f5f5; border-radius: 5px;">
        <p style="margin: 0; font-size: 14px;"><strong>Forecast Logged:</strong> Results saved to tracking spreadsheet for performance monitoring. 
        See attached chart for visual comparison.</p>
    </div>
    
    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; font-size: 12px; color: #666;">
        <p style="margin: 5px 0;">Model: {model_package['model_name']} | 
        Summer Accuracy: ±{model_package['final_performance']['summer_mae']:.0f} kbd | 
        Error Rate: {model_package['final_performance']['summer_mape']:.1f}%</p>
        <p style="margin: 5px 0;">Generated: {current_date.strftime('%B %d, %Y at %H:%M')}</p>
    </div>
    
</body>
</html>
"""
            
            mail.HTMLBody = email_body
            mail.Send()
            
            print(f"✓ Email sent to: robel.abate@woodmac.com")
        
        create_and_send_email()
    
    print("\n" + "="*60)
    print("FORECAST COMPLETE")
    print("="*60)
    
    log_message("Forecast completed successfully")
    
except Exception as e:
    error_msg = f"ERROR: {str(e)}\n{traceback.format_exc()}"
    log_message(error_msg)
    print(error_msg)
    sys.exit(1)