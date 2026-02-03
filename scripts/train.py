"""
Training Script Template for SageMaker Pipeline
Combines: Data Preprocessing + Model Training + Evaluation
"""

import os
import argparse
import pickle
import json
import shutil
import boto3
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from math import sqrt

# Model-specific imports
import xgboost as xgb
import requests
from datetime import datetime, timedelta
from math import sqrt
import warnings
warnings.filterwarnings('ignore')


def upload_evaluation_to_s3(evaluation_data: dict, eval_s3_path: str) -> str:
    """
    Upload evaluation results to S3 evaluation_data folder.

    Args:
        evaluation_data: Dict with metrics, predictions, actuals
        eval_s3_path: S3 URI of evaluation_data folder (e.g., s3://bucket/projects/user/project/project/evaluation_data/)

    Returns:
        S3 URI where evaluation was saved
    """
    try:
        if not eval_s3_path.startswith("s3://"):
            print(f"  [WARN] Cannot upload to S3: invalid URI {eval_s3_path}")
            return ""

        # Extract bucket and prefix
        parts = eval_s3_path[5:].split("/", 1)
        bucket = parts[0]
        prefix = parts[1] if len(parts) > 1 else ""

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        eval_key = f"{prefix.rstrip('/')}/evaluation_{timestamp}.json"

        # Upload to S3
        s3 = boto3.client("s3")
        s3.put_object(
            Bucket=bucket,
            Key=eval_key,
            Body=json.dumps(evaluation_data, indent=2),
            ContentType="application/json"
        )

        s3_uri = f"s3://{bucket}/{eval_key}"
        print(f"  Uploaded evaluation to: {s3_uri}")
        return s3_uri

    except Exception as e:
        print(f"  [WARN] Failed to upload evaluation to S3: {e}")
        return ""


def parse_args():
    """Parse hyperparameters"""
    parser = argparse.ArgumentParser()

    # XGBoost model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--min-child-weight", type=int, default=5)

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--project-s3-path", type=str, default="")  # S3 path for saving evaluation

    return parser.parse_args()


# =============================================================================
# STEP 1: DATA PREPROCESSING
# =============================================================================

def fetch_weather_data(burn_df):
    """Get historical weather data for training"""

    cities = {
        'Riyadh': {'lat': 24.7136, 'lon': 46.6753, 'weight': 0.25, 'type': 'capital'},
        'Jeddah': {'lat': 21.5433, 'lon': 39.1728, 'weight': 0.15, 'type': 'coastal'},
        'Dammam': {'lat': 26.3927, 'lon': 49.9777, 'weight': 0.15, 'type': 'industrial'},
        'Mecca': {'lat': 21.4225, 'lon': 39.8262, 'weight': 0.15, 'type': 'inland'},
        'Jubail': {'lat': 27.0174, 'lon': 49.6225, 'weight': 0.15, 'type': 'industrial'},
        'Yanbu': {'lat': 24.0943, 'lon': 38.0493, 'weight': 0.15, 'type': 'industrial'}
    }

    base_url = "https://archive-api.open-meteo.com/v1/archive"
    start_date = burn_df['date'].min().strftime('%Y-%m-%d')
    end_date = burn_df['date'].max().strftime('%Y-%m-%d')

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
            response = requests.get(base_url, params=params, timeout=30)
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

    return daily_weather


def load_and_preprocess_data(args):
    """Load, preprocess, and split data into test/train sets"""

    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Load burn data from SageMaker input path
    data_file = None
    input_path = args.train
    for filename in os.listdir(input_path):
        if filename.endswith('.parquet') or filename.endswith('.xlsx') or filename.endswith('.csv'):
            data_file = os.path.join(input_path, filename)
            break

    if data_file is None:
        raise ValueError(f"No data file found in {input_path}")

    # Load data based on file type
    if data_file.endswith('.parquet'):
        burn_df = pd.read_parquet(data_file)
    elif data_file.endswith('.xlsx'):
        burn_df = pd.read_excel(data_file)
    else:
        burn_df = pd.read_csv(data_file)

    # Standardize column names
    if burn_df.shape[1] == 2:
        burn_df.columns = ['date', 'burn_bpd']
    else:
        date_col = [col for col in burn_df.columns if 'date' in col.lower()][0] if any('date' in col.lower() for col in burn_df.columns) else burn_df.columns[0]
        burn_col = [col for col in burn_df.columns if 'saudi' in col.lower() or 'burn' in col.lower()][0] if any('saudi' in col.lower() or 'burn' in col.lower() for col in burn_df.columns) else burn_df.columns[1]
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

    print(f"✓ Loaded {len(burn_df)} months of burn data")

    # Fetch weather data
    print("\n[Fetching weather data...]")
    weather_daily = fetch_weather_data(burn_df)

    # Monthly aggregation
    weather_monthly = weather_daily.groupby(['year', 'month']).agg({
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

    # Merge burn and weather data
    merged_df = pd.merge(
        burn_df[['year', 'month', 'burn_kbd', 'is_summer', 'is_peak_summer']],
        weather_monthly,
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

    print(f"✓ Created {len(merged_df.columns)} total features")
    print(f"  Final dataset: {len(merged_df)} samples")

    # Prepare features
    exclude_cols = ['date', 'year', 'month', 'burn_kbd', 'is_summer', 'is_peak_summer']
    feature_cols = [col for col in merged_df.columns if col not in exclude_cols]

    X = merged_df[feature_cols].values
    y = merged_df['burn_kbd'].values
    is_summer = merged_df['is_summer'].values
    is_peak = merged_df['is_peak_summer'].values

    # Use last 12 months as test set for out-of-sample validation
    split_idx = len(X) - 12
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    is_summer_train, is_summer_test = is_summer[:split_idx], is_summer[split_idx:]
    is_peak_train, is_peak_test = is_peak[:split_idx], is_peak[split_idx:]

    print(f"\nTrain/Test split: {len(X_train)}/{len(X_test)} months")

    # Save test predictions data for later use
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'is_summer_test': is_summer_test,
        'is_peak_test': is_peak_test,
        'dates': merged_df.iloc[split_idx:]['date'].tolist()
    }

    train_data = {
        'X_train': X_train,
        'y_train': y_train,
        'is_summer_train': is_summer_train,
        'is_peak_train': is_peak_train,
        'feature_cols': feature_cols
    }

    return train_data, test_data


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def train_model(train_data, args):
    """Train XGBoost model with sample weighting for summer months"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    X_train = train_data['X_train']
    y_train = train_data['y_train']
    is_summer_train = train_data['is_summer_train']
    is_peak_train = train_data['is_peak_train']

    # Create sample weights emphasizing summer and peak months
    sample_weights = np.ones(len(y_train))
    sample_weights[is_summer_train == 1] = 5
    sample_weights[is_peak_train == 1] = 10

    print(f"Training XGBoost model with {len(X_train)} samples...")
    print(f"  n_estimators: {args.n_estimators}")
    print(f"  max_depth: {args.max_depth}")
    print(f"  learning_rate: {args.learning_rate}")

    # Train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        gamma=args.gamma,
        min_child_weight=args.min_child_weight,
        random_state=42,
        verbosity=0
    )

    model.fit(X_train, y_train, sample_weight=sample_weights)

    print("✓ Model training completed")

    return model


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, test_data):
    """Evaluate model on test set and return key metrics"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    X_test = test_data['X_test']
    y_test = test_data['y_test']
    is_summer_test = test_data['is_summer_test']
    is_peak_test = test_data['is_peak_test']
    dates = test_data['dates']

    # Generate predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    overall_mae = mean_absolute_error(y_test, y_pred)

    # Summer-specific metrics
    if is_summer_test.sum() > 0:
        summer_mae = mean_absolute_error(
            y_test[is_summer_test == 1],
            y_pred[is_summer_test == 1]
        )
        summer_mape = np.mean(
            np.abs((y_pred[is_summer_test == 1] - y_test[is_summer_test == 1]) / y_test[is_summer_test == 1])
        ) * 100
    else:
        summer_mae = overall_mae
        summer_mape = 0

    # Peak summer metrics
    if is_peak_test.sum() > 0:
        peak_mae = mean_absolute_error(
            y_test[is_peak_test == 1],
            y_pred[is_peak_test == 1]
        )
    else:
        peak_mae = overall_mae

    print(f"\nModel Performance:")
    print(f"  Overall MAE: {overall_mae:.1f} kbd")
    print(f"  Summer MAE: {summer_mae:.1f} kbd")
    print(f"  Peak MAE: {peak_mae:.1f} kbd")
    print(f"  Summer MAPE: {summer_mape:.1f}%")

    # Create test predictions list for logging
    test_predictions = []
    for i in range(len(y_test)):
        test_predictions.append({
            'date': str(dates[i]),
            'actual_kbd': float(y_test[i]),
            'predicted_kbd': float(y_pred[i])
        })

    evaluation = {
        'overall_mae': float(overall_mae),
        'summer_mae': float(summer_mae),
        'peak_mae': float(peak_mae),
        'summer_mape': float(summer_mape),
        'test_predictions': test_predictions,
        'n_test_samples': len(y_test),
        'n_summer_samples': int(is_summer_test.sum()),
        'n_peak_samples': int(is_peak_test.sum())
    }

    return evaluation


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, args, evaluation, train_data):
    """Save model and evaluation outputs"""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save model
    print(f"\nSaving model to: {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)

    # Package model with metadata
    model_package = {
        'model': model,
        'model_name': 'XGBoost_Saudi_Crude_Burn',
        'feature_cols': train_data['feature_cols'],
        'final_performance': {
            'overall_mae': evaluation['overall_mae'],
            'summer_mae': evaluation['summer_mae'],
            'peak_mae': evaluation['peak_mae'],
            'summer_mape': evaluation['summer_mape']
        }
    }

    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_package, f)

    print(f"  ✓ Model saved with {len(train_data['feature_cols'])} features")

    # Create code directory for inference artifacts
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # Copy inference.py to code directory (SageMaker convention)
    inference_src = "/opt/ml/code/inference.py"
    inference_dst = os.path.join(code_dir, "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_dst)
        print(f"  Copied inference.py to code/")

    requirements_src = "/opt/ml/code/requirements.txt"
    requirements_root_dst = args.model_dir
    requirements_dst = os.path.join(code_dir, "requirements.txt")
    if os.path.exists(requirements_src):
        shutil.copy(requirements_src, requirements_dst)
        shutil.copy(requirements_src, requirements_root_dst)
        print(f"  Copied requirements.txt to code/ and root")

    # Create setup.py at BOTH root level and code/ directory
    # SageMaker looks for setup.py at root when SAGEMAKER_SUBMIT_DIRECTORY points to model.tar.gz
    setup_py_content = '''from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="inference",
    version="1.0.0",
    py_modules=["inference"],
    install_requires=required,
)
'''
    # Put setup.py in code/ directory
    setup_py_code_path = os.path.join(code_dir, "setup.py")
    with open(setup_py_code_path, "w") as f:
        f.write(setup_py_content)
    print(f"  Created code/setup.py")

    # ALSO put setup.py and inference.py at model root level for SageMaker to find
    # (SageMaker looks at root when SAGEMAKER_SUBMIT_DIRECTORY points to model.tar.gz)
    setup_py_root_path = os.path.join(args.model_dir, "setup.py")
    with open(setup_py_root_path, "w") as f:
        f.write(setup_py_content)
    print(f"  Created setup.py at model root")

    # Copy inference.py to root as well (setup.py references it)
    inference_root_dst = os.path.join(args.model_dir, "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_root_dst)
        print(f"  Copied inference.py to model root")

    # Add metadata
    metadata = {
        'model_name': 'XGBoost_Saudi_Crude_Burn',
        'model_type': 'XGBoost Regressor',
        'n_features': len(train_data['feature_cols']),
        'hyperparameters': {
            'n_estimators': args.n_estimators,
            'max_depth': args.max_depth,
            'learning_rate': args.learning_rate,
            'subsample': args.subsample,
            'colsample_bytree': args.colsample_bytree,
            'gamma': args.gamma,
            'min_child_weight': args.min_child_weight
        },
        'performance': {
            'overall_mae': evaluation['overall_mae'],
            'summer_mae': evaluation['summer_mae'],
            'peak_mae': evaluation['peak_mae'],
            'summer_mape': evaluation['summer_mape']
        },
        'training_date': datetime.now().isoformat()
    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Metadata saved")

    # Save evaluation report (for CallbackStep)
    eval_output_dir = args.output_data_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    eval_report = evaluation

    eval_path = os.path.join(eval_output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved evaluation report to: {eval_path}")

    # Upload evaluation to S3 evaluation_data folder
    if args.project_s3_path:
        # project_s3_path format: s3://bucket/projects/user/project/project/
        # Append evaluation_data/ to get the target folder
        eval_s3_path = args.project_s3_path.rstrip("/") + "/evaluation_data/"
        upload_evaluation_to_s3(eval_report, eval_s3_path)

    return eval_report


def main():
    """Main training logic"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("SAUDI CRUDE BURN FORECASTER - TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate → Save")
    print("=" * 60)

    # Step 1: Data Preprocessing
    train_data, test_data = load_and_preprocess_data(args)

    # Step 2: Model Training
    model = train_model(train_data, args)

    # Step 3: Model Evaluation
    evaluation = evaluate_model(model, test_data)

    # Save outputs
    save_outputs(model, args, evaluation, train_data)

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n✓ Model trained successfully!")
    print(f"  Overall MAE: {evaluation['overall_mae']:.1f} kbd")
    print(f"  Summer MAE: {evaluation['summer_mae']:.1f} kbd")
    print(f"  Model saved to: {args.model_dir}")


if __name__ == "__main__":
    main()
