"""
Training Script Template for SageMaker Pipeline
Combines: Data Preprocessing + Model Training + Evaluation
"""

import os
import argparse
import argparse
import pickle
import json
import shutil
import boto3
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
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

def parse_args():
    """Parse hyperparameters"""
    parser = argparse.ArgumentParser()

    # SARIMA model hyperparameters
    parser.add_argument("--seasonal-periods", type=int, default=12, help="Seasonal period for SARIMA (12 for monthly data)")
    parser.add_argument("--auto-arima", type=bool, default=True, help="Use auto parameter selection for SARIMA")
    parser.add_argument("--max-p", type=int, default=3, help="Maximum p value for ARIMA")
    parser.add_argument("--max-d", type=int, default=2, help="Maximum d value for ARIMA") 
    parser.add_argument("--max-q", type=int, default=3, help="Maximum q value for ARIMA")
    parser.add_argument("--max-P", type=int, default=2, help="Maximum P value for seasonal ARIMA")
    parser.add_argument("--max-D", type=int, default=1, help="Maximum D value for seasonal ARIMA")
    parser.add_argument("--max-Q", type=int, default=2, help="Maximum Q value for seasonal ARIMA")

    # Data split ratios - for time series, we typically use temporal splits
    parser.add_argument("--train-split", type=float, default=0.8, help="Train split ratio for time series")
    parser.add_argument("--test-split", type=float, default=0.2, help="Test split ratio for time series")

    # SageMaker environment variables
    # TODO: Add your model hyperparameters here
    # Example: parser.add_argument("--learning-rate", type=float, default=0.01)

    # Data split ratios
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

# STEP 1: DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_data(args):
    """Load, preprocess, and split time series data into train/test sets"""

    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Load data
    data_path = os.path.join(args.train, "data.csv")
    print(f"Loading data from: {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
    except FileNotFoundError:
        print("No training data found. Creating synthetic oil consumption data for demonstration...")
        # Create synthetic Saudi oil consumption data
        date_range = pd.date_range(start='2000-01-01', end='2023-12-31', freq='MS')
        np.random.seed(42)
        
        # Base consumption with trend and seasonality (thousand barrels per day)
        base_consumption = 2800 + np.arange(len(date_range)) * 2  # Growing trend
        seasonal_pattern = 200 * np.sin(2 * np.pi * np.arange(len(date_range)) / 12)  # Annual seasonality
        noise = np.random.normal(0, 50, len(date_range))
        
        oil_consumption = base_consumption + seasonal_pattern + noise
        
        df = pd.DataFrame({
            'date': date_range,
            'oil_consumption_kbd': oil_consumption
        })
        print(f"Created synthetic data shape: {df.shape}")
    
    # Ensure date column is datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    elif df.index.dtype == 'object':
        df.index = pd.to_datetime(df.index)
        df = df.reset_index()
        df.columns = ['date'] + df.columns[1:].tolist()
    
    # Identify the target column (oil consumption)
    target_col = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['consumption', 'burn', 'oil', 'demand', 'usage']):
            target_col = col
            break
    
    if target_col is None:
        target_col = [col for col in df.columns if col != 'date'][0]
        print(f"No obvious target column found, using: {target_col}")
    else:
        print(f"Using target column: {target_col}")
    
    # Sort by date and set as index
    df = df.sort_values('date')
    df.set_index('date', inplace=True)
    
    # Create time series
    ts = df[target_col]
    print(f"Time series range: {ts.index[0]} to {ts.index[-1]}")
    print(f"Time series length: {len(ts)}")
    print(f"Missing values: {ts.isna().sum()}")
    
    # Handle missing values
    if ts.isna().sum() > 0:
        ts = ts.interpolate(method='linear')
        print("Interpolated missing values")
    
    # Time-based split (important for time series)
    train_size = int(len(ts) * args.train_split)
    train_data = ts[:train_size]
    test_data = ts[train_size:]
    
    print(f"Train data: {len(train_data)} points ({train_data.index[0]} to {train_data.index[-1]})")
    print(f"Test data: {len(test_data)} points ({test_data.index[0]} to {test_data.index[-1]})")
    
    # Store metadata
    metadata = {
        'target_column': target_col,
        'frequency': pd.infer_freq(ts.index),
        'start_date': str(ts.index[0]),
        'end_date': str(ts.index[-1]),
        'total_observations': len(ts),
        'train_observations': len(train_data),
        'test_observations': len(test_data),
        'mean_value': float(ts.mean()),
        'std_value': float(ts.std())
    }
    
    print(f"Detected frequency: {metadata['frequency']}")

    return train_data, test_data, metadata


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def check_stationarity(ts, title):
    """Check if time series is stationary using Augmented Dickey-Fuller test"""
    result = adfuller(ts.dropna())
    print(f'\n{title} - Stationarity Test:')
    print(f'  ADF Statistic: {result[0]:.6f}')
    print(f'  p-value: {result[1]:.6f}')
    
    if result[1] <= 0.05:
        print("  Result: Stationary (reject null hypothesis)")
        return True
    else:
        print("  Result: Non-stationary (fail to reject null hypothesis)")
        return False

def find_best_sarima_params(ts, seasonal_periods=12, max_p=3, max_d=2, max_q=3, max_P=2, max_D=1, max_Q=2):
    """Find best SARIMA parameters using grid search with AIC"""
    print("Searching for optimal SARIMA parameters...")
    
    best_aic = float('inf')
    best_params = None
    best_seasonal_params = None
    
    # Grid search
    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                for P in range(max_P + 1):
                    for D in range(max_D + 1):
                        for Q in range(max_Q + 1):
                            try:
                                model = SARIMAX(ts, 
                                              order=(p, d, q),
                                              seasonal_order=(P, D, Q, seasonal_periods),
                                              enforce_stationarity=False,
                                              enforce_invertibility=False)
                                fitted_model = model.fit(disp=False)
                                
                                if fitted_model.aic < best_aic:
                                    best_aic = fitted_model.aic
                                    best_params = (p, d, q)
                                    best_seasonal_params = (P, D, Q, seasonal_periods)
                                    
                            except Exception:
                                continue
    
    print(f"Best SARIMA parameters: order={best_params}, seasonal_order={best_seasonal_params}")
    print(f"Best AIC: {best_aic:.2f}")
    
    return best_params, best_seasonal_params

def train_model(train_data, metadata, args):
    """Train SARIMA model for oil consumption forecasting"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    # Analyze time series characteristics
    print("Analyzing time series characteristics...")
    
    # Check stationarity
    is_stationary = check_stationarity(train_data, "Original Series")
    
    # If auto ARIMA is enabled, find best parameters
    if args.auto_arima:
        best_order, best_seasonal_order = find_best_sarima_params(
            train_data, 
            seasonal_periods=args.seasonal_periods,
            max_p=args.max_p,
            max_d=args.max_d, 
            max_q=args.max_q,
            max_P=args.max_P,
            max_D=args.max_D,
            max_Q=args.max_Q
        )
    else:
        # Use default parameters
        best_order = (1, 1, 1)
        best_seasonal_order = (1, 1, 1, args.seasonal_periods)
        print(f"Using default SARIMA parameters: order={best_order}, seasonal_order={best_seasonal_order}")
    
    # Train SARIMA model
    print("\nTraining SARIMA model...")
    try:
        model = SARIMAX(train_data,
                       order=best_order,
                       seasonal_order=best_seasonal_order,
                       enforce_stationarity=False,
                       enforce_invertibility=False)
        
        fitted_model = model.fit(disp=False)
        
        # Model diagnostics
        print(f"Model AIC: {fitted_model.aic:.2f}")
        print(f"Model BIC: {fitted_model.bic:.2f}")
        print(f"Log Likelihood: {fitted_model.llf:.2f}")
        
        # Ljung-Box test for residuals
        ljung_box = acorr_ljungbox(fitted_model.resid, lags=10, return_df=True)
        print(f"Ljung-Box test p-value: {ljung_box['lb_pvalue'].iloc[-1]:.4f}")
        
        model_info = {
            'fitted_model': fitted_model,
            'order': best_order,
            'seasonal_order': best_seasonal_order,
            'aic': fitted_model.aic,
            'bic': fitted_model.bic,
            'metadata': metadata
        }
        
        print("SARIMA model trained successfully!")
        return model_info
        
    except Exception as e:
        print(f"SARIMA training failed: {str(e)}")
        print("Falling back to seasonal naive model...")
        
        # Simple seasonal naive fallback
        seasonal_naive = {
            'type': 'seasonal_naive',
            'seasonal_periods': args.seasonal_periods,
            'train_data': train_data,
            'metadata': metadata
        }
        return seasonal_naive

    return model_info


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, test_data):
    """Evaluate forecasting model on test set and return key metrics"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    if isinstance(model, dict) and model.get('type') == 'seasonal_naive':
        # Evaluate seasonal naive model
        print("Evaluating Seasonal Naive model...")
        seasonal_periods = model['seasonal_periods']
        train_data = model['train_data']
        
        # Generate forecasts using seasonal naive approach
        forecasts = []
        for i in range(len(test_data)):
            # Use value from same season in previous year
            seasonal_idx = len(train_data) - seasonal_periods + (i % seasonal_periods)
            if seasonal_idx >= 0:
                forecast = train_data.iloc[seasonal_idx]
            else:
                forecast = train_data.mean()
            forecasts.append(forecast)
        
        forecasts = pd.Series(forecasts, index=test_data.index)
        
    else:
        # Evaluate SARIMA model
        print("Evaluating SARIMA model...")
        fitted_model = model['fitted_model']
        
        # Generate out-of-sample forecasts
        forecast_result = fitted_model.forecast(steps=len(test_data))
        forecasts = pd.Series(forecast_result, index=test_data.index)
    
    # Calculate evaluation metrics
    mae = np.mean(np.abs(test_data - forecasts))
    mse = np.mean((test_data - forecasts) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((test_data - forecasts) / test_data)) * 100
    
    # Additional time series metrics
    mean_test = test_data.mean()
    scaled_mae = mae / mean_test * 100
    
    print(f"\nForecast Evaluation Metrics:")
    print(f"  MAE: {mae:.2f}")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Scaled MAE: {scaled_mae:.2f}% of mean")
    
    evaluation = {
        'mae': float(mae),
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'scaled_mae': float(scaled_mae),
        'mean_actual': float(mean_test),
        'mean_forecast': float(forecasts.mean()),
        'test_periods': len(test_data),
        'model_type': model.get('type', 'sarima'),
        'forecast_summary': {
            'min_forecast': float(forecasts.min()),
            'max_forecast': float(forecasts.max()),
            'std_forecast': float(forecasts.std())
        }
    }
    
    # Add model-specific information
    if 'order' in model:
        evaluation['model_parameters'] = {
            'order': model['order'],
            'seasonal_order': model['seasonal_order'],
            'aic': model['aic'],
            'bic': model['bic']
        }

    return evaluation

def evaluate_model(model, test_data):
    """Evaluate model on test set and return key metrics"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    evaluation = {}

    return evaluation


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, args, evaluation):
    """Save model and evaluation outputs"""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    requirements_dst = os.path.join(code_dir, "requirements.txt")
    if os.path.exists(requirements_src):
        shutil.copy(requirements_src, requirements_dst)
    print(f"\nSaving model to: {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

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
        f.write(setup_py_content)
    print(f"  Copied inference.py to model root")

    metadata = {
        'model_type': 'time_series_forecasting',
        'algorithm': 'SARIMA',
        'target_variable': 'oil_consumption_kbd',
        'frequency': 'monthly',
        'training_end_date': str(max(evaluation.get('test_periods', 0) and 
                                   pd.Timestamp.now() or pd.Timestamp.now())),
        'evaluation_metrics': {
            'rmse': evaluation.get('rmse', 0),
            'mae': evaluation.get('mae', 0),
            'mape': evaluation.get('mape', 0)
        },
        'created_at': str(pd.Timestamp.now())
    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
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

    #TODO: add metadata as needed
    metadata = {

    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save evaluation report (for CallbackStep)
    eval_output_dir = args.output_data_dir
    os.makedirs(eval_output_dir, exist_ok=True)
    print("TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    train_data, test_data, metadata = load_and_preprocess_data(args)

    # Step 2: Model Training
    model = train_model(train_data, metadata, args)

    # Step 3: Model Evaluation
    evaluation = evaluate_model(model, test_data)

    # Save outputs


def main():
    """Main training logic"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)

    #TODO: Modify the function calls as needed.

    # Step 1: Data Preprocessing
    train_data, test_data = load_and_preprocess_data()

    # Step 2: Model Training
    model = train_model(train_data)
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":

    # Print final summary
    print("\\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
