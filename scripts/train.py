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
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# Oil price forecasting model imports
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox
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

    # Model hyperparameters for ARIMA
    parser.add_argument("--arima-order-p", type=int, default=2, help="ARIMA p parameter (AR order)")
    parser.add_argument("--arima-order-d", type=int, default=1, help="ARIMA d parameter (differencing)")
    parser.add_argument("--arima-order-q", type=int, default=2, help="ARIMA q parameter (MA order)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size as proportion")
    
    # Oil price data parameters
    parser.add_argument("--oil-symbol", type=str, default="CL=F", help="Oil futures symbol (default: WTI crude)")
    parser.add_argument("--period", type=str, default="2y", help="Data period (1y, 2y, 5y, max)")

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--project-s3-path", type=str, default="")  # S3 path for saving evaluation

    return parser.parse_args()


# =============================================================================
# STEP 1: DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_data(args):
    """Load, preprocess, and split oil price data into test/train sets"""

    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Download oil price data from Yahoo Finance
    print(f"Downloading oil price data for symbol: {args.oil_symbol}")
    print(f"Period: {args.period}")
    
    try:
        ticker = yf.Ticker(args.oil_symbol)
        df = ticker.history(period=args.period)
        
        if df.empty:
            raise ValueError(f"No data found for symbol {args.oil_symbol}")
            
        print(f"Downloaded {len(df)} data points")
        print(f"Date range: {df.index.min()} to {df.index.max()}")
        
        # Use closing price as our target variable
        oil_prices = df['Close'].dropna()
        
        # Convert to daily frequency and forward fill any missing values
        oil_prices = oil_prices.asfreq('D', method='ffill')
        
        # Remove any remaining NaN values
        oil_prices = oil_prices.dropna()
        
        print(f"After preprocessing: {len(oil_prices)} data points")
        print(f"Price range: ${oil_prices.min():.2f} - ${oil_prices.max():.2f}")
        
        # Split into train and test sets
        split_index = int(len(oil_prices) * (1 - args.test_size))
        train_data = oil_prices[:split_index]
        test_data = oil_prices[split_index:]
        
        print(f"Training set: {len(train_data)} points ({train_data.index.min()} to {train_data.index.max()})")
        print(f"Test set: {len(test_data)} points ({test_data.index.min()} to {test_data.index.max()})")
        
        # Check for stationarity (for informational purposes)
        from statsmodels.tsa.stattools import adfuller
        
        adf_result = adfuller(train_data)
        print(f"\nStationarity test (ADF): p-value = {adf_result[1]:.4f}")
        if adf_result[1] <= 0.05:
            print("Series appears to be stationary")
        else:
            print("Series appears to be non-stationary (differencing may be needed)")
        
        return train_data, test_data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        print("Creating synthetic oil price data for demonstration...")
        
        # Create synthetic oil price data if real data fails
        dates = pd.date_range(start='2022-01-01', end='2024-01-01', freq='D')
        np.random.seed(42)
        
        # Generate realistic oil price movements
        base_price = 75.0
        returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            # Add some trend and seasonality
            trend = 0.0001 * len(prices)  # Slight upward trend
            seasonal = 5 * np.sin(2 * np.pi * len(prices) / 365)  # Annual seasonality
            new_price += trend + seasonal
            prices.append(max(new_price, 30))  # Floor at $30
            
        oil_prices = pd.Series(prices, index=dates, name='oil_price')
        
        # Split into train and test sets
        split_index = int(len(oil_prices) * (1 - args.test_size))
        train_data = oil_prices[:split_index]
        test_data = oil_prices[split_index:]
        
        print(f"Created synthetic data: {len(oil_prices)} points")
        print(f"Training set: {len(train_data)} points")
        print(f"Test set: {len(test_data)} points")
        
        return train_data, test_data


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def train_model(train_data, args):
    """Train ARIMA model for oil price forecasting"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    arima_order = (args.arima_order_p, args.arima_order_d, args.arima_order_q)
    print(f"Training ARIMA model with order: {arima_order}")
    
    try:
        # Fit ARIMA model
        model = ARIMA(train_data, order=arima_order)
        fitted_model = model.fit()
        
        print("Model training completed successfully")
        print(f"AIC: {fitted_model.aic:.2f}")
        print(f"BIC: {fitted_model.bic:.2f}")
        
        # Model diagnostics
        print("\nModel Summary:")
        print(fitted_model.summary().tables[1])  # Coefficient table
        
        # Check residuals
        residuals = fitted_model.resid
        ljung_box = acorr_ljungbox(residuals, lags=10, return_df=True)
        print(f"\nResidual diagnostics (Ljung-Box test p-value): {ljung_box['lb_pvalue'].iloc[-1]:.4f}")
        
        return fitted_model
        
    except Exception as e:
        print(f"Error training ARIMA model: {e}")
        print("Falling back to simpler ARIMA(1,1,1) model...")
        
        # Fallback to simpler model
        model = ARIMA(train_data, order=(1, 1, 1))
        fitted_model = model.fit()
        
        print("Fallback model training completed")
        print(f"AIC: {fitted_model.aic:.2f}")
        
        return fitted_model


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, train_data, test_data):
    """Evaluate model on test set and return key metrics"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    try:
        # Generate forecasts for test period
        forecast_steps = len(test_data)
        forecast = model.forecast(steps=forecast_steps)
        
        # Get confidence intervals
        forecast_ci = model.get_forecast(steps=forecast_steps).conf_int()
        
        # Calculate metrics
        mse = mean_squared_error(test_data.values, forecast)
        rmse = sqrt(mse)
        mae = np.mean(np.abs(test_data.values - forecast))
        mape = np.mean(np.abs((test_data.values - forecast) / test_data.values)) * 100
        
        # Direction accuracy (did we predict up/down correctly?)
        actual_directions = np.diff(test_data.values) > 0
        predicted_directions = np.diff(forecast) > 0
        direction_accuracy = np.mean(actual_directions == predicted_directions) * 100
        
        print(f"Forecast generated for {forecast_steps} days")
        print(f"Mean Squared Error (MSE): {mse:.4f}")
        print(f"Root Mean Squared Error (RMSE): ${rmse:.2f}")
        print(f"Mean Absolute Error (MAE): ${mae:.2f}")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Directional Accuracy: {direction_accuracy:.1f}%")
        
        # Price level analysis
        actual_mean = test_data.mean()
        forecast_mean = forecast.mean()
        print(f"\nActual mean price: ${actual_mean:.2f}")
        print(f"Forecast mean price: ${forecast_mean:.2f}")
        print(f"Bias: ${forecast_mean - actual_mean:.2f}")
        
        evaluation = {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "direction_accuracy": float(direction_accuracy),
            "actual_mean_price": float(actual_mean),
            "forecast_mean_price": float(forecast_mean),
            "bias": float(forecast_mean - actual_mean),
            "forecast_length": int(forecast_steps),
            "model_aic": float(model.aic),
            "model_bic": float(model.bic),
            "predictions": forecast.tolist(),
            "actuals": test_data.values.tolist(),
            "forecast_dates": test_data.index.strftime('%Y-%m-%d').tolist(),
            "confidence_intervals": {
                "lower": forecast_ci.iloc[:, 0].tolist(),
                "upper": forecast_ci.iloc[:, 1].tolist()
            }
        }
        
        return evaluation
        
    except Exception as e:
        print(f"Error in model evaluation: {e}")
        return {
            "error": str(e),
            "mse": float('inf'),
            "rmse": float('inf'),
            "mae": float('inf'),
            "mape": float('inf')
        }


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, args, evaluation):
    """Save model and evaluation outputs"""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save model
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

    # Model metadata
    metadata = {
        "model_type": "time_series_arima",
        "target": "oil_prices",
        "arima_order": (args.arima_order_p, args.arima_order_d, args.arima_order_q),
        "oil_symbol": args.oil_symbol,
        "period": args.period,
        "test_size": args.test_size,
        "training_date": datetime.now().isoformat(),
        "model_performance": {
            "rmse": evaluation.get("rmse", None),
            "mape": evaluation.get("mape", None),
            "direction_accuracy": evaluation.get("direction_accuracy", None)
        }
    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

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
    print("OIL PRICES AUSTIN - TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)

    # Step 1: Data Preprocessing
    train_data, test_data = load_and_preprocess_data(args)

    # Step 2: Model Training
    model = train_model(train_data, args)

    # Step 3: Model Evaluation
    evaluation = evaluate_model(model, train_data, test_data)

    # Save outputs
    save_outputs(model, args, evaluation)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()