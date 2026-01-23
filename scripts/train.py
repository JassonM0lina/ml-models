"""
Combined Training Script for SageMaker Pipeline
Combines: Data Preprocessing + Model Training + Evaluation
Single step = Single cold start
"""

import os
import argparse
import pickle
import json
import shutil
import pandas as pd
from datetime import datetime
from sklearn.metrics import mean_squared_error
from math import sqrt

# Install statsmodels
import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels==0.13.5", "-q"])

from statsmodels.tsa.arima.model import ARIMA


def parse_args():
    """Parse hyperparameters"""
    parser = argparse.ArgumentParser()

    # ARIMA hyperparameters
    parser.add_argument("-p", "--p", type=int, default=5, help="AR order")
    parser.add_argument("-d", "--d", type=int, default=1, help="Differencing order")
    parser.add_argument("-q", "--q", type=int, default=0, help="MA order")

    # Data split ratios
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    return parser.parse_args()


# =============================================================================
# STEP 1: DATA PREPROCESSING
# =============================================================================

def load_and_preprocess_data(input_path, train_split, validation_split, test_split):
    """Load raw data and split into train/validation/test"""
    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Find data file
    data_file = None
    for filename in os.listdir(input_path):
        if filename.endswith('.csv'):
            data_file = os.path.join(input_path, filename)
            break

    if not data_file:
        raise FileNotFoundError(f"No CSV file found in {input_path}")

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file, header=0, index_col=0)

    # Parse dates: "1-01" -> "1901-01", "10-01" -> "1910-01"
    def parse_date(x):
        parts = str(x).split('-')
        year = int(parts[0])
        month = int(parts[1])
        return datetime(1900 + year, month, 1)

    df.index = df.index.map(parse_date)
    series = df.squeeze()

    print(f"  Loaded {len(series)} data points")
    print(f"  Date range: {series.index[0]} to {series.index[-1]}")

    # Data quality checks
    print("\nRunning data quality checks...")
    if series.isnull().sum() > 0:
        print(f"  Found {series.isnull().sum()} missing values - filling with mean")
        series = series.fillna(series.mean())

    # Check for outliers
    mean = series.mean()
    std = series.std()
    outliers = series[(series < mean - 3*std) | (series > mean + 3*std)]
    if len(outliers) > 0:
        print(f"  Found {len(outliers)} potential outliers (>3 std from mean)")

    # Split data
    print(f"\nSplitting data (train: {train_split}, val: {validation_split}, test: {test_split})...")
    n = len(series)
    train_size = int(n * train_split)
    val_size = int(n * validation_split)

    train_data = series[:train_size]
    val_data = series[train_size:train_size + val_size]
    test_data = series[train_size + val_size:]

    # Convert to period index for ARIMA
    train_data.index = pd.PeriodIndex(train_data.index, freq='M')
    val_data.index = pd.PeriodIndex(val_data.index, freq='M')
    test_data.index = pd.PeriodIndex(test_data.index, freq='M')

    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    data_quality_report = {
        "total_samples": int(n),
        "train_samples": int(len(train_data)),
        "validation_samples": int(len(val_data)),
        "test_samples": int(len(test_data)),
        "missing_values_filled": int(series.isnull().sum()),
        "outliers_detected": int(len(outliers)),
        "statistics": {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max())
        }
    }

    return train_data, val_data, test_data, data_quality_report


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def train_arima(train_data, val_data, p, d, q):
    """Train ARIMA model and validate"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    print(f"\nTraining ARIMA({p},{d},{q}) model...")
    model = ARIMA(train_data, order=(p, d, q))
    model_fit = model.fit()

    print("  Model trained successfully")
    print(f"  AIC: {model_fit.aic:.2f}")
    print(f"  BIC: {model_fit.bic:.2f}")

    # Validate on validation set
    print(f"\nValidating on {len(val_data)} samples...")
    val_predictions = []
    val_actuals = []

    for i in range(len(val_data)):
        forecast = model_fit.forecast(steps=1)
        pred = forecast[0] if hasattr(forecast, '__getitem__') else forecast
        val_predictions.append(pred)
        val_actuals.append(val_data.iloc[i])

    val_rmse = sqrt(mean_squared_error(val_actuals, val_predictions))
    print(f"  Validation RMSE: {val_rmse:.2f}")

    return model_fit, val_rmse


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model_fit, test_data):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    print(f"\nEvaluating on {len(test_data)} test samples...")

    predictions = []
    actuals = []
    errors = []

    for i in range(len(test_data)):
        forecast = model_fit.forecast(steps=1)
        pred = forecast[0] if hasattr(forecast, '__getitem__') else forecast
        actual = test_data.iloc[i]

        predictions.append(float(pred))
        actuals.append(float(actual))
        errors.append(float(abs(pred - actual)))

        if i < 5 or i >= len(test_data) - 3:
            print(f"  Step {i+1}: Predicted={pred:.2f}, Actual={actual:.2f}")

    # Calculate metrics
    rmse = sqrt(mean_squared_error(actuals, predictions))
    mae = sum(errors) / len(errors)

    mape_values = [abs((a - p) / a) * 100 for a, p in zip(actuals, predictions) if a != 0]
    mape = sum(mape_values) / len(mape_values) if mape_values else 0

    ss_res = sum((a - p)**2 for a, p in zip(actuals, predictions))
    ss_tot = sum((a - sum(actuals)/len(actuals))**2 for a in actuals)
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")

    return {
        "metrics": {
            "rmse": rmse,
            "mae": mae,
            "mape": mape,
            "r2": r2
        },
        "predictions": predictions,
        "actuals": actuals,
        "errors": errors
    }


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model_fit, args, data_quality_report, evaluation_results):
    """Save model and evaluation outputs"""
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # Save model
    print(f"\nSaving model to: {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)

    model_path = os.path.join(args.model_dir, "arima_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model_fit, f)

    # Create code directory for inference artifacts
    code_dir = os.path.join(args.model_dir, "code")
    os.makedirs(code_dir, exist_ok=True)

    # Copy inference.py to code directory (SageMaker convention)
    inference_src = "/opt/ml/code/inference.py"
    inference_dst = os.path.join(code_dir, "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_dst)
        print(f"  Copied inference.py to code/")

    # Create setup.py at BOTH root level and code/ directory
    # SageMaker looks for setup.py at root when SAGEMAKER_SUBMIT_DIRECTORY points to model.tar.gz
    setup_py_content = '''from setuptools import setup

setup(
    name="inference",
    version="1.0.0",
    py_modules=["inference"],
    install_requires=[
        "statsmodels==0.13.5",
        "patsy>=0.5.2",
    ],
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

    # Save metadata
    metadata = {
        "model_type": "ARIMA",
        "order": [args.p, args.d, args.q],
        "data_quality": data_quality_report,
        "aic": float(model_fit.aic),
        "bic": float(model_fit.bic),
        "test_metrics": evaluation_results["metrics"]
    }

    metadata_path = os.path.join(args.model_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save evaluation predictions (actuals vs predictions)
    evaluation_details = {
        "predictions": evaluation_results["predictions"],
        "actuals": evaluation_results["actuals"],
        "errors": evaluation_results["errors"],
        "num_test_samples": len(evaluation_results["actuals"]),
        "metrics": evaluation_results["metrics"]
    }

    eval_details_path = os.path.join(args.model_dir, "evaluation_results.json")
    with open(eval_details_path, "w") as f:
        json.dump(evaluation_details, f, indent=2)
    print(f"  Saved evaluation predictions to: evaluation_results.json")

    # Save evaluation report (for CallbackStep)
    eval_output_dir = args.output_data_dir
    os.makedirs(eval_output_dir, exist_ok=True)

    eval_report = {
        "metrics": evaluation_results["metrics"],
        "test_samples": len(evaluation_results["actuals"]),
        "predictions_summary": {
            "mean": sum(evaluation_results["predictions"]) / len(evaluation_results["predictions"]),
            "min": min(evaluation_results["predictions"]),
            "max": max(evaluation_results["predictions"])
        },
        "actuals_summary": {
            "mean": sum(evaluation_results["actuals"]) / len(evaluation_results["actuals"]),
            "min": min(evaluation_results["actuals"]),
            "max": max(evaluation_results["actuals"])
        }
    }

    eval_path = os.path.join(eval_output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved evaluation report to: {eval_path}")

    return eval_report


def main():
    """Main combined training logic"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("ARIMA COMBINED TRAINING PIPELINE")
    print("Single Step: Preprocess → Train → Evaluate")
    print("=" * 60)

    # Step 1: Data Preprocessing
    train_data, val_data, test_data, data_quality_report = load_and_preprocess_data(
        args.train,
        args.train_split,
        args.validation_split,
        args.test_split
    )

    # Step 2: Model Training
    model_fit, val_rmse = train_arima(train_data, val_data, args.p, args.d, args.q)

    # Step 3: Model Evaluation
    evaluation_results = evaluate_model(model_fit, test_data)

    # Save outputs
    eval_report = save_outputs(model_fit, args, data_quality_report, evaluation_results)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  RMSE: {eval_report['metrics']['rmse']:.2f}")
    print(f"  MAE: {eval_report['metrics']['mae']:.2f}")
    print(f"  MAPE: {eval_report['metrics']['mape']:.2f}%")
    print(f"  R²: {eval_report['metrics']['r2']:.4f}")

    # Output for SageMaker metrics
    print(f"\ntest:rmse={eval_report['metrics']['rmse']:.4f};")


if __name__ == "__main__":
    main()
