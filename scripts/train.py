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
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from math import sqrt

# Energy price regression specific imports
from sklearn.feature_selection import SelectKBest, f_regression


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

    # Model hyperparameters for energy price regression
    parser.add_argument("--model-type", type=str, default="random_forest", choices=["linear", "random_forest"])
    parser.add_argument("--n-estimators", type=int, default=100, help="Number of trees for RandomForest")
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum depth for RandomForest")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size")

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
    """Load, preprocess, and split data into test/train sets"""

    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    # Generate synthetic Orlando gas price data for demonstration
    # In production, this would load from actual data sources
    print("Generating synthetic Orlando gas price data...")
    
    # Create date range for 3 years of daily data
    start_date = datetime.now() - timedelta(days=1095)
    end_date = datetime.now()
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Base gas price in Orlando (around $3.20/gallon historically)
    base_price = 3.20
    n_samples = len(dates)
    
    # Create synthetic features that influence gas price
    np.random.seed(42)
    
    # Seasonal trends (summer driving season)
    seasonal_factor = 0.2 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + 0.1 * np.sin(4 * np.pi * np.arange(n_samples) / 365.25)
    
    # Economic indicators (synthetic)
    crude_oil_price = 70 + 20 * np.sin(2 * np.pi * np.arange(n_samples) / 180) + np.random.normal(0, 5, n_samples)
    demand_index = 100 + 10 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 3, n_samples)
    supply_disruption = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% chance of disruption
    
    # Weather factors
    temperature = 75 + 15 * np.sin(2 * np.pi * np.arange(n_samples) / 365.25) + np.random.normal(0, 5, n_samples)
    hurricane_season = ((pd.to_datetime(dates).month >= 6) & (pd.to_datetime(dates).month <= 11)).astype(int)
    
    # Market factors
    day_of_week = pd.to_datetime(dates).dayofweek
    weekend = (day_of_week >= 5).astype(int)
    holiday_effect = np.random.choice([0, 1], n_samples, p=[0.97, 0.03])  # 3% chance of holiday
    
    # Calculate gas price based on features
    gas_price = (base_price + 
                0.6 * seasonal_factor +
                0.02 * (crude_oil_price - 70) +
                0.01 * (demand_index - 100) +
                0.15 * supply_disruption +
                0.005 * (temperature - 75) +
                0.08 * hurricane_season +
                0.05 * weekend +
                0.12 * holiday_effect +
                np.random.normal(0, 0.05, n_samples))  # noise
    
    # Ensure prices are realistic (between $2.50 and $5.00)
    gas_price = np.clip(gas_price, 2.5, 5.0)
    
    # Create DataFrame
    data = pd.DataFrame({
        'date': dates,
        'gas_price': gas_price,
        'crude_oil_price': crude_oil_price,
        'demand_index': demand_index,
        'supply_disruption': supply_disruption,
        'temperature': temperature,
        'hurricane_season': hurricane_season,
        'day_of_week': day_of_week,
        'weekend': weekend,
        'holiday_effect': holiday_effect
    })
    
    # Add time-based features
    data['month'] = pd.to_datetime(data['date']).dt.month
    data['quarter'] = pd.to_datetime(data['date']).dt.quarter
    data['day_of_year'] = pd.to_datetime(data['date']).dt.dayofyear
    
    # Add lag features (previous day's price)
    data['gas_price_lag1'] = data['gas_price'].shift(1)
    data['gas_price_lag7'] = data['gas_price'].shift(7)  # Previous week
    
    # Remove rows with NaN values (due to lag features)
    data = data.dropna().reset_index(drop=True)
    
    print(f"Generated {len(data)} samples with {data.shape[1]-2} features")
    print(f"Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"Gas price range: ${data['gas_price'].min():.2f} - ${data['gas_price'].max():.2f}")
    
    # Prepare features and target
    feature_columns = ['crude_oil_price', 'demand_index', 'supply_disruption', 
                      'temperature', 'hurricane_season', 'day_of_week', 
                      'weekend', 'holiday_effect', 'month', 'quarter', 
                      'day_of_year', 'gas_price_lag1', 'gas_price_lag7']
    
    X = data[feature_columns]
    y = data['gas_price']
    
    # Split data chronologically (more realistic for time series)
    split_idx = int(len(data) * (1 - args.test_size))
    
    train_data = {
        'X': X.iloc[:split_idx],
        'y': y.iloc[:split_idx],
        'dates': data['date'].iloc[:split_idx]
    }
    
    test_data = {
        'X': X.iloc[split_idx:],
        'y': y.iloc[split_idx:],
        'dates': data['date'].iloc[split_idx:]
    }
    
    print(f"\nTrain set: {len(train_data['X'])} samples")
    print(f"Test set: {len(test_data['X'])} samples")
    print(f"Features: {list(X.columns)}")
    
    return train_data, test_data


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def train_model(train_data, args):
    """Train energy price regression model"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    X_train = train_data['X']
    y_train = train_data['y']
    
    print(f"Training {args.model_type} model...")
    print(f"Training samples: {len(X_train)}")
    print(f"Features: {X_train.shape[1]}")
    
    # Feature scaling (important for linear regression)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Feature selection to reduce overfitting
    selector = SelectKBest(score_func=f_regression, k=min(10, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    
    # Train the model
    if args.model_type == "linear":
        model_core = LinearRegression()
    else:  # random_forest
        model_core = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state,
            n_jobs=-1
        )
        # For random forest, use original features (not scaled)
        X_train_selected = selector.fit_transform(X_train, y_train)
    
    model_core.fit(X_train_selected, y_train)
    
    # Create model pipeline
    model = {
        'scaler': scaler,
        'selector': selector,
        'model': model_core,
        'feature_names': list(X_train.columns),
        'model_type': args.model_type
    }
    
    # Print feature importance (for random forest)
    if args.model_type == "random_forest":
        selected_features = np.array(X_train.columns)[selector.get_support()]
        importance_scores = model_core.feature_importances_
        feature_importance = sorted(zip(selected_features, importance_scores), 
                                  key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 5 Feature Importances:")
        for feature, importance in feature_importance[:5]:
            print(f"  {feature}: {importance:.4f}")
    
    print(f"\nModel training completed successfully!")
    return model


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model, test_data):
    """Evaluate model on test set and return key metrics"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    X_test = test_data['X']
    y_test = test_data['y']
    dates_test = test_data['dates']
    
    # Preprocess test data same as training
    if model['model_type'] == "linear":
        X_test_scaled = model['scaler'].transform(X_test)
        X_test_selected = model['selector'].transform(X_test_scaled)
    else:  # random_forest
        X_test_selected = model['selector'].transform(X_test)
    
    # Make predictions
    y_pred = model['model'].predict(X_test_selected)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate percentage errors
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # Price-specific metrics
    avg_actual_price = np.mean(y_test)
    avg_predicted_price = np.mean(y_pred)
    price_difference = abs(avg_predicted_price - avg_actual_price)
    
    print(f"Model Performance Metrics:")
    print(f"  RMSE: ${rmse:.4f}")
    print(f"  MAE: ${mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  Average Actual Price: ${avg_actual_price:.3f}")
    print(f"  Average Predicted Price: ${avg_predicted_price:.3f}")
    print(f"  Price Difference: ${price_difference:.3f}")
    
    # Prediction range analysis
    pred_min, pred_max = np.min(y_pred), np.max(y_pred)
    actual_min, actual_max = np.min(y_test), np.max(y_test)
    
    print(f"\nPrice Range Analysis:")
    print(f"  Actual Range: ${actual_min:.3f} - ${actual_max:.3f}")
    print(f"  Predicted Range: ${pred_min:.3f} - ${pred_max:.3f}")
    
    evaluation = {
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape)
        },
        "price_analysis": {
            "avg_actual_price": float(avg_actual_price),
            "avg_predicted_price": float(avg_predicted_price),
            "price_difference": float(price_difference),
            "actual_range": {"min": float(actual_min), "max": float(actual_max)},
            "predicted_range": {"min": float(pred_min), "max": float(pred_max)}
        },
        "model_info": {
            "model_type": model['model_type'],
            "test_samples": len(y_test),
            "selected_features": len(model['selector'].get_support()),
            "total_features": len(model['feature_names'])
        },
        "predictions_sample": {
            "actual": [float(x) for x in y_test.iloc[:10].tolist()],
            "predicted": [float(x) for x in y_pred[:10].tolist()],
            "dates": [str(d) for d in dates_test.iloc[:10].tolist()]
        }
    }

    return evaluation


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
        "model_type": model['model_type'],
        "features": model['feature_names'],
        "selected_features": int(len(model['selector'].get_support())),
        "training_date": datetime.now().isoformat(),
        "model_version": "1.0.0",
        "domain": "energy_prices",
        "location": "Orlando, FL",
        "target": "gas_price_usd_per_gallon"
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
    print("ORLANDO GAS PRICE REGRESSION PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)

    # Step 1: Data Preprocessing
    train_data, test_data = load_and_preprocess_data(args)

    # Step 2: Model Training
    model = train_model(train_data, args)

    # Step 3: Model Evaluation
    evaluation = evaluate_model(model, test_data)

    # Save outputs
    save_outputs(model, args, evaluation)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"RMSE: ${evaluation['metrics']['rmse']:.4f}")
    print(f"R² Score: {evaluation['metrics']['r2_score']:.4f}")
    print(f"MAPE: {evaluation['metrics']['mape']:.2f}%")


if __name__ == "__main__":
    main()