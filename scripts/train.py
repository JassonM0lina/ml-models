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
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
import numpy as np

# TODO: Add your model-specific imports here
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


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

    # TODO: Add your model hyperparameters here
    parser.add_argument("--fit-intercept", type=bool, default=True, help="Whether to calculate the intercept")
    parser.add_argument("--normalize", type=bool, default=True, help="Whether to normalize features")

    # Data split ratios
    parser.add_argument("--train-split", type=float, default=0.7)
    parser.add_argument("--validation-split", type=float, default=0.15)
    parser.add_argument("--test-split", type=float, default=0.15)

    # SageMaker environment variables
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--output-data-dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--project-s3-path", type=str, default="")  # S3 path for saving evaluation

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
    df = pd.read_csv(data_file)

    # TODO: Add your data preprocessing logic here
    print(f"  Original columns: {list(df.columns)}")
    print(f"  Data types: {df.dtypes.to_dict()}")
    
    # Drop date column if it exists (not needed for property feature prediction)
    if 'date' in df.columns:
        df = df.drop('date', axis=1)
        print(f"  Dropped date column - not needed for property prediction")
    
    # Handle categorical variables if any
    categorical_columns = df.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        print(f"  Found categorical columns: {list(categorical_columns)}")
        # For simplicity, drop categorical columns (could use encoding in practice)
        df = df.select_dtypes(exclude=['object'])
        print(f"  Dropped categorical columns for simplicity")
    
    # Ensure we have the expected columns that match inference schema
    expected_feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lot_size', 'garage_spaces']
    expected_target_col = 'price'
    
    missing_cols = []
    for col in expected_feature_cols + [expected_target_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        raise ValueError(f"Missing expected columns: {missing_cols}. Available: {list(df.columns)}")
    
    # Keep only the expected columns to match inference schema exactly
    df = df[expected_feature_cols + [expected_target_col]]
    
    print(f"  Final columns for modeling: {list(df.columns)}")

    print(f"  Loaded {len(df)} rows")

    # Data quality checks
    print("\nRunning data quality checks...")
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"  Found {missing_values} missing values")
        # TODO: Handle missing values (fill, drop, etc.)
        df = df.fillna(df.median())
        print(f"  Filled missing values with median")

    # Validate minimum rows for splitting
    n = len(df)
    min_rows = 20  # Minimum rows needed for train/validation/test split (70%/15%/15%)
    if n < min_rows:
        raise ValueError(f"Dataset too small: {n} rows. Need at least {min_rows} rows to split into train/validation/test sets.")

    # Split data
    print(f"\nSplitting data (train: {train_split}, val: {validation_split}, test: {test_split})...")
    train_size = int(n * train_split)
    val_size = int(n * validation_split)

    # Ensure each split has at least 1 sample
    if train_size < 1:
        train_size = 1
    if val_size < 1:
        val_size = 1
    test_size = n - train_size - val_size
    if test_size < 1:
        raise ValueError(f"Dataset too small for split ratios. Got {n} rows but need at least {train_size + val_size + 1} for train({train_split})/val({validation_split})/test({test_split}).")

    train_data = df[:train_size]
    val_data = df[train_size:train_size + val_size]
    test_data = df[train_size + val_size:]

    print(f"  Train: {len(train_data)} samples")
    print(f"  Validation: {len(val_data)} samples")
    print(f"  Test: {len(test_data)} samples")

    data_quality_report = {
        "total_samples": int(n),
        "train_samples": int(len(train_data)),
        "validation_samples": int(len(val_data)),
        "test_samples": int(len(test_data)),
        "missing_values": int(missing_values),
        "columns": list(df.columns)
    }

    return train_data, val_data, test_data, data_quality_report


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

def train_model(train_data, val_data, args):
    """Train model and validate"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    # TODO: Implement your model training logic here
    # Use property features that match inference schema exactly
    feature_cols = ['bedrooms', 'bathrooms', 'sqft', 'year_built', 'lot_size', 'garage_spaces']
    target_col = 'price'
    
    print(f"  Feature columns: {feature_cols}")
    print(f"  Target column: {target_col}")
    
    # Prepare training data
    X_train = train_data[feature_cols]
    y_train = train_data[target_col]
    
    X_val = val_data[feature_cols]
    y_val = val_data[target_col]
    
    print(f"  Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Validation set: {X_val.shape[0]} samples")
    
    # Feature scaling
    scaler = StandardScaler() if args.normalize else None
    if scaler:
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        print(f"  Applied feature scaling")
    else:
        X_train_scaled = X_train.values
        X_val_scaled = X_val.values
    
    # Train linear regression model
    model = LinearRegression(fit_intercept=args.fit_intercept)
    model.fit(X_train_scaled, y_train)
    
    # Store scaler with model for inference
    model_dict = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_cols
    }

    print("  Model trained successfully")
    print(f"  Intercept: {model.intercept_:.2f}")
    print(f"  Coefficients shape: {model.coef_.shape}")

    # Validate on validation set
    print(f"\nValidating on {len(val_data)} samples...")
    # TODO: Calculate validation metrics
    val_predictions = model.predict(X_val_scaled)
    val_rmse = sqrt(mean_squared_error(y_val, val_predictions))
    val_r2 = r2_score(y_val, val_predictions)

    print(f"  Validation RMSE: {val_rmse:.2f}")
    print(f"  Validation R²: {val_r2:.4f}")

    return model_dict, val_rmse


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================

def evaluate_model(model_dict, test_data):
    """Evaluate model on test set"""
    print("\n" + "=" * 60)
    print("STEP 3: MODEL EVALUATION")
    print("=" * 60)

    print(f"\nEvaluating on {len(test_data)} test samples...")

    # TODO: Implement your evaluation logic here
    model = model_dict['model']
    scaler = model_dict['scaler']
    feature_cols = model_dict['feature_columns']
    
    # Prepare test data
    X_test = test_data[feature_cols]
    y_test = test_data['price']
    
    # Apply scaling if used during training
    if scaler:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test.values
    
    # Generate predictions
    predictions = model.predict(X_test_scaled)
    actuals = y_test.values
    errors = predictions - actuals

    # Calculate metrics
    # TODO: Replace with actual metric calculations
    rmse = sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    # MAPE calculation (avoid division by zero)
    mape = np.mean(np.abs((actuals - predictions) / np.maximum(np.abs(actuals), 1e-8))) * 100
    r2 = r2_score(actuals, predictions)

    print(f"\nTest Set Metrics:")
    print(f"  RMSE: {rmse:.2f}")
    print(f"  MAE: {mae:.2f}")
    print(f"  MAPE: {mape:.2f}%")
    print(f"  R²: {r2:.4f}")

    return {
        "metrics": {
            "rmse": float(rmse),
            "mae": float(mae),
            "mape": float(mape),
            "r2": float(r2)
        },
        "predictions": predictions.tolist(),
        "actuals": actuals.tolist(),
        "errors": errors.tolist()
    }


# =============================================================================
# SAVE OUTPUTS
# =============================================================================

def save_outputs(model, args, data_quality_report, evaluation_results):
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

    # Copy inference.py to model root as well
    inference_root_dst = os.path.join(args.model_dir, "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_root_dst)
        print(f"  Copied inference.py to model root")

    # Save metadata
    metadata = {
        "model_type": "Linear Regression",
        "data_quality": data_quality_report,
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
            "mean": sum(evaluation_results["predictions"]) / len(evaluation_results["predictions"]) if evaluation_results["predictions"] else 0,
            "min": min(evaluation_results["predictions"]) if evaluation_results["predictions"] else 0,
            "max": max(evaluation_results["predictions"]) if evaluation_results["predictions"] else 0
        }
    }

    eval_path = os.path.join(eval_output_dir, "evaluation.json")
    with open(eval_path, "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"\nSaved evaluation report to: {eval_path}")

    # Upload evaluation to S3 evaluation_data folder
    if args.project_s3_path:
        # project_s3_path format: s3://bucket/projects/user/project/project/
        # Append evaluation_data/ to get the target folder
        eval_s3_path = args.project_s3_path.rstrip("/") + "/evaluation_data/"
        upload_evaluation_to_s3(evaluation_details, eval_s3_path)

    return eval_report


def main():
    """Main training logic"""
    args = parse_args()

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)

    # Step 1: Data Preprocessing
    train_data, val_data, test_data, data_quality_report = load_and_preprocess_data(
        args.train,
        args.train_split,
        args.validation_split,
        args.test_split
    )

    # Step 2: Model Training
    model, val_rmse = train_model(train_data, val_data, args)

    # Step 3: Model Evaluation
    evaluation_results = evaluate_model(model, test_data)

    # Save outputs
    eval_report = save_outputs(model, args, data_quality_report, evaluation_results)

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