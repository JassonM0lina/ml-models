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
from sklearn.metrics import mean_squared_error
from math import sqrt

# TODO: Add your model-specific imports here
# Example: from sklearn.linear_model import LinearRegression
# Example: from statsmodels.tsa.arima.model import ARIMA


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
    # Example: parser.add_argument("--learning-rate", type=float, default=0.01)

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

# TODO: modify load_and_preprocess_data to suit your data and model, modify parameters as needed
def load_and_preprocess_data():
    """Load, preprocess, and split data into test/train sets"""

    print("=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    train_data = []

    test_data = []
    
    return train_data, test_data


# =============================================================================
# STEP 2: MODEL TRAINING
# =============================================================================

# TODO: modify train_model to suit your model, modify parameters as needed
def train_model(train_data):
    """Train model"""
    print("\n" + "=" * 60)
    print("STEP 2: MODEL TRAINING")
    print("=" * 60)

    model = None

    return model


# =============================================================================
# STEP 3: MODEL EVALUATION
# =============================================================================
#TODO: modify evaluate_model to suit your model, add parameters as needed
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

    #TODO: add metadata as needed
    metadata = {

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
    print("TRAINING PIPELINE")
    print("Preprocess → Train → Evaluate")
    print("=" * 60)

    #TODO: Modify the function calls as needed.

    # Step 1: Data Preprocessing
    train_data, test_data = load_and_preprocess_data()

    # Step 2: Model Training
    model = train_model(train_data)

    # Step 3: Model Evaluation
    evaluation = evaluate_model(model, test_data)

    # Save outputs
    save_outputs(model, args, evaluation)

    # Print final summary
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
