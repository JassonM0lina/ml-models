# MLOps Standards Compliance Guide for Saudi Crude Burn Forecaster

## Project Overview

Your **Saudi Crude Burn Forecaster** project is a sophisticated machine learning system for predicting Saudi Arabia's crude oil burn for power generation based on weather conditions. The project contains:

- **Complete ML Implementation**: XGBoost model with weather API integration
- **Production-Ready Code**: Automated forecasting with email reporting  
- **Domain-Specific Features**: Temperature-based overrides, cooling degree day calculations
- **Historical Analysis**: Multi-city weather data processing and extreme heat modeling

## Required Changes for IIM Compliance

To deploy this project on the Integrated Intelligent Modelling (IIM) platform, you need to restructure your code to comply with SageMaker and IIM standards. Here are the specific changes required:

### 1. Create SageMaker-Compatible `train.py`

Your current model training is in a Jupyter notebook (`Saudi Direct Burn Forecast Model Creation.ipynb`). You need to convert this to a standalone `train.py` script with the following structure:

```python
import argparse
import os
import pandas as pd
import numpy as np
import pickle
import json
import shutil
import boto3
from datetime import datetime
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import requests

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Required IIM arguments
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--project-s3-path", type=str, default="")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    
    # Add your model hyperparameters
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    
    return parser.parse_args()

def load_and_preprocess_data(args):
    """Load training data from SageMaker input path"""
    # Load data from the training input path (not hardcoded local path)
    data_file = None
    input_path = args.train
    for filename in os.listdir(input_path):
        if filename.endswith('.xlsx') or filename.endswith('.csv'):
            data_file = os.path.join(input_path, filename)
            break
    
    if data_file.endswith('.xlsx'):
        burn_df = pd.read_excel(data_file)
    else:
        burn_df = pd.read_csv(data_file)
    
    # Your existing preprocessing logic here
    # ...
    
    return train_data, test_data

def fetch_weather_data():
    """Your existing weather fetching logic"""
    # Move your weather API logic here
    # ...

def train_model(train_data, args):
    """Train XGBoost model"""
    model = xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        # ... other parameters
    )
    
    # Your existing training logic
    # ...
    
    return model

def evaluate_model(model, test_data):
    """Evaluate model and return metrics"""
    # Your existing evaluation logic
    evaluation_data = {
        'overall_mae': overall_mae,
        'summer_mae': summer_mae,
        'peak_mae': peak_mae,
        'summer_mape': summer_mape,
        'test_predictions': test_predictions.to_dict('records')
    }
    
    return evaluation_data

def save_model_and_artifacts(model, args, evaluation_data):
    """Save model with SageMaker/IIM requirements"""
    
    # Save the model
    model_path = os.path.join(args.model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({
            'model': model,
            'model_name': 'XGBoost_Saudi_Crude_Burn',
            'feature_cols': feature_cols,
            'final_performance': evaluation_data
        }, f)
    
    # Save evaluation data to S3
    eval_s3_path = args.project_s3_path.rstrip("/") + "/evaluation_data/"
    parts = eval_s3_path[5:].split("/", 1)
    bucket = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_key = f"{prefix.rstrip('/')}/evaluation_{timestamp}.json"
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=eval_key,
        Body=json.dumps(evaluation_data, indent=2),
        ContentType="application/json"
    )
    
    # Copy inference artifacts (REQUIRED for SageMaker)
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
    
    # Copy requirements.txt
    requirements_src = "/opt/ml/code/requirements.txt"
    requirements_dst = args.model_dir
    if os.path.exists(requirements_src):
        shutil.copy(requirements_src, requirements_dst)
    
    # Copy inference.py
    inference_src = "/opt/ml/code/inference.py"
    inference_dst = os.path.join(args.model_dir, "inference.py")
    if os.path.exists(inference_src):
        shutil.copy(inference_src, inference_dst)
    
    # Create setup.py
    setup_dst = os.path.join(args.model_dir, "setup.py")
    with open(setup_dst, "w") as f:
        f.write(setup_py_content)

def main():
    args = parse_args()
    
    # Load and preprocess data
    train_data, test_data = load_and_preprocess_data(args)
    
    # Fetch weather data
    weather_data = fetch_weather_data()
    
    # Train model
    model = train_model(train_data, args)
    
    # Evaluate model
    evaluation_data = evaluate_model(model, test_data)
    
    # Save outputs
    save_model_and_artifacts(model, args, evaluation_data)

if __name__ == "__main__":
    main()
```

### 2. Create SageMaker-Compatible `inference.py`

Convert your forecasting logic from `saudi_burn_forecast.py` to a SageMaker inference script:

```python
import os
import json
import pickle
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

def model_fn(model_dir):
    """Load the trained model"""
    model_path = os.path.join(model_dir, "model.pkl")
    
    with open(model_path, "rb") as f:
        model_package = pickle.load(f)
    
    return model_package

def input_fn(request_body, request_content_type):
    """Parse input parameters"""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")
    
    input_data = json.loads(request_body)
    return input_data

def predict_fn(input_data, model_package):
    """Generate crude burn forecast"""
    model = model_package['model']
    feature_cols = model_package['feature_cols']
    
    # Get parameters from input
    forecast_months = input_data.get('forecast_months', 3)
    target_year = input_data.get('target_year', datetime.now().year)
    target_month = input_data.get('target_month', datetime.now().month)
    
    # Fetch current weather data (your existing logic)
    weather_data = fetch_current_weather()
    
    # Prepare features for prediction (your existing logic)
    features = prepare_monthly_features(weather_data, target_year, target_month)
    
    # Make prediction
    X_pred = pd.DataFrame([features])
    
    # Ensure all required features are present
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = 0
    
    X_pred = X_pred[feature_cols]
    
    # Get base prediction
    base_prediction = model.predict(X_pred.values)[0]
    
    # Apply temperature-based rules (your existing logic)
    final_prediction, adjustment_type, confidence = apply_temperature_rules(
        base_prediction, features
    )
    
    # Calculate confidence intervals
    model_mae = model_package['final_performance']['summer_mae'] if target_month in [6,7,8,9] else model_package['final_performance']['overall_mae']
    lower_bound = final_prediction - 2 * model_mae
    upper_bound = final_prediction + 2 * model_mae
    
    return {
        'prediction_kbd': final_prediction,
        'base_prediction_kbd': base_prediction,
        'adjustment_type': adjustment_type,
        'confidence': confidence,
        'confidence_interval': {
            'lower': lower_bound,
            'upper': upper_bound
        },
        'weather_conditions': {
            'max_temp_c': features['temp_max_max_max'],
            'extreme_days_44c': int(features['extreme_44_sum']),
            'cooling_degree_days': features['cdd_24_sum']
        }
    }

def fetch_current_weather():
    """Your existing weather fetching logic"""
    # Move your weather API logic here
    pass

def prepare_monthly_features(weather_data, target_year, target_month):
    """Your existing feature preparation logic"""
    # Move your feature engineering logic here
    pass

def apply_temperature_rules(model_prediction, features):
    """Your existing temperature override logic"""
    # Move your rule-based adjustment logic here
    pass

def output_fn(predictions, response_content_type):
    """Format output response"""
    if response_content_type == "application/json":
        return json.dumps(predictions)
    else:
        raise ValueError(f"Unsupported response type: {response_content_type}")
```

### 3. Update `requirements.txt`

Your current requirements include Windows-specific packages that won't work in SageMaker. Update to:

```txt
# Additional packages for SageMaker sklearn container
# (pandas, numpy, scikit-learn already included)

xgboost>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
requests>=2.31.0
urllib3>=2.0.0
python-dateutil>=2.8.0

# Remove Windows-specific packages:
# pywin32 - not needed in SageMaker Linux environment  
# win32com.client - Windows only, remove email functionality for SageMaker
```

### 4. Create Inference Schema

Based on your forecasting logic, create an `inference_schema.json`:

```json
{
    "parameters": {
        "forecast_months": {
            "name": "Forecast Months",
            "tooltip": "Number of months to forecast (1-6)",
            "display_type": "slider",
            "default_value": 3,
            "required": true,
            "min_value": 1,
            "max_value": 6,
            "step_value": 1
        },
        "target_year": {
            "name": "Target Year",
            "tooltip": "Year for forecast (current year if not specified)",
            "display_type": "number",
            "default_value": 2025,
            "required": false,
            "min_value": 2024,
            "max_value": 2030
        },
        "target_month": {
            "name": "Starting Month",
            "tooltip": "Starting month for forecast (1=Jan, 12=Dec)",
            "display_type": "slider", 
            "default_value": 1,
            "required": true,
            "min_value": 1,
            "max_value": 12,
            "step_value": 1
        },
        "use_temperature_override": {
            "name": "Enable Temperature Override",
            "tooltip": "Apply rule-based adjustments for extreme heat conditions",
            "display_type": "boolean",
            "default_value": true,
            "required": false
        }
    }
}
```

### 5. Prepare Training Data

The training data has been uploaded as a parquet file to its proper location. This parquet file was created directly from Saudi Crude Burn History.xlsx, so it should have the same content as the excel file.

## Key Architectural Changes

### Data Flow Changes
- **Before**: Local file paths and Windows environment
- **After**: SageMaker managed paths and Linux container environment

### Model Artifacts
- **Before**: Single pickle file with model
- **After**: Model + inference code + setup.py + requirements.txt packaged together

### API Integration  
- **Before**: Direct weather API calls in production script
- **After**: Weather API calls within SageMaker inference endpoint

### Evaluation Storage
- **Before**: Local Excel log files
- **After**: JSON evaluation data stored in S3 `evaluation_data/` folder

## Domain-Specific Considerations

### Weather Data Integration
Your existing Open-Meteo API integration should work in SageMaker, but consider:
- Adding error handling for API timeouts
- Caching weather data to reduce API calls
- Using fallback historical averages if API is unavailable

### Temperature Override Logic
Your sophisticated rule-based adjustments for extreme heat are valuable:
- Keep the temperature threshold logic (48°C, 46°C, 44°C)
- Maintain the confidence-weighted blending approach
- Preserve the cooling degree day calculations

### Feature Engineering
Your 54 engineered features are well-designed:
- Weather aggregations across multiple Saudi cities
- Lag features and moving averages for burn history
- Extreme heat indicators and consecutive hot day tracking
- Summer-specific and Ramadan adjustments

## Implementation Steps

1. **Extract Core Logic**: Move your model training from Jupyter notebook to `train.py`
2. **Adapt Inference**: Convert production forecasting script to SageMaker `inference.py`
3. **Update Dependencies**: Remove Windows-specific packages from `requirements.txt`
4. **Test Data Flow**: Ensure training data loads correctly from SageMaker paths
5. **Validate Weather API**: Test Open-Meteo integration in containerized environment

## Expected Benefits

After compliance:
- **Automated Training**: Model retraining through SageMaker pipelines
- **Scalable Inference**: Serverless endpoints with auto-scaling
- **Model Monitoring**: Built-in performance tracking and evaluation storage
- **Version Control**: Automated model versioning and artifact management
- **Integration Ready**: Direct integration with IIM platform interface

Your domain expertise in Saudi energy markets and sophisticated weather modeling will be preserved while gaining enterprise-grade MLOps capabilities.