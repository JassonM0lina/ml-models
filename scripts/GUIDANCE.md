# MLOps Standards Compliance Guide
## Saudi Oil Burn Forecaster Project

### Executive Summary

You have developed an impressive and comprehensive Saudi Arabia crude oil burn forecasting system. Your project includes sophisticated weather data integration, XGBoost/LightGBM ensemble modeling, temperature-based rule overrides, and automated email reporting. However, to deploy this on our Integrated Intelligent Modelling (IIM) platform, several structural changes are required to meet SageMaker and IIM compliance standards.

### Current Project Strengths

Your existing system demonstrates excellent domain expertise:
- **Advanced Feature Engineering**: Weather data from 7 Saudi cities, cooling degree days, extreme heat indicators
- **Robust Modeling**: XGBoost with time-weighted training, temperature override rules, 95% confidence intervals
- **Performance Tracking**: MAE ~40-50 kbd, MAPE ~7-9% on test set
- **Operational Integration**: Daily automated forecasting, email reports, Excel logging

### Required Changes for MLOps Compliance

#### 1. **CRITICAL: Restructure for SageMaker Architecture**

Your current `saudi_burn_forecast.py` (1,100+ lines) combines training, inference, and operational tasks. This must be separated into two distinct SageMaker-compliant scripts:

**A. train.py Structure Required:**
```python
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--project-s3-path", type=str, default="")
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    return parser.parse_args()
```

**What needs to move from your existing code to train.py:**
- Historical data loading (`Saudi Crude Burn History.xlsx` → CSV format)
- Weather API integration and feature engineering
- XGBoost/LightGBM model training with cross-validation
- Temperature override rule development
- Model evaluation and performance metrics
- Model artifact saving (your existing `.pkl` files)

**B. inference.py Structure Required:**
```python
def model_fn(model_dir):
    # Load your trained XGBoost model and feature definitions

def predict_fn(input_data, model):
    # Weather-based prediction logic
    # Temperature override rules
    # Return forecast with confidence intervals
```

**What needs to move to inference.py:**
- Model loading (`summer_peak_model_cv.pkl`)
- Real-time weather data fetching
- Monthly feature preparation
- Temperature-based rule application
- Prediction generation with confidence bounds

#### 2. **Data Pipeline Compliance**

**Current State:** Direct Excel file access, hardcoded Windows paths
```python
WORKING_DIR = r"C:\Users\i31851\OneDrive..."
burn_df = pd.read_excel(f"{WORKING_DIR}/Saudi Crude Burn History.xlsx")
```

**Required Changes:**
- Convert Excel files to CSV format in `training_data/` folder
- Load training data from SageMaker input path:
```python
args = parse_args()
data_file = None
input_path = args.train
for filename in os.listdir(input_path):
    data_file = os.path.join(input_path, filename)
df = pd.read_csv(data_file)
```

#### 3. **Evaluation Data Storage**

**Current State:** Local Excel logging
**Required:** S3 evaluation data storage

```python
evaluation_data = {
    "overall_mae": 45.2,
    "summer_mae": 41.8,
    "peak_mae": 44.6,
    "summer_mape": 7.1,
    "model_performance": model_metrics,
    "feature_importance": feature_rankings
}

eval_s3_path = args.project_s3_path.rstrip("/") + "/evaluation_data/"
s3 = boto3.client("s3")
# Upload evaluation metrics
```

#### 4. **Dependencies Management**

**Current requirements.txt issues:**
- Windows-specific packages (`pywin32`)
- Email/visualization packages not needed for inference
- Version conflicts with SageMaker container

**Required requirements.txt (inference-only):**
```
xgboost>=2.0.0
requests>=2.31.0
openmeteo-requests>=1.1.0
python-dateutil>=2.8.0
```

**Note:** pandas, numpy, scikit-learn are pre-installed in SageMaker container

#### 5. **Inference Schema Definition**

Based on your weather feature engineering, the inference schema should capture the key parameters users can configure:

```json
{
    "name": "Saudi Oil Burn Forecaster",
    "info": "Forecasts monthly Saudi Arabia crude oil direct burn for power generation using weather data and machine learning",
    "model_type": "time_series_forecasting",
    "parameters": {
        "forecast_months": {
            "name": "Number of Forecast Months",
            "tooltip": "How many months ahead to forecast (1-6 months)",
            "display_type": "slider",
            "default_value": 3,
            "min_value": 1,
            "max_value": 6,
            "step_value": 1,
            "required": true
        },
        "use_weather_forecast": {
            "name": "Use Weather Forecast",
            "tooltip": "Use predicted weather vs historical averages for future months",
            "display_type": "boolean",
            "default_value": true,
            "required": false
        },
        "confidence_level": {
            "name": "Confidence Level",
            "tooltip": "Confidence interval percentage for predictions",
            "display_type": "selectbox",
            "options": [90, 95, 99],
            "default_value": 95,
            "required": false
        }
    },
    "output": {
        "monthly_forecasts": {
            "name": "Monthly Burn Forecasts",
            "type": "array",
            "unit": "kbd"
        },
        "confidence_bounds": {
            "name": "Confidence Intervals",
            "type": "array",
            "unit": "kbd"
        },
        "weather_drivers": {
            "name": "Key Weather Metrics",
            "type": "object"
        }
    }
}
```

#### 6. **Code Architecture Recommendations**

**Break down your 1,100-line script into modular functions:**

**train.py modules:**
- `fetch_historical_weather()`: Weather API integration
- `prepare_features()`: Feature engineering pipeline  
- `train_ensemble_model()`: XGBoost/LightGBM training
- `develop_temperature_rules()`: Rule-based override logic
- `evaluate_performance()`: Cross-validation and testing

**inference.py modules:**
- `load_model_artifacts()`: Model + feature definitions
- `fetch_current_weather()`: Real-time weather data
- `apply_temperature_overrides()`: Rule application
- `generate_forecasts()`: Multi-month prediction

#### 7. **Operational Features to Preserve**

Your automation and monitoring capabilities are valuable but need restructuring:

**Move to separate operational scripts:**
- Email reporting → Post-inference Lambda function
- Excel logging → CloudWatch metrics + S3 storage
- Chart generation → Post-processing visualization service
- Daily scheduling → EventBridge rules

#### 8. **Migration Strategy**

**Phase 1: Core Model Extraction**
1. Extract model training logic to `train.py`
2. Extract prediction logic to `inference.py`
3. Convert Excel data to CSV format
4. Test basic SageMaker training job

**Phase 2: Feature Integration**
1. Integrate weather API calls in inference
2. Implement temperature override rules
3. Add confidence interval calculations
4. Test endpoint deployment

**Phase 3: Operational Integration**
1. Set up evaluation data pipeline
2. Configure automated retraining triggers  
3. Implement monitoring and alerting
4. Migrate reporting to cloud services

### Implementation Priorities

**HIGH PRIORITY (Required for deployment):**
- ✅ Separate train/inference code architecture
- ✅ SageMaker argument parser implementation
- ✅ Training data CSV conversion and S3 integration
- ✅ Model artifact setup.py configuration
- ✅ Clean requirements.txt (remove Windows dependencies)

**MEDIUM PRIORITY (Enhanced functionality):**
- Weather API error handling and fallbacks
- Model versioning and A/B testing setup
- Automated retraining based on performance degradation
- Cloud-native monitoring and alerting

**LOW PRIORITY (Operational efficiency):**
- Email/reporting service migration
- Visualization service setup
- Historical log data migration

### Technical Debt Considerations

Your current system has some technical debt that should be addressed:

1. **Hardcoded paths and configurations** → Environment variables and configuration files
2. **Windows-specific dependencies** → Cross-platform alternatives
3. **Monolithic architecture** → Microservices design
4. **File-based logging** → Structured logging with CloudWatch

### Conclusion

Your Saudi oil burn forecasting system represents sophisticated domain expertise and strong predictive modeling. The primary work required is architectural restructuring to align with SageMaker's training/inference separation and IIM platform standards. The core modeling logic and weather integration can be preserved while adapting to cloud-native MLOps practices.

The migration will enhance your system's scalability, maintainability, and integration capabilities while preserving the advanced forecasting features that make it valuable for crude oil market analysis.