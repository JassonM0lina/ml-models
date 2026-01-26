Saudi Arabia Crude Oil Direct Burn Forecasting Model
Model Overview
An automated machine learning system that forecasts monthly crude oil direct burn for power generation in Saudi Arabia. The model combines historical burn data (from JODI) with weather patterns from seven major Saudi cities to predict direct burn of the current month, with particular focus on summer peak months (June-September) when extreme heat drives peak power demand. This is especially relevant, as deviations in crude burn can swing oil balances, and JODI data is released at a multi-month lag.
How the Model Works
●	1. Weather Data Collection: Fetches real-time weather from Open-Meteo API for 7 cities (Riyadh, Jeddah, Dammam, Mecca, Jubail, Yanbu, Khobar) with population and GPD-weighted aggregation.
●	2. Feature Engineering: Calculates cooling degree days, extreme heat indicators (>111°F, >115°F, >118°F), temperature polynomials, and seasonal patterns.
●	3. Machine Learning: XGBoost/LightGBM ensemble trained on historical data with lag features from previous months to capture consumption patterns.
●	4. Temperature Override: Applies automatic adjustments when extreme conditions (>118°F with high extreme day counts) are detected.
●	5. Daily Forecasting Process: Runs daily via Task Scheduler. Early in the month, forecasts use weather predictions from API. As days pass, weather forecasts become actual, increasing forecast confidence. Each day's forecast is logged to track accuracy.
Required Files and Workflow
File Name	Purpose
Saudi Crude Burn History.xlsx	Historical monthly burn data from JODI - PRIMARY INPUT
summer_peak_model_cv.pkl	Trained XGBoost model file with feature definitions
model_test_predictions.csv	Test set predictions for chart visualization
Saudi_Burn_Forecast_Log.xlsx	Output log tracking all daily forecasts and performance
saudi_burn_forecast.py	Daily automated script - fetches weather, generates forecast, sends email
Saudi Direct Burn Forecast Model Creation.ipynb	Model training notebook - run manually when retraining is needed
Monthly Update Process
1.	1. Add new monthly actuals from JODI to Saudi Crude Burn History.xlsx
2.	2. Automated Python script (saudi_burn_forecast.py) runs daily via Windows Task Scheduler
3.	3. Daily forecasts generated with increasing confidence as weather forecasts become actuals
4.	4. Automated email report sent to stakeholders with charts and predictions
5.	5. Model retraining: Run Saudi Direct Burn Forecast Model Creation.ipynb manually when consistent forecast errors are observed vs JODI actuals
Model Performance
●	Overall Test Set: MAE ~40-50 kbd, MAPE ~7-9%
●	Summer Months (Jun-Sep): MAE 41-46 kbd, MAPE 6.5-7.4%
●	Peak Summer (Jul-Aug): MAE ~45 kbd, MAPE ~6%
●	Model Type: XGBoost with time-weighted training (2024-2025 data weighted 20x)
●	Key Features: Max temperature, cooling degree days, extreme heat days (>111°F), lag features
●	Confidence Intervals: 95% CI calculated as ±2 × MAE
