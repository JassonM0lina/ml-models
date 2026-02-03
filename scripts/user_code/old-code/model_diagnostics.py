import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ============================================================
# CONFIGURATION
# ============================================================
WORKING_DIR = r"C:\Users\i31851\OneDrive - Wood Mackenzie Limited\Documents\Automation Work For Others\Jim Mitchell\Saudi Direct Burn V2"

print("="*80)
print("MODEL DIAGNOSTIC ANALYSIS - WHY ARE PREDICTIONS TOO HIGH?")
print("="*80)

# ============================================================
# LOAD CURRENT MODEL AND DATA
# ============================================================

print("\n[1] LOADING MODEL AND DATA")
print("-"*60)

# Load model
with open(f"{WORKING_DIR}/summer_peak_model_cv.pkl", 'rb') as f:
    model_package = pickle.load(f)

model = model_package['model']
feature_cols = model_package['feature_cols']

print(f"✓ Model loaded: {model_package['model_name']}")
print(f"  Training metrics:")
print(f"    Overall MAE: {model_package['final_performance']['overall_mae']:.1f} kbd")
print(f"    Summer MAE: {model_package['final_performance']['summer_mae']:.1f} kbd")

# Load historical data
burn_df = pd.read_excel(f"{WORKING_DIR}/Saudi Crude Burn History.xlsx")

# Handle different possible formats
if burn_df.shape[1] == 2:
    burn_df.columns = ['date', 'burn_bpd']
else:
    # Find date and burn columns
    date_col = None
    burn_col = None
    
    for col in burn_df.columns:
        if 'date' in str(col).lower():
            date_col = col
        elif 'saudi' in str(col).lower():
            burn_col = col
    
    if date_col is None:
        date_col = burn_df.columns[0]
    if burn_col is None:
        burn_col = burn_df.columns[1]
    
    burn_df = burn_df[[date_col, burn_col]]
    burn_df.columns = ['date', 'burn_bpd']

# Clean data
burn_df['date'] = pd.to_datetime(burn_df['date'], errors='coerce')
burn_df['burn_bpd'] = pd.to_numeric(burn_df['burn_bpd'], errors='coerce')
burn_df = burn_df.dropna()  # Remove any rows with invalid data

# Convert to kbd
if burn_df['burn_bpd'].mean() > 10000:
    burn_df['burn_kbd'] = burn_df['burn_bpd'] / 1000
else:
    burn_df['burn_kbd'] = burn_df['burn_bpd']
burn_df['year'] = burn_df['date'].dt.year
burn_df['month'] = burn_df['date'].dt.month
burn_df = burn_df.sort_values('date')

print(f"\n✓ Historical data loaded")
print(f"  Date range: {burn_df['date'].min().strftime('%Y-%m')} to {burn_df['date'].max().strftime('%Y-%m')}")
print(f"  Total months: {len(burn_df)}")

# Load forecast log
try:
    forecast_log = pd.read_excel(f"{WORKING_DIR}/Saudi_Burn_Forecast_Log.xlsx")
    forecast_log['Date'] = pd.to_datetime(forecast_log['Date'])
    forecast_log['Target Month'] = pd.to_datetime(forecast_log['Target Month'])
    print(f"\n✓ Forecast log loaded: {len(forecast_log)} forecasts on record")
except:
    forecast_log = pd.DataFrame()
    print("\n⚠ No forecast log found")

# ============================================================
# ANALYZE RECENT PERFORMANCE
# ============================================================

print("\n[2] RECENT FORECAST PERFORMANCE ANALYSIS")
print("-"*60)

if len(forecast_log) > 0:
    # Get forecasts where we now have actuals
    forecast_log['target_year'] = forecast_log['Target Month'].dt.year
    forecast_log['target_month'] = forecast_log['Target Month'].dt.month
    
    # Merge with actuals
    comparison = forecast_log.merge(
        burn_df[['year', 'month', 'burn_kbd']],
        left_on=['target_year', 'target_month'],
        right_on=['year', 'month'],
        how='inner'
    )
    
    if len(comparison) > 0:
        comparison['error'] = comparison['Forecast (kbd)'] - comparison['burn_kbd']
        comparison['abs_error'] = comparison['error'].abs()
        comparison['pct_error'] = (comparison['error'] / comparison['burn_kbd']) * 100
        
        print(f"\nForecasts with actuals available: {len(comparison)}")
        print(f"\n{'Month':<15} {'Forecast':>10} {'Actual':>10} {'Error':>10} {'% Error':>10}")
        print("-"*60)
        
        for _, row in comparison.tail(12).iterrows():
            print(f"{row['Target Month'].strftime('%Y-%m'):<15} {row['Forecast (kbd)']:>10.0f} {row['burn_kbd']:>10.0f} {row['error']:>10.0f} {row['pct_error']:>10.1f}%")
        
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Mean Error (bias): {comparison['error'].mean():+.1f} kbd")
        print(f"Mean Absolute Error: {comparison['abs_error'].mean():.1f} kbd")
        print(f"Mean % Error: {comparison['pct_error'].mean():+.1f}%")
        print(f"Median Error: {comparison['error'].median():+.1f} kbd")
        
        # Check if model is consistently high
        pct_overestimate = (comparison['error'] > 0).sum() / len(comparison) * 100
        print(f"\nOverestimation frequency: {pct_overestimate:.1f}% of forecasts")
        
        if comparison['error'].mean() > 50:
            print("\n⚠️ CRITICAL: Model has significant POSITIVE BIAS (consistently too high)")
        elif comparison['error'].mean() < -50:
            print("\n⚠️ CRITICAL: Model has significant NEGATIVE BIAS (consistently too low)")
        
        # Recent trend
        recent = comparison.tail(6)
        if len(recent) > 0:
            print(f"\nLast 6 months:")
            print(f"  Mean error: {recent['error'].mean():+.1f} kbd")
            print(f"  Mean % error: {recent['pct_error'].mean():+.1f}%")

# ============================================================
# ANALYZE TRAINING DATA DISTRIBUTION
# ============================================================

print("\n[3] TRAINING DATA ANALYSIS")
print("-"*60)

# Show burn trends over time
recent_years = burn_df[burn_df['year'] >= 2020].copy()

print("\nRecent burn history by year:")
for year in sorted(recent_years['year'].unique()):
    year_data = recent_years[recent_years['year'] == year]
    print(f"\n{year}:")
    print(f"  Mean: {year_data['burn_kbd'].mean():.0f} kbd")
    print(f"  Summer (Jun-Sep): {year_data[year_data['month'].isin([6,7,8,9])]['burn_kbd'].mean():.0f} kbd")
    print(f"  Peak (Jul-Aug): {year_data[year_data['month'].isin([7,8])]['burn_kbd'].mean():.0f} kbd")

# Check for structural breaks
print("\n" + "="*60)
print("POTENTIAL ISSUES")
print("="*60)

# Issue 1: Has burn declined recently?
last_12_months = burn_df.tail(12)['burn_kbd'].mean()
previous_12_months = burn_df.iloc[-24:-12]['burn_kbd'].mean()
decline = previous_12_months - last_12_months

if decline > 50:
    print(f"\n⚠️ Issue 1: SIGNIFICANT RECENT DECLINE")
    print(f"   Last 12 months avg: {last_12_months:.0f} kbd")
    print(f"   Previous 12 months: {previous_12_months:.0f} kbd")
    print(f"   Decline: {decline:.0f} kbd ({decline/previous_12_months*100:.1f}%)")
    print(f"   → Model trained on higher historical burns, may not reflect new normal")

# Issue 2: Lag features anchoring to outdated values
recent_summer = burn_df[(burn_df['year'] == 2025) & (burn_df['month'].isin([6,7,8]))]['burn_kbd']
historical_summer = burn_df[(burn_df['year'] < 2025) & (burn_df['month'].isin([6,7,8]))]['burn_kbd']

if len(recent_summer) > 0 and len(historical_summer) > 0:
    summer_gap = historical_summer.mean() - recent_summer.mean()
    if summer_gap > 100:
        print(f"\n⚠️ Issue 2: SUMMER BURN STRUCTURAL SHIFT")
        print(f"   Historical summer avg (pre-2025): {historical_summer.mean():.0f} kbd")
        print(f"   Recent summer (2025): {recent_summer.mean():.0f} kbd")
        print(f"   Gap: {summer_gap:.0f} kbd ({summer_gap/historical_summer.mean()*100:.1f}%)")
        print(f"   → Lag features pulling predictions toward outdated higher values")

# Issue 3: Check recent extreme heat vs burn relationship
recent_data = burn_df.tail(12)
print(f"\n⚠️ Issue 3: RECENT MONTHS BEHAVIOR")
print(f"   Recent 12-month average: {recent_data['burn_kbd'].mean():.0f} kbd")
print(f"   Recent max burn: {recent_data['burn_kbd'].max():.0f} kbd")
print(f"   Recent min burn: {recent_data['burn_kbd'].min():.0f} kbd")

# ============================================================
# VISUALIZATIONS
# ============================================================

print("\n[4] GENERATING DIAGNOSTIC CHARTS")
print("-"*60)

fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# Chart 1: Time series with forecast errors
ax1 = axes[0, 0]
recent_plot = burn_df[burn_df['year'] >= 2020]
ax1.plot(recent_plot['date'], recent_plot['burn_kbd'], 'o-', linewidth=2, label='Actual', color='#2C3E50')

if len(forecast_log) > 0 and len(comparison) > 0:
    for _, row in comparison.iterrows():
        ax1.plot([row['Target Month'], row['Target Month']], 
                [row['burn_kbd'], row['Forecast (kbd)']], 
                'r--', alpha=0.5, linewidth=1)
    ax1.scatter(comparison['Target Month'], comparison['Forecast (kbd)'], 
               color='red', s=100, alpha=0.6, label='Forecast', zorder=5)

ax1.set_title('Actual vs Forecasts (2020+)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Burn (kbd)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Chart 2: Error distribution
if len(forecast_log) > 0 and len(comparison) > 0:
    ax2 = axes[0, 1]
    ax2.hist(comparison['error'], bins=15, edgecolor='black', alpha=0.7)
    ax2.axvline(comparison['error'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {comparison["error"].mean():.0f}')
    ax2.axvline(0, color='green', linestyle='-', linewidth=2, label='Perfect')
    ax2.set_title('Forecast Error Distribution', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Error (kbd)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

# Chart 3: Yearly trends
ax3 = axes[1, 0]
yearly_avg = burn_df[burn_df['year'] >= 2015].groupby('year')['burn_kbd'].mean()
ax3.bar(yearly_avg.index, yearly_avg.values, alpha=0.7, edgecolor='black')
ax3.set_title('Annual Average Burn Trend', fontsize=12, fontweight='bold')
ax3.set_xlabel('Year')
ax3.set_ylabel('Average Burn (kbd)')
ax3.grid(True, alpha=0.3, axis='y')

# Chart 4: Monthly pattern
ax4 = axes[1, 1]
monthly_recent = burn_df[burn_df['year'] >= 2022].groupby('month')['burn_kbd'].mean()
monthly_historical = burn_df[burn_df['year'] < 2022].groupby('month')['burn_kbd'].mean()
months = range(1, 13)
ax4.plot(months, monthly_historical.reindex(months), 'o-', label='Historical (pre-2022)', linewidth=2)
ax4.plot(months, monthly_recent.reindex(months), 's-', label='Recent (2022+)', linewidth=2)
ax4.set_title('Seasonal Pattern: Historical vs Recent', fontsize=12, fontweight='bold')
ax4.set_xlabel('Month')
ax4.set_ylabel('Average Burn (kbd)')
ax4.set_xticks(months)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
chart_path = f"{WORKING_DIR}/model_diagnostic_analysis.png"
plt.savefig(chart_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Diagnostic charts saved to: {chart_path}")
plt.close()

# ============================================================
# RECOMMENDATIONS
# ============================================================

print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("\nBased on diagnostic analysis, here are potential fixes:\n")

if len(forecast_log) > 0 and len(comparison) > 0:
    if comparison['error'].mean() > 50:
        print("1. MODEL BIAS CORRECTION")
        print("   → Retrain model with updated data through 2025")
        print("   → Consider reducing weight on lag features (they anchor to old highs)")
        print("   → Add year-based trend to capture declining burn pattern")
        
        print("\n2. FEATURE ENGINEERING")
        print("   → Remove or reduce importance of historical lag features")
        print("   → Add 'recent_trend' feature based on last 6 months only")
        print("   → Increase weight on current weather vs historical patterns")
        
        print("\n3. STRUCTURAL BREAK HANDLING")
        print("   → Train separate model on post-2022 data only")
        print("   → Or add 'regime_change' dummy variable for 2023+")
        print("   → Consider weighted training (recent data weighted higher)")

print("\n4. IMMEDIATE TACTICAL FIX")
print("   → Apply bias correction: Subtract mean error from predictions")
bias_correction = comparison['error'].mean() if len(comparison) > 0 and len(forecast_log) > 0 else 0
if abs(bias_correction) > 20:
    print(f"   → Current bias: {bias_correction:+.0f} kbd")
    print(f"   → Quick fix: prediction - {bias_correction:.0f}")

print("\n" + "="*80)
print("Next step: Review diagnostic chart and decide on retraining strategy")
print("="*80)