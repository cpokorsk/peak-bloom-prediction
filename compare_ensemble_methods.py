"""
Compare Different Ensemble Methods for Peak Bloom Prediction

This script demonstrates various ensemble techniques and compares their performance.
Run this after executing the model_comparison.ipynb notebook.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup
sns.set_style('whitegrid')
model_outputs_dir = Path('data/model_outputs')

def doy_to_date(doy):
    """Convert day of year to calendar date"""
    if pd.isna(doy):
        return None
    return (pd.to_datetime('2026-01-01') + pd.Timedelta(days=int(doy)-1)).strftime('%b %d')

# Load all predictions
print("Loading predictions...")
linear_ols = pd.read_csv(model_outputs_dir / 'final_2026_predictions.csv')
ridge_lasso = pd.read_csv(model_outputs_dir / 'final_2026_predictions_ridge_lasso.csv')
bayesian = pd.read_csv(model_outputs_dir / 'final_2026_predictions_bayesian_ridge.csv')
arimax_pred = pd.read_csv(model_outputs_dir / 'final_2026_arimax_predictions.csv')
dts_pred = pd.read_csv(model_outputs_dir / 'final_2026_dts_predictions.csv')
process_pred = pd.read_csv(model_outputs_dir / 'final_2026_process_based_predictions.csv')

# Create base comparison dataframe
comparison = pd.DataFrame()
comparison['location'] = linear_ols['location']
comparison['Linear_OLS'] = linear_ols['predicted_doy'].round(1)
comparison['Ridge'] = ridge_lasso['predicted_doy_ridge'].round(1)
comparison['Lasso'] = ridge_lasso['predicted_doy_lasso'].round(1)
comparison['Bayesian_Ridge'] = bayesian['predicted_doy'].round(1)
comparison['ARIMAX'] = arimax_pred['predicted_doy'].round(1)

dts_col = 'predicted_doy' if 'predicted_doy' in dts_pred.columns else 'predicted_bloom_doy'
process_col = 'predicted_doy' if 'predicted_doy' in process_pred.columns else 'predicted_bloom_doy'
comparison['DTS'] = dts_pred[dts_col].round(1)
comparison['Process_Based'] = process_pred[process_col].round(1)

pred_cols = ['Linear_OLS', 'Ridge', 'Lasso', 'Bayesian_Ridge', 'ARIMAX', 'DTS', 'Process_Based']

# Holdout MAE values (from notebook results)
holdout_mae = {
    'Lasso': 6.40,
    'Bayesian Ridge': 6.60,
    'Ridge': 6.63,
    'ARIMAX': 7.21,
    'Linear OLS': 7.25,
    'Process_Based': 9.29,
    'DTS': 11.60
}

# Map to column names
model_col_map = {
    'Lasso': 'Lasso',
    'Bayesian Ridge': 'Bayesian_Ridge',
    'Ridge': 'Ridge',
    'Linear OLS': 'Linear_OLS',
    'ARIMAX': 'ARIMAX',
    'DTS': 'DTS',
    'Process_Based': 'Process_Based'
}

print("\n" + "="*70)
print("ENSEMBLE METHODS COMPARISON")
print("="*70)

# ============================================================================
# METHOD 1: Simple Average (Equal Weights)
# ============================================================================
print("\n1. SIMPLE AVERAGE (Equal Weights)")
print("-" * 70)

simple_avg = comparison[pred_cols].mean(axis=1, skipna=True).round(1)
print("All models weighted equally (1/7 = 14.3% each)")
print("\nPredictions:")
for loc, pred in zip(comparison['location'], simple_avg):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# METHOD 2: Weighted Average (Inverse MAE)
# ============================================================================
print("\n\n2. WEIGHTED AVERAGE (Inverse MAE) - RECOMMENDED ✓")
print("-" * 70)

# Get top 3 models
sorted_models = sorted(holdout_mae.items(), key=lambda x: x[1])[:3]
top3_models = [m[0] for m in sorted_models]

# Calculate weights
weights = {model: 1/mae for model, mae in sorted_models}
total_weight = sum(weights.values())
weights = {model: w/total_weight for model, w in weights.items()}

print(f"Using top 3 models:")
for model, weight in weights.items():
    print(f"  {model:20s}: {weight:6.1%} (MAE: {holdout_mae[model]:.2f} days)")

# Calculate weighted predictions
weighted_avg = pd.Series(0.0, index=comparison.index)
for model, weight in weights.items():
    col = model_col_map[model]
    weighted_avg += comparison[col].fillna(comparison[pred_cols].mean(axis=1)) * weight

weighted_avg = weighted_avg.round(1)
print("\nPredictions:")
for loc, pred in zip(comparison['location'], weighted_avg):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# METHOD 3: Median Ensemble
# ============================================================================
print("\n\n3. MEDIAN ENSEMBLE (Robust to Outliers)")
print("-" * 70)

median_pred = comparison[pred_cols].median(axis=1, skipna=True).round(1)
print("Uses median of all predictions (middle value)")
print("\nPredictions:")
for loc, pred in zip(comparison['location'], median_pred):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# METHOD 4: Trimmed Mean
# ============================================================================
print("\n\n4. TRIMMED MEAN (Remove Extremes)")
print("-" * 70)

def trimmed_mean(row):
    valid = row.dropna()
    if len(valid) <= 2:
        return valid.mean()
    sorted_vals = valid.sort_values()
    trimmed = sorted_vals.iloc[1:-1]
    return trimmed.mean()

trimmed_avg = comparison[pred_cols].apply(trimmed_mean, axis=1).round(1)
print("Removes highest and lowest prediction, then averages")
print("\nPredictions:")
for loc, pred in zip(comparison['location'], trimmed_avg):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# METHOD 5: Best Model Only (No Ensemble)
# ============================================================================
print("\n\n5. BEST MODEL ONLY (No Ensemble - for comparison)")
print("-" * 70)

best_model = sorted_models[0][0]
best_col = model_col_map[best_model]
best_only = comparison[best_col].round(1)
print(f"Using only: {best_model} (MAE: {holdout_mae[best_model]:.2f} days)")
print("\nPredictions:")
for loc, pred in zip(comparison['location'], best_only):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# METHOD 6: Top-3 Simple Average
# ============================================================================
print("\n\n6. TOP-3 SIMPLE AVERAGE")
print("-" * 70)

top3_cols = [model_col_map[m] for m in top3_models]
top3_avg = comparison[top3_cols].mean(axis=1, skipna=True).round(1)
print(f"Simple average of top 3 models: {', '.join(top3_models)}")
print("\nPredictions:")
for loc, pred in zip(comparison['location'], top3_avg):
    print(f"  {loc:15s}: {doy_to_date(pred)} (DOY {pred})")

# ============================================================================
# COMPARISON TABLE
# ============================================================================
print("\n\n" + "="*70)
print("ENSEMBLE METHODS SIDE-BY-SIDE COMPARISON")
print("="*70)

ensemble_comparison = pd.DataFrame({
    'Location': comparison['location'],
    'Simple_Avg': simple_avg,
    'Weighted_Avg': weighted_avg,
    'Median': median_pred,
    'Trimmed_Mean': trimmed_avg,
    'Best_Only': best_only,
    'Top3_Avg': top3_avg
})

# Calculate agreement statistics
ensemble_cols = ['Simple_Avg', 'Weighted_Avg', 'Median', 'Trimmed_Mean', 'Best_Only', 'Top3_Avg']
ensemble_comparison['Mean'] = ensemble_comparison[ensemble_cols].mean(axis=1).round(1)
ensemble_comparison['Std'] = ensemble_comparison[ensemble_cols].std(axis=1).round(1)
ensemble_comparison['Range'] = (
    ensemble_comparison[ensemble_cols].max(axis=1) - 
    ensemble_comparison[ensemble_cols].min(axis=1)
).round(1)

print("\nDay of Year (DOY) Predictions:")
print(ensemble_comparison.to_string(index=False))

# Convert to dates
date_comparison = ensemble_comparison.copy()
for col in ensemble_cols:
    date_comparison[col] = date_comparison[col].apply(doy_to_date)
date_comparison['Mean'] = date_comparison['Mean'].apply(doy_to_date)

print("\n\nCalendar Date Predictions:")
print(date_comparison[['Location'] + ensemble_cols].to_string(index=False))

# ============================================================================
# AGREEMENT ANALYSIS
# ============================================================================
print("\n\n" + "="*70)
print("ENSEMBLE METHOD AGREEMENT ANALYSIS")
print("="*70)

agreement_stats = ensemble_comparison[['Location', 'Mean', 'Std', 'Range']].copy()
agreement_stats['Agreement_Level'] = pd.cut(
    agreement_stats['Range'],
    bins=[0, 3, 7, 100],
    labels=['High', 'Medium', 'Low']
)

print("\nAgreement Statistics:")
print(agreement_stats.to_string(index=False))

print("\nInterpretation:")
print("  High Agreement (Range ≤ 3 days)  : All methods very consistent")
print("  Medium Agreement (3-7 days)      : Some variation across methods")
print("  Low Agreement (>7 days)          : Significant disagreement")

avg_range = agreement_stats['Range'].mean()
max_range = agreement_stats['Range'].max()

print(f"\nOverall Statistics:")
print(f"  Average range: {avg_range:.1f} days")
print(f"  Maximum range: {max_range:.1f} days")

# ============================================================================
# RECOMMENDATION
# ============================================================================
print("\n\n" + "="*70)
print("RECOMMENDATION")
print("="*70)

if avg_range < 3:
    consensus_level = "HIGH"
    recommendation = "Any method will work well. Use Weighted Average for optimal performance."
elif avg_range < 7:
    consensus_level = "MODERATE"
    recommendation = "Use Weighted Average or Median for balance of accuracy and robustness."
else:
    consensus_level = "LOW"
    recommendation = "Report range of predictions. Consider location-specific model selection."

print(f"\nConsensus Level: {consensus_level} (avg range: {avg_range:.1f} days)")
print(f"Recommendation: {recommendation}")

print("\n✓ SELECTED METHOD: Weighted Average (Inverse MAE)")
print("\nRationale:")
print("  1. Theoretically optimal (minimizes expected error)")
print("  2. Gives appropriate weight to best models")
print("  3. Still includes diversity from multiple model types")
print("  4. Validated on holdout data")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n\nSaving results...")

# Save comprehensive ensemble comparison
ensemble_comparison.to_csv(
    model_outputs_dir / 'ensemble_methods_comparison.csv',
    index=False
)
print("✓ Saved: data/model_outputs/ensemble_methods_comparison.csv")

# Save final recommendations (weighted average)
final_recommendations = pd.DataFrame({
    'location': comparison['location'],
    'predicted_doy': weighted_avg,
    'predicted_date': weighted_avg.apply(doy_to_date),
    'ensemble_method': 'Weighted Average (Top 3)',
    'top_models': ', '.join(top3_models),
    'agreement_range_days': agreement_stats['Range'],
    'agreement_level': agreement_stats['Agreement_Level']
})

# Add individual model predictions for reference
for col in pred_cols:
    final_recommendations[f'model_{col}'] = comparison[col]

final_recommendations.to_csv(
    model_outputs_dir / 'final_2026_ensemble_recommendations.csv',
    index=False
)
print("✓ Saved: data/model_outputs/final_2026_ensemble_recommendations.csv")

# ============================================================================
# VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")

# Plot 1: Ensemble methods comparison
fig, axes = plt.subplots(1, 5, figsize=(18, 5), sharey=True)

for idx, location in enumerate(ensemble_comparison['Location']):
    ax = axes[idx]
    loc_data = ensemble_comparison[ensemble_comparison['Location'] == location][ensemble_cols].iloc[0]
    
    colors = ['#3498db'] * len(ensemble_cols)
    colors[1] = '#2ecc71'  # Highlight weighted average
    
    bars = ax.bar(range(len(ensemble_cols)), loc_data, color=colors, alpha=0.8,
                  edgecolor='black', linewidth=1.2)
    
    for bar, val in zip(bars, loc_data):
        if not pd.isna(val):
            ax.text(bar.get_x() + bar.get_width()/2, val + 1,
                   doy_to_date(val), ha='center', fontsize=7, rotation=45)
    
    ax.set_title(location, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(ensemble_cols)))
    ax.set_xticklabels([c.replace('_', '\n') for c in ensemble_cols],
                        rotation=45, ha='right', fontsize=7)
    ax.grid(axis='y', alpha=0.3)
    
    if idx == 0:
        ax.set_ylabel('Predicted DOY', fontsize=11, fontweight='bold')

fig.suptitle('Ensemble Methods Comparison by Location\n(Green = Recommended Weighted Average)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(model_outputs_dir / 'ensemble_methods_visualization.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: data/model_outputs/ensemble_methods_visualization.png")

# Plot 2: Agreement heatmap
fig, ax = plt.subplots(figsize=(10, 6))
agreement_matrix = ensemble_comparison[ensemble_cols].T
agreement_matrix.columns = ensemble_comparison['Location']

sns.heatmap(agreement_matrix, annot=True, fmt='.1f', cmap='RdYlGn_r',
            cbar_kws={'label': 'Predicted DOY'}, ax=ax,
            linewidths=0.5, linecolor='black')

ax.set_xlabel('Location', fontsize=12, fontweight='bold')
ax.set_ylabel('Ensemble Method', fontsize=12, fontweight='bold')
ax.set_title('Ensemble Method Predictions Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(model_outputs_dir / 'ensemble_agreement_heatmap.png',
            dpi=300, bbox_inches='tight')
print("✓ Saved: data/model_outputs/ensemble_agreement_heatmap.png")

plt.close('all')

print("\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print("\nFinal 2026 Peak Bloom Predictions (Weighted Ensemble):")
for _, row in final_recommendations.iterrows():
    print(f"  {row['location']:15s}: {row['predicted_date']} "
          f"(±{row['agreement_range_days']:.1f} day range, {row['agreement_level']} consensus)")

print("\n📊 All results saved to data/model_outputs/")
