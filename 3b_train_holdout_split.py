"""
Test different train/holdout splitting strategies for time series phenology data.

This script compares various splitting methods to help select the optimal strategy
for holdout validation in cherry blossom bloom prediction.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from phenology_config import (
    MODEL_FEATURES_FILE,
    MODEL_OUTPUT_DIR,
    HOLDOUT_LAST_N_YEARS,
    HOLDOUT_RANDOM_SEED,
)


OUTPUT_SUMMARY = os.path.join(MODEL_OUTPUT_DIR, "split_method_comparison_summary.csv")
OUTPUT_PLOT = os.path.join(MODEL_OUTPUT_DIR, "split_method_comparison.png")


def load_features():
    """Load the feature-engineered dataset."""
    if not os.path.exists(MODEL_FEATURES_FILE):
        raise FileNotFoundError(
            f"Model features file not found: {MODEL_FEATURES_FILE}\n"
            "Run 3_feature_engineering.py first to generate features."
        )
    
    df = pd.read_csv(MODEL_FEATURES_FILE)
    print(f"Loaded {len(df)} observations from {MODEL_FEATURES_FILE}")
    print(f"Years: {df['year'].min()} - {df['year'].max()}")
    print(f"Locations: {df['location'].nunique()} unique")
    print(f"Features: {df.shape[1]} columns")
    return df


def get_feature_columns(df):
    """Extract numeric feature columns (exclude metadata and target)."""
    exclude_cols = {'location', 'year', 'bloom_doy', 'bloom_date', 'species', 
                    'lat', 'long', 'alt', 'country', 'continent', 'country_code'}
    
    # Only use numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    return feature_cols


def evaluate_split(y_train, y_pred_train, y_test, y_pred_test, split_name):
    """Calculate metrics for a given split."""
    train_mae = mean_absolute_error(y_train, y_pred_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    train_r2 = r2_score(y_train, y_pred_train)
    
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n{split_name}:")
    print(f"  Train: MAE={train_mae:.2f}, RMSE={train_rmse:.2f}, R²={train_r2:.3f}")
    print(f"  Test:  MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R²={test_r2:.3f}")
    print(f"  Generalization gap (MAE): {test_mae - train_mae:.2f} days")
    
    return {
        'split_method': split_name,
        'train_n': len(y_train),
        'test_n': len(y_test),
        'train_mae': round(train_mae, 3),
        'train_rmse': round(train_rmse, 3),
        'train_r2': round(train_r2, 3),
        'test_mae': round(test_mae, 3),
        'test_rmse': round(test_rmse, 3),
        'test_r2': round(test_r2, 3),
        'generalization_gap_mae': round(test_mae - train_mae, 3),
    }


def split_last_n_years(df, n_years=None):
    """Current method: last N years as holdout."""
    if n_years is None:
        n_years = HOLDOUT_LAST_N_YEARS
    
    max_year = df['year'].max()
    cutoff_year = max_year - n_years + 1
    
    train_mask = df['year'] < cutoff_year
    test_mask = df['year'] >= cutoff_year
    
    print(f"\nLast {n_years} years split:")
    print(f"  Train: years < {cutoff_year} ({train_mask.sum()} samples)")
    print(f"  Test: years >= {cutoff_year} ({test_mask.sum()} samples)")
    
    return train_mask, test_mask


def split_percentage(df, test_pct=0.2, random_state=None):
    """Random percentage split (stratified by location to maintain balance)."""
    if random_state is None:
        random_state = HOLDOUT_RANDOM_SEED
    
    np.random.seed(random_state)
    
    train_indices = []
    test_indices = []
    
    for location in df['location'].unique():
        loc_indices = df[df['location'] == location].index.tolist()
        n_test = max(1, int(len(loc_indices) * test_pct))
        
        np.random.shuffle(loc_indices)
        test_indices.extend(loc_indices[:n_test])
        train_indices.extend(loc_indices[n_test:])
    
    train_mask = df.index.isin(train_indices)
    test_mask = df.index.isin(test_indices)
    
    print(f"\nRandom {int(test_pct*100)}% split (stratified by location):")
    print(f"  Train: {train_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    return train_mask, test_mask


def split_recent_vs_historical(df, recent_start_year=2010):
    """Split into historical vs recent data."""
    train_mask = df['year'] < recent_start_year
    test_mask = df['year'] >= recent_start_year
    
    print(f"\nHistorical vs Recent split (cutoff={recent_start_year}):")
    print(f"  Train: years < {recent_start_year} ({train_mask.sum()} samples)")
    print(f"  Test: years >= {recent_start_year} ({test_mask.sum()} samples)")
    
    return train_mask, test_mask


def split_leave_one_location_out(df, location):
    """Leave-one-location-out: train on all locations except one."""
    train_mask = df['location'] != location
    test_mask = df['location'] == location
    
    print(f"\nLeave-one-location-out ({location}):")
    print(f"  Train: all locations except {location} ({train_mask.sum()} samples)")
    print(f"  Test: {location} only ({test_mask.sum()} samples)")
    
    return train_mask, test_mask


def split_by_year_blocks(df, n_folds=5):
    """Split by contiguous year blocks for time series cross-validation."""
    years = sorted(df['year'].unique())
    n_years = len(years)
    fold_size = n_years // n_folds
    
    # Use middle fold as test
    test_fold_idx = n_folds // 2
    test_start = test_fold_idx * fold_size
    test_end = test_start + fold_size
    
    test_years = years[test_start:test_end]
    train_mask = ~df['year'].isin(test_years)
    test_mask = df['year'].isin(test_years)
    
    print(f"\nYear block split ({n_folds} folds, testing fold {test_fold_idx+1}):")
    print(f"  Train years: {[y for y in years if y not in test_years]}")
    print(f"  Test years: {test_years}")
    print(f"  Train: {train_mask.sum()} samples")
    print(f"  Test: {test_mask.sum()} samples")
    
    return train_mask, test_mask


def train_and_evaluate(df, train_mask, test_mask, split_name, feature_cols):
    """Train a simple model and evaluate on the split."""
    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    
    # Drop rows with missing values
    train_df = train_df.dropna(subset=feature_cols + ['bloom_doy'])
    test_df = test_df.dropna(subset=feature_cols + ['bloom_doy'])
    
    if len(train_df) == 0 or len(test_df) == 0:
        print(f"  WARNING: No valid data for {split_name}")
        return None
    
    X_train = train_df[feature_cols].values
    y_train = train_df['bloom_doy'].values
    X_test = test_df[feature_cols].values
    y_test = test_df['bloom_doy'].values
    
    # Simple linear regression for comparison
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    return evaluate_split(y_train, y_pred_train, y_test, y_pred_test, split_name)


def compare_all_splits(df):
    """Compare all splitting strategies."""
    print("=" * 80)
    print("TRAIN/HOLDOUT SPLITTING METHOD COMPARISON")
    print("=" * 80)
    
    feature_cols = get_feature_columns(df)
    print(f"\nUsing {len(feature_cols)} features: {feature_cols[:5]}...")
    
    results = []
    
    # 1. Last N years (current default)
    train_mask, test_mask = split_last_n_years(df)
    result = train_and_evaluate(df, train_mask, test_mask, f"Last {HOLDOUT_LAST_N_YEARS} years", feature_cols)
    if result:
        results.append(result)
    
    # 2. Last 10 years (faster iteration)
    train_mask, test_mask = split_last_n_years(df, n_years=10)
    result = train_and_evaluate(df, train_mask, test_mask, "Last 10 years", feature_cols)
    if result:
        results.append(result)
    
    # 3. Random 20% split
    train_mask, test_mask = split_percentage(df, test_pct=0.2)
    result = train_and_evaluate(df, train_mask, test_mask, "Random 20% (stratified)", feature_cols)
    if result:
        results.append(result)
    
    # 4. Recent vs Historical
    train_mask, test_mask = split_recent_vs_historical(df, recent_start_year=2010)
    result = train_and_evaluate(df, train_mask, test_mask, "Historical vs Recent (2010+)", feature_cols)
    if result:
        results.append(result)
    
    # 5. Year blocks
    train_mask, test_mask = split_by_year_blocks(df, n_folds=5)
    result = train_and_evaluate(df, train_mask, test_mask, "Year blocks (5-fold middle)", feature_cols)
    if result:
        results.append(result)
    
    # 6. Leave-one-location-out (example with one location)
    locations = df['location'].unique()
    if len(locations) >= 3:
        test_location = locations[0]
        train_mask, test_mask = split_leave_one_location_out(df, test_location)
        result = train_and_evaluate(df, train_mask, test_mask, f"LOLO ({test_location})", feature_cols)
        if result:
            results.append(result)
    
    return pd.DataFrame(results)


def plot_comparison(results_df):
    """Create comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: MAE comparison
    ax = axes[0, 0]
    x = np.arange(len(results_df))
    width = 0.35
    ax.bar(x - width/2, results_df['train_mae'], width, label='Train MAE', alpha=0.8)
    ax.bar(x + width/2, results_df['test_mae'], width, label='Test MAE', alpha=0.8)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('MAE (days)')
    ax.set_title('Mean Absolute Error by Split Method')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['split_method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 2: R² comparison
    ax = axes[0, 1]
    ax.bar(x - width/2, results_df['train_r2'], width, label='Train R²', alpha=0.8)
    ax.bar(x + width/2, results_df['test_r2'], width, label='Test R²', alpha=0.8)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('R²')
    ax.set_title('R² Score by Split Method')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['split_method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 3: Generalization gap
    ax = axes[1, 0]
    colors = ['green' if gap < 2 else 'orange' if gap < 4 else 'red' 
              for gap in results_df['generalization_gap_mae']]
    ax.bar(x, results_df['generalization_gap_mae'], color=colors, alpha=0.7)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Generalization Gap (days)')
    ax.set_title('Generalization Gap (Test MAE - Train MAE)')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['split_method'], rotation=45, ha='right')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Sample sizes
    ax = axes[1, 1]
    ax.bar(x - width/2, results_df['train_n'], width, label='Train samples', alpha=0.8)
    ax.bar(x + width/2, results_df['test_n'], width, label='Test samples', alpha=0.8)
    ax.set_xlabel('Split Method')
    ax.set_ylabel('Number of Samples')
    ax.set_title('Train/Test Split Sizes')
    ax.set_xticks(x)
    ax.set_xticklabels(results_df['split_method'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_PLOT, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved comparison plot: {OUTPUT_PLOT}")


def main():
    df = load_features()
    
    results_df = compare_all_splits(df)
    
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(results_df.to_string(index=False))
    
    # Save results
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    results_df.to_csv(OUTPUT_SUMMARY, index=False)
    print(f"\nSaved results: {OUTPUT_SUMMARY}")
    
    # Create visualization
    plot_comparison(results_df)
    
    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    best_test_mae = results_df.loc[results_df['test_mae'].idxmin()]
    best_r2 = results_df.loc[results_df['test_r2'].idxmax()]
    smallest_gap = results_df.loc[results_df['generalization_gap_mae'].abs().idxmin()]
    
    print(f"\nBest test MAE: {best_test_mae['split_method']} ({best_test_mae['test_mae']:.2f} days)")
    print(f"Best test R²: {best_r2['split_method']} (R²={best_r2['test_r2']:.3f})")
    print(f"Smallest generalization gap: {smallest_gap['split_method']} ({smallest_gap['generalization_gap_mae']:.2f} days)")
    
    print("\nConsiderations:")
    print("- Time series data: prefer temporal splits (last N years, year blocks)")
    print("- Spatial generalization: test leave-one-location-out")
    print("- Small generalization gap suggests good model stability")
    print("- Larger holdout = more reliable test metrics but less training data")


if __name__ == "__main__":
    main()
