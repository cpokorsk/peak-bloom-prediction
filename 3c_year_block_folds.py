"""
Generate year-block folds for time series cross-validation.

This script divides the available years into K contiguous blocks (folds) for
cross-validation. Each fold can be used as a holdout set while training on the
remaining folds, enabling robust temporal cross-validation.
"""

import os
import numpy as np
import pandas as pd

from phenology_config import (
    MODEL_FEATURES_FILE,
    MODEL_INPUT_DIR,
    MODEL_OUTPUT_DIR,
)


# Configuration
N_FOLDS = 5
OUTPUT_FOLDS_FILE = os.path.join(MODEL_INPUT_DIR, "year_block_folds.csv")
OUTPUT_CONFIG_FILE = os.path.join(MODEL_INPUT_DIR, "cv_fold_config.csv")


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
    return df


def create_year_block_folds(df, n_folds=N_FOLDS):
    """Create contiguous year-block folds for cross-validation."""
    years = sorted(df['year'].unique())
    n_years = len(years)
    
    print(f"\nCreating {n_folds} year-block folds from {n_years} unique years")
    print(f"Year range: {years[0]} - {years[-1]}")
    
    # Calculate fold boundaries
    fold_size = n_years // n_folds
    remainder = n_years % n_folds
    
    year_folds = []
    start_idx = 0
    
    for fold_idx in range(n_folds):
        # Distribute remainder years across first folds
        current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        end_idx = start_idx + current_fold_size
        
        fold_years = years[start_idx:end_idx]
        
        for year in fold_years:
            year_folds.append({
                'year': year,
                'fold': fold_idx + 1,  # 1-indexed for readability
                'fold_name': f"fold_{fold_idx + 1}",
            })
        
        print(f"  Fold {fold_idx + 1}: years {fold_years[0]}-{fold_years[-1]} ({len(fold_years)} years)")
        start_idx = end_idx
    
    folds_df = pd.DataFrame(year_folds)
    return folds_df


def create_cv_configurations(folds_df, n_folds=N_FOLDS):
    """Create CV configuration specifying train/test splits for each fold."""
    configs = []
    
    for test_fold in range(1, n_folds + 1):
        train_folds = [f for f in range(1, n_folds + 1) if f != test_fold]
        
        test_years = folds_df[folds_df['fold'] == test_fold]['year'].tolist()
        train_years = folds_df[folds_df['fold'].isin(train_folds)]['year'].tolist()
        
        configs.append({
            'cv_split': test_fold,
            'test_fold': test_fold,
            'train_folds': ','.join(map(str, train_folds)),
            'n_train_years': len(train_years),
            'n_test_years': len(test_years),
            'train_year_min': min(train_years) if train_years else None,
            'train_year_max': max(train_years) if train_years else None,
            'test_year_min': min(test_years) if test_years else None,
            'test_year_max': max(test_years) if test_years else None,
        })
    
    config_df = pd.DataFrame(configs)
    return config_df


def summarize_folds(folds_df, config_df):
    """Print summary of fold configuration."""
    print("\n" + "=" * 80)
    print("YEAR BLOCK FOLD SUMMARY")
    print("=" * 80)
    
    print("\nFold Assignments:")
    print(folds_df.groupby('fold').agg({
        'year': ['min', 'max', 'count']
    }).to_string())
    
    print("\n\nCross-Validation Configurations:")
    print(config_df.to_string(index=False))
    
    print("\n\nUsage Notes:")
    print("- Each CV split uses one fold as test, remaining folds as train")
    print("- Maintains temporal ordering (no data leakage)")
    print("- Stage 4 models can iterate through all CV splits for full evaluation")
    print("- Or use a single split (e.g., middle fold) for production holdout")


def main():
    print("=" * 80)
    print("GENERATE YEAR BLOCK FOLDS FOR CROSS-VALIDATION")
    print("=" * 80)
    
    # Load data
    df = load_features()
    
    # Create folds
    folds_df = create_year_block_folds(df, n_folds=N_FOLDS)
    
    # Create CV configurations
    config_df = create_cv_configurations(folds_df, n_folds=N_FOLDS)
    
    # Save outputs
    os.makedirs(MODEL_INPUT_DIR, exist_ok=True)
    folds_df.to_csv(OUTPUT_FOLDS_FILE, index=False)
    print(f"\nSaved fold assignments: {OUTPUT_FOLDS_FILE}")
    
    config_df.to_csv(OUTPUT_CONFIG_FILE, index=False)
    print(f"Saved CV configurations: {OUTPUT_CONFIG_FILE}")
    
    # Summary
    summarize_folds(folds_df, config_df)
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Stage 4 models can now load year_block_folds.csv")
    print("2. For each CV split in cv_fold_config.csv:")
    print("   - Filter training data to train_folds years")
    print("   - Filter test data to test_fold years")
    print("   - Train model and evaluate on holdout")
    print("3. Aggregate metrics across all CV splits for robust evaluation")


if __name__ == "__main__":
    main()
