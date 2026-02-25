import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.metrics import mean_absolute_error, r2_score
from phenology_config import (
    MODEL_FEATURES_FILE,
    HOLDOUT_LOCATIONS,
    HOLDOUT_EXTRA_COUNTRIES,
    HOLDOUT_PER_COUNTRY,
    HOLDOUT_RANDOM_SEED,
    MIN_MODEL_YEAR,
    TARGET_PREDICTION_LOCATIONS,
    normalize_location,
)

def main():
    print("--- Loading Data ---")
    features_df = pd.read_csv(MODEL_FEATURES_FILE)
    features_df['location'] = features_df['location'].apply(normalize_location)
    if 'is_future' not in features_df.columns:
        features_df['is_future'] = False

    required_predictors = [
        'max_tmax_early_spring',
        'total_prcp_early_spring',
        'species',
        'continent'
    ]

    df = features_df[(features_df['is_future'] == False) & (features_df['year'] >= MIN_MODEL_YEAR)].copy()
    df = df.dropna(subset=['bloom_doy'] + required_predictors)
    
    print("\n--- Splitting Data ---")
    # Replicate the same split logic as 4_lm_train_and_predict.py
    base_holdout = set(HOLDOUT_LOCATIONS)
    rng = np.random.default_rng(HOLDOUT_RANDOM_SEED)

    location_meta = (
        df[['location', 'country_code']]
        .dropna(subset=['location'])
        .drop_duplicates(subset=['location'])
        .copy()
    )

    extra_holdout = set()
    for country in HOLDOUT_EXTRA_COUNTRIES:
        candidates = location_meta[location_meta['country_code'] == country]['location']
        candidates = [loc for loc in candidates if loc not in base_holdout and loc not in TARGET_PREDICTION_LOCATIONS]
        if candidates:
            pick_count = min(HOLDOUT_PER_COUNTRY, len(candidates))
            extra_holdout.update(rng.choice(candidates, size=pick_count, replace=False).tolist())

    holdout_locations = base_holdout.union(extra_holdout)
    holdout_mask = df['location'].isin(holdout_locations)
    df_holdout = df[holdout_mask].copy()
    df_main = df[~holdout_mask].copy()
    
    # Split main pool by time
    years = sorted(df_main['year'].dropna().unique().tolist())
    train_cut = int(len(years) * 0.70)
    val_cut = int(len(years) * 0.85)

    train_years = set(years[:train_cut])
    val_years = set(years[train_cut:val_cut])
    test_years = set(years[val_cut:])

    train = df_main[df_main['year'].isin(train_years)].copy()
    test_main = df_main[df_main['year'].isin(test_years)].copy()
    
    print(f"Training set: {len(train)} records")
    print(f"Main Test set: {len(test_main)} records")
    print(f"Holdout Test set: {len(df_holdout)} records")

    print("\n--- Training Model ---")
    formula = "bloom_doy ~ observed_gdd_to_bloom + chill_days_oct1_dec31 + total_prcp_early_spring + C(species)"
    model = smf.ols(formula=formula, data=train).fit()
    
    # Make predictions on test and holdout
    test_main['predicted_doy'] = model.predict(test_main)
    test_main['dataset'] = 'Test'
    
    df_holdout['predicted_doy'] = model.predict(df_holdout)
    df_holdout['dataset'] = 'Holdout'
    
    # Combine test and holdout for plotting
    combined = pd.concat([test_main, df_holdout], ignore_index=True)
    combined['observed_doy'] = combined['bloom_doy']
    combined['abs_error_days'] = np.abs(combined['observed_doy'] - combined['predicted_doy'])
    
    # Calculate overall metrics
    mae = mean_absolute_error(combined['observed_doy'], combined['predicted_doy'])
    rmse = np.sqrt(((combined['observed_doy'] - combined['predicted_doy'])**2).mean())
    r2 = r2_score(combined['observed_doy'], combined['predicted_doy'])
    
    # Create visualization - time series with subplots by location
    print("\n--- Creating Visualization ---")
    
    # Filter for target prediction locations only
    plot_locs = [loc for loc in TARGET_PREDICTION_LOCATIONS if loc in combined['location'].values]
    
    # Calculate subplot grid
    n_locs = len(plot_locs)
    n_cols = 3
    n_rows = int(np.ceil(n_locs / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten() if n_locs > 1 else [axes]
    
    for idx, loc in enumerate(plot_locs):
        ax = axes[idx]
        loc_data = combined[combined['location'] == loc].sort_values('year')
        
        # Plot observed and predicted
        ax.plot(loc_data['year'], loc_data['observed_doy'], 
                marker='o', linestyle='-', linewidth=2, markersize=6,
                color='steelblue', label='Observed', alpha=0.8)
        ax.plot(loc_data['year'], loc_data['predicted_doy'], 
                marker='s', linestyle='--', linewidth=2, markersize=6,
                color='orange', label='Predicted', alpha=0.8)
        
        # Calculate location-specific MAE
        loc_mae = loc_data['abs_error_days'].mean()
        dataset_label = loc_data['dataset'].iloc[0]
        
        # Title with location and metrics
        ax.set_title(f"{loc}\n({dataset_label}, MAE: {loc_mae:.1f}d)", 
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Year', fontsize=8)
        ax.set_ylabel('Bloom DOY', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        
        # Add legend only to first subplot
        if idx == 0:
            ax.legend(loc='best', fontsize=7, framealpha=0.9)
    
    # Remove empty subplots
    for idx in range(n_locs, len(axes)):
        fig.delaxes(axes[idx])
    
    # Overall title with metrics
    fig.suptitle(f'Linear Model: Observed vs Predicted Over Time (Test + Holdout)\n' +
                 f'Overall MAE: {mae:.2f} days | RMSE: {rmse:.2f} days | R²: {r2:.3f} | n = {len(combined)}',
                 fontsize=14, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save figure
    output_path = 'data/model_outputs/lm_test_holdout_timeseries.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f'Plot saved to: {output_path}')
    
    # Print detailed results
    print("\n--- Linear Model Test + Holdout Results ---")
    print(f"Total predictions: {len(combined)}")
    print(f"MAE: {mae:.2f} days")
    print(f"RMSE: {rmse:.2f} days")
    print(f"R²: {r2:.3f}")
    
    print("\nBy dataset:")
    print(combined.groupby('dataset')['abs_error_days'].agg(['count', 'mean', 'std']).round(2))
    
    print("\nTarget locations only:")
    target_data = combined[combined['location'].isin(TARGET_PREDICTION_LOCATIONS)]
    if not target_data.empty:
        print(target_data.groupby('location')['abs_error_days'].agg(['count', 'mean', 'std']).round(2))
        target_mae = target_data['abs_error_days'].mean()
        print(f"\nTarget locations MAE: {target_mae:.2f} days")
    
    print("\nBy location and dataset (target locations):")
    if not target_data.empty:
        pivot = target_data.groupby(['location', 'dataset'])['abs_error_days'].mean().unstack(fill_value=np.nan).round(2)
        print(pivot)

if __name__ == "__main__":
    main()
