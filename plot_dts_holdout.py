import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score


lm = pd.read_csv('data/model_outputs/holdout/holdout_last10y_linear_ols.csv')
br = pd.read_csv('data/model_outputs/holdout/holdout_last10y_bayesian_ridge.csv')
ar = pd.read_csv('data/model_outputs/holdout/holdout_last10y_arimax.csv')

holdout_df = lm[['location', 'year', 'actual_bloom_doy', 'predicted_doy']].rename(
    columns={'predicted_doy': 'pred_linear_ols'}
)
holdout_df = holdout_df.merge(
    br[['location', 'year', 'predicted_doy']].rename(columns={'predicted_doy': 'pred_bayesian_ridge'}),
    on=['location', 'year'],
    how='inner',
)
holdout_df = holdout_df.merge(
    ar[['location', 'year', 'predicted_doy']].rename(columns={'predicted_doy': 'pred_arimax'}),
    on=['location', 'year'],
    how='inner',
)

meta_features = ['pred_linear_ols', 'pred_bayesian_ridge', 'pred_arimax']
ok_df = holdout_df.dropna(subset=['actual_bloom_doy'] + meta_features).copy()

if ok_df.empty:
    print('No ensemble holdout rows available to plot.')
    raise SystemExit(0)

X_meta = ok_df[meta_features].values
y_meta = ok_df['actual_bloom_doy'].values
meta_model = RidgeCV(alphas=np.logspace(-3, 3, 25), cv=5)
meta_model.fit(X_meta, y_meta)
ok_df['expected_doy'] = meta_model.predict(X_meta)

ok_df['observed_doy'] = ok_df['actual_bloom_doy']
ok_df['abs_error_days'] = (ok_df['observed_doy'] - ok_df['expected_doy']).abs()

mae = mean_absolute_error(ok_df['observed_doy'], ok_df['expected_doy'])
r2 = r2_score(ok_df['observed_doy'], ok_df['expected_doy'])
rmse = np.sqrt(((ok_df['observed_doy'] - ok_df['expected_doy']) ** 2).mean())

fig, ax = plt.subplots(figsize=(10, 8))

locations = ok_df['location'].unique()
colors = plt.cm.Set2(np.linspace(0, 1, len(locations)))

for loc, color in zip(locations, colors):
    loc_data = ok_df[ok_df['location'] == loc]
    ax.scatter(
        loc_data['observed_doy'],
        loc_data['expected_doy'],
        label=loc,
        s=100,
        alpha=0.7,
        color=color,
        edgecolors='black',
        linewidth=1,
    )

min_doy = min(ok_df['observed_doy'].min(), ok_df['expected_doy'].min()) - 5
max_doy = max(ok_df['observed_doy'].max(), ok_df['expected_doy'].max()) + 5
ax.plot(
    [min_doy, max_doy],
    [min_doy, max_doy],
    'k--',
    linewidth=2,
    label='Perfect Agreement',
    alpha=0.5,
)

ax.fill_between(
    [min_doy, max_doy],
    [min_doy - 7, max_doy - 7],
    [min_doy + 7, max_doy + 7],
    alpha=0.1,
    color='gray',
    label='±7 day tolerance',
)

ax.set_xlabel('Observed Bloom Date (Day of Year)', fontsize=14, fontweight='bold')
ax.set_ylabel('Expected Bloom Date (Day of Year)', fontsize=14, fontweight='bold')
ax.set_title(
    'Stacked Ensemble Model: Observed vs Expected\nLast 10-Year Holdout Evaluation',
    fontsize=16,
    fontweight='bold',
    pad=20,
)

metrics_text = f'MAE: {mae:.2f} days\nRMSE: {rmse:.2f} days\nR²: {r2:.3f}\nn = {len(ok_df)}'
ax.text(
    0.05,
    0.95,
    metrics_text,
    transform=ax.transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
)

ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
ax.set_aspect('equal', adjustable='box')

plt.tight_layout()

output_path = 'data/model_outputs/stacked_ensemble_holdout_observed_vs_expected.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f'Plot saved to: {output_path}')

print('\n--- Stacked Ensemble Holdout Results ---')
print(f'Total predictions: {len(ok_df)}')
print(f'MAE: {mae:.2f} days')
print(f'RMSE: {rmse:.2f} days')
print(f'R²: {r2:.3f}')
print('\nBy location:')
print(ok_df.groupby('location')['abs_error_days'].agg(['count', 'mean', 'std']).round(2))
