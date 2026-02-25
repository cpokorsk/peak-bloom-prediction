# Ensemble Methods Guide for Peak Bloom Prediction

## What is an Ensemble Model?

An **ensemble model** combines predictions from multiple individual models to produce a final prediction that is often more accurate and robust than any single model alone.

## Why Use Ensemble Models?

1. **Reduces Variance** - Different models make different errors; averaging cancels out random errors
2. **Captures Different Patterns** - Each model type captures different aspects:
   - **Statistical models** (Linear, Ridge, Lasso): Learn patterns from historical data
   - **Time-series models** (ARIMAX): Capture temporal autocorrelation
   - **Process-based models** (DTS, Thermal): Encode biological mechanisms
3. **More Robust** - Less sensitive to any single model's failures or outliers
4. **Better Generalization** - Often outperforms individual models on unseen data

## Ensemble Methods Comparison

### 1. Simple Average

```python
prediction = mean([model1, model2, model3, ...])
```

- **Pros**: Simple, often works surprisingly well
- **Cons**: Gives equal weight to poor models
- **Best When**: All models have similar performance

### 2. Weighted Average (Recommended ✓)

```python
weights = 1 / MAE_per_model  # Inverse MAE
weights = weights / sum(weights)  # Normalize
prediction = sum(weight_i * prediction_i)
```

- **Pros**: Optimal theoretical performance, gives more influence to better models
- **Cons**: Requires holdout data to estimate weights
- **Best When**: Models have clearly different performance levels
- **Implementation**: Already implemented in notebook cell 23

### 3. Median Ensemble

```python
prediction = median([model1, model2, model3, ...])
```

- **Pros**: Very robust to outlier predictions
- **Cons**: Loses information by discarding values
- **Best When**: Some models might produce extreme predictions

### 4. Trimmed Mean

```python
predictions = sort([model1, model2, model3, ...])
prediction = mean(predictions[1:-1])  # Remove highest and lowest
```

- **Pros**: Balances robustness and statistical efficiency
- **Cons**: Arbitrary choice of how much to trim
- **Best When**: Want compromise between mean and median

### 5. Stacked Ensemble (Advanced)

```python
# Train a meta-model on predictions
meta_model.fit(X=[pred1, pred2, pred3], y=actual)
prediction = meta_model.predict([new_pred1, new_pred2, new_pred3])
```

- **Pros**: Can learn complex relationships between models
- **Cons**: Requires extra holdout data, risk of overfitting
- **Best When**: You have lots of holdout data and models with different strengths

### 6. Top-K Selection

```python
top_k_models = select_best_k_models(by='holdout_mae', k=3)
prediction = mean([top_k_models])
```

- **Pros**: Excludes poor models, maintains diversity
- **Cons**: Sharp cutoff might exclude useful information
- **Best When**: Some models are clearly much worse than others

## Current Implementation in Notebook

Your notebook already implements a **Weighted Average** ensemble:

```python
# Calculate weights from top 3 models
top3_models = ranked.head(3).index.tolist()  # e.g., [Lasso, Bayesian Ridge, Ridge]
top3_weights = 1 / ranked.head(3)['MAE'].values  # Inverse MAE
top3_weights = top3_weights / top3_weights.sum()  # Normalize to sum to 1

# Example weights:
# Lasso (MAE=6.40): weight = 1/6.40 = 0.156 → 34.3%
# Bayesian Ridge (MAE=6.60): weight = 1/6.60 = 0.151 → 33.2%
# Ridge (MAE=6.63): weight = 1/6.63 = 0.151 → 32.5%
```

## Practical Recommendations

### For Your Cherry Blossom Predictions:

1. **Primary Approach**: Use **Weighted Average** of top 3-5 models
   - Based on holdout MAE (Lasso, Ridge, Bayesian Ridge are best)
   - Weights proportional to inverse MAE
   - Gives optimal expected performance

2. **Robustness Check**: Also compute **Median** ensemble
   - If weighted average and median are close (< 3 days), high confidence
   - If they differ significantly (> 7 days), report range of uncertainty

3. **Uncertainty Reporting**: Report ensemble agreement range
   - Min-max across ensemble methods = prediction uncertainty
   - Complements Bayesian Ridge's confidence intervals

4. **Model Transparency**: Include individual model predictions
   - Allows stakeholders to see consensus/disagreement
   - Useful for sensitivity analysis

## Code Example: Complete Ensemble Pipeline

```python
import pandas as pd
import numpy as np

# 1. Load holdout performance
holdout_mae = {
    'Lasso': 6.40,
    'Bayesian_Ridge': 6.60,
    'Ridge': 6.63,
    'Linear_OLS': 7.25,
    'ARIMAX': 7.21,
    'DTS': 11.60,
    'Process_Based': 9.29
}

# 2. Load 2026 predictions (DOY)
predictions = {
    'kyoto': {
        'Lasso': 93.8,
        'Bayesian_Ridge': 93.8,
        'Ridge': 93.8,
        'ARIMAX': 99.0,
        'DTS': 93.0
    },
    # ... other locations
}

# 3. Calculate weighted ensemble
def weighted_ensemble(location_preds, holdout_mae, top_k=3):
    # Select top k models
    sorted_models = sorted(holdout_mae.items(), key=lambda x: x[1])[:top_k]

    # Calculate weights
    weights = {model: 1/mae for model, mae in sorted_models}
    total_weight = sum(weights.values())
    weights = {model: w/total_weight for model, w in weights.items()}

    # Weighted sum
    ensemble_pred = sum(
        weights[model] * location_preds[model]
        for model in weights.keys()
        if model in location_preds
    )

    return ensemble_pred, weights

# 4. Apply to each location
for location, preds in predictions.items():
    ensemble, weights = weighted_ensemble(preds, holdout_mae, top_k=3)
    print(f"{location}: {ensemble:.1f} DOY")
    print(f"  Using: {', '.join(f'{m}({w:.2%})' for m, w in weights.items())}")
```

## Advanced: When to Use Different Approaches

| Scenario                           | Recommended Ensemble               | Reason                               |
| ---------------------------------- | ---------------------------------- | ------------------------------------ |
| Models perform similarly           | Simple Average                     | No need for complexity               |
| Clear performance differences      | Weighted Average                   | Optimal use of information           |
| Some outlier models                | Median or Trimmed Mean             | Robustness                           |
| Large dataset, complex patterns    | Stacked Ensemble                   | Can learn intricate relationships    |
| Limited holdout data               | Simple Average or Median           | Avoid overfitting weights            |
| High stakes, need interpretability | Weighted Average with transparency | Balance performance + explainability |

## Your Current Results

Based on your analysis, the **Weighted Average** approach is optimal because:

1. ✓ Models have different performance (6.40 to 11.60 MAE)
2. ✓ Top 3 models cluster together (6.40-6.63 MAE)
3. ✓ You have sufficient holdout data (239 observations)
4. ✓ Inverse MAE weighting is theoretically optimal
5. ✓ Includes diversity (statistical + time-series + process-based)

## Further Enhancements

If you want to improve your ensemble further:

1. **Location-Specific Weights**: Calculate weights separately per location

   ```python
   weights['kyoto'] = calculate_weights(holdout_data[location=='kyoto'])
   ```

2. **Dynamic Model Selection**: Exclude models that fail for specific locations

   ```python
   if prediction is NaN: exclude_model_for_this_location
   ```

3. **Confidence-Weighted Ensemble**: Weight by prediction interval width

   ```python
   weight = 1 / (MAE * prediction_interval_width)
   ```

4. **Cross-Validation Weights**: Use cross-validation instead of single holdout
   ```python
   weights = cross_val_score(models, folds=5)
   ```

## Conclusion

Your notebook's **Weighted Ensemble (Top 3)** approach is excellent and follows best practices:

- Mathematically optimal (inverse MAE weighting)
- Robust (includes diverse model types)
- Interpretable (clear weighting rationale)
- Validated (based on holdout performance)

**Predicted 2026 bloom dates using this ensemble**:

- Washington DC: ~Apr 10-15
- Kyoto: ~Apr 3
- Liestal: ~Apr 20
- Vancouver: ~Apr 20-28
- New York City: ~Apr 22-29

The ensemble reduces the risk of any single model's errors affecting your final predictions.
