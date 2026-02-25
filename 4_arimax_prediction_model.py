import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from phenology_config import (
	MODEL_FEATURES_FILE,
	MODEL_OUTPUT_DIR,
	HOLDOUT_OUTPUT_DIR,
	PREDICTIONS_OUTPUT_DIR,
	HOLDOUT_LAST_N_YEARS,
	MIN_MODEL_YEAR,
	TARGET_YEAR,
	TARGET_PREDICTION_LOCATIONS,
	USE_CV_FOLDS,
	CV_FOLDS_FILE,
	CV_CONFIG_FILE,
	CV_ACTIVE_SPLIT,
	normalize_location,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
MIN_YEAR = MIN_MODEL_YEAR

EXOG_COLUMNS = [
	"max_tmax_early_spring",
	"total_prcp_early_spring",
]

OUTPUT_PREDICTIONS_FILE = os.path.join(
	PREDICTIONS_OUTPUT_DIR,
	f"final_{TARGET_YEAR}_predictions_arimax.csv",
)
OUTPUT_HOLDOUT_FILE = os.path.join(
	HOLDOUT_OUTPUT_DIR,
	f"holdout_last{HOLDOUT_LAST_N_YEARS}y_arimax.csv",
)
OUTPUT_MODEL_SUMMARY_FILE = os.path.join(
	MODEL_OUTPUT_DIR,
	"arimax_location_model_summary.csv",
)
OUTPUT_CV_METRICS = os.path.join(MODEL_OUTPUT_DIR, "cv_metrics_arimax.csv")


def doy_to_date(year, doy):
	if pd.isna(year) or pd.isna(doy):
		return pd.NaT
	start = pd.to_datetime(f"{int(year)}-01-01")
	return start + pd.to_timedelta(int(round(float(doy))) - 1, unit="D")


# ==========================================
# CV UTILITIES
# ==========================================
def load_cv_configuration():
	if not os.path.exists(CV_FOLDS_FILE):
		raise FileNotFoundError(f"CV folds file not found: {CV_FOLDS_FILE}\nRun 3c_year_block_folds.py first.")
	if not os.path.exists(CV_CONFIG_FILE):
		raise FileNotFoundError(f"CV config file not found: {CV_CONFIG_FILE}\nRun 3c_year_block_folds.py first.")
	return pd.read_csv(CV_FOLDS_FILE), pd.read_csv(CV_CONFIG_FILE)

def get_cv_splits(folds_df, config_df, active_split=None):
	splits = []
	split_indices = [active_split] if active_split else config_df['cv_split'].tolist()
	for split_idx in split_indices:
		config_row = config_df[config_df['cv_split'] == split_idx].iloc[0]
		test_fold = config_row['test_fold']
		train_folds = [int(f) for f in config_row['train_folds'].split(',')]
		train_years = folds_df[folds_df['fold'].isin(train_folds)]['year'].tolist()
		test_years = folds_df[folds_df['fold'] == test_fold]['year'].tolist()
		splits.append({
			'split_id': split_idx,
			'train_years': set(train_years),
			'test_years': set(test_years),
			'test_fold': test_fold,
		})
	return splits


# ==========================================
# 2. MODEL HELPERS
# ==========================================
def _fit_arimax(train_df):
	train_df = train_df.sort_values("year").drop_duplicates(subset=["year"], keep="last").copy()
	annual_index = pd.PeriodIndex(train_df["year"].astype(int), freq="Y")

	y_train = pd.Series(train_df["bloom_doy"].astype(float).values, index=annual_index)
	x_train = train_df[EXOG_COLUMNS].astype(float).copy()
	x_train.index = annual_index

	model = SARIMAX(
		endog=y_train,
		exog=x_train,
		order=(1, 0, 0),
		trend="c",
		enforce_stationarity=False,
		enforce_invertibility=False,
	)

	with warnings.catch_warnings():
		warnings.filterwarnings("ignore")
		results = model.fit(disp=False)
	return results


def _predict_with_interval(results, exog_df, alpha=0.10):
	forecast = results.get_forecast(steps=len(exog_df), exog=exog_df[EXOG_COLUMNS].astype(float))
	frame = forecast.summary_frame(alpha=alpha)
	return pd.DataFrame(
		{
			"predicted_doy": frame["mean"].values,
			"pi90_lower": frame["mean_ci_lower"].values,
			"pi90_upper": frame["mean_ci_upper"].values,
		}
	)


def _fit_pooled_exog_model(train_df):
	formula = "bloom_doy ~ max_tmax_early_spring + total_prcp_early_spring"
	return smf.ols(formula=formula, data=train_df).fit()


def _predict_pooled_with_interval(results, exog_df, alpha=0.10):
	pred = results.get_prediction(exog_df)
	frame = pred.summary_frame(alpha=alpha)
	return pd.DataFrame(
		{
			"predicted_doy": frame["mean"].values,
			"pi90_lower": frame["obs_ci_lower"].values,
			"pi90_upper": frame["obs_ci_upper"].values,
		}
	)


# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def run_arimax_split(historical, future, train_years, test_years, split_name=""):
	"""Run ARIMAX for a given train/test split."""
	train_data = historical[historical["year"].isin(train_years)].copy()
	test_data = historical[historical["year"].isin(test_years)].copy()
	
	print(f"\n{split_name}")
	print(f"Train: {len(train_data)} rows ({min(train_years)}-{max(train_years)})")
	print(f"Test: {len(test_data)} rows ({min(test_years)}-{max(test_years)})")
	
	holdout_records = []
	pred_records = []
	model_summary_records = []
	
	# Fit pooled fallback model
	pooled_model = None
	if len(train_data) >= 20:
		pooled_model = _fit_pooled_exog_model(train_data)
	
	# Fit per-location models
	for location in tqdm(historical["location"].unique(), desc=f"{split_name} - ARIMAX by location", unit="loc"):
		loc_hist = historical[historical["location"] == location].sort_values("year").copy()
		loc_train = loc_hist[loc_hist["year"].isin(train_years)].copy()
		loc_test = loc_hist[loc_hist["year"].isin(test_years)].copy()
		
		# Insufficient training data - use fallback
		if len(loc_train) < 12:
			if pooled_model is not None and not loc_test.empty:
				holdout_forecast = _predict_pooled_with_interval(pooled_model, loc_test)
				holdout_eval = loc_test[["location", "year", "bloom_doy"]].reset_index(drop=True).copy()
				holdout_eval["predicted_doy"] = holdout_forecast["predicted_doy"].round(1)
				holdout_eval["pi90_lower"] = holdout_forecast["pi90_lower"].round(1)
				holdout_eval["pi90_upper"] = holdout_forecast["pi90_upper"].round(1)
				holdout_eval["abs_error_days"] = (holdout_eval["predicted_doy"] - holdout_eval["bloom_doy"]).abs().round(1)
				holdout_eval["model_type"] = "pooled_fallback"
				holdout_records.append(holdout_eval)
			continue
		
		# Fit ARIMAX
		try:
			results = _fit_arimax(loc_train)
		except Exception:
			continue
		
		# Evaluate on test set
		if not loc_test.empty:
			holdout_forecast = _predict_with_interval(results, loc_test)
			holdout_eval = loc_test[["location", "year", "bloom_doy"]].reset_index(drop=True).copy()
			holdout_eval["predicted_doy"] = holdout_forecast["predicted_doy"].round(1)
			holdout_eval["pi90_lower"] = holdout_forecast["pi90_lower"].round(1)
			holdout_eval["pi90_upper"] = holdout_forecast["pi90_upper"].round(1)
			holdout_eval["abs_error_days"] = (holdout_eval["predicted_doy"] - holdout_eval["bloom_doy"]).abs().round(1)
			holdout_eval["model_type"] = "arimax"
			holdout_records.append(holdout_eval)
	
	# Aggregate holdout results
	if holdout_records:
		holdout_df = pd.concat(holdout_records, ignore_index=True)
		test_mae = holdout_df["abs_error_days"].mean()
		test_rmse = np.sqrt((holdout_df["abs_error_days"] ** 2).mean())
		if len(holdout_df) >= 2:
			test_r2 = r2_score(holdout_df["bloom_doy"], holdout_df["predicted_doy"])
		else:
			test_r2 = np.nan
		
		print(f"Test: MAE={test_mae:.2f}, RMSE={test_rmse:.2f}, R²={test_r2:.3f} (n={len(holdout_df)})")
		
		metrics = {
			'test_n': len(holdout_df),
			'test_mae': round(test_mae, 3),
			'test_rmse': round(test_rmse, 3),
			'test_r2': round(test_r2, 3) if not np.isnan(test_r2) else np.nan,
		}
		
		return holdout_df, metrics
	else:
		return pd.DataFrame(), {}

def main():
	print("1. Loading model features...")
	if not os.path.exists(FEATURES_FILE):
		raise FileNotFoundError(f"Features file not found: {FEATURES_FILE}")

	df = pd.read_csv(FEATURES_FILE)
	df["location"] = df["location"].apply(normalize_location)

	if "is_future" not in df.columns:
		raise ValueError("model_features.csv must include 'is_future'. Re-run feature engineering.")

	required_columns = ["location", "year", "bloom_doy", "is_future"] + EXOG_COLUMNS
	missing = [c for c in required_columns if c not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns in features: {missing}")

	historical = df[
		(df["is_future"] == False)
		& (df["location"].isin(TARGET_PREDICTION_LOCATIONS))
		& (df["year"] >= MIN_YEAR)
	].copy()
	future = df[(df["is_future"] == True) & (df["location"].isin(TARGET_PREDICTION_LOCATIONS))].copy()

	historical = historical.dropna(subset=["bloom_doy"] + EXOG_COLUMNS)
	future = future.dropna(subset=EXOG_COLUMNS)

	if historical.empty:
		raise ValueError("No historical rows available for target locations after filtering.")

	if USE_CV_FOLDS:
		print(f"\n{'='*80}")
		print("MODE: Year-Block Cross-Validation")
		print(f"{'='*80}")
		
		folds_df, config_df = load_cv_configuration()
		splits = get_cv_splits(folds_df, config_df, active_split=CV_ACTIVE_SPLIT)
		print(f"\nRunning {len(splits)} CV split(s)...")
		
		cv_metrics = []
		all_holdout_outputs = []
		
		for split_info in splits:
			split_id = split_info['split_id']
			train_years = split_info['train_years']
			test_years = split_info['test_years']
			
			holdout_df, metrics = run_arimax_split(
				historical, future, train_years, test_years, 
				split_name=f"CV Split {split_id} (Test Fold {split_info['test_fold']})"
			)
			
			if not holdout_df.empty:
				metrics['split_id'] = split_id
				metrics['test_fold'] = split_info['test_fold']
				cv_metrics.append(metrics)
				holdout_df['cv_split'] = split_id
				all_holdout_outputs.append(holdout_df)
		
		if cv_metrics:
			cv_metrics_df = pd.DataFrame(cv_metrics)
			mean_metrics = cv_metrics_df[['test_mae', 'test_rmse', 'test_r2']].mean()
			std_metrics = cv_metrics_df[['test_mae', 'test_rmse', 'test_r2']].std()
			
			print(f"\n{'='*80}")
			print("CROSS-VALIDATION SUMMARY")
			print(f"{'='*80}")
			print(f"Test MAE:  {mean_metrics['test_mae']:.2f} ± {std_metrics['test_mae']:.2f} days")
			print(f"Test RMSE: {mean_metrics['test_rmse']:.2f} ± {std_metrics['test_rmse']:.2f} days")
			print(f"Test R²:   {mean_metrics['test_r2']:.3f} ± {std_metrics['test_r2']:.3f}")
			
			os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
			cv_metrics_df.to_csv(OUTPUT_CV_METRICS, index=False)
			print(f"\nCV metrics saved to: {OUTPUT_CV_METRICS}")
			
			all_holdout_df = pd.concat(all_holdout_outputs, ignore_index=True)
			output_holdout_cv = os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_arimax.csv")
			os.makedirs(os.path.dirname(output_holdout_cv), exist_ok=True)
			all_holdout_df.to_csv(output_holdout_cv, index=False)
			print(f"Holdout predictions saved to: {output_holdout_cv}")
	else:
		print(f"\n{'='*80}")
		print(f"MODE: Simple Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)")
		print(f"{'='*80}")
		
		max_year = int(historical["year"].max())
		holdout_start_year = max_year - HOLDOUT_LAST_N_YEARS + 1
		train_years = set(historical[historical["year"] < holdout_start_year]["year"].unique())
		test_years = set(historical[historical["year"] >= holdout_start_year]["year"].unique())
		
		holdout_df, metrics = run_arimax_split(
			historical, future, train_years, test_years,
			split_name=f"Holdout (Last {HOLDOUT_LAST_N_YEARS} Years)"
		)
		
		if not holdout_df.empty:
			holdout_df["actual_bloom_doy"] = holdout_df.pop("bloom_doy")
			holdout_df["model_name"] = "arimax"
			os.makedirs(os.path.dirname(OUTPUT_HOLDOUT_FILE), exist_ok=True)
			holdout_df.to_csv(OUTPUT_HOLDOUT_FILE, index=False)
			print(f"Holdout saved to: {OUTPUT_HOLDOUT_FILE}")
	
	# Train final model on all data and generate 2026 predictions
	print("\n3. Training final models on all historical data for 2026 predictions...")
	pred_records = []
	pooled_model = _fit_pooled_exog_model(historical) if len(historical) >= 20 else None
	
	for location in tqdm(historical["location"].unique(), desc="Final models by location", unit="loc"):
		loc_hist = historical[historical["location"] == location].copy()
		
		if len(loc_hist) < 12:
			if pooled_model is not None:
				loc_future = future[future["location"] == location].copy()
				if not loc_future.empty:
					fut_forecast = _predict_pooled_with_interval(pooled_model, loc_future)
					loc_pred = loc_future[["location", "year"]].reset_index(drop=True).copy()
					loc_pred["predicted_doy"] = fut_forecast["predicted_doy"].round(1)
					loc_pred["pi90_lower"] = fut_forecast["pi90_lower"].round(1)
					loc_pred["pi90_upper"] = fut_forecast["pi90_upper"].round(1)
					loc_pred["interval_halfwidth_days"] = ((loc_pred["pi90_upper"] - loc_pred["pi90_lower"]) / 2.0).round(1)
					pred_records.append(loc_pred)
			continue
		
		try:
			results = _fit_arimax(loc_hist)
		except Exception:
			continue
		
		loc_future = future[future["location"] == location].copy()
		if not loc_future.empty:
			fut_forecast = _predict_with_interval(results, loc_future)
			loc_pred = loc_future[["location", "year"]].reset_index(drop=True).copy()
			loc_pred["predicted_doy"] = fut_forecast["predicted_doy"].round(1)
			loc_pred["pi90_lower"] = fut_forecast["pi90_lower"].round(1)
			loc_pred["pi90_upper"] = fut_forecast["pi90_upper"].round(1)
			loc_pred["interval_halfwidth_days"] = ((loc_pred["pi90_upper"] - loc_pred["pi90_lower"]) / 2.0).round(1)
			pred_records.append(loc_pred)
	
	if pred_records:
		pred_df = pd.concat(pred_records, ignore_index=True)
		pred_df["predicted_date"] = pred_df.apply(lambda r: doy_to_date(r["year"], r["predicted_doy"]), axis=1)
		os.makedirs(os.path.dirname(OUTPUT_PREDICTIONS_FILE), exist_ok=True)
		pred_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
		print(f"\n{TARGET_YEAR} ARIMAX predictions saved to: {OUTPUT_PREDICTIONS_FILE}")
		print(pred_df[["location", "predicted_date", "predicted_doy"]].to_string(index=False))
	else:
		print("No 2026 predictions produced.")


if __name__ == "__main__":
	main()
