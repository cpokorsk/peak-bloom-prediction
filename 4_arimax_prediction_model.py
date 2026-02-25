import os
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error
import statsmodels.formula.api as smf
from statsmodels.tsa.statespace.sarimax import SARIMAX

from phenology_config import (
	MODEL_FEATURES_FILE,
	MODEL_OUTPUT_DIR,
	HOLDOUT_OUTPUT_DIR,
	PREDICTIONS_OUTPUT_DIR,
	MIN_MODEL_YEAR,
	TARGET_YEAR,
	TARGET_PREDICTION_LOCATIONS,
	normalize_location,
)


# ==========================================
# 1. CONFIGURATION
# ==========================================
FEATURES_FILE = MODEL_FEATURES_FILE
HOLDOUT_LAST_N_YEARS = 10
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


def doy_to_date(year, doy):
	if pd.isna(year) or pd.isna(doy):
		return pd.NaT
	start = pd.to_datetime(f"{int(year)}-01-01")
	return start + pd.to_timedelta(int(round(float(doy))) - 1, unit="D")


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

	max_year = int(historical["year"].max())
	holdout_start_year = max_year - HOLDOUT_LAST_N_YEARS + 1
	print(
		f"Using years < {holdout_start_year} for training and years >= {holdout_start_year} "
		f"as last-{HOLDOUT_LAST_N_YEARS}-years holdout."
	)

	holdout_records = []
	pred_records = []
	model_summary_records = []

	global_train = historical[historical["year"] < holdout_start_year].copy()
	pooled_model = None
	if len(global_train) >= 20:
		pooled_model = _fit_pooled_exog_model(global_train)
		print("Using pooled exogenous fallback model for sparse locations.")
	else:
		print("Warning: insufficient global training rows for pooled fallback model.")

	grouped_hist = list(historical.groupby("location"))
	print("2. Fitting per-location ARIMAX (state-space SARIMAX) models...")
	for location, loc_hist in tqdm(grouped_hist, desc="ARIMAX by location", unit="location"):
		loc_hist = loc_hist.sort_values("year").copy()
		loc_train = loc_hist[loc_hist["year"] < holdout_start_year].copy()
		loc_holdout = loc_hist[loc_hist["year"] >= holdout_start_year].copy()

		if len(loc_train) < 12:
			used_fallback = False
			if pooled_model is not None:
				used_fallback = True
				if not loc_holdout.empty:
					holdout_forecast = _predict_pooled_with_interval(pooled_model, loc_holdout)
					holdout_eval = loc_holdout[["location", "year", "bloom_doy"]].reset_index(drop=True).copy()
					holdout_eval["predicted_doy"] = holdout_forecast["predicted_doy"].round(1)
					holdout_eval["pi90_lower"] = holdout_forecast["pi90_lower"].round(1)
					holdout_eval["pi90_upper"] = holdout_forecast["pi90_upper"].round(1)
					holdout_eval["abs_error_days"] = (holdout_eval["predicted_doy"] - holdout_eval["bloom_doy"]).abs().round(1)
					holdout_eval["model_type"] = "pooled_exog_fallback"
					holdout_records.append(holdout_eval)

				loc_future = future[future["location"] == location].sort_values("year").copy()
				if not loc_future.empty:
					fut_forecast = _predict_pooled_with_interval(pooled_model, loc_future)
					loc_pred = loc_future[["location", "year"]].reset_index(drop=True).copy()
					loc_pred["predicted_doy"] = fut_forecast["predicted_doy"].round(1)
					loc_pred["pi90_lower"] = fut_forecast["pi90_lower"].round(1)
					loc_pred["pi90_upper"] = fut_forecast["pi90_upper"].round(1)
					loc_pred["interval_halfwidth_days"] = ((loc_pred["pi90_upper"] - loc_pred["pi90_lower"]) / 2.0).round(1)
					loc_pred["model_type"] = "pooled_exog_fallback"
					pred_records.append(loc_pred)

			model_summary_records.append(
				{
					"location": location,
					"n_train": len(loc_train),
					"n_holdout": len(loc_holdout),
					"aic": np.nan,
					"holdout_mae_days": (
						float(mean_absolute_error(holdout_eval["bloom_doy"], holdout_eval["predicted_doy"]))
						if used_fallback and not loc_holdout.empty else np.nan
					),
					"status": "fallback_pooled_exog" if used_fallback else "insufficient_training_history",
				}
			)
			continue

		try:
			results = _fit_arimax(loc_train)
		except Exception as exc:
			model_summary_records.append(
				{
					"location": location,
					"n_train": len(loc_train),
					"n_holdout": len(loc_holdout),
					"aic": np.nan,
					"holdout_mae_days": np.nan,
					"status": f"fit_failed: {type(exc).__name__}",
				}
			)
			continue

		holdout_mae = np.nan
		if not loc_holdout.empty:
			holdout_forecast = _predict_with_interval(results, loc_holdout)
			holdout_eval = loc_holdout[["location", "year", "bloom_doy"]].reset_index(drop=True).copy()
			holdout_eval["predicted_doy"] = holdout_forecast["predicted_doy"].round(1)
			holdout_eval["pi90_lower"] = holdout_forecast["pi90_lower"].round(1)
			holdout_eval["pi90_upper"] = holdout_forecast["pi90_upper"].round(1)
			holdout_eval["abs_error_days"] = (holdout_eval["predicted_doy"] - holdout_eval["bloom_doy"]).abs().round(1)
			holdout_records.append(holdout_eval)

			holdout_mae = mean_absolute_error(holdout_eval["bloom_doy"], holdout_eval["predicted_doy"])

		loc_future = future[future["location"] == location].sort_values("year").copy()
		if not loc_future.empty:
			fut_forecast = _predict_with_interval(results, loc_future)
			loc_pred = loc_future[["location", "year"]].reset_index(drop=True).copy()
			loc_pred["predicted_doy"] = fut_forecast["predicted_doy"].round(1)
			loc_pred["pi90_lower"] = fut_forecast["pi90_lower"].round(1)
			loc_pred["pi90_upper"] = fut_forecast["pi90_upper"].round(1)
			loc_pred["interval_halfwidth_days"] = ((loc_pred["pi90_upper"] - loc_pred["pi90_lower"]) / 2.0).round(1)
			loc_pred["model_type"] = "arimax"
			pred_records.append(loc_pred)

		if not loc_holdout.empty:
			holdout_eval["model_type"] = "arimax"

		model_summary_records.append(
			{
				"location": location,
				"n_train": len(loc_train),
				"n_holdout": len(loc_holdout),
				"aic": float(results.aic) if np.isfinite(results.aic) else np.nan,
				"holdout_mae_days": float(holdout_mae) if np.isfinite(holdout_mae) else np.nan,
				"status": "ok",
			}
		)

	print("3. Saving outputs...")
	os.makedirs(HOLDOUT_OUTPUT_DIR, exist_ok=True)
	os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)

	holdout_df = pd.concat(holdout_records, ignore_index=True) if holdout_records else pd.DataFrame()
	pred_df = pd.concat(pred_records, ignore_index=True) if pred_records else pd.DataFrame()
	model_summary_df = pd.DataFrame(model_summary_records)

	if not holdout_df.empty:
		holdout_df["actual_bloom_doy"] = holdout_df["bloom_doy"].round(1)
		holdout_df["predicted_bloom_date"] = holdout_df.apply(
			lambda r: doy_to_date(r["year"], r["predicted_doy"]), axis=1
		)
		holdout_df["observed_bloom_date"] = holdout_df.apply(
			lambda r: doy_to_date(r["year"], r["bloom_doy"]), axis=1
		)
		holdout_df["pi90_lower_date"] = holdout_df.apply(
			lambda r: doy_to_date(r["year"], r["pi90_lower"]), axis=1
		)
		holdout_df["pi90_upper_date"] = holdout_df.apply(
			lambda r: doy_to_date(r["year"], r["pi90_upper"]), axis=1
		)
		holdout_df["model_name"] = "arimax"
		holdout_df = holdout_df[
			[
				"location",
				"year",
				"actual_bloom_doy",
				"predicted_doy",
				"pi90_lower",
				"pi90_upper",
				"abs_error_days",
				"model_name",
				"observed_bloom_date",
				"predicted_bloom_date",
				"pi90_lower_date",
				"pi90_upper_date",
			]
		]
		holdout_df.to_csv(OUTPUT_HOLDOUT_FILE, index=False)
		overall_mae = holdout_df["abs_error_days"].mean()
		print(f"Holdout MAE (last {HOLDOUT_LAST_N_YEARS} years): {overall_mae:.2f} days over {len(holdout_df)} rows")
	else:
		print("No holdout predictions produced.")

	if not pred_df.empty:
		pred_df["predicted_doy"] = pred_df["predicted_doy"].round(1)
		pred_df["predicted_date"] = pred_df.apply(
			lambda r: doy_to_date(r["year"], r["predicted_doy"]), axis=1
		)
		pred_df["pi90_lower_date"] = pred_df.apply(
			lambda r: doy_to_date(r["year"], r["pi90_lower"]), axis=1
		)
		pred_df["pi90_upper_date"] = pred_df.apply(
			lambda r: doy_to_date(r["year"], r["pi90_upper"]), axis=1
		)
		pred_df = pred_df[
			[
				"location",
				"year",
				"predicted_date",
				"predicted_doy",
				"pi90_lower",
				"pi90_upper",
				"interval_halfwidth_days",
				"model_type",
				"pi90_lower_date",
				"pi90_upper_date",
			]
		]
		pred_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)
		print(f"{TARGET_YEAR} ARIMAX predictions:")
		print(pred_df.to_string(index=False))
	else:
		print("No future predictions produced for target locations.")

	model_summary_df.to_csv(OUTPUT_MODEL_SUMMARY_FILE, index=False)

	print(f"Saved holdout file: {OUTPUT_HOLDOUT_FILE}")
	print(f"Saved prediction file: {OUTPUT_PREDICTIONS_FILE}")
	print(f"Saved model summary file: {OUTPUT_MODEL_SUMMARY_FILE}")


if __name__ == "__main__":
	main()
