import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import r2_score

from phenology_config import (
	AGGREGATED_BLOOM_FILE,
	AGGREGATED_CLIMATE_FILE,
	PROJECTED_CLIMATE_FILE,
	MODEL_OUTPUT_DIR,
	HOLDOUT_OUTPUT_DIR,
	PREDICTIONS_OUTPUT_DIR,
	HOLDOUT_LAST_N_YEARS,
	MIN_MODEL_YEAR,
	TARGET_YEAR,
	TARGET_PREDICTION_LOCATIONS,
	WINTER_START_MONTH_DAY,
	USE_CV_FOLDS,
	get_species_thresholds,
	normalize_location,
)


BLOOM_FILE = AGGREGATED_BLOOM_FILE
CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
FORECAST_FILE = PROJECTED_CLIMATE_FILE

OUTPUT_PREDICTIONS_FILE = os.path.join(
	PREDICTIONS_OUTPUT_DIR,
	f"final_{TARGET_YEAR}_predictions_dts.csv",
)
OUTPUT_PARAMS_FILE = os.path.join(
	MODEL_OUTPUT_DIR,
	"dts_species_parameters.csv",
)
OUTPUT_HOLDOUT_FILE = os.path.join(
	HOLDOUT_OUTPUT_DIR,
	f"holdout_last{HOLDOUT_LAST_N_YEARS}y_dts.csv",
)

MIN_YEAR = MIN_MODEL_YEAR

# DTS Model uses Arrhenius exponential rate function
# Rate(T) = exp(a - b/T_kelvin)
# We'll calibrate critical thresholds for rest-breaking and flowering development
REST_THRESHOLD_CANDIDATES = np.arange(10.0, 100.0, 5.0)  # Critical rest accumulation
FLOWER_THRESHOLD_CANDIDATES = np.arange(5.0, 50.0, 2.5)  # Critical flowering development


def _celsius_to_kelvin(temp_c):
	"""Convert Celsius to Kelvin."""
	return temp_c + 273.15


def _dts_rate(temp_c, a, b):
	"""
	Calculate DTS development rate using Arrhenius equation.
	Rate = exp(a - b/T_kelvin)
	
	Args:
		temp_c: Temperature in Celsius
		a, b: Species-specific Arrhenius parameters
	
	Returns:
		Development rate (dimensionless daily contribution)
	"""
	temp_k = _celsius_to_kelvin(temp_c)
	return np.exp(a - b / temp_k)


def _build_event_series(location_climate, year, a_rest, b_rest, a_flower, b_flower):
	"""
	Build daily DTS accumulation series for rest-breaking and flowering development.
	
	Uses exponential Arrhenius approach rather than linear GDD.
	"""
	start = pd.to_datetime(f"{year-1}-{WINTER_START_MONTH_DAY}")
	end = pd.to_datetime(f"{year}-06-30")
	window = location_climate[
		(location_climate["date"] >= start) & (location_climate["date"] <= end)
	].copy()

	if window.empty:
		return None

	window = window.sort_values("date")
	
	# Calculate daily DTS rates for rest-breaking and flowering
	window["rest_rate"] = window["tmean_c"].apply(lambda t: _dts_rate(t, a_rest, b_rest))
	window["flower_rate"] = window["tmean_c"].apply(lambda t: _dts_rate(t, a_flower, b_flower))
	
	return window


def _get_rest_break_date(series_df, rest_threshold):
	"""Find date when accumulated rest-breaking DTS reaches threshold."""
	rest_cumsum = series_df["rest_rate"].cumsum()
	hit = series_df.loc[rest_cumsum >= rest_threshold, "date"]
	if hit.empty:
		return None
	return hit.iloc[0]


def _flower_development_to_bloom(series_df, bloom_date, rest_break_date):
	"""Calculate required flowering DTS from rest-break to observed bloom."""
	post_rest = series_df[
		(series_df["date"] >= rest_break_date) & (series_df["date"] <= bloom_date)
	]
	if post_rest.empty:
		return np.nan
	return float(post_rest["flower_rate"].sum())


def _predict_bloom_date(series_df, rest_threshold, flower_threshold):
	"""Predict bloom date using DTS two-phase model."""
	rest_break_date = _get_rest_break_date(series_df, rest_threshold)
	if rest_break_date is None:
		return pd.NaT

	post_rest = series_df[series_df["date"] >= rest_break_date].copy()
	post_rest["flower_cumsum"] = post_rest["flower_rate"].cumsum()
	hit = post_rest.loc[post_rest["flower_cumsum"] >= flower_threshold, "date"]
	if hit.empty:
		return pd.NaT
	return hit.iloc[0]


def calibrate_species_params(bloom_df, climate_df):
	"""
	Calibrate DTS model parameters for each species using grid search.
	
	Uses Japan Meteorological Agency DTS approach with Arrhenius exponential rates.
	"""
	climate_by_loc = {
		loc: grp.sort_values("date") for loc, grp in climate_df.groupby("location")
	}

	species_params = []
	species_groups = list(bloom_df.groupby("species"))

	for species, species_rows in tqdm(
		species_groups,
		desc="Calibrating DTS species parameters",
		unit="species",
	):
		# Initialize with literature-based Arrhenius parameters
		# Rest-breaking: moderate temperature sensitivity
		# Flowering: higher temperature sensitivity (faster exponential acceleration)
		a_rest = 18.5
		b_rest = 4500.0
		a_flower = 20.0
		b_flower = 5000.0

		event_cache = []
		for _, row in species_rows.iterrows():
			loc = row["location"]
			year = int(row["year"])
			bloom_date = row["bloom_date"]

			if loc not in climate_by_loc:
				continue

			series_df = _build_event_series(
				climate_by_loc[loc], year, a_rest, b_rest, a_flower, b_flower
			)
			if series_df is None:
				continue

			event_cache.append(
				{
					"location": loc,
					"year": year,
					"bloom_date": bloom_date,
					"series": series_df,
				}
			)

		best = {
			"species": species,
			"a_rest": a_rest,
			"b_rest": b_rest,
			"a_flower": a_flower,
			"b_flower": b_flower,
			"rest_threshold": np.nan,
			"flower_threshold": np.nan,
			"calibration_mae_days": np.nan,
			"calibration_event_count": 0,
		}

		if not event_cache:
			species_params.append(best)
			continue

		best_mae = np.inf
		for rest_thresh in tqdm(
			REST_THRESHOLD_CANDIDATES,
			desc=f"{species} DTS calibration",
			unit="candidate",
			leave=False,
		):
			flower_requirements = []
			valid_events = []

			for ev in event_cache:
				rest_break = _get_rest_break_date(ev["series"], rest_thresh)
				if rest_break is None or rest_break > ev["bloom_date"]:
					continue

				required_flower = _flower_development_to_bloom(
					ev["series"], ev["bloom_date"], rest_break
				)
				if np.isnan(required_flower):
					continue

				flower_requirements.append(required_flower)
				valid_events.append(ev)

			if len(flower_requirements) < 10:
				continue

			flower_thresh = float(np.median(flower_requirements))
			errors = []

			for ev in valid_events:
				pred_date = _predict_bloom_date(ev["series"], rest_thresh, flower_thresh)
				if pd.isna(pred_date):
					continue
				errors.append(abs((pred_date - ev["bloom_date"]).days))

			if not errors:
				continue

			mae = float(np.mean(errors))
			if mae < best_mae:
				best_mae = mae
				best.update(
					{
						"rest_threshold": rest_thresh,
						"flower_threshold": flower_thresh,
						"calibration_mae_days": mae,
						"calibration_event_count": len(errors),
					}
				)

		species_params.append(best)

	return pd.DataFrame(species_params)


def predict_2026(dts_params_df, bloom_df, forecast_df):
	"""Generate 2026 predictions using calibrated DTS model."""
	forecast_by_loc = {
		loc: grp.sort_values("date") for loc, grp in forecast_df.groupby("location")
	}

	latest_meta = (
		bloom_df.sort_values(["location", "year"])
		.groupby("location", as_index=False)
		.agg(species=("species", "last"), country_code=("country_code", "last"))
	)
	latest_meta = latest_meta[latest_meta["location"].isin(TARGET_PREDICTION_LOCATIONS)]

	param_lookup = dts_params_df.set_index("species").to_dict("index")
	records = []

	for _, row in latest_meta.iterrows():
		loc = row["location"]
		species = row["species"]

		if loc not in forecast_by_loc:
			records.append(
				{
					"location": loc,
					"species": species,
					"predicted_bloom_date": pd.NaT,
					"predicted_bloom_doy": np.nan,
					"status": "missing_forecast_climate",
				}
			)
			continue

		params = param_lookup.get(species)
		if params is None or np.isnan(params.get("rest_threshold", np.nan)):
			records.append(
				{
					"location": loc,
					"species": species,
					"predicted_bloom_date": pd.NaT,
					"predicted_bloom_doy": np.nan,
					"status": "missing_species_parameters",
				}
			)
			continue

		series_df = _build_event_series(
			forecast_by_loc[loc],
			TARGET_YEAR,
			params["a_rest"],
			params["b_rest"],
			params["a_flower"],
			params["b_flower"],
		)

		if series_df is None:
			records.append(
				{
					"location": loc,
					"species": species,
					"predicted_bloom_date": pd.NaT,
					"predicted_bloom_doy": np.nan,
					"status": "missing_forecast_window",
				}
			)
			continue

		pred_date = _predict_bloom_date(
			series_df,
			rest_threshold=float(params["rest_threshold"]),
			flower_threshold=float(params["flower_threshold"]),
		)

		if pd.isna(pred_date):
			records.append(
				{
					"location": loc,
					"species": species,
					"predicted_bloom_date": pd.NaT,
					"predicted_bloom_doy": np.nan,
					"status": "not_reached_within_forecast_window",
				}
			)
			continue

		records.append(
			{
				"location": loc,
				"species": species,
				"predicted_bloom_date": pred_date,
				"predicted_bloom_doy": int(pred_date.dayofyear),
				"status": "ok",
			}
		)

	return pd.DataFrame(records)


def evaluate_holdout(dts_params_df, holdout_bloom_df, climate_df):
	"""Evaluate DTS model on holdout years."""
	climate_by_loc = {
		loc: grp.sort_values("date") for loc, grp in climate_df.groupby("location")
	}
	param_lookup = dts_params_df.set_index("species").to_dict("index")

	records = []
	for _, row in holdout_bloom_df.iterrows():
		loc = row["location"]
		species = row["species"]
		year = int(row["year"])
		obs_date = row["bloom_date"]

		params = param_lookup.get(species)
		if loc not in climate_by_loc or params is None or np.isnan(params.get("rest_threshold", np.nan)):
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
					"abs_error_days": np.nan,
					"status": "missing_inputs",
				}
			)
			continue

		series_df = _build_event_series(
			climate_by_loc[loc],
			year,
			params["a_rest"],
			params["b_rest"],
			params["a_flower"],
			params["b_flower"],
		)
		if series_df is None:
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
					"abs_error_days": np.nan,
					"status": "missing_climate_window",
				}
			)
			continue

		pred_date = _predict_bloom_date(
			series_df,
			rest_threshold=float(params["rest_threshold"]),
			flower_threshold=float(params["flower_threshold"]),
		)

		if pd.isna(pred_date):
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
					"abs_error_days": np.nan,
					"status": "not_reached",
				}
			)
			continue

		records.append(
			{
				"location": loc,
				"species": species,
				"year": year,
				"observed_bloom_date": obs_date,
				"predicted_bloom_date": pred_date,
				"abs_error_days": abs((pred_date - obs_date).days),
				"status": "ok",
			}
		)

	return pd.DataFrame(records)


def main():
	print("=" * 80)
	print("DTS (Development rate Temperature Summation) Model")
	print("Using Arrhenius exponential approach (Japan Meteorological Agency method)")
	print("=" * 80)
	if USE_CV_FOLDS:
		print("Note: CV mode not supported for this model. Using simple holdout.")
	print(f"Holdout: Last {HOLDOUT_LAST_N_YEARS} years")
	print("=" * 80)
	
	print("\n1. Loading bloom, climate, and forecast datasets...")
	if not os.path.exists(BLOOM_FILE) or not os.path.exists(CLIMATE_FILE) or not os.path.exists(FORECAST_FILE):
		raise FileNotFoundError("Missing required input files. Run earlier pipeline steps first.")

	bloom_df = pd.read_csv(BLOOM_FILE)
	climate_df = pd.read_csv(CLIMATE_FILE)
	forecast_df = pd.read_csv(FORECAST_FILE)

	bloom_df["location"] = bloom_df["location"].apply(normalize_location)
	bloom_df["bloom_date"] = pd.to_datetime(bloom_df["bloom_date"])
	bloom_df = bloom_df[bloom_df["year"] >= MIN_YEAR].copy()
	bloom_target_df = bloom_df[bloom_df["location"].isin(TARGET_PREDICTION_LOCATIONS)].copy()

	if bloom_target_df.empty:
		raise ValueError("No target-location bloom rows found after filtering.")

	max_year = int(bloom_target_df["year"].max())
	holdout_start_year = max_year - HOLDOUT_LAST_N_YEARS + 1
	bloom_train_df = bloom_target_df[bloom_target_df["year"] < holdout_start_year].copy()
	bloom_holdout_df = bloom_target_df[bloom_target_df["year"] >= holdout_start_year].copy()

	if bloom_train_df.empty:
		raise ValueError(f"No training rows left after last-{HOLDOUT_LAST_N_YEARS}-years holdout split.")

	print(
		f"Using years < {holdout_start_year} for calibration and years >= {holdout_start_year} "
		f"as last-{HOLDOUT_LAST_N_YEARS}-years holdout."
	)
	print(
		f"Calibration rows: {len(bloom_train_df)} (targets only), "
		f"Holdout rows: {len(bloom_holdout_df)} (targets only)."
	)

	climate_df["location"] = climate_df["location"].apply(normalize_location)
	climate_df["date"] = pd.to_datetime(climate_df["date"])
	calibration_locations = set(bloom_train_df["location"].unique())
	target_locations = set(TARGET_PREDICTION_LOCATIONS)
	needed_locations = calibration_locations.union(target_locations)
	climate_df = climate_df[climate_df["location"].isin(needed_locations)].copy()

	forecast_df["location"] = forecast_df["location"].apply(normalize_location)
	forecast_df["date"] = pd.to_datetime(forecast_df["date"])
	forecast_df = forecast_df[forecast_df["location"].isin(TARGET_PREDICTION_LOCATIONS)].copy()

	print("2. Calibrating species-level DTS thresholds on pre-holdout years...")
	params_df = calibrate_species_params(bloom_train_df, climate_df)

	print(f"\n3. Evaluating last-{HOLDOUT_LAST_N_YEARS}-years holdout performance...")
	holdout_eval_df = evaluate_holdout(params_df, bloom_holdout_df, climate_df)
	holdout_ok = holdout_eval_df[holdout_eval_df["status"] == "ok"]
	if not holdout_ok.empty:
		print(f"Holdout MAE (days): {holdout_ok['abs_error_days'].mean():.2f} over {len(holdout_ok)} events")
		if len(holdout_ok) >= 2:
			holdout_obs_doy = holdout_ok["observed_bloom_date"].dt.dayofyear
			holdout_pred_doy = holdout_ok["predicted_bloom_date"].dt.dayofyear
			holdout_r2 = r2_score(holdout_obs_doy, holdout_pred_doy)
			print(f"Holdout R²: {holdout_r2:.3f}")
	else:
		print("Warning: No valid holdout predictions were produced.")

	print("\n4. Predicting 2026 bloom dates with DTS model...")
	pred_df = predict_2026(params_df, bloom_target_df, forecast_df)

	print(f"\n5. Saving DTS species parameters to {OUTPUT_PARAMS_FILE}")
	os.makedirs(os.path.dirname(OUTPUT_PARAMS_FILE), exist_ok=True)
	os.makedirs(HOLDOUT_OUTPUT_DIR, exist_ok=True)
	os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
	params_df.to_csv(OUTPUT_PARAMS_FILE, index=False)

	print(f"6. Saving holdout evaluation to {OUTPUT_HOLDOUT_FILE}")
	holdout_eval_df.to_csv(OUTPUT_HOLDOUT_FILE, index=False)

	print(f"7. Saving 2026 DTS predictions to {OUTPUT_PREDICTIONS_FILE}")
	pred_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)

	print("\n--- DTS Model Complete ---")
	print(pred_df.to_string(index=False))
	print("\nDTS Model uses exponential Arrhenius rates instead of linear GDD accumulation.")


if __name__ == "__main__":
	main()
