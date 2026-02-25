import os
import numpy as np
import pandas as pd
from tqdm import tqdm

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
	get_species_thresholds,
	normalize_location,
)


BLOOM_FILE = AGGREGATED_BLOOM_FILE
CLIMATE_FILE = AGGREGATED_CLIMATE_FILE
FORECAST_FILE = PROJECTED_CLIMATE_FILE

OUTPUT_PREDICTIONS_FILE = os.path.join(
	PREDICTIONS_OUTPUT_DIR,
	f"final_{TARGET_YEAR}_predictions_process_based_thermal.csv",
)
OUTPUT_PARAMS_FILE = os.path.join(
	MODEL_OUTPUT_DIR,
	"process_based_species_parameters.csv",
)
OUTPUT_HOLDOUT_FILE = os.path.join(
	HOLDOUT_OUTPUT_DIR,
	f"holdout_last{HOLDOUT_LAST_N_YEARS}y_process_based_thermal.csv",
)

MIN_YEAR = MIN_MODEL_YEAR
CHILL_REQ_CANDIDATES = list(range(5, 161, 5))


def _build_event_series(location_climate, year, chill_temp_c, forcing_base_c):
	start = pd.to_datetime(f"{year-1}-{WINTER_START_MONTH_DAY}")
	end = pd.to_datetime(f"{year}-06-30")
	window = location_climate[
		(location_climate["date"] >= start) & (location_climate["date"] <= end)
	].copy()

	if window.empty:
		return None

	window = window.sort_values("date")
	window["chill_unit"] = (window["tmean_c"] <= chill_temp_c).astype(int)
	window["forcing_unit"] = np.maximum(window["tmean_c"] - forcing_base_c, 0.0)
	return window


def _get_chill_break_date(series_df, chill_requirement):
	chill_cumsum = series_df["chill_unit"].cumsum()
	hit = series_df.loc[chill_cumsum >= chill_requirement, "date"]
	if hit.empty:
		return None
	return hit.iloc[0]


def _forcing_to_observed_bloom(series_df, bloom_date, chill_break_date):
	post_break = series_df[(series_df["date"] >= chill_break_date) & (series_df["date"] <= bloom_date)]
	if post_break.empty:
		return np.nan
	return float(post_break["forcing_unit"].sum())


def _predict_bloom_date(series_df, chill_requirement, forcing_requirement):
	chill_break_date = _get_chill_break_date(series_df, chill_requirement)
	if chill_break_date is None:
		return pd.NaT

	post_break = series_df[series_df["date"] >= chill_break_date].copy()
	post_break["forcing_cumsum"] = post_break["forcing_unit"].cumsum()
	hit = post_break.loc[post_break["forcing_cumsum"] >= forcing_requirement, "date"]
	if hit.empty:
		return pd.NaT
	return hit.iloc[0]


def calibrate_species_params(bloom_df, climate_df):
	climate_by_loc = {
		loc: grp.sort_values("date") for loc, grp in climate_df.groupby("location")
	}

	species_params = []
	species_groups = list(bloom_df.groupby("species"))

	for species, species_rows in tqdm(
		species_groups,
		desc="Calibrating species parameters",
		unit="species",
	):
		thresholds = get_species_thresholds(species)
		chill_temp_c = thresholds["chill_temp_c"]
		forcing_base_c = thresholds["forcing_base_c"]

		event_cache = []
		for _, row in species_rows.iterrows():
			loc = row["location"]
			year = int(row["year"])
			bloom_date = row["bloom_date"]

			if loc not in climate_by_loc:
				continue

			series_df = _build_event_series(
				climate_by_loc[loc], year, chill_temp_c, forcing_base_c
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
			"chill_temp_c": chill_temp_c,
			"forcing_base_c": forcing_base_c,
			"chill_requirement": np.nan,
			"forcing_requirement": np.nan,
			"calibration_mae_days": np.nan,
			"calibration_event_count": 0,
		}

		if not event_cache:
			species_params.append(best)
			continue

		best_mae = np.inf
		for chill_req in tqdm(
			CHILL_REQ_CANDIDATES,
			desc=f"{species} chill search",
			unit="candidate",
			leave=False,
		):
			forcing_to_bloom = []
			valid_events = []

			for ev in event_cache:
				break_date = _get_chill_break_date(ev["series"], chill_req)
				if break_date is None or break_date > ev["bloom_date"]:
					continue

				required_forcing = _forcing_to_observed_bloom(
					ev["series"], ev["bloom_date"], break_date
				)
				if np.isnan(required_forcing):
					continue

				forcing_to_bloom.append(required_forcing)
				valid_events.append(ev)

			if len(forcing_to_bloom) < 10:
				continue

			forcing_req = float(np.median(forcing_to_bloom))
			errors = []

			for ev in valid_events:
				pred_date = _predict_bloom_date(ev["series"], chill_req, forcing_req)
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
						"chill_requirement": chill_req,
						"forcing_requirement": forcing_req,
						"calibration_mae_days": mae,
						"calibration_event_count": len(errors),
					}
				)

		species_params.append(best)

	return pd.DataFrame(species_params)


def predict_2026(process_params_df, bloom_df, forecast_df):
	forecast_by_loc = {
		loc: grp.sort_values("date") for loc, grp in forecast_df.groupby("location")
	}

	latest_meta = (
		bloom_df.sort_values(["location", "year"])
		.groupby("location", as_index=False)
		.agg(species=("species", "last"), country_code=("country_code", "last"))
	)
	latest_meta = latest_meta[latest_meta["location"].isin(TARGET_PREDICTION_LOCATIONS)]

	param_lookup = process_params_df.set_index("species").to_dict("index")
	records = []

	for _, row in latest_meta.iterrows():
		loc = row["location"]
		species = row["species"]

		if loc not in forecast_by_loc:
			records.append(
				{
					"location": loc,
					"year": TARGET_YEAR,
					"species": species,
					"predicted_date": pd.NaT,
					"predicted_doy": np.nan,
					"model_type": "process_based_thermal",
					"status": "missing_forecast_climate",
				}
			)
			continue

		params = param_lookup.get(species)
		if params is None or np.isnan(params.get("chill_requirement", np.nan)):
			records.append(
				{
					"location": loc,
					"year": TARGET_YEAR,
					"species": species,
					"predicted_date": pd.NaT,
					"predicted_doy": np.nan,
					"model_type": "process_based_thermal",
					"status": "missing_species_parameters",
				}
			)
			continue

		thresholds = get_species_thresholds(species)
		series_df = forecast_by_loc[loc].copy()
		series_df["chill_unit"] = (series_df["tmean_c"] <= thresholds["chill_temp_c"]).astype(int)
		series_df["forcing_unit"] = np.maximum(series_df["tmean_c"] - thresholds["forcing_base_c"], 0.0)

		pred_date = _predict_bloom_date(
			series_df,
			chill_requirement=float(params["chill_requirement"]),
			forcing_requirement=float(params["forcing_requirement"]),
		)

		if pd.isna(pred_date):
			records.append(
				{
					"location": loc,
					"year": TARGET_YEAR,
					"species": species,
					"predicted_date": pd.NaT,
					"predicted_doy": np.nan,
					"model_type": "process_based_thermal",
					"status": "not_reached_within_forecast_window",
				}
			)
			continue

		records.append(
			{
				"location": loc,
				"year": TARGET_YEAR,
				"species": species,
				"predicted_date": pred_date,
				"predicted_doy": int(pred_date.dayofyear),
				"model_type": "process_based_thermal",
				"status": "ok",
			}
		)

	return pd.DataFrame(records)


def evaluate_holdout(process_params_df, holdout_bloom_df, climate_df):
	climate_by_loc = {
		loc: grp.sort_values("date") for loc, grp in climate_df.groupby("location")
	}
	param_lookup = process_params_df.set_index("species").to_dict("index")

	records = []
	for _, row in holdout_bloom_df.iterrows():
		loc = row["location"]
		species = row["species"]
		year = int(row["year"])
		obs_date = row["bloom_date"]

		params = param_lookup.get(species)
		if loc not in climate_by_loc or params is None or np.isnan(params.get("chill_requirement", np.nan)):
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
						"actual_bloom_doy": float(obs_date.dayofyear),
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
						"predicted_doy": np.nan,
					"abs_error_days": np.nan,
						"model_name": "process_based_thermal",
					"status": "missing_inputs",
				}
			)
			continue

		thresholds = get_species_thresholds(species)
		series_df = _build_event_series(
			climate_by_loc[loc], year, thresholds["chill_temp_c"], thresholds["forcing_base_c"]
		)
		if series_df is None:
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
					"actual_bloom_doy": float(obs_date.dayofyear),
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
					"predicted_doy": np.nan,
					"abs_error_days": np.nan,
					"model_name": "process_based_thermal",
					"status": "missing_climate_window",
				}
			)
			continue

		pred_date = _predict_bloom_date(
			series_df,
			chill_requirement=float(params["chill_requirement"]),
			forcing_requirement=float(params["forcing_requirement"]),
		)

		if pd.isna(pred_date):
			records.append(
				{
					"location": loc,
					"species": species,
					"year": year,
					"actual_bloom_doy": float(obs_date.dayofyear),
					"observed_bloom_date": obs_date,
					"predicted_bloom_date": pd.NaT,
					"predicted_doy": np.nan,
					"abs_error_days": np.nan,
					"model_name": "process_based_thermal",
					"status": "not_reached",
				}
			)
			continue

		records.append(
			{
				"location": loc,
				"species": species,
				"year": year,
				"actual_bloom_doy": float(obs_date.dayofyear),
				"observed_bloom_date": obs_date,
				"predicted_bloom_date": pred_date,
				"predicted_doy": float(pred_date.dayofyear),
				"abs_error_days": abs((pred_date - obs_date).days),
				"model_name": "process_based_thermal",
				"status": "ok",
			}
		)

	return pd.DataFrame(records)


def main():
	print("1. Loading bloom, climate, and forecast datasets...")
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

	print("2. Calibrating species-level chill/forcing requirements on pre-holdout years (target locations only)...")
	params_df = calibrate_species_params(bloom_train_df, climate_df)

	print(f"3. Evaluating last-{HOLDOUT_LAST_N_YEARS}-years holdout performance...")
	holdout_eval_df = evaluate_holdout(params_df, bloom_holdout_df, climate_df)
	holdout_ok = holdout_eval_df[holdout_eval_df["status"] == "ok"]
	if not holdout_ok.empty:
		print(f"Holdout MAE (days): {holdout_ok['abs_error_days'].mean():.2f} over {len(holdout_ok)} events")
	else:
		print("Warning: No valid holdout predictions were produced.")

	print("4. Predicting 2026 bloom dates with process-based model...")
	pred_df = predict_2026(params_df, bloom_target_df, forecast_df)
	pred_df["predicted_doy"] = pd.to_numeric(pred_df["predicted_doy"], errors="coerce").round(1)
	pred_df["predicted_date"] = pd.to_datetime(pred_df["predicted_date"], errors="coerce")

	holdout_eval_df["predicted_doy"] = pd.to_numeric(holdout_eval_df["predicted_doy"], errors="coerce").round(1)
	holdout_eval_df["actual_bloom_doy"] = pd.to_numeric(holdout_eval_df["actual_bloom_doy"], errors="coerce").round(1)
	holdout_eval_df = holdout_eval_df[
		[
			"location",
			"year",
			"actual_bloom_doy",
			"predicted_doy",
			"abs_error_days",
			"model_name",
			"species",
			"status",
			"observed_bloom_date",
			"predicted_bloom_date",
		]
	]

	pred_df = pred_df[
		[
			"location",
			"year",
			"predicted_date",
			"predicted_doy",
			"model_type",
			"species",
			"status",
		]
	]

	print(f"5. Saving species parameters to {OUTPUT_PARAMS_FILE}")
	os.makedirs(os.path.dirname(OUTPUT_PARAMS_FILE), exist_ok=True)
	os.makedirs(HOLDOUT_OUTPUT_DIR, exist_ok=True)
	os.makedirs(PREDICTIONS_OUTPUT_DIR, exist_ok=True)
	params_df.to_csv(OUTPUT_PARAMS_FILE, index=False)

	print(f"6. Saving holdout evaluation to {OUTPUT_HOLDOUT_FILE}")
	holdout_eval_df.to_csv(OUTPUT_HOLDOUT_FILE, index=False)

	print(f"7. Saving 2026 process-based predictions to {OUTPUT_PREDICTIONS_FILE}")
	pred_df.to_csv(OUTPUT_PREDICTIONS_FILE, index=False)

	print("\n--- Process-Based Thermal Prediction Complete ---")
	print(pred_df.to_string(index=False))


if __name__ == "__main__":
	main()
