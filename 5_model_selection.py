import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from phenology_config import HOLDOUT_OUTPUT_DIR, HOLDOUT_LAST_N_YEARS, MODEL_OUTPUT_DIR, USE_CV_FOLDS


OUTPUT_SUMMARY = os.path.join(MODEL_OUTPUT_DIR, "model_selection_metrics_summary.csv")
OUTPUT_RECOMMENDED = os.path.join(MODEL_OUTPUT_DIR, "model_selection_recommended_for_ensemble.csv")


def _metrics(y_true, y_pred):
	mae = float(mean_absolute_error(y_true, y_pred))
	rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
	r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else np.nan
	return mae, rmse, r2


def _load_numeric_holdout(file_path, model_name, pred_col="predicted_doy", actual_col="actual_bloom_doy"):
	df = pd.read_csv(file_path)
	df = df.copy()

	if "status" in df.columns:
		df = df[df["status"] == "ok"].copy()

	# Handle both CV and simple holdout column names
	if actual_col not in df.columns and "bloom_doy" in df.columns:
		actual_col = "bloom_doy"
	
	y_true = pd.to_numeric(df.get(actual_col), errors="coerce")
	y_pred = pd.to_numeric(df.get(pred_col), errors="coerce")
	valid = y_true.notna() & y_pred.notna()

	y_true = y_true[valid]
	y_pred = y_pred[valid]

	if y_true.empty:
		return {
			"model": model_name,
			"n": 0,
			"mae_days": np.nan,
			"rmse_days": np.nan,
			"r2": np.nan,
		}

	mae, rmse, r2 = _metrics(y_true, y_pred)
	return {
		"model": model_name,
		"n": int(len(y_true)),
		"mae_days": round(mae, 4),
		"rmse_days": round(rmse, 4),
		"r2": round(r2, 4),
	}


def _load_dts_holdout(file_path):
	df = pd.read_csv(file_path)
	df = df.copy()

	if "status" in df.columns:
		df = df[df["status"] == "ok"].copy()

	obs = pd.to_datetime(df.get("observed_bloom_date"), errors="coerce")
	pred = pd.to_datetime(df.get("predicted_bloom_date"), errors="coerce")
	y_true = obs.dt.dayofyear
	y_pred = pred.dt.dayofyear
	valid = y_true.notna() & y_pred.notna()

	y_true = y_true[valid]
	y_pred = y_pred[valid]

	if y_true.empty:
		return {
			"model": "dts",
			"n": 0,
			"mae_days": np.nan,
			"rmse_days": np.nan,
			"r2": np.nan,
		}

	mae, rmse, r2 = _metrics(y_true, y_pred)
	return {
		"model": "dts",
		"n": int(len(y_true)),
		"mae_days": round(mae, 4),
		"rmse_days": round(rmse, 4),
		"r2": round(r2, 4),
	}


def build_model_metrics():
	"""Build model metrics using holdout files based on USE_CV_FOLDS flag."""
	suffix = f"last{HOLDOUT_LAST_N_YEARS}y"
	
	# Define model file paths based on USE_CV_FOLDS flag
	if USE_CV_FOLDS:
		print("  Mode: Using CV holdouts (USE_CV_FOLDS=True)")
		cv_files = {
			"linear_ols": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_linear_ols.csv"),
			"weighted_lm": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_weighted_lm.csv"),
			"bayesian_ridge": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_bayesian_ridge.csv"),
			"ridge_lasso": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_ridge_lasso.csv"),
			"gradient_boosting_quantile": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_gradient_boosting_quantile.csv"),
			"arimax": os.path.join(HOLDOUT_OUTPUT_DIR, "holdout_cv_arimax.csv"),
		}
		# Process-based models still use simple holdout (no CV support)
		files = cv_files.copy()
		files["process_based_thermal"] = os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_process_based_thermal.csv")
		files["dts"] = os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_dts.csv")
		files["random_forest"] = os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_random_forest.csv")
	else:
		print(f"  Mode: Using simple holdouts (USE_CV_FOLDS=False, last {HOLDOUT_LAST_N_YEARS} years)")
		files = {
			"linear_ols": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_linear_ols.csv"),
			"weighted_lm": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_weighted_lm.csv"),
			"bayesian_ridge": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_bayesian_ridge.csv"),
			"ridge_lasso": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_ridge_lasso.csv"),
			"gradient_boosting_quantile": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_gradient_boosting_quantile.csv"),
			"arimax": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_arimax.csv"),
			"process_based_thermal": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_process_based_thermal.csv"),
			"dts": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_dts.csv"),
			"random_forest": os.path.join(HOLDOUT_OUTPUT_DIR, f"holdout_{suffix}_random_forest.csv"),
		}

	missing = [path for path in files.values() if not os.path.exists(path)]
	if missing:
		missing_text = "\n".join(missing)
		raise FileNotFoundError(f"Missing holdout files:\n{missing_text}")

	rows = []
	rows.append(_load_numeric_holdout(files["linear_ols"], "linear_ols"))
	rows.append(_load_numeric_holdout(files["weighted_lm"], "weighted_lm"))
	rows.append(_load_numeric_holdout(files["bayesian_ridge"], "bayesian_ridge"))
	rows.append(_load_numeric_holdout(files["ridge_lasso"], "ridge", pred_col="predicted_doy_ridge"))
	rows.append(_load_numeric_holdout(files["ridge_lasso"], "lasso", pred_col="predicted_doy_lasso"))
	rows.append(_load_numeric_holdout(files["gradient_boosting_quantile"], "gradient_boosting_quantile"))
	rows.append(_load_numeric_holdout(files["arimax"], "arimax"))
	rows.append(_load_numeric_holdout(files["process_based_thermal"], "process_based_thermal"))
	rows.append(_load_dts_holdout(files["dts"]))
	rows.append(_load_numeric_holdout(files["random_forest"], "random_forest"))

	metrics_df = pd.DataFrame(rows)
	metrics_df["mae_rank"] = metrics_df["mae_days"].rank(method="min", ascending=True)
	metrics_df["rmse_rank"] = metrics_df["rmse_days"].rank(method="min", ascending=True)
	metrics_df["r2_rank"] = metrics_df["r2"].rank(method="min", ascending=False)
	metrics_df["composite_rank_score"] = (metrics_df["mae_rank"] + metrics_df["rmse_rank"] + metrics_df["r2_rank"]) / 3.0
	metrics_df = metrics_df.sort_values(
		["composite_rank_score", "mae_days", "rmse_days"],
		ascending=[True, True, True],
	).reset_index(drop=True)

	return metrics_df


def main():
	print("=== Step 5: Model Selection for Ensemble Inputs ===")
	print(f"Config: USE_CV_FOLDS={USE_CV_FOLDS}")

	metrics_df = build_model_metrics()

	os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
	metrics_df.to_csv(OUTPUT_SUMMARY, index=False)

	recommended = metrics_df.head(5).copy()
	recommended.insert(1, "include_in_ensemble", True)
	recommended.to_csv(OUTPUT_RECOMMENDED, index=False)

	print("\n--- Model Performance Summary (sorted) ---")
	print(
		metrics_df[
			[
				"model",
				"n",
				"mae_days",
				"rmse_days",
				"r2",
				"composite_rank_score",
			]
		].to_string(index=False)
	)

	print("\n--- Recommended Models for Ensemble (Top 5 by composite rank) ---")
	print(recommended[["model", "mae_days", "rmse_days", "r2"]].to_string(index=False))

	print(f"\nSaved summary: {OUTPUT_SUMMARY}")
	print(f"Saved recommendations: {OUTPUT_RECOMMENDED}")


if __name__ == "__main__":
	main()
