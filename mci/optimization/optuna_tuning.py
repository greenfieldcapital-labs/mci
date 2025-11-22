"""Optuna-based hyperparameter optimization for time series prediction."""

from typing import Any, Callable

import numpy as np
import pandas as pd
from fastdtw import fastdtw
from optuna import Trial
from scipy.spatial.distance import euclidean

from ..core.dtw import _get_estimates
from ..core.minima import find_local_minima_with_separation
from ..forecasting.predictions import generate_sudo_predictions_for_frame
from .config import OptimizationConfig


__all__ = ["create_objective", "extract_config_from_params"]


def extract_config_from_params(
    best_params: dict[str, Any], df: pd.DataFrame
) -> tuple[int, dict[str, str], dict[str, float]]:
    """Extract configuration from Optuna best trial parameters.

    Args:
        best_params: Dictionary of best parameters from Optuna study.best_params
        df: DataFrame with columns to extract choices and weights for

    Returns:
        Tuple of (window_size, column_choices, weights) where:
        - window_size: The win_length parameter from best_params
        - column_choices: Dict mapping column names to their choice values
        - weights: Dict mapping column names to their weight values

    Example:
        >>> study = optuna.create_study()
        >>> # ... run optimization ...
        >>> win_size, column_choices, weights = extract_config_from_params(
        ...     study.best_params, df
        ... )
    """
    win_size = best_params["win_length"]

    # Get non-date columns
    non_date_cols = [col for col in df.columns if col != "date"]

    # Extract column choices
    column_choices = {}
    for col in non_date_cols:
        column_choices[col] = best_params[f"choice_{col}"]

    # Extract weights
    weights = {}
    for col in non_date_cols:
        weights[col] = best_params[f"weight_{col}"]

    return win_size, column_choices, weights


def create_objective(config: OptimizationConfig) -> Callable[[Trial], float]:
    """Create an Optuna objective function with captured configuration.

    Args:
        config: Configuration object containing all user parameters and data.

    Returns:
        Objective function that takes an Optuna trial and returns a loss metric.

    Example:
        >>> config = OptimizationConfig(
        ...     df=my_dataframe,
        ...     df_line_col="price",
        ...     horizon=10
        ... )
        >>> study = optuna.create_study(direction="minimize")
        >>> study.optimize(create_objective(config), n_trials=50)
    """
    df = config.df
    horizon = config.horizon
    calc_every_n = config.calc_every_n
    df_line_col = config.df_line_col
    min_current_idx = config.min_current_idx

    def objective(trial: Trial) -> float:
        """Objective function for Optuna optimization.

        Evaluates parameter combinations by computing DTW distances between
        predicted and actual values across the time series.

        Args:
            trial: Optuna trial object for suggesting parameters.

        Returns:
            Mean DTW distance (lower is better). Returns np.inf for invalid trials.
        """
        # Suggest hyperparameters from configured ranges
        win_length = trial.suggest_int("win_length", *config.win_length_range)
        N = trial.suggest_int("N", *config.N_range)
        min_separation = trial.suggest_int(
            "min_separation", *config.min_separation_range
        )
        top_k = trial.suggest_int("top_k", *config.top_k_range)
        prediction_averaging_range = trial.suggest_int(
            "prediction_averaging_range", *config.prediction_averaging_range
        )

        # Get non-date columns dynamically
        non_date_cols = [col for col in df.columns if col != "date"]

        # Suggest choices for each column
        column_choices = {}
        for col in non_date_cols:
            column_choices[col] = trial.suggest_categorical(
                f"choice_{col}", ["use", "not_use", "use_normalized"]
            )

        col_weights = {}
        for col in non_date_cols:
            col_weights[col] = trial.suggest_float(
                f"weight_{col}", 1e-10, 1.0, log=True
            )

        # Ensure at least one column is used
        if sum(1 for choice in column_choices.values() if choice != "not_use") == 0:
            return np.inf

        # Collect squared errors and DTW distances across all predictions
        squared_errors = []
        errors = []

        for current_idx in range(min_current_idx, len(df) - horizon, calc_every_n):
            history_df = df.iloc[: current_idx + 1]
            current_date = history_df["date"].iloc[-1]

            try:
                # Get DTW estimates for historical windows
                estimates = _get_estimates(
                    history_df.loc[
                        :,
                        ["date"]
                        + [k for k, v in column_choices.items() if v != "not_use"],
                    ],
                    win_length,
                    normalize=[
                        k for k, v in column_choices.items() if v == "use_normalized"
                    ],
                    weights=[
                        col_weights[k]
                        for k, v in column_choices.items()
                        if v == "use_normalized"
                    ],
                )

                # Find local minima (LMs)
                lms = find_local_minima_with_separation(
                    estimates,
                    N=N,
                    min_separation=min_separation,
                    top_k=top_k,
                )

                # Generate sudo predictions
                # Logic to generate sudo-predictions might be completely different!
                # One might even want to try some ML model using distances as weights
                # in the training and make actual 'forecasts'
                predictions = generate_sudo_predictions_for_frame(
                    current_date,
                    lms,
                    history_df,
                    df_line_col=df_line_col,
                    horizon=horizon,
                    prediction_averaging_range=prediction_averaging_range,
                )
            except Exception:
                return np.inf

            # Collect all predicted values arrays
            all_pred_arrays = [pred["predicted_values"] for pred in predictions]
            arr = np.array(all_pred_arrays)  # shape: [n_preds, horizon]

            # For weights: use inverse dist (lower dist = higher weight)
            epsilon = 1e-8

            if lms.shape[0] == 0:
                return np.inf

            dists = lms["dist"].values
            weights = 1 / (dists + epsilon)
            weights /= weights.sum()  # Normalize

            # Weighted average (instead of np.nanmean)
            avg_pred = np.average(arr, axis=0, weights=weights)

            # Actual values (from full df, for evaluation)
            max_pred_length = min(horizon, len(df) - current_idx - 1)
            actuals = (
                df[df_line_col]
                .iloc[current_idx + 1 : current_idx + 1 + max_pred_length]
                .values
            )

            try:
                errors.append(config.error_function(
                    avg_pred, actuals
                ))
            except Exception:
                return np.inf

        if not errors:
            return np.inf  # No valid predictions

        return np.mean(errors) 

    return objective
