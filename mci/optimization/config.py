"""Configuration for Optuna-based hyperparameter optimization."""

from dataclasses import dataclass

import pandas as pd


__all__ = ["OptimizationConfig"]


@dataclass(kw_only=True)
class OptimizationConfig:
    """User-configurable parameters for Optuna optimization.

    Attributes:
        df: Input time series dataframe with 'date' column and feature columns.
        df_line_col: Name of the column to use for predictions and evaluation.
        horizon: Number of time steps ahead to predict.
        calc_every_n: Evaluate predictions every N time steps (for speed).
        min_current_idx: Minimum index to start evaluation from. If None, defaults to horizon * 5.
        win_length_range: (min, max) range for window length parameter search.
        N_range: (min, max) range for N parameter (number of candidates for local minima).
        min_separation_range: (min, max) range for minimum separation between local minima.
        top_k_range: (min, max) range for top K local minima to use.
        prediction_averaging_range: (min, max) range for prediction averaging window.
    """

    # Data configuration
    df: pd.DataFrame
    df_line_col: str = "x"

    # Optimization targets
    horizon: int
    calc_every_n: int = 20
    min_current_idx: int | None = None

    # Parameter search ranges
    win_length_range: tuple[int, int] = (5, 25)
    N_range: tuple[int, int] = (1, 31)
    min_separation_range: tuple[int, int] = (1, 14)
    top_k_range: tuple[int, int] = (1, 10)
    prediction_averaging_range: tuple[int, int] = (1, 5)

    def __post_init__(self) -> None:
        """Set defaults and validate configuration."""
        if self.min_current_idx is None:
            self.min_current_idx = self.horizon * 5
