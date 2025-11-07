"""Time series processing utilities for expanding window analysis."""

from dataclasses import dataclass
from typing import Literal

import pandas as pd
from tqdm import tqdm

from .dtw import _get_estimates

__all__ = ["ExpandingWindowConfig", "run_expanding_window_analysis"]


@dataclass(kw_only=True)
class ExpandingWindowConfig:
    """Configuration for expanding window DTW analysis.

    Args:
        df: Input DataFrame with time series data
        window_size: Size of the sliding window
        start_index: Starting index for the analysis (default: window_size)
        step: Step size for iteration (default: 1, process every row)
        column_choices: Dictionary mapping column names to usage choices
            Options: "use_normalized", "use_raw", "not_use"
        weights: Dictionary mapping column names to their weights (for normalized columns)
        show_progress: Whether to show progress bar (default: True)
    """
    df: pd.DataFrame
    window_size: int
    start_index: int | None = None
    step: int = 1
    column_choices: dict[str, Literal["use_normalized", "use_raw", "not_use"]] | None = None
    weights: dict[str, float] | None = None
    show_progress: bool = True


def run_expanding_window_analysis(config: ExpandingWindowConfig) -> pd.DataFrame:
    """Run DTW analysis over expanding windows of time series data.

    This function performs expanding window analysis by:
    1. Iterating through the dataframe with progressively larger windows
    2. Computing DTW distances using _get_estimates for each window
    3. Filtering out invalid results (None or empty DataFrames)
    4. Concatenating all results into a single DataFrame

    Args:
        config: Configuration for the expanding window analysis

    Returns:
        DataFrame with DTW distance estimates for all windows

    Example:
        >>> # Simple usage
        >>> config = ExpandingWindowConfig(df=df, window_size=10)
        >>> results = run_expanding_window_analysis(config)

        >>> # Advanced usage with column selection
        >>> config = ExpandingWindowConfig(
        ...     df=df,
        ...     window_size=25,
        ...     column_choices={"x": "use_normalized", "x2": "not_use"},
        ...     weights={"x": 0.5}
        ... )
        >>> results = run_expanding_window_analysis(config)
    """
    df = config.df
    win_size = config.window_size
    start_idx = config.start_index if config.start_index is not None else win_size
    step = config.step

    # Determine which columns to use and how
    if config.column_choices is not None:
        # Filter columns based on choices
        selected_cols = [k for k, v in config.column_choices.items() if v != "not_use"]
        normalize_cols = [k for k, v in config.column_choices.items() if v == "use_normalized"]

        # Build weights list in same order as normalize_cols
        if config.weights is not None:
            weights_list = [config.weights.get(col, 1.0) for col in normalize_cols]
        else:
            weights_list = None

        # Create iterator with tqdm
        indices = range(start_idx, df.shape[0], step)
        iterator = tqdm(indices) if config.show_progress else indices

        # Process each window with column filtering
        results = [
            _get_estimates(
                df.loc[:i, ["date"] + selected_cols],
                win_size,
                normalize=normalize_cols if normalize_cols else None,
                weights=weights_list
            )
            for i in iterator
        ]
    else:
        # Simple case: use all columns
        indices = range(start_idx, df.shape[0], step)
        iterator = tqdm(indices) if config.show_progress else indices

        results = [
            _get_estimates(df[:i], win_size)
            for i in iterator
        ]

    # Filter out None and empty DataFrames
    valid_results = [x for x in results if x is not None and not x.empty]

    # Concatenate all results
    if valid_results:
        return pd.concat(valid_results, ignore_index=True)
    else:
        return pd.DataFrame()
