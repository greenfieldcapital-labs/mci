"""Data generation utilities for testing and examples."""

from dataclasses import dataclass
from typing import Literal, Callable
import datetime

import numpy as np
import pandas as pd

__all__ = ["RandomDataConfig", "generate_random_timeseries"]


@dataclass(kw_only=True)
class RandomDataConfig:
    """Configuration for random time series generation.

    Args:
        count: Number of data points to generate
        start_date: Starting date for the time series
        window_size: Default window size for processing
        columns: Dictionary mapping column names to generation functions
            Available built-in functions: 'sine_wave', 'random', 'abs_random'
        seed: Random seed for reproducibility (optional)
    """
    count: int = 150
    start_date: datetime.datetime | None = None
    window_size: int = 10
    columns: dict[str, Literal["sine_wave", "random", "abs_random"] | Callable] | None = None
    seed: int | None = None


def generate_random_timeseries(config: RandomDataConfig | None = None) -> tuple[pd.DataFrame, int]:
    """Generate random time series data for testing.

    Args:
        config: Configuration for data generation. Uses defaults if None.

    Returns:
        Tuple of (DataFrame with date and value columns, window_size)

    Example:
        >>> config = RandomDataConfig(count=100, window_size=15)
        >>> df, win_size = generate_random_timeseries(config)
    """
    if config is None:
        config = RandomDataConfig()

    if config.seed is not None:
        np.random.seed(config.seed)

    base = config.start_date or datetime.datetime.today()

    # Default columns if not specified
    if config.columns is None:
        columns = {
            "x": "sine_wave",
            "x2": "abs_random"
        }
    else:
        columns = config.columns

    # Generate date column
    data = {
        "date": [base + datetime.timedelta(days=x) for x in range(config.count)]
    }

    # Generate value columns
    for col_name, generator in columns.items():
        if generator == "sine_wave":
            data[col_name] = [10 + np.random.random() + np.sin(i / 3.14) for i in range(config.count)]
        elif generator == "random":
            data[col_name] = [np.random.random() for _ in range(config.count)]
        elif generator == "abs_random":
            data[col_name] = np.abs([np.random.random() for _ in range(config.count)])
        elif callable(generator):
            data[col_name] = [generator(i) for i in range(config.count)]
        else:
            raise ValueError(f"Unknown generator type: {generator}")

    df = pd.DataFrame(data)

    return df, config.window_size
