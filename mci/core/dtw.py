from typing import List
from typing import Dict
from typing import Tuple
from typing import Union
from typing import Optional

from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import warnings

from numpy import any as np_any
from numpy import isinf as np_isinf
from numpy import isnan as np_isnan

from pandas import DataFrame as pd_DataFrame
from pandas import Timestamp as pd_Timestamp

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# Suppress numpy warnings for invalid values during normalization
warnings.filterwarnings('ignore', 'invalid value encountered')

MAX_NA_THRESHOLD = 10
DATE_FORMAT = '%Y-%m-%d'


def _process_window(
    i: int,
    df: pd_DataFrame,
    ref_frame: pd_DataFrame,
    win_size: int,
    use_cols: List[str]
) -> Optional[Dict[str, Union[pd_Timestamp, float]]]:
    """
    Process a single window of time series data and calculate DTW distance.

    Args:
        i: Starting index of the window
        df: Input dataframe
        ref_frame: Reference frame for comparison
        win_size: Size of the window
        use_cols: Columns to use for calculation

    Returns:
        Dictionary containing window dates and DTW distance, or None if invalid
    """

    # Validate use_cols is not empty
    if len(use_cols) == 0:
        raise ValueError("use_cols cannot be empty")

    window = df.iloc[i:i+win_size].copy()
    min_date = window['date'].min()
    max_date = window['date'].max()

    # Validate columns
    window_use_cols = [col for col in window.columns[1:] if window[col].isna().sum() == 0]
    if len(window_use_cols) < (len(ref_frame.columns) - 1):
        return None

    # Check for inf/nan values BEFORE normalization
    if (np_any(np_isnan(window[use_cols].values)) or
        np_any(np_isinf(window[use_cols].values)) or
        np_any(np_isnan(ref_frame[use_cols].values)) or
        np_any(np_isinf(ref_frame[use_cols].values))):
        return None

    # Normalize window data
    for col in use_cols:
        std = window[col].std()
        if std > 0:
            window[col] = (window[col] - window[col].mean()) / std
        else:
            window[col] = 0

    # Calculate normalized DTW distance
    dist, _ = fastdtw(window[use_cols].values,
                        ref_frame[use_cols].values,
                        dist=euclidean)
    dist = dist / win_size

    return {
        'min_date': min_date,
        'max_date': max_date,
        'dist': dist
    }

def _get_estimates(df: pd_DataFrame, win_size: int, normalize=None, weights =None) -> pd_DataFrame:
    """
    Calculate DTW distances for all possible windows in the dataset.

    Args:
        df: Input dataframe
        win_size: Size of the window

    Returns:
        DataFrame with calculated distances
    """


    # Prepare reference frame
    ref_frame = df.iloc[-win_size:].copy()
    df = df.iloc[:-win_size].copy()

    # Clean data
    to_rm = [col for col in ref_frame.columns[1:]
             if ref_frame[col].isna().sum() >= MAX_NA_THRESHOLD]
    df = df.drop(columns=to_rm)
    ref_frame = ref_frame.drop(columns=to_rm)
    if normalize is None:
        normalize = ref_frame.columns[1:]

    if weights is None:
        weights = [1 for _ in ref_frame.columns[1:]]

    # Normalize reference frame
    for col in normalize:
        std = ref_frame[col].std()
        if std > 0:
            ref_frame[col] = (ref_frame[col] - ref_frame[col].mean()) / std
        else:
            ref_frame[col] = 0

    for i, col in enumerate(normalize):
        ref_frame[col] *= weights[i]

    # Prepare for parallel processing
    total_windows = len(df) - win_size + 1
    use_cols = [col for col in ref_frame.columns[1:]
                if ref_frame[col].isna().sum() == 0]

    # Process windows in parallel


    results = []
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
        process_window_partial = partial(
            _process_window,
            df=df,
            ref_frame=ref_frame,
            win_size=win_size,
            use_cols=use_cols
        )

        futures = [executor.submit(process_window_partial, i)
                  for i in range(total_windows)]

        results = [future.result() for future in futures
                  if future.result() is not None]

    # Process results
    dists = pd_DataFrame(results)

    if len(dists) == 0:
        return pd_DataFrame(columns=['max_date', 'dist'])

    merged = (dists[['max_date', 'dist']]
             .copy()
             .dropna()
             .sort_values('max_date'))
    merged['current_date'] = ref_frame['date'].max()
    return merged
