import pandas as pd
import numpy as np


def find_local_minima_with_separation(
    df, N=14, min_separation=7, top_k=5,
):
    df = df.sort_values(['current_date', 'max_date']).reset_index(drop=True)
    lms_all = []
    for curr_date, group in df.groupby('current_date'):
        vals = group['dist'].values
        dates = pd.to_datetime(group['max_date'].values)
        idx_minima = []
        for i in range(len(vals)):
            left = max(0, i - N)
            right = min(len(vals), i + N + 1)
            window = vals[left:right]
            if len(window) == 0:
                continue
            if vals[i] == np.min(window) and np.count_nonzero(window == vals[i]) == 1:
                idx_minima.append(i)
        minima_dates = []
        for i in idx_minima:
            date_i = dates[i]
            if all(abs((date_i - d).days) >= min_separation for d in minima_dates):
                minima_dates.append(date_i)
        minima = [
            {
                'current_date': curr_date,
                'max_date': date,
                'dist': vals[list(dates).index(date)]
            }
            for date in minima_dates
        ]
        # minima = [m for m in minima if m['dist'] <= lm_dist_max]
        minima = sorted(minima, key=lambda x: x['dist'])[:top_k]
        lms_all.extend(minima)
    return pd.DataFrame(lms_all)
