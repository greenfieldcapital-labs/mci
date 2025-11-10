import pandas as pd
import numpy as np
import warnings


def get_avg_predicted_values_ratios_over_window(df_line, center_date, range_days, horizon, df_line_col):
    """
    For all dates in [center_date - range_days, center_date + range_days], compute
    ratios up to horizon, and average them at each offset.
    """
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    window_dates = [center_date + pd.Timedelta(days=delta) for delta in range(-range_days, range_days+1)]
    ratio_matrix = []
    for d in window_dates:
        base_row = df_line[df_line['date'] == d]
        if base_row.empty:
            continue
        base_value = base_row[df_line_col].iloc[0]
        ratios = []
        for k in range(1, horizon+1):
            future_date = d + pd.Timedelta(days=k)
            future_row = df_line[df_line['date'] == future_date]
            if future_row.empty:
                ratios.append(np.nan)
            else:
                ratios.append(future_row[df_line_col].iloc[0] / base_value)
        ratio_matrix.append(ratios)
    if len(ratio_matrix) == 0:
        return [np.nan] * horizon
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Mean of empty slice')
        return list(np.nanmean(np.array(ratio_matrix), axis=0))


def generate_sudo_predictions_for_frame(current_date, lms, df_line, df_line_col, horizon=62, prediction_averaging_range=3):
    base_row = df_line[df_line['date'] == current_date]
    if base_row.empty:
        base_value = np.nan
    else:
        base_value = base_row[df_line_col].iloc[0]

    assert not np.isnan(base_value), f'Cannot find value for date = {current_date}'
    predictions = []
    for _, row in lms.iterrows():
        avg_ratios = get_avg_predicted_values_ratios_over_window(
            df_line, row['max_date'], prediction_averaging_range, horizon, df_line_col
        )
        predicted_values = [base_value * r if not np.isnan(r) else np.nan for r in avg_ratios]
        predictions.append({
            'lm_date': row['max_date'],
            'predicted_values': predicted_values,
            'ratios': avg_ratios
        })
    return predictions
