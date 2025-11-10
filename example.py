# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: nbformat,nbformat_minor
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   nbformat: 4
#   nbformat_minor: 5
# ---

# %% [markdown]
# # 🚀 Setup (Run this first in Google Colab)
#
# This cell installs the `mci` package and downloads the Bitcoin dataset from GitHub.
# **Skip this cell if running locally** (your local environment already has these).

# %%
# Google Colab Setup - Install package and download data from GitHub
import sys
import os

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("📦 Installing mci package from GitHub...")
    # Install mci package from GitHub repository (release-v1 branch)
    # !pip install -q git+https://github.com/greenfieldcapital-labs/mci.git@release-v1

    print("📥 Downloading Bitcoin dataset...")
    # Download data from GitHub raw content
    # !wget -q https://raw.githubusercontent.com/greenfieldcapital-labs/mci/release-v1/data/bitcoin_prices.csv -P data/

    print("✅ Setup complete! You can now run all cells below.")
else:
    print("ℹ️ Running locally - skipping Colab setup")

# %%
# Import from mci package
from mci.core import ExpandingWindowConfig, run_expanding_window_analysis
from mci.visualization.plotting import plot_result
from mci.optimization import OptimizationConfig, create_objective, extract_config_from_params
from mci.data import RandomDataConfig, generate_random_timeseries

# %% editable=true slideshow={"slide_type": ""}
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.colors
from tqdm import tqdm
import optuna
from random import random
import datetime
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler

# Reduce optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

lm_palette = plotly.colors.qualitative.Safe

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # DTW-Based Time Series sudo-predictions
#
# **Method**: Uses Dynamic Time Warping (DTW) to identify historical patterns similar to current data, then generates predictions by averaging outcomes from top matches.
#
# **Workflow**:
# 1. **Quick Demo** - Run on synthetic data to verify setup
# 2. **Optimization** - Find optimal hyperparameters via Optuna (50 trials)
# 3. **Production** - Apply to your data with tuned parameters
#
# **Key Parameters**:
# - `window_size`: Historical lookback period for pattern matching
# - `N`: Number of candidate patterns to search
# - `top_k`: Number of best matches to average for prediction
# - `min_separation`: Temporal gap between selected patterns (prevents autocorrelation)
# - `horizon`: Forecast steps ahead
#
# ---
#
# # Example usage on random data

# %% editable=true slideshow={"slide_type": ""}
# Generate random time series data using configurable parameters
df, win_size = generate_random_timeseries(
    RandomDataConfig(
        count=150,
        window_size=10,
        columns={"x": "sine_wave", "x2": "abs_random"}
    )
)

print(f"Generated {len(df)} data points | Window size: {win_size}")
print(f"Columns: {list(df.columns)}")

# Run expanding window analysis
res = run_expanding_window_analysis(
    ExpandingWindowConfig(df=df, window_size=win_size)
)

pred_horizon = 10

# Visualize with baseline parameters
frames = plot_result(
    res,
    df,
    'x',  # Target column to predict
    N = 5,  # Search 5 historical patterns
    horizon = pred_horizon,  # Predict 10 steps ahead
    top_k = 2,  # Average top 2 matches
    calc_every_nth=10  # Compute predictions every 10 steps
)

print(f"\n✓ Demo complete | Predictions generated for horizon={pred_horizon}")

# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Hyperparameter Optimization
# ## Suggested for users with at leats basic data science/optization knowledge
# **Objective**: Minimize prediction error (based on provided metric) across validation windows
#
# **Tuning Guidelines**:
# - `win_length_range`: Start with 5-10% of data length, increase for smoother series, decrease for more dynamic results
# - `N_range`: Higher values increase computation but find more diverse patterns
# - `min_separation`: Higher value lead to sudo-predictions based on diverse samples
# - `top_k_range`: 3-7 works well; too many dilutes signal
# - `prediction_averaging_range`: Smooths predictions from similar patterns
#
# **For Your Data**:
# 1. Replace `df` with your DataFrame (must have 'date' column + numeric features)
# 2. Set `df_line_col` to your target prediction column
# 3. Adjust ranges based on data frequency and volatility
# 4. Increase `n_trials` (line 149) for better convergence

# %%
# Configure optimization parameters, 

# Error metric, can be eg. MAE, RMSE, DT
def my_error_metric(avg_pred, actuals):
    dist, _ = fastdtw(
                    avg_pred[:, np.newaxis], actuals[:, np.newaxis], dist=euclidean
                )
    return dist

config = OptimizationConfig(
    df=df,
    df_line_col="x",  # Column to predict - CHANGE THIS for your data
    horizon=pred_horizon,
    calc_every_n=20,  # Validation frequency (higher = faster but less robust)
    # Parameter search ranges - ADJUST based on your data characteristics
    win_length_range=(5, 25),  # Pattern length to match
    N_range=(1, 31),  # Number of candidates to consider
    min_separation_range=(1, 14),  # Min distance between patterns
    top_k_range=(1, 10),  # How many matches to average
    prediction_averaging_range=(1, 5),  # Smooth similar predictions
    error_function=my_error_metric, # Way to calculate error
)

# Create Optuna study (persists to SQLite for resume capability)
study = optuna.create_study(
    direction="minimize",  # Minimize prediction error
    storage="sqlite:///optuna_study.db",
    study_name="time_series_prediction_study_v8",  # CHANGE THIS for new experiments
    load_if_exists=True  # Resume previous runs
)

print(f"Study loaded: {len(study.trials)} previous trials found")
print(f"Target: {config.df_line_col} | Horizon: {config.horizon} | Validation every {config.calc_every_n} steps")




# %% editable=true slideshow={"slide_type": ""}
# Optional: Seed with known-good configurations (delete if not needed)
study.enqueue_trial({
    "win_length": 14,
    "N": 7,
    "min_separation": 8,
    "top_k": 6,
    "prediction_averaging_range": 3,
    "choice_x": "use_normalized",
    "choice_x2": "use_normalized",
    "weight_x": 0.5196458224617007,
    "weight_x2": 0.07359009630233702
})

study.enqueue_trial({
    "N": 14,
    "choice_x": "use_normalized",
    "choice_x2": "not_use",
    "min_separation": 7,
    "prediction_averaging_range": 3,
    "top_k": 3,
    "weight_x": 1.0,
    "weight_x2": 1.0,
    "win_length": 10
})

# Run enqueued trials first (sequential to avoid race conditions)
print("▶ Running 2 enqueued trials...")
study.optimize(create_objective(config), n_trials=2, n_jobs=1, show_progress_bar=True)

# Run main optimization with parallelism
# Adjust n_trials and n_jobs based on available resources
print("\n▶ Running 48 trials (4 parallel jobs)...")
study.optimize(create_objective(config), n_trials=48, n_jobs=4, show_progress_bar=True)

# Summary statistics
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if len(completed_trials) > 0:
    values = [t.value for t in completed_trials if not np.isinf(t.value)]
    print(f"\n{'='*60}")
    print(f"✓ Optimization Complete")
    print(f"{'='*60}")
    print(f"Total trials: {len(completed_trials)}")
    print(f"Best Error: {study.best_value:.6f}")
    print(f"Mean Error: {np.mean(values):.6f} ± {np.std(values):.6f}")
    print(f"Worst Error: {max(values):.6f}")
    print(f"Improvement: {((max(values) - study.best_value) / max(values) * 100):.1f}% from worst")
    print(f"{'='*60}\n")



# %% editable=true slideshow={"slide_type": ""}
optuna.visualization.plot_optimization_history(study)

# %% editable=true slideshow={"slide_type": ""}
# Display optimization results
print(f"\n{'='*60}")
print(f"Best Configuration")
print(f"{'='*60}")
print(f"Error Score: {study.best_value:.6f}")
print(f"\nOptimal Parameters:")
for key, value in study.best_params.items():
    if isinstance(value, float):
        print(f"  {key:30s}: {value:.4f}")
    else:
        print(f"  {key:30s}: {value}")
print(f"{'='*60}\n")

# %% [markdown]
# # Apply Optimized Parameters
#
# Re-run analysis using best configuration from optimization.
# This generates predictions at higher temporal resolution (calc_every_nth=5) for detailed visualization.

# %%
best_params = study.best_params

# Extract configuration from best trial
win_size, column_choices, weights = extract_config_from_params(best_params, df)

print(f"▶ Running analysis with optimized parameters...")
print(f"  Window: {win_size} | N: {best_params['N']} | top_k: {best_params['top_k']}")

# Run analysis with best parameters
res = run_expanding_window_analysis(
    ExpandingWindowConfig(
        df=df,
        window_size=win_size,
        column_choices=column_choices,
        weights=weights
    )
)

# Generate visualization
frames = plot_result(
    res,
    df,
    'x',
    N = best_params['N'],
    horizon = pred_horizon,
    top_k = best_params['top_k'],
    min_separation = best_params['min_separation'],
    prediction_averaging_range = best_params['prediction_averaging_range'],
    calc_every_nth=5  # Higher resolution for visualization
)

print(f"✓ Optimized model visualization complete")



# %% [markdown] editable=true slideshow={"slide_type": ""}
# # Example: Bitcoin Price
#
# **Data Requirements**:
# - CSV with 'date' column (datetime) + numeric features
# - Sufficient history
# - Regular sampling frequency (daily, hourly, etc.)
#
# **Parameter Selection**:
# ###It is suggested to run optimisation for each new dataset
# - `window_size=180`: 6-month pattern matching (seasonal cycles)
# - `N=14`: Two weeks of candidate patterns
# - `top_k=5`: Average top 5 matches (reduces noise)
# - `min_separation=14`: 2-week gap (avoids temporal leakage)
# - `step=60`: Run analysis every 60 days (balances speed/coverage)
#

# %% editable=true slideshow={"slide_type": ""}
# Configuration for Bitcoin price
win_size = 180  # 6-month lookback
pred_horizon_btc = 180  # 6-month forecast

# Load and prepare data
# Replace with your data: ensure 'date' column + target numeric column(s)
df = pd.read_csv("data/bitcoin_prices.csv", parse_dates=["date"])
df = df.sort_values(['date']).reset_index(drop=True)
df = df[df.date>='2017-01-01']  # Filter to relevant period

print(f"Data loaded: {len(df)} rows from {df.date.min().date()} to {df.date.max().date()}")
print(f"Columns: {list(df.columns)}")
print(f"Target: avg_predicted_values | Window: {win_size} days | Horizon: {pred_horizon_btc} days\n")

# Run expanding window analysis
print("▶ Running expanding window analysis (this may take ~1 hour)...")
res = run_expanding_window_analysis(
    ExpandingWindowConfig(
        df=df,
        window_size=win_size,
        start_index=0,
        step=60  # Analyze every 60 days for efficiency
    )
)

print("✓ Analysis complete | Generating visualization...\n")

# Generate interactive visualization
plot_result(
    res,
    df,
    'avg_price',  # Target column - CHANGE for your data
    N = 14,  # Search last 14 patterns
    horizon = pred_horizon_btc,
    top_k = 5,  # Average 5 best matches
    min_separation = 14,  # 2-week minimum gap
    prediction_averaging_range = 3,  # Smooth 3 similar predictions
    calc_every_nth=1,  # Calculate all frames (set >1 for faster preview)
    frame_duration=150,  # Animation speed (ms)
    transition_duration=150
)

print(f"✓ Visualization complete")

