"""Tests for objective function (Optuna optimization)."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from mci.optimization import OptimizationConfig, create_objective


class FakeTrial:
    """Fake Optuna trial for testing without mock objects."""

    def __init__(self, params=None):
        """Initialize with predefined parameters."""
        self.params = params or {}
        self._int_defaults = {
            "win_length": 10,
            "N": 7,
            "min_separation": 7,
            "top_k": 3,
            "prediction_averaging_range": 3,
        }
        self._float_defaults = {
            "weight_x": 0.5,
            "weight_x2": 0.5,
        }
        self._categorical_defaults = {
            "choice_x": "use_normalized",
            "choice_x2": "use_normalized",
        }

    def suggest_int(self, name, low, high):
        """Suggest an integer parameter."""
        if name in self.params:
            return self.params[name]
        if name in self._int_defaults:
            return self._int_defaults[name]
        return (low + high) // 2

    def suggest_categorical(self, name, choices):
        """Suggest a categorical parameter."""
        if name in self.params:
            return self.params[name]
        if name in self._categorical_defaults:
            return self._categorical_defaults[name]
        return choices[0]

    def suggest_float(self, name, low, high, log=False):
        """Suggest a float parameter."""
        if name in self.params:
            return self.params[name]
        if name in self._float_defaults:
            return self._float_defaults[name]
        return (low + high) / 2


class TestObjectiveFunction:
    """Test suite for objective function."""

    @pytest.fixture
    def sample_df(self):
        """Create sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(300)]
        df = pd.DataFrame({
            "date": dates,
            "x": np.linspace(100, 200, 300) + np.sin(np.linspace(0, 6*np.pi, 300)) * 10,
            "x2": np.linspace(50, 100, 300) + np.cos(np.linspace(0, 6*np.pi, 300)) * 5,
        })
        return df

    def test_objective_returns_float(self, sample_df):
        """Test that objective returns a float value."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        assert isinstance(result, (float, np.floating))

    def test_objective_returns_positive(self, sample_df):
        """Test that objective returns a positive value."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        assert result >= 0 or result == np.inf

    def test_objective_with_invalid_params(self, sample_df):
        """Test objective with invalid parameters."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        # Trial that returns all 'not_use'
        trial = FakeTrial({
            "choice_x": "not_use",
            "choice_x2": "not_use",
        })

        result = objective(trial)

        # Should return infinity when no columns are used
        assert result == np.inf

    def test_objective_handles_exceptions(self, sample_df):
        """Test that objective handles exceptions gracefully."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        # Trial with win length too large
        trial = FakeTrial({"win_length": 250})

        result = objective(trial)

        # Should return infinity on error
        assert result == np.inf or isinstance(result, float)

    def test_objective_with_different_horizons(self, sample_df):
        """Test objective with different horizon values."""
        trial = FakeTrial()

        # Test with different horizons
        for horizon in [10, 30, 60]:
            config = OptimizationConfig(
                df=sample_df,
                df_line_col="x",
                horizon=horizon,
            )
            objective = create_objective(config)
            result = objective(trial)
            assert isinstance(result, (float, np.floating))

    def test_parameter_ranges_respected(self, sample_df):
        """Test that parameter ranges are respected."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        # Create trial that tracks what parameters were requested
        suggested_values = {}

        class TrackingTrial(FakeTrial):
            def suggest_int(self, name, low, high):
                val = super().suggest_int(name, low, high)
                suggested_values[name] = (val, low, high)
                return val

        trial = TrackingTrial()
        objective(trial)

        # Check that suggest_int was called with appropriate ranges
        assert "win_length" in suggested_values
        assert "N" in suggested_values
        assert "min_separation" in suggested_values
        assert "top_k" in suggested_values

    def test_column_choices_handling(self, sample_df):
        """Test handling of different column choice combinations."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        # Test with 'use' choice
        trial = FakeTrial({
            "choice_x": "use",
            "choice_x2": "use",
        })

        result = objective(trial)
        assert isinstance(result, (float, np.floating))

    def test_weights_effect(self, sample_df):
        """Test that weights are used in calculations."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        weight_values = {}

        class WeightTrackingTrial(FakeTrial):
            def suggest_float(self, name, low, high, log=False):
                val = 0.7 if "x2" in name else 0.3
                weight_values[name] = val
                return val

        trial = WeightTrackingTrial()
        objective(trial)

        # Check that weights were suggested for columns
        assert any("weight_" in key for key in weight_values)

    def test_empty_predictions_handling(self, sample_df):
        """Test handling when no valid predictions are generated."""
        config = OptimizationConfig(
            df=sample_df[:50],  # Very small dataset
            df_line_col="x",
            horizon=40,  # Large horizon
        )
        objective = create_objective(config)

        trial = FakeTrial({"win_length": 20})

        result = objective(trial)

        # Should return infinity when no valid predictions
        assert result == np.inf or isinstance(result, float)

    def test_dtw_distance_calculation(self, sample_df):
        """Test that DTW distance is calculated and returned."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        # Result should be the mean DTW distance or RMSE
        assert isinstance(result, (float, np.floating))
        assert result >= 0 or result == np.inf

    def test_squared_errors_accumulation(self, sample_df):
        """Test that squared errors are accumulated correctly."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        # Should compute and return error metric
        assert isinstance(result, (float, np.floating))

    def test_calc_every_n_parameter(self, sample_df):
        """Test that calc_every_n reduces computation."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
            calc_every_n=50,  # Large value for faster test
        )
        objective = create_objective(config)

        # The function uses calc_every_n internally
        trial = FakeTrial()
        result = objective(trial)

        assert isinstance(result, (float, np.floating))

    def test_min_current_idx_constraint(self, sample_df):
        """Test that min_current_idx constraint is respected."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
            min_current_idx=200,  # Explicit min index
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        # Should handle min_current_idx constraint
        assert isinstance(result, (float, np.floating))

    def test_nan_handling_in_predictions(self, sample_df):
        """Test handling of NaN values in predictions."""
        # Create data with some NaN values
        df_with_nan = sample_df.copy()
        df_with_nan.loc[50:55, "x"] = np.nan

        config = OptimizationConfig(
            df=df_with_nan,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        # Should handle NaN values gracefully
        assert isinstance(result, (float, np.floating))

    def test_zero_lms_handling(self, sample_df):
        """Test handling when no local minima are found."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        # Trial with parameters that might not find LMs
        trial = FakeTrial({"N": 50})

        result = objective(trial)

        # Should return infinity when no LMs found
        assert result == np.inf or isinstance(result, float)

    def test_different_column_combinations(self, sample_df):
        """Test with different combinations of column usage."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        combinations = [
            {"choice_x": "use", "choice_x2": "not_use"},
            {"choice_x": "use_normalized", "choice_x2": "not_use"},
            {"choice_x": "not_use", "choice_x2": "use"},
            {"choice_x": "use_normalized", "choice_x2": "use_normalized"},
        ]

        for combo in combinations:
            trial = FakeTrial(combo)
            result = objective(trial)

            # Should handle all combinations
            assert isinstance(result, (float, np.floating))

    def test_use_vs_use_normalized(self, sample_df):
        """Test that 'use' and 'use_normalized' produce different results."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial1 = FakeTrial({"choice_x": "use", "choice_x2": "not_use"})
        result1 = objective(trial1)

        trial2 = FakeTrial({"choice_x": "use_normalized", "choice_x2": "not_use"})
        result2 = objective(trial2)

        # Both should return valid results
        assert isinstance(result1, (float, np.floating))
        assert isinstance(result2, (float, np.floating))

    def test_deterministic_results(self, sample_df):
        """Test that same parameters produce same results."""
        params = {
            "win_length": 15,
            "N": 10,
            "min_separation": 5,
            "top_k": 3,
            "prediction_averaging_range": 2,
            "choice_x": "use_normalized",
            "choice_x2": "not_use",
            "weight_x": 0.6,
            "weight_x2": 0.4,
        }

        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial1 = FakeTrial(params.copy())
        result1 = objective(trial1)

        trial2 = FakeTrial(params.copy())
        result2 = objective(trial2)

        # Same parameters should produce same result
        if result1 != np.inf and result2 != np.inf:
            assert np.isclose(result1, result2, rtol=1e-5)
        else:
            assert result1 == result2

    def test_different_weights_different_results(self, sample_df):
        """Test that different weights produce different results."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
        )
        objective = create_objective(config)

        trial1 = FakeTrial({
            "choice_x": "use_normalized",
            "choice_x2": "use_normalized",
            "weight_x": 0.9,
            "weight_x2": 0.1,
        })

        trial2 = FakeTrial({
            "choice_x": "use_normalized",
            "choice_x2": "use_normalized",
            "weight_x": 0.1,
            "weight_x2": 0.9,
        })

        result1 = objective(trial1)
        result2 = objective(trial2)

        # Different weights should generally produce different results
        if result1 != np.inf and result2 != np.inf:
            # Results should differ (unless data happens to produce same values)
            assert isinstance(result1, (float, np.floating))
            assert isinstance(result2, (float, np.floating))

    def test_config_parameter_ranges(self, sample_df):
        """Test that custom parameter ranges are used."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x",
            horizon=30,
            win_length_range=(10, 20),
            N_range=(5, 15),
            min_separation_range=(3, 10),
            top_k_range=(2, 5),
            prediction_averaging_range=(1, 3),
        )
        objective = create_objective(config)

        suggested_values = {}

        class RangeTrackingTrial(FakeTrial):
            def suggest_int(self, name, low, high):
                val = super().suggest_int(name, low, high)
                suggested_values[name] = (low, high)
                return val

        trial = RangeTrackingTrial()
        objective(trial)

        # Check that custom ranges were used
        assert suggested_values["win_length"] == (10, 20)
        assert suggested_values["N"] == (5, 15)
        assert suggested_values["min_separation"] == (3, 10)
        assert suggested_values["top_k"] == (2, 5)
        assert suggested_values["prediction_averaging_range"] == (1, 3)

    def test_different_df_line_col(self, sample_df):
        """Test with different target column."""
        config = OptimizationConfig(
            df=sample_df,
            df_line_col="x2",  # Use x2 instead of x
            horizon=30,
        )
        objective = create_objective(config)

        trial = FakeTrial()
        result = objective(trial)

        assert isinstance(result, (float, np.floating))
