"""Tests for get_avg_predicted_values_ratios_over_window function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.forecasting.predictions import get_avg_predicted_values_ratios_over_window


class TestGetAvgValueRatiosOverWindow:
    """Test suite for get_avg_predicted_values_ratios_over_window function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(100, 200, 200),  # Linear growth
        })
        return df

    def test_basic_ratio_calculation(self, sample_data):
        """Test basic ratio calculation."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=10
        )

        assert isinstance(result, list)
        assert len(result) == 10
        # Ratios should be > 1 for increasing prices
        assert all(r > 1 or np.isnan(r) for r in result)

    def test_horizon_length(self, sample_data):
        """Test that result has correct horizon length."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        horizon = 30
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=5,
            df_line_col='price',
            horizon=horizon
        )

        assert len(result) == horizon

    def test_zero_range_days(self, sample_data):
        """Test with zero range_days (single date)."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=0,
            df_line_col='price',
            horizon=10
        )

        assert len(result) == 10
        # Should still return valid ratios
        assert any(not np.isnan(r) for r in result)

    def test_center_date_not_in_data(self, sample_data):
        """Test with center date not in the data."""
        center_date = datetime(2025, 1, 1)  # Future date
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=10
        )

        # Should return all NaN when no data available
        assert all(np.isnan(r) for r in result)

    def test_horizon_beyond_data(self, sample_data):
        """Test when horizon extends beyond available data."""
        center_date = datetime(2024, 1, 1) + timedelta(days=149)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=100
        )

        assert len(result) == 100
        # Later values should be NaN when data runs out
        assert np.isnan(result[-1])

    def test_constant_prices(self):
        """Test with constant prices."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            'date': dates,
            'price': [100] * 100,  # Constant price
        })

        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=df,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=10
        )

        # Ratios should be 1.0 for constant prices
        assert all(np.isclose(r, 1.0) or np.isnan(r) for r in result)

    def test_range_days_larger_than_data(self, sample_data):
        """Test with range_days larger than available data."""
        center_date = datetime(2024, 1, 10)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=500,
            df_line_col='price',
            horizon=10
        )

        assert len(result) == 10
        # Should still compute ratios from available dates
        assert isinstance(result, list)

    def test_negative_horizon(self, sample_data):
        """Test error handling with invalid horizon."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        with pytest.raises((ValueError, IndexError)):
            get_avg_predicted_values_ratios_over_window(
                df_line=sample_data,
                center_date=center_date,
                range_days=3,
                df_line_col='price',
                horizon=-10
            )

    def test_single_day_horizon(self, sample_data):
        """Test with horizon of 1 day."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=1
        )

        assert len(result) == 1
        assert not np.isnan(result[0])

    def test_missing_data_column(self, sample_data):
        """Test with non-existent column name."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        with pytest.raises(KeyError):
            get_avg_predicted_values_ratios_over_window(
                df_line=sample_data,
                center_date=center_date,
                range_days=3,
                df_line_col='nonexistent',
                horizon=10
            )

    def test_window_averaging(self, sample_data):
        """Test that ratios are averaged across window."""
        center_date = datetime(2024, 1, 1) + timedelta(days=49)

        # Single date result
        result_single = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=0,
            df_line_col='price',
            horizon=10
        )

        # Window result
        result_window = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=10,
            df_line_col='price',
            horizon=10
        )

        # Results should be different due to averaging
        assert not np.allclose(result_single, result_window, equal_nan=True)

    def test_sparse_data_with_gaps(self):
        """Test with data that has gaps."""
        dates = [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
            datetime(2024, 1, 5),  # Gap
            datetime(2024, 1, 6),
            datetime(2024, 1, 10),  # Gap
        ]
        df = pd.DataFrame({
            'date': dates,
            'price': [100, 101, 105, 106, 110],
        })

        center_date = datetime(2024, 1, 2)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=df,
            center_date=center_date,
            range_days=1,
            df_line_col='price',
            horizon=10
        )

        # Should handle gaps with NaN values
        assert len(result) == 10
        assert any(np.isnan(r) for r in result)

    def test_decreasing_prices(self):
        """Test with decreasing prices."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(200, 100, 100),  # Decreasing
        })

        center_date = datetime(2024, 1, 1) + timedelta(days=49)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=df,
            center_date=center_date,
            range_days=3,
            df_line_col='price',
            horizon=10
        )

        # Ratios should be < 1 for decreasing prices
        assert all(r < 1 or np.isnan(r) for r in result)

    def test_large_range_days(self, sample_data):
        """Test with large range_days."""
        center_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = get_avg_predicted_values_ratios_over_window(
            df_line=sample_data,
            center_date=center_date,
            range_days=50,
            df_line_col='price',
            horizon=20
        )

        assert len(result) == 20
        assert any(not np.isnan(r) for r in result)
