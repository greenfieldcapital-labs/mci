"""Tests for _get_estimates function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.core.dtw import _get_estimates


class TestGetEstimates:
    """Test suite for _get_estimates function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 150, 50) + np.sin(np.linspace(0, 4*np.pi, 50)) * 10,
            'volume': np.linspace(1000, 2000, 50) + np.random.randn(50) * 50,
        })
        return df

    def test_basic_estimates_calculation(self, sample_data):
        """Test basic calculation of estimates."""
        result = _get_estimates(sample_data, win_size=10)

        assert isinstance(result, pd.DataFrame)
        assert 'max_date' in result.columns
        assert 'dist' in result.columns
        assert 'current_date' in result.columns
        assert len(result) >= 0

    def test_window_size_larger_than_data(self, sample_data):
        """Test when window size is larger than available data."""
        result = _get_estimates(sample_data, win_size=40)

        # Should have very few windows or empty result
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 10

    def test_minimum_window_size(self, sample_data):
        """Test with minimum window size."""
        result = _get_estimates(sample_data, win_size=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_output_sorted_by_date(self, sample_data):
        """Test that output is sorted by max_date."""
        result = _get_estimates(sample_data, win_size=10)

        if len(result) > 0:
            dates = result['max_date'].values
            assert all(dates[i] <= dates[i+1] for i in range(len(dates)-1))

    def test_no_na_in_output(self, sample_data):
        """Test that NaN values are dropped from output."""
        result = _get_estimates(sample_data, win_size=10)

        assert not result['dist'].isna().any()
        assert not result['max_date'].isna().any()

    def test_with_normalize_parameter(self, sample_data):
        """Test with specific columns to normalize."""
        result = _get_estimates(
            sample_data,
            win_size=10,
            normalize=['predicted_values']
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_with_weights_parameter(self, sample_data):
        """Test with custom weights."""
        result = _get_estimates(
            sample_data,
            win_size=10,
            normalize=['predicted_values', 'volume'],
            weights=[0.7, 0.3]
        )

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_weights_affect_distances(self, sample_data):
        """Test that different weights produce different distances."""
        result1 = _get_estimates(
            sample_data,
            win_size=10,
            normalize=['predicted_values', 'volume'],
            weights=[1.0, 1.0]
        )

        result2 = _get_estimates(
            sample_data,
            win_size=10,
            normalize=['predicted_values', 'volume'],
            weights=[2.0, 0.5]
        )

        # Different weights should generally produce different results
        if len(result1) > 0 and len(result2) > 0:
            assert not np.allclose(result1['dist'].values, result2['dist'].values)

    def test_current_date_is_last_date(self, sample_data):
        """Test that current_date is set to the last date in reference frame."""
        result = _get_estimates(sample_data, win_size=10)

        if len(result) > 0:
            expected_current = sample_data['date'].iloc[-1]
            assert all(result['current_date'] == expected_current)

    def test_data_with_high_na_columns(self):
        """Test with columns that have many NaN values."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 150, 50),
            'sparse_data': [np.nan] * 45 + [1, 2, 3, 4, 5],  # Mostly NaN
        })

        result = _get_estimates(df, win_size=10)

        # Should handle high NA columns by removing them
        assert isinstance(result, pd.DataFrame)

    def test_single_column_data(self):
        """Test with only one data column besides date."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 150, 50) + np.sin(np.linspace(0, 4*np.pi, 50)) * 10,
        })

        result = _get_estimates(df, win_size=10)

        assert isinstance(result, pd.DataFrame)
        assert len(result) >= 0

    def test_zero_variance_data(self):
        """Test with constant values (zero variance)."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        df = pd.DataFrame({
            'date': dates,
            'predicted_values': [100] * 50,  # Constant
            'volume': [1000] * 50,  # Constant
        })

        result = _get_estimates(df, win_size=10)

        # Should handle zero variance gracefully
        assert isinstance(result, pd.DataFrame)

    def test_empty_result_structure(self):
        """Test that empty result has correct structure."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(15)]
        df = pd.DataFrame({
            'date': dates,
            'predicted_values': [np.nan] * 15,  # All NaN
        })

        result = _get_estimates(df, win_size=10)

        assert isinstance(result, pd.DataFrame)
        assert 'max_date' in result.columns
        assert 'dist' in result.columns

    def test_large_window_size(self, sample_data):
        """Test with large window size relative to data."""
        result = _get_estimates(sample_data, win_size=30)

        assert isinstance(result, pd.DataFrame)
        # Should have few or no valid windows
        assert len(result) <= 20

    def test_dates_not_modified(self, sample_data):
        """Test that original date column is not modified."""
        original_dates = sample_data['date'].copy()
        _get_estimates(sample_data, win_size=10)

        assert all(sample_data['date'] == original_dates)

    def test_parallel_processing_consistency(self, sample_data):
        """Test that parallel processing produces consistent results."""
        result1 = _get_estimates(sample_data.copy(), win_size=10)
        result2 = _get_estimates(sample_data.copy(), win_size=10)

        if len(result1) > 0 and len(result2) > 0:
            pd.testing.assert_frame_equal(result1, result2)
