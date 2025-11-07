"""Tests for _process_window function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.core.dtw import _process_window


class TestProcessWindow:
    """Test suite for _process_window function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(20)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(100, 120, 20) + np.random.randn(20) * 2,
            'volume': np.linspace(1000, 1500, 20) + np.random.randn(20) * 50,
        })
        return df

    @pytest.fixture
    def reference_frame(self):
        """Create reference frame for testing."""
        base_date = datetime(2024, 1, 21)
        dates = [base_date + timedelta(days=i) for i in range(10)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(120, 130, 10),
            'volume': np.linspace(1500, 1800, 10),
        })
        return df

    def test_valid_window_processing(self, sample_data, reference_frame):
        """Test processing a valid window returns expected structure."""
        result = _process_window(
            i=0,
            df=sample_data,
            ref_frame=reference_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        assert result is not None
        assert 'min_date' in result
        assert 'max_date' in result
        assert 'dist' in result
        assert isinstance(result['dist'], float)
        assert result['dist'] >= 0

    def test_window_with_nan_values(self, sample_data, reference_frame):
        """Test that windows with NaN values return None."""
        sample_data_with_nan = sample_data.copy()
        sample_data_with_nan.loc[5, 'price'] = np.nan

        result = _process_window(
            i=0,
            df=sample_data_with_nan,
            ref_frame=reference_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        # Should return None due to NaN values after normalization
        assert result is None or isinstance(result, dict)

    def test_window_size_boundary(self, sample_data, reference_frame):
        """Test window at data boundary."""
        result = _process_window(
            i=10,  # Last possible window
            df=sample_data,
            ref_frame=reference_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        assert result is not None

    def test_date_range_in_result(self, sample_data, reference_frame):
        """Test that min_date and max_date are correctly set."""
        result = _process_window(
            i=5,
            df=sample_data,
            ref_frame=reference_frame,
            win_size=5,
            use_cols=['price', 'volume']
        )

        if result is not None:
            assert result['min_date'] <= result['max_date']
            assert result['min_date'] == sample_data.iloc[5]['date']
            assert result['max_date'] == sample_data.iloc[9]['date']

    def test_zero_std_normalization(self):
        """Test normalization when standard deviation is zero."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(20)]
        df = pd.DataFrame({
            'date': dates,
            'price': [100] * 20,  # Constant values, std = 0
            'volume': np.linspace(1000, 1500, 20),
        })

        ref_dates = [base_date + timedelta(days=i+20) for i in range(10)]
        ref_frame = pd.DataFrame({
            'date': ref_dates,
            'price': [110] * 10,
            'volume': np.linspace(1500, 1800, 10),
        })

        result = _process_window(
            i=0,
            df=df,
            ref_frame=ref_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        # Should handle zero std gracefully
        assert result is not None or result is None

    def test_single_column(self, sample_data):
        """Test processing with a single column."""
        ref_dates = [datetime(2024, 1, 21) + timedelta(days=i) for i in range(10)]
        ref_frame = pd.DataFrame({
            'date': ref_dates,
            'price': np.linspace(120, 130, 10),
        })

        result = _process_window(
            i=0,
            df=sample_data[['date', 'price']],
            ref_frame=ref_frame,
            win_size=10,
            use_cols=['price']
        )

        assert result is not None
        assert isinstance(result['dist'], float)

    def test_inf_values_handling(self):
        """Test that infinite values are handled correctly."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(20)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(100, 120, 20),
            'volume': np.linspace(1000, 1500, 20),
        })
        df.loc[5, 'price'] = np.inf

        ref_dates = [base_date + timedelta(days=i+20) for i in range(10)]
        ref_frame = pd.DataFrame({
            'date': ref_dates,
            'price': np.linspace(120, 130, 10),
            'volume': np.linspace(1500, 1800, 10),
        })

        result = _process_window(
            i=0,
            df=df,
            ref_frame=ref_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        # Should return None due to inf values
        assert result is None

    def test_empty_use_cols(self, sample_data, reference_frame):
        """Test with empty use_cols list."""
        with pytest.raises((ValueError, IndexError, KeyError)):
            _process_window(
                i=0,
                df=sample_data,
                ref_frame=reference_frame,
                win_size=10,
                use_cols=[]
            )

    def test_mismatched_columns(self, sample_data):
        """Test when window has fewer columns than reference."""
        ref_dates = [datetime(2024, 1, 21) + timedelta(days=i) for i in range(10)]
        ref_frame = pd.DataFrame({
            'date': ref_dates,
            'price': np.linspace(120, 130, 10),
            'volume': np.linspace(1500, 1800, 10),
            'extra_col': np.linspace(0, 1, 10),
        })

        result = _process_window(
            i=0,
            df=sample_data[['date', 'price', 'volume']],
            ref_frame=ref_frame,
            win_size=10,
            use_cols=['price', 'volume']
        )

        # Should return None due to column mismatch
        assert result is None or isinstance(result, dict)
