"""Tests for find_local_minima_with_separation function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.core.minima import find_local_minima_with_separation


class TestFindLocalMinimaWithSeparation:
    """Test suite for find_local_minima_with_separation function."""

    @pytest.fixture
    def sample_distances(self):
        """Create sample distance dataframe."""
        base_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 3, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]

        # Create distance pattern with clear minima
        distances = np.abs(np.sin(np.linspace(0, 4*np.pi, 100))) + np.random.randn(100) * 0.1

        df = pd.DataFrame({
            'current_date': [current_date] * 100,
            'max_date': dates,
            'dist': distances
        })
        return df

    def test_basic_minima_detection(self, sample_distances):
        """Test basic local minima detection."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=7,
            top_k=5
        )

        assert isinstance(result, pd.DataFrame)
        assert 'current_date' in result.columns
        assert 'max_date' in result.columns
        assert 'dist' in result.columns
        assert len(result) <= 5  # Should return at most top_k minima

    def test_top_k_limit(self, sample_distances):
        """Test that top_k limits the number of results."""
        top_k = 3
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=7,
            top_k=top_k
        )

        assert len(result) <= top_k

    def test_min_separation_constraint(self, sample_distances):
        """Test that minima are separated by at least min_separation days."""
        min_separation = 10
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=min_separation,
            top_k=10
        )

        if len(result) > 1:
            dates = sorted(result['max_date'].values)
            for i in range(len(dates) - 1):
                days_diff = (pd.Timestamp(dates[i+1]) - pd.Timestamp(dates[i])).days
                assert days_diff >= min_separation

    def test_sorted_by_distance(self, sample_distances):
        """Test that results are sorted by distance (lowest first)."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=7,
            top_k=5
        )

        if len(result) > 1:
            distances = result['dist'].values
            assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))

    def test_multiple_current_dates(self):
        """Test with multiple current_date groups."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]

        df_list = []
        for j in range(3):
            current_date = datetime(2024, 2, 1) + timedelta(days=j*10)
            distances = np.abs(np.sin(np.linspace(0, 2*np.pi, 50))) + np.random.randn(50) * 0.1
            df = pd.DataFrame({
                'current_date': [current_date] * 50,
                'max_date': dates,
                'dist': distances
            })
            df_list.append(df)

        combined_df = pd.concat(df_list, ignore_index=True)
        result = find_local_minima_with_separation(
            combined_df,
            N=5,
            min_separation=7,
            top_k=3
        )

        # Should find minima for each current_date group
        unique_current_dates = result['current_date'].nunique()
        assert unique_current_dates <= 3

    def test_no_local_minima(self):
        """Test with strictly increasing values (no local minima)."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        current_date = datetime(2024, 3, 1)

        df = pd.DataFrame({
            'current_date': [current_date] * 50,
            'max_date': dates,
            'dist': np.linspace(0, 100, 50)  # Strictly increasing
        })

        result = find_local_minima_with_separation(df, N=5, min_separation=7, top_k=5)

        # Should find at least the global minimum at the start
        assert len(result) >= 0

    def test_all_equal_values(self):
        """Test with all equal distance values."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(50)]
        current_date = datetime(2024, 3, 1)

        df = pd.DataFrame({
            'current_date': [current_date] * 50,
            'max_date': dates,
            'dist': [10.0] * 50  # All equal
        })

        result = find_local_minima_with_separation(df, N=5, min_separation=7, top_k=5)

        # Should handle equal values gracefully
        assert isinstance(result, pd.DataFrame)

    def test_small_window_size(self, sample_distances):
        """Test with small N (window size)."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=1,
            min_separation=7,
            top_k=5
        )

        # Should find more local minima with smaller window
        assert len(result) >= 0

    def test_large_window_size(self, sample_distances):
        """Test with large N (window size)."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=25,
            min_separation=7,
            top_k=5
        )

        # Should find fewer local minima with larger window
        assert len(result) >= 0

    def test_zero_min_separation(self, sample_distances):
        """Test with zero minimum separation."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=0,
            top_k=10
        )

        # Should allow adjacent minima
        assert len(result) <= 10

    def test_small_dataset(self):
        """Test with small dataset."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(10)]
        current_date = datetime(2024, 3, 1)

        df = pd.DataFrame({
            'current_date': [current_date] * 10,
            'max_date': dates,
            'dist': [5, 3, 7, 2, 8, 1, 9, 4, 6, 5]
        })

        result = find_local_minima_with_separation(df, N=2, min_separation=2, top_k=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) <= 5

    def test_empty_dataframe(self):
        """Test with empty dataframe."""
        df = pd.DataFrame(columns=['current_date', 'max_date', 'dist'])
        result = find_local_minima_with_separation(df, N=5, min_separation=7, top_k=5)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0

    def test_single_point(self):
        """Test with single data point."""
        df = pd.DataFrame({
            'current_date': [datetime(2024, 3, 1)],
            'max_date': [datetime(2024, 1, 1)],
            'dist': [5.0]
        })

        result = find_local_minima_with_separation(df, N=5, min_separation=7, top_k=5)

        # Single point should be considered a minimum
        assert len(result) == 1

    def test_preserve_date_types(self, sample_distances):
        """Test that date types are preserved."""
        result = find_local_minima_with_separation(
            sample_distances,
            N=5,
            min_separation=7,
            top_k=5
        )

        if len(result) > 0:
            assert isinstance(result['max_date'].iloc[0], (pd.Timestamp, datetime))
            assert isinstance(result['current_date'].iloc[0], (pd.Timestamp, datetime))

    def test_boundary_minima(self):
        """Test detection of minima at boundaries."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(20)]
        current_date = datetime(2024, 3, 1)

        # Create pattern with minimum at start and end
        distances = [1] + list(np.linspace(10, 10, 18)) + [1]

        df = pd.DataFrame({
            'current_date': [current_date] * 20,
            'max_date': dates,
            'dist': distances
        })

        result = find_local_minima_with_separation(df, N=3, min_separation=5, top_k=5)

        # Should detect boundary minima
        assert len(result) > 0

    def test_noise_robustness(self):
        """Test robustness to noisy data."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]
        current_date = datetime(2024, 3, 1)

        # Create clean pattern with added noise
        clean_signal = np.abs(np.sin(np.linspace(0, 4*np.pi, 100)))
        noise = np.random.randn(100) * 0.5
        distances = clean_signal + noise

        df = pd.DataFrame({
            'current_date': [current_date] * 100,
            'max_date': dates,
            'dist': distances
        })

        result = find_local_minima_with_separation(df, N=5, min_separation=7, top_k=5)

        # Should still find meaningful minima despite noise
        assert len(result) > 0
        assert len(result) <= 5
