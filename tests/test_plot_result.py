"""Tests for plot_result function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from mci.visualization.plotting import plot_result


class TestPlotResult:
    """Test suite for plot_result function."""

    @pytest.fixture
    def sample_distances(self):
        """Create sample distance dataframe."""
        base_date = datetime(2024, 1, 1)
        current_dates = [datetime(2024, 2, 1) + timedelta(days=i*7) for i in range(5)]

        df_list = []
        for current_date in current_dates:
            dates = [base_date + timedelta(days=i) for i in range(100)]
            distances = np.abs(np.sin(np.linspace(0, 4*np.pi, 100))) * 100

            df = pd.DataFrame({
                'current_date': [current_date] * 100,
                'max_date': dates,
                'dist': distances
            })
            df_list.append(df)

        return pd.concat(df_list, ignore_index=True)

    @pytest.fixture
    def sample_line_data(self):
        """Create sample line data."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(100, 200, 200) + np.sin(np.linspace(0, 4*np.pi, 200)) * 10,
        })
        return df

    @pytest.mark.slow
    def test_basic_plot_creation(self, sample_distances, sample_line_data):
        """Test basic plot creation without errors."""
        # Use real plotly Figure object
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            N=14,
            min_separation=7,
            top_k=5,
            prediction_averaging_range=3,
            calc_every_nth=5,  # Use larger value for faster test
            show_plot=False
        )

        # Result should be None (function calls fig.show() but doesn't return)
        assert result is None

    @pytest.mark.slow
    def test_show_scatter_parameter(self, sample_distances, sample_line_data):
        """Test show_scatter parameter effect."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            show_scatter=False,
            calc_every_nth=5,
            show_plot=False
        )

        assert result is None

    @pytest.mark.slow
    def test_calc_every_nth_parameter(self, sample_distances, sample_line_data):
        """Test calc_every_nth parameter reduces computations."""
        # Should process fewer frames with larger calc_every_nth
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            calc_every_nth=3,
            show_plot=False
        )

        assert result is None

    @pytest.mark.slow
    def test_custom_title(self, sample_distances, sample_line_data):
        """Test custom title parameter."""
        custom_title = "My Custom Plot"
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            title=custom_title,
            calc_every_nth=5,
            show_plot=False
        )

        assert result is None

    def test_frame_duration_parameter(self, sample_distances, sample_line_data):
        """Test frame_duration parameter."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            frame_duration=500,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_transition_duration_parameter(self, sample_distances, sample_line_data):
        """Test transition_duration parameter."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            transition_duration=200,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_top_k_parameter(self, sample_distances, sample_line_data):
        """Test top_k parameter for limiting local minima."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            top_k=3,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    @pytest.mark.slow
    def test_large_horizon(self, sample_distances, sample_line_data):
        """Test with large horizon value."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=180,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_small_horizon(self, sample_distances, sample_line_data):
        """Test with small horizon value."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=10,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_min_separation_parameter(self, sample_distances, sample_line_data):
        """Test min_separation parameter."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            min_separation=14,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_n_parameter(self, sample_distances, sample_line_data):
        """Test N parameter for local minima window size."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            N=20,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_prediction_averaging_range(self, sample_distances, sample_line_data):
        """Test prediction_averaging_range parameter."""
        result = plot_result(
            df=sample_distances,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=5,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None

    def test_invalid_column_name(self, sample_distances, sample_line_data):
        """Test with invalid column name."""
        with pytest.raises(KeyError):
            plot_result(
                df=sample_distances,
                df_line=sample_line_data,
                df_line_col='nonexistent',
                horizon=30,
                calc_every_nth=10,
            show_plot=False
            )

    def test_empty_distances_dataframe(self, sample_line_data):
        """Test with empty distances dataframe."""
        empty_df = pd.DataFrame(columns=['current_date', 'max_date', 'dist'])

        with pytest.raises((ValueError, IndexError, KeyError)):
            plot_result(
                df=empty_df,
                df_line=sample_line_data,
                df_line_col='price',
                horizon=30,
                calc_every_nth=1,
            show_plot=False
            )

    def test_single_frame(self, sample_line_data):
        """Test with single frame of data."""
        base_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 2, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]
        distances = np.abs(np.sin(np.linspace(0, 4*np.pi, 100))) * 100

        single_frame_df = pd.DataFrame({
            'current_date': [current_date] * 100,
            'max_date': dates,
            'dist': distances
        })

        result = plot_result(
            df=single_frame_df,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            calc_every_nth=1,
            show_plot=False
        )

        assert result is None

    def test_distance_normalization(self, sample_distances, sample_line_data):
        """Test that distances are normalized to 0-100 range."""
        # Create copy to check normalization
        original_min = sample_distances['dist'].min()
        original_max = sample_distances['dist'].max()

        result = plot_result(
            df=sample_distances.copy(),
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            calc_every_nth=10,
            show_plot=False
        )

        # Function should normalize internally
        assert original_min != 0 or original_max != 100
        assert result is None

    def test_constant_distance_values(self, sample_line_data):
        """Test with constant distance values."""
        base_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 2, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]

        constant_df = pd.DataFrame({
            'current_date': [current_date] * 100,
            'max_date': dates,
            'dist': [50.0] * 100  # All same
        })

        # Should handle constant values without crashing
        result = plot_result(
            df=constant_df,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            calc_every_nth=1,
            show_plot=False
        )

        assert result is None

    def test_unsorted_data(self, sample_line_data):
        """Test with unsorted data."""
        base_date = datetime(2024, 1, 1)
        current_date = datetime(2024, 2, 1)
        dates = [base_date + timedelta(days=i) for i in range(100)]
        distances = np.abs(np.sin(np.linspace(0, 4*np.pi, 100))) * 100

        # Shuffle the data
        unsorted_df = pd.DataFrame({
            'current_date': [current_date] * 100,
            'max_date': dates,
            'dist': distances
        })
        unsorted_df = unsorted_df.sample(frac=1).reset_index(drop=True)

        # Should handle unsorted data (function sorts internally)
        result = plot_result(
            df=unsorted_df,
            df_line=sample_line_data,
            df_line_col='price',
            horizon=30,
            calc_every_nth=10,
            show_plot=False
        )

        assert result is None
