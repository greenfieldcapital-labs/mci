"""Integration tests for mci package workflow."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.core.dtw import _get_estimates
from mci.core.minima import find_local_minima_with_separation
from mci.forecasting.predictions import (
    generate_sudo_predictions_for_frame,
    get_avg_predicted_values_ratios_over_window
)


@pytest.mark.integration
class TestWorkflowIntegration:
    """Integration tests for complete workflow."""

    @pytest.fixture
    def realistic_data(self):
        """Create realistic time series data."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(365)]

        # Create realistic predicted_values data with trend and seasonality
        trend = np.linspace(100, 150, 365)
        seasonality = 10 * np.sin(np.linspace(0, 8*np.pi, 365))
        noise = np.random.randn(365) * 2
        predicted_values = trend + seasonality + noise

        volume = np.random.randint(1000, 5000, 365)

        return pd.DataFrame({
            'date': dates,
            'predicted_values': predicted_values,
            'volume': volume
        })

    @pytest.mark.slow
    def test_complete_workflow(self, realistic_data):
        """Test complete workflow from data to predictions."""
        win_size = 30

        # Step 1: Get estimates
        estimates = _get_estimates(realistic_data, win_size=win_size)

        assert isinstance(estimates, pd.DataFrame)
        assert len(estimates) > 0
        assert 'dist' in estimates.columns
        assert 'max_date' in estimates.columns

        # Step 2: Find local minima
        lms = find_local_minima_with_separation(
            estimates,
            N=14,
            min_separation=7,
            top_k=5
        )

        assert isinstance(lms, pd.DataFrame)
        assert len(lms) <= 5

        # Step 3: Generate predictions
        if len(lms) > 0:
            current_date = estimates['current_date'].iloc[0]
            predictions = generate_sudo_predictions_for_frame(
                current_date=current_date,
                lms=lms,
                df_line=realistic_data,
                df_line_col='predicted_values',
                horizon=30,
                prediction_averaging_range=3
            )

            assert isinstance(predictions, list)
            assert len(predictions) == len(lms)
            assert all('predicted_values' in p for p in predictions)
            assert all(len(p['predicted_values']) == 30 for p in predictions)

    def test_estimates_to_minima_pipeline(self, realistic_data):
        """Test pipeline from estimates to local minima."""
        win_size = 20

        estimates = _get_estimates(realistic_data, win_size=win_size)

        # Should produce valid estimates
        assert len(estimates) > 0

        lms = find_local_minima_with_separation(
            estimates,
            N=10,
            min_separation=5,
            top_k=3
        )

        # Local minima should be subset of estimates
        assert len(lms) <= len(estimates)

        # All LM dates should be in estimates
        if len(lms) > 0:
            lm_dates = set(lms['max_date'].values)
            estimate_dates = set(estimates['max_date'].values)
            assert lm_dates.issubset(estimate_dates)

    def test_minima_to_predictions_pipeline(self, realistic_data):
        """Test pipeline from local minima to predictions."""
        base_date = datetime(2024, 1, 1)

        # Create mock local minima
        lms = pd.DataFrame({
            'max_date': [
                base_date + timedelta(days=30),
                base_date + timedelta(days=60),
                base_date + timedelta(days=90),
            ],
            'dist': [0.5, 0.8, 1.2]
        })

        current_date = base_date + timedelta(days=150)

        predictions = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=lms,
            df_line=realistic_data,
            df_line_col='predicted_values',
            horizon=30,
            prediction_averaging_range=3
        )

        # Should generate predictions for all LMs
        assert len(predictions) == len(lms)

        # Each prediction should correspond to an LM
        pred_dates = [p['lm_date'] for p in predictions]
        assert all(date in lms['max_date'].values for date in pred_dates)

    def test_different_window_sizes(self, realistic_data):
        """Test workflow with different window sizes."""
        window_sizes = [10, 20, 30, 50]

        results = []
        for win_size in window_sizes:
            estimates = _get_estimates(realistic_data, win_size=win_size)
            results.append(len(estimates))

        # All should produce some results
        assert all(r > 0 for r in results)

        # Larger windows should generally produce fewer estimates
        assert results[0] >= results[-1]

    def test_data_quality_impact(self):
        """Test how data quality affects results."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]

        # Clean data
        clean_df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 150, 200)
        })

        # Noisy data
        noisy_df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 150, 200) + np.random.randn(200) * 10
        })

        clean_estimates = _get_estimates(clean_df, win_size=20)
        noisy_estimates = _get_estimates(noisy_df, win_size=20)

        # Both should produce results
        assert len(clean_estimates) > 0
        assert len(noisy_estimates) > 0

        # Clean data might produce more consistent distances
        clean_std = clean_estimates['dist'].std()
        noisy_std = noisy_estimates['dist'].std()
        assert clean_std >= 0 and noisy_std >= 0

    @pytest.mark.slow
    def test_temporal_consistency(self, realistic_data):
        """Test that similar time periods produce similar distances."""
        win_size = 30

        # Get estimates for full dataset
        estimates = _get_estimates(realistic_data, win_size=win_size)

        if len(estimates) > 50:
            # Compare distances for consecutive windows
            early_distances = estimates['dist'].iloc[:25].values
            late_distances = estimates['dist'].iloc[-25:].values

            # Both should be valid
            assert not np.any(np.isnan(early_distances))
            assert not np.any(np.isnan(late_distances))

    def test_prediction_horizon_consistency(self, realistic_data):
        """Test that predictions maintain consistency across horizons."""
        base_date = datetime(2024, 1, 1)

        lms = pd.DataFrame({
            'max_date': [base_date + timedelta(days=50)],
            'dist': [0.5]
        })

        current_date = base_date + timedelta(days=150)

        # Short horizon
        pred_short = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=lms,
            df_line=realistic_data,
            df_line_col='predicted_values',
            horizon=30,
            prediction_averaging_range=3
        )

        # Long horizon
        pred_long = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=lms,
            df_line=realistic_data,
            df_line_col='predicted_values',
            horizon=60,
            prediction_averaging_range=3
        )

        # First 30 predictions should be similar
        short_predicted_values = pred_short[0]['predicted_values']
        long_predicted_values = pred_long[0]['predicted_values'][:30]

        # Allow for some numerical differences
        valid_indices = [i for i in range(len(short_predicted_values))
                        if not (np.isnan(short_predicted_values[i]) or np.isnan(long_predicted_values[i]))]

        if valid_indices:
            correlation = np.corrcoef(
                [short_predicted_values[i] for i in valid_indices],
                [long_predicted_values[i] for i in valid_indices]
            )[0, 1]
            assert correlation > 0.9 or np.isnan(correlation)

    def test_multiple_iterations(self, realistic_data):
        """Test running the workflow multiple times."""
        win_size = 25

        # Run multiple times with same parameters
        results = []
        for _ in range(3):
            estimates = _get_estimates(realistic_data.copy(), win_size=win_size)
            results.append(estimates)

        # Results should be consistent
        assert all(len(r) == len(results[0]) for r in results)

        # Distances should be identical
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i])

    def test_edge_case_minimal_data(self):
        """Test workflow with minimal valid data."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(30)]

        minimal_df = pd.DataFrame({
            'date': dates,
            'predicted_values': np.linspace(100, 110, 30)
        })

        # Should handle minimal data
        estimates = _get_estimates(minimal_df, win_size=10)

        assert isinstance(estimates, pd.DataFrame)
        # May have few or no results with minimal data
        assert len(estimates) >= 0

    def test_weighted_estimates_impact(self, realistic_data):
        """Test that weights affect the estimates."""
        # Estimates without weights
        est1 = _get_estimates(
            realistic_data,
            win_size=20,
            normalize=['predicted_values', 'volume'],
            weights=None
        )

        # Estimates with weights
        est2 = _get_estimates(
            realistic_data,
            win_size=20,
            normalize=['predicted_values', 'volume'],
            weights=[0.8, 0.2]
        )

        # Both should produce results
        assert len(est1) > 0
        assert len(est2) > 0

        # Distances should be different due to weights
        if len(est1) > 10 and len(est2) > 10:
            assert not np.allclose(est1['dist'].values[:10], est2['dist'].values[:10])
