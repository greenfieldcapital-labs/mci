"""Tests for generate_sudo_predictions_for_frame function."""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from mci.forecasting.predictions import generate_sudo_predictions_for_frame


class TestGenerateSudoPredictionsForFrame:
    """Test suite for generate_sudo_predictions_for_frame function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample dataframe for testing."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]
        df = pd.DataFrame({
            'date': dates,
            'price': np.linspace(100, 200, 200) + np.sin(np.linspace(0, 4*np.pi, 200)) * 10,
        })
        return df

    @pytest.fixture
    def sample_lms(self):
        """Create sample local minima dataframe."""
        base_date = datetime(2024, 1, 1)
        lms = pd.DataFrame({
            'max_date': [
                base_date + timedelta(days=20),
                base_date + timedelta(days=40),
                base_date + timedelta(days=60),
            ],
            'dist': [0.5, 0.8, 1.2]
        })
        return lms

    def test_basic_prediction_generation(self, sample_data, sample_lms):
        """Test basic prediction generation."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        assert isinstance(result, list)
        assert len(result) == len(sample_lms)

    def test_prediction_structure(self, sample_data, sample_lms):
        """Test that predictions have correct structure."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        for pred in result:
            assert 'lm_date' in pred
            assert 'predicted_prices' in pred
            assert 'ratios' in pred
            assert len(pred['predicted_prices']) == 30
            assert len(pred['ratios']) == 30

    def test_horizon_length(self, sample_data, sample_lms):
        """Test that predictions match specified horizon."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        horizon = 50
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=horizon,
            prediction_averaging_range=3
        )

        for pred in result:
            assert len(pred['predicted_prices']) == horizon
            assert len(pred['ratios']) == horizon

    def test_empty_lms(self, sample_data):
        """Test with empty local minima dataframe."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        empty_lms = pd.DataFrame(columns=['max_date', 'dist'])

        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=empty_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        assert isinstance(result, list)
        assert len(result) == 0

    def test_single_lm(self, sample_data):
        """Test with single local minimum."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        single_lm = pd.DataFrame({
            'max_date': [datetime(2024, 1, 1) + timedelta(days=49)],
            'dist': [0.5]
        })

        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=single_lm,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        assert len(result) == 1
        assert len(result[0]['predicted_prices']) == 30

    def test_current_date_not_in_data(self, sample_data, sample_lms):
        """Test with current_date not in the data."""
        current_date = datetime(2025, 1, 1)  # Future date

        with pytest.raises(AssertionError):
            generate_sudo_predictions_for_frame(
                current_date=current_date,
                lms=sample_lms,
                df_line=sample_data,
                df_line_col='price',
                horizon=30,
                prediction_averaging_range=3
            )

    def test_prediction_averaging_range_effect(self, sample_data, sample_lms):
        """Test that different averaging ranges produce different results."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)

        result1 = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=1
        )

        result2 = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=5
        )

        # Different averaging ranges should produce different predictions
        pred1 = result1[0]['predicted_prices']
        pred2 = result2[0]['predicted_prices']
        # Allow some similarity but they should differ
        assert not np.allclose(pred1, pred2, equal_nan=True) or np.any(np.isnan(pred1))

    def test_lm_date_preservation(self, sample_data, sample_lms):
        """Test that lm_date is correctly preserved in results."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        for i, pred in enumerate(result):
            assert pred['lm_date'] == sample_lms.iloc[i]['max_date']

    def test_predictions_based_on_current_price(self, sample_data, sample_lms):
        """Test that predictions are based on current date price."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        current_price = sample_data[sample_data['date'] == current_date]['price'].iloc[0]

        # Predicted prices should be related to current price via ratios
        for pred in result:
            for i, (price, ratio) in enumerate(zip(pred['predicted_prices'], pred['ratios'])):
                if not np.isnan(price) and not np.isnan(ratio):
                    expected_price = current_price * ratio
                    assert np.isclose(price, expected_price, rtol=1e-5)

    def test_nan_handling(self, sample_data):
        """Test handling of NaN values in predictions."""
        # Use an lm_date late in the dataset so horizon extends beyond data
        late_lms = pd.DataFrame({
            'max_date': [datetime(2024, 1, 1) + timedelta(days=150)],
            'dist': [0.5]
        })
        current_date = datetime(2024, 1, 1) + timedelta(days=100)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=late_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=100,  # Extends beyond data
            prediction_averaging_range=3
        )

        # Later predictions should contain NaN when data runs out
        # LM is at day 150, horizon 100 means looking at days 151-250
        # Data only goes to day 199, so days 200-250 should be NaN
        for pred in result:
            assert any(np.isnan(p) for p in pred['predicted_prices'][-10:])

    def test_zero_horizon(self, sample_data, sample_lms):
        """Test with horizon of zero."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        with pytest.raises((ValueError, IndexError)):
            generate_sudo_predictions_for_frame(
                current_date=current_date,
                lms=sample_lms,
                df_line=sample_data,
                df_line_col='price',
                horizon=0,
                prediction_averaging_range=3
            )

    def test_invalid_column_name(self, sample_data, sample_lms):
        """Test with invalid column name."""
        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        with pytest.raises(KeyError):
            generate_sudo_predictions_for_frame(
                current_date=current_date,
                lms=sample_lms,
                df_line=sample_data,
                df_line_col='nonexistent',
                horizon=30,
                prediction_averaging_range=3
            )

    def test_constant_prices(self, sample_lms):
        """Test with constant price data."""
        base_date = datetime(2024, 1, 1)
        dates = [base_date + timedelta(days=i) for i in range(200)]
        df = pd.DataFrame({
            'date': dates,
            'price': [100] * 200,  # Constant
        })

        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=df,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        # All predictions should be constant (ratio = 1.0)
        for pred in result:
            non_nan_prices = [p for p in pred['predicted_prices'] if not np.isnan(p)]
            if non_nan_prices:
                assert all(np.isclose(p, 100, rtol=0.01) for p in non_nan_prices)

    def test_large_horizon(self, sample_data, sample_lms):
        """Test with very large horizon."""
        current_date = datetime(2024, 1, 1) + timedelta(days=49)
        horizon = 200
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=sample_lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=horizon,
            prediction_averaging_range=3
        )

        for pred in result:
            assert len(pred['predicted_prices']) == horizon

    def test_multiple_lms_ordering(self, sample_data):
        """Test with multiple LMs in different order."""
        lms = pd.DataFrame({
            'max_date': [
                datetime(2024, 1, 1) + timedelta(days=59),
                datetime(2024, 1, 20),
                datetime(2024, 1, 1) + timedelta(days=39),
            ],
            'dist': [1.2, 0.5, 0.8]
        })

        current_date = datetime(2024, 1, 1) + timedelta(days=99)
        result = generate_sudo_predictions_for_frame(
            current_date=current_date,
            lms=lms,
            df_line=sample_data,
            df_line_col='price',
            horizon=30,
            prediction_averaging_range=3
        )

        # Should process all LMs regardless of order
        assert len(result) == len(lms)
        assert all(pred['lm_date'] in lms['max_date'].values for pred in result)
