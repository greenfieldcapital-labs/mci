"""Pytest configuration and shared fixtures."""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def basic_time_series():
    """Create basic time series data for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(100)]
    values = np.linspace(100, 200, 100)
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def noisy_time_series():
    """Create noisy time series data for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(100)]
    values = np.linspace(100, 200, 100) + np.random.randn(100) * 5
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def cyclic_time_series():
    """Create cyclic time series data for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(200)]
    values = 100 + np.sin(np.linspace(0, 4*np.pi, 200)) * 20
    return pd.DataFrame({'date': dates, 'value': values})


@pytest.fixture
def multi_column_time_series():
    """Create multi-column time series data for testing."""
    base_date = datetime(2024, 1, 1)
    dates = [base_date + timedelta(days=i) for i in range(100)]
    return pd.DataFrame({
        'date': dates,
        'price': np.linspace(100, 200, 100),
        'volume': np.linspace(1000, 2000, 100),
        'volatility': np.random.randn(100).cumsum() + 10
    })


@pytest.fixture
def sparse_time_series():
    """Create time series with missing dates."""
    dates = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 2),
        datetime(2024, 1, 5),
        datetime(2024, 1, 6),
        datetime(2024, 1, 10),
        datetime(2024, 1, 15),
    ]
    values = [100, 102, 108, 110, 115, 120]
    return pd.DataFrame({'date': dates, 'value': values})


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
