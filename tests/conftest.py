"""Pytest configuration and fixtures for tests."""

from pathlib import Path

import pytest

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.utils.config import Config


@pytest.fixture
def data_reader():
    """Fixture providing a DataReader instance."""
    return DataReader()


@pytest.fixture
def sf1_files():
    """Fixture providing list of SF1 snapshot files."""
    return sorted(Config.SF1_DIR.glob("sf1_snapshot_*.parquet"))


@pytest.fixture
def sep_files():
    """Fixture providing list of SEP data files."""
    return sorted(Config.SEP_DIR.glob("sep_*.parquet"))


@pytest.fixture
def ticker_files():
    """Fixture providing list of TICKERS snapshot files."""
    return sorted(Config.TICKERS_DIR.glob("tickers_snapshot_*.parquet"))


@pytest.fixture
def latest_sf1_snapshot(sf1_files):
    """Fixture providing the latest SF1 snapshot date."""
    if not sf1_files:
        pytest.skip("No SF1 data files found - run main_fetch.py first")
    latest_file = sf1_files[-1]
    return latest_file.stem.replace("sf1_snapshot_", "")


@pytest.fixture
def latest_ticker_snapshot(ticker_files):
    """Fixture providing the latest TICKERS snapshot date."""
    if not ticker_files:
        pytest.skip("No TICKERS data files found - run main_fetch.py first")
    latest_file = ticker_files[-1]
    return latest_file.stem.replace("tickers_snapshot_", "")


@pytest.fixture
def first_sep_month(sep_files):
    """Fixture providing the first available SEP month."""
    if not sep_files:
        pytest.skip("No SEP data files found - run main_fetch.py first")
    first_file = sep_files[0]
    return first_file.stem.replace("sep_", "")
