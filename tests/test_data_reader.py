"""Basic sanity tests for DataReader class."""

from datetime import datetime, timedelta

import polars as pl
import pytest

pytestmark = pytest.mark.integration


class TestDataReaderSF1:
    """Tests for SF1 fundamental data reading."""

    def test_read_sf1_basic(self, data_reader, latest_sf1_snapshot):
        """Sanity check: Can read SF1 data when files exist."""
        # Use a 6-month date range to ensure we get data
        end = latest_sf1_snapshot
        start = (datetime.strptime(latest_sf1_snapshot, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")

        df = data_reader.read_sf1(start_date=start, end_date=end)

        # Basic assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0, "Should have some data"

        # Check required columns exist
        required_cols = ["ticker", "reportperiod", "datekey", "dimension"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_read_sf1_datekey_filtering(self, data_reader, latest_sf1_snapshot):
        """Sanity check: Datekey filtering works correctly."""
        start = (datetime.strptime(latest_sf1_snapshot, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        end = latest_sf1_snapshot

        df = data_reader.read_sf1(start_date=start, end_date=end)

        # All datekeys should be within the range
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()

        datekeys = df.select("datekey").unique()
        for row in datekeys.iter_rows(named=True):
            datekey = row["datekey"]
            assert start_dt <= datekey <= end_dt, \
                f"Datekey {datekey} outside range {start_dt} to {end_dt}"

    def test_read_sf1_deduplication(self, data_reader, latest_sf1_snapshot):
        """Sanity check: Deduplication removes duplicate (ticker, reportperiod) pairs."""
        start = (datetime.strptime(latest_sf1_snapshot, "%Y-%m-%d") - timedelta(days=60)).strftime("%Y-%m-%d")
        end = latest_sf1_snapshot

        df = data_reader.read_sf1(start_date=start, end_date=end)

        # Check no duplicates exist
        duplicates = df.group_by(["ticker", "reportperiod"]).agg(
            pl.len().alias("count")
        ).filter(pl.col("count") > 1)

        assert len(duplicates) == 0, \
            f"Found {len(duplicates)} duplicate (ticker, reportperiod) pairs after deduplication"

    def test_read_sf1_with_max_delay(self, data_reader, latest_sf1_snapshot):
        """Sanity check: max_data_delay_days parameter filters correctly."""
        start = (datetime.strptime(latest_sf1_snapshot, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
        end = latest_sf1_snapshot

        # Read with and without delay filter
        df_no_filter = data_reader.read_sf1(
            start_date=start,
            end_date=end,
            max_data_delay_days=None
        )

        df_with_filter = data_reader.read_sf1(
            start_date=start,
            end_date=end,
            max_data_delay_days=60  # 60 days max delay
        )

        # With filter should have <= records than without
        assert len(df_with_filter) <= len(df_no_filter), \
            "Delay filter should reduce or maintain record count"

    def test_read_sf1_empty_result(self, data_reader, sf1_files):
        """Sanity check: Returns empty DataFrame when no data matches filters."""
        if not sf1_files:
            pytest.skip("No SF1 data files found")

        # Use dates in the far future that won't have any datekeys
        future_start = (datetime.now() + timedelta(days=365*10)).strftime("%Y-%m-%d")
        future_end = (datetime.now() + timedelta(days=365*10 + 30)).strftime("%Y-%m-%d")

        # Should return empty DataFrame, not raise error (uses latest snapshot)
        df = data_reader.read_sf1(start_date=future_start, end_date=future_end)
        assert len(df) == 0, "Should have no data for future dates"


class TestDataReaderSEP:
    """Tests for SEP price data reading."""

    def test_read_sep_basic(self, data_reader, first_sep_month):
        """Sanity check: Can read SEP data when files exist."""
        # Read first half of the month
        start_date = f"{first_sep_month}-01"
        end_date = f"{first_sep_month}-15"

        df = data_reader.read_sep(start_date=start_date, end_date=end_date)

        # Basic assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0, "Should have some price data"

        # Check required columns
        required_cols = ["ticker", "date", "close"]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"

    def test_read_sep_date_filtering(self, data_reader, first_sep_month):
        """Sanity check: Date filtering works correctly for SEP."""
        start = f"{first_sep_month}-01"
        end = f"{first_sep_month}-15"

        df = data_reader.read_sep(start_date=start, end_date=end)

        # All dates should be within range
        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end, "%Y-%m-%d").date()

        dates = df.select("date").unique()
        for row in dates.iter_rows(named=True):
            date = row["date"]
            assert start_dt <= date <= end_dt, \
                f"Date {date} outside range {start_dt} to {end_dt}"

    def test_read_sep_missing_files_error(self, data_reader):
        """Sanity check: Appropriate error when no SEP files exist for date range."""
        # Try to read from year 1900 where no data should exist
        with pytest.raises(FileNotFoundError):
            data_reader.read_sep(start_date="1900-01-01", end_date="1900-01-31")


class TestDataReaderTickers:
    """Tests for TICKERS metadata reading."""

    def test_read_tickers_basic(self, data_reader, latest_ticker_snapshot):
        """Sanity check: Can read tickers metadata when files exist."""
        df = data_reader.read_tickers(snapshot_date=latest_ticker_snapshot)

        # Basic assertions
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0, "Should have ticker data"

        # Check required columns
        assert "ticker" in df.columns, "Missing ticker column"

    def test_read_tickers_file_selection(self, data_reader, ticker_files):
        """Sanity check: Selects most recent file <= snapshot_date."""
        if len(ticker_files) < 2:
            pytest.skip("Need at least 2 ticker files for this test")

        # Get the second-to-last file
        target_file = ticker_files[-2]
        snapshot_date = target_file.stem.replace("tickers_snapshot_", "")

        df = data_reader.read_tickers(snapshot_date=snapshot_date)

        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_read_tickers_missing_file_error(self, data_reader):
        """Sanity check: Appropriate error when no ticker file exists for date."""
        # Try to read from a very old date
        with pytest.raises(FileNotFoundError):
            data_reader.read_tickers(snapshot_date="1900-01-01")


class TestDataReaderIntegration:
    """Integration tests checking data consistency."""

    def test_sf1_and_tickers_consistency(self, data_reader, latest_sf1_snapshot):
        """Sanity check: Tickers in SF1 data should exist in TICKERS metadata."""
        # Use 6-month range to ensure we get data
        start = (datetime.strptime(latest_sf1_snapshot, "%Y-%m-%d") - timedelta(days=180)).strftime("%Y-%m-%d")
        end = latest_sf1_snapshot

        # Read small sample of SF1 data
        df_sf1 = data_reader.read_sf1(start_date=start, end_date=end)

        # Read tickers
        df_tickers = data_reader.read_tickers(snapshot_date=latest_sf1_snapshot)

        # Get unique tickers from both
        sf1_tickers = set(df_sf1["ticker"].unique().to_list())
        all_tickers = set(df_tickers["ticker"].unique().to_list())

        # SF1 tickers should be a subset of all tickers (allowing for some drift)
        # At least 80% should match
        matching = len(sf1_tickers & all_tickers)
        coverage = matching / len(sf1_tickers) if len(sf1_tickers) > 0 else 0

        assert coverage > 0.8, \
            f"Only {coverage*100:.1f}% of SF1 tickers found in TICKERS metadata"
