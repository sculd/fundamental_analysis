"""Data reader for loading and filtering saved Sharadar data."""

from datetime import datetime
from typing import Optional

import polars as pl

from fundamental_analysis.utils.config import Config
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataReader:
    """Reads and filters locally stored financial data."""

    def __init__(self):
        """Initialize DataReader."""
        pass

    def read_tickers(
        self,
        snapshot_date: str,
    ) -> pl.DataFrame:
        """
        Read ticker metadata.

        Finds the most recent tickers file with date <= snapshot_date.
        """
        end_dt = datetime.strptime(snapshot_date, "%Y-%m-%d")

        # Find all tickers files
        tickers_files = sorted(Config.TICKERS_DIR.glob("tickers_snapshot_*.parquet"))

        if not tickers_files:
            raise FileNotFoundError(f"No tickers files found in {Config.TICKERS_DIR}")

        # Find the most recent file with date <= snapshot_date
        selected_file = None
        for file_path in tickers_files:
            file_date_str = file_path.stem.replace("tickers_snapshot_", "")
            file_dt = datetime.strptime(file_date_str, "%Y-%m-%d")

            if file_dt <= end_dt:
                selected_file = file_path
            else:
                break

        if selected_file is None:
            raise FileNotFoundError(
                f"No tickers file found with date <= {snapshot_date}"
            )

        logger.info(f"Reading tickers from {selected_file.name}")
        return pl.read_parquet(selected_file)

    def read_sf1(
        self,
        start_date: str,
        end_date: str,
        max_data_delay_days: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Read SF1 fundamental data filtered by datekey.

        Args:
            start_date: Start date for datekey filter (YYYY-MM-DD)
            end_date: End date for datekey filter (YYYY-MM-DD)
            max_data_delay_days: If set, exclude entries where
                (datekey - reportperiod) exceeds this value

        Returns:
            Filtered SF1 data. If same ticker has multiple datekeys for
            single reportperiod, only the first (earliest) datekey is kept.
        """
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")

        # Find the most recent snapshot file with date <= end_date
        snapshot_files = sorted(Config.SF1_DIR.glob("sf1_snapshot_*.parquet"))

        if not snapshot_files:
            raise FileNotFoundError(f"No SF1 snapshot files found in {Config.SF1_DIR}")

        selected_file = None
        for file_path in snapshot_files:
            file_date_str = file_path.stem.replace("sf1_snapshot_", "")
            file_dt = datetime.strptime(file_date_str, "%Y-%m-%d")

            if file_dt <= end_dt:
                selected_file = file_path
            else:
                break

        if selected_file is None:
            raise FileNotFoundError(
                f"No SF1 snapshot file found with date <= {end_date}"
            )

        logger.info(f"Reading SF1 from {selected_file.name}")
        df = pl.read_parquet(selected_file)

        # Filter by datekey
        df = df.filter(
            (pl.col("datekey") >= pl.lit(start_dt).cast(pl.Date)) &
            (pl.col("datekey") <= pl.lit(end_dt).cast(pl.Date))
        )

        logger.info(f"After datekey filter ({start_date} to {end_date}): {len(df)} records")

        # Filter by max data delay if specified
        if max_data_delay_days is not None:
            df = df.with_columns(
                (pl.col("datekey") - pl.col("reportperiod")).dt.total_days().alias("_delay_days")
            )

            original_count = len(df)
            df = df.filter(pl.col("_delay_days") <= max_data_delay_days)
            df = df.drop("_delay_days")

            logger.info(
                f"After max_data_delay_days filter ({max_data_delay_days} days): "
                f"{len(df)} records (removed {original_count - len(df)})"
            )

        # Keep only first datekey for each (ticker, reportperiod) combination
        df = df.sort(["ticker", "reportperiod", "datekey"])
        df = df.unique(subset=["ticker", "reportperiod"], keep="first")

        logger.info(f"After deduplication (first datekey per ticker-reportperiod): {len(df)} records")

        return df

    def read_sep(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """
        Read SEP price data filtered by date.

        Args:
            start_date: Start date for price data (YYYY-MM-DD)
            end_date: End date for price data (YYYY-MM-DD)

        Returns:
            SEP price data filtered to the date range
        """
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Determine which monthly files to read
        start_month = start_dt.strftime("%Y-%m")
        end_month = end_dt.strftime("%Y-%m")

        # Generate list of months to read
        current_dt = start_dt.replace(day=1)
        months_to_read = []

        while current_dt <= end_dt:
            month_str = current_dt.strftime("%Y-%m")
            file_path = Config.SEP_DIR / f"sep_{month_str}.parquet"

            if file_path.exists():
                months_to_read.append(file_path)

            # Move to next month
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)

        if not months_to_read:
            raise FileNotFoundError(
                f"No SEP files found for date range {start_date} to {end_date}"
            )

        logger.info(f"Reading {len(months_to_read)} monthly SEP files from {start_month} to {end_month}")

        # Read all monthly files and concatenate
        dfs = [pl.read_parquet(file_path) for file_path in months_to_read]
        df = pl.concat(dfs)

        # Filter by exact date range
        df = df.filter(
            (pl.col("date") >= pl.lit(start_dt).cast(pl.Date)) &
            (pl.col("date") <= pl.lit(end_dt).cast(pl.Date))
        )

        logger.info(f"After date filter ({start_date} to {end_date}): {len(df)} records")

        return df
