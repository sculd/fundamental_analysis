"""Data fetcher orchestration for downloading and saving Sharadar data."""

from datetime import datetime, timedelta
from typing import Optional

import polars as pl

from fundamental_analysis.data_acquisition.sharadar_client import \
    SharadarClient
from fundamental_analysis.utils.config import Config
from fundamental_analysis.utils.logger import setup_logger

_FETCH_START_DATE = "1998-01-01"
logger = setup_logger(__name__)


class DataFetcher:
    """Orchestrates fetching and saving financial data."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = SharadarClient(api_key)
        Config.validate()

    def fetch_and_save_tickers(
        self,
        end_date: str,
        overwrite: bool = False
    ) -> pl.DataFrame:
        """Fetch and save ticker metadata with end_date in filename."""
        logger.info("Fetching and saving TICKERS data...")

        output_path = Config.TICKERS_DIR / f"tickers_snapshot_{end_date}.parquet"

        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists. Skipping (use --overwrite to replace).")
            return pl.read_parquet(output_path)

        df_tickers = self.client.fetch_tickers()
        df_tickers.write_parquet(output_path)
        logger.info(f"Saved {len(df_tickers)} tickers to {output_path}")

        return df_tickers

    def fetch_and_save_sf1(
        self,
        end_date: str,
        tickers: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Fetch and save SF1 fundamentals as a point-in-time snapshot.

        Fetches all historical data from 1990-01-01 to (end_date - REPORTING_DELAY_DAYS).
        Filters by datekey to ensure data was actually available on the snapshot date.
        Saves as sf1_snapshot_{end_date}.parquet.
        """

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        adjusted_end_dt = end_dt - timedelta(days=Config.REPORTING_DELAY_DAYS)
        adjusted_end_date = adjusted_end_dt.strftime("%Y-%m-%d")

        logger.info(f"Creating SF1 snapshot as of {end_date}")
        logger.info(f"Applying {Config.REPORTING_DELAY_DAYS}-day reporting delay: data cutoff = {adjusted_end_date}")

        output_path = Config.SF1_DIR / f"sf1_snapshot_{end_date}.parquet"

        if output_path.exists() and not overwrite:
            logger.info(f"Snapshot {output_path.name} already exists. Skipping (use --overwrite to replace).")
            return

        logger.info(f"Fetching all historical data from {_FETCH_START_DATE} to {adjusted_end_date}")

        df_sf1 = self.client.fetch_sf1(
            start_date=_FETCH_START_DATE,
            end_date=adjusted_end_date,
            tickers=tickers,
        )

        if len(df_sf1) == 0:
            logger.warning("No SF1 data fetched. Skipping save.")
            return

        # Date fields in SF1:
        # - reportperiod: actual fiscal quarter end (varies by company)
        # - calendardate: normalized to calendar quarters (Mar/Jun/Sep/Dec 31)
        # - datekey: when data became available in Sharadar (critical for point-in-time)
        df_sf1 = df_sf1.with_columns([
            pl.col("calendardate").cast(pl.Date),
            pl.col("datekey").cast(pl.Date),
            pl.col("reportperiod").cast(pl.Date)
        ])

        logger.info(f"Fetched {len(df_sf1)} records, filtering by datekey <= {adjusted_end_date}")

        # Filter by datekey to ensure point-in-time correctness
        df_sf1 = df_sf1.filter(pl.col("datekey") <= pl.lit(adjusted_end_dt).cast(pl.Date))

        logger.info(f"After datekey filter: {len(df_sf1)} records")

        df_sf1.write_parquet(output_path)
        logger.info(f"Saved SF1 snapshot to {output_path}")

    def fetch_and_save_sep(
        self,
        end_date: str,
        tickers: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Fetch and save SEP price data, partitioned by month.

        Fetches from _FETCH_START_DATE to the last complete month before end_date.
        Excludes the incomplete month containing end_date.
        """

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        # Get last day of previous complete month
        first_day_current_month = end_dt.replace(day=1)
        last_day_previous_month = first_day_current_month - timedelta(days=1)

        logger.info(f"Fetching SEP price data from {_FETCH_START_DATE} to {last_day_previous_month.strftime('%Y-%m-%d')}")
        logger.info(f"Excluding incomplete month: {end_dt.strftime('%Y-%m')}")

        start_dt = datetime.strptime(_FETCH_START_DATE, "%Y-%m-%d")
        end_dt = last_day_previous_month

        current_dt = start_dt.replace(day=1)
        fetched_count = 0
        skipped_count = 0

        while current_dt <= end_dt:
            month_str = current_dt.strftime("%Y-%m")
            output_path = Config.SEP_DIR / f"sep_{month_str}.parquet"

            # Check if file exists and skip if overwrite is False
            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path.name} already exists. Skipping.")
                skipped_count += 1
            else:
                # Calculate first and last day of current month
                fetch_start = current_dt.strftime("%Y-%m-01")

                if current_dt.month == 12:
                    next_month = current_dt.replace(year=current_dt.year + 1, month=1, day=1)
                else:
                    next_month = current_dt.replace(month=current_dt.month + 1, day=1)
                last_day = next_month - timedelta(days=1)
                fetch_end = last_day.strftime("%Y-%m-%d")

                logger.info(f"Fetching SEP data for {month_str}: {fetch_start} to {fetch_end}")

                df_sep = self.client.fetch_sep(
                    start_date=fetch_start,
                    end_date=fetch_end,
                    tickers=tickers,
                )

                if len(df_sep) == 0:
                    logger.warning(f"No SEP data fetched for {month_str}")
                else:
                    # Cast date to Date type
                    df_sep = df_sep.with_columns(
                        pl.col("date").cast(pl.Date)
                    )

                    df_sep.write_parquet(output_path)
                    logger.info(f"Saved {len(df_sep)} SEP records to {output_path.name}")
                    fetched_count += 1

            # Move to next month
            if current_dt.month == 12:
                current_dt = current_dt.replace(year=current_dt.year + 1, month=1)
            else:
                current_dt = current_dt.replace(month=current_dt.month + 1)

        logger.info(f"SEP fetch complete: {fetched_count} months fetched, {skipped_count} months skipped")

    def fetch_all(
        self,
        end_date: str,
        overwrite: bool = False,
    ) -> None:
        """Fetch all data (TICKERS, SF1, SEP) up to the given end_date."""
        logger.info(f"Starting full data fetch up to {end_date}")

        df_tickers = self.fetch_and_save_tickers(
            end_date=end_date,
            overwrite=overwrite
        )
        active_tickers = df_tickers["ticker"].to_list()
        logger.info(f"Working with {len(active_tickers)} active tickers")

        self.fetch_and_save_sf1(
            end_date=end_date,
            tickers=active_tickers,
            overwrite=overwrite,
        )

        self.fetch_and_save_sep(
            end_date=end_date,
            tickers=active_tickers,
            overwrite=overwrite,
        )

        logger.info("Full data fetch completed successfully")
