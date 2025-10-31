"""Data fetcher orchestration for downloading and saving Sharadar data."""

from datetime import datetime
from pathlib import Path
from typing import Optional
import polars as pl

from fundamental_analysis.data_acquisition.sharadar_client import SharadarClient
from fundamental_analysis.utils.config import Config
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


class DataFetcher:
    """Orchestrates fetching and saving financial data."""

    def __init__(self, api_key: Optional[str] = None):
        self.client = SharadarClient(api_key)
        Config.validate()

    def fetch_and_save_tickers(
        self,
        exclude_delisted: bool = True,
        overwrite: bool = False
    ) -> pl.DataFrame:
        """Fetch and save ticker metadata with today's date in filename."""
        logger.info("Fetching and saving TICKERS data...")

        today = datetime.now().strftime("%Y-%m-%d")
        output_path = Config.TICKERS_DIR / f"tickers_{today}.parquet"

        if output_path.exists() and not overwrite:
            logger.info(f"File {output_path} already exists. Skipping (use --overwrite to replace).")
            return pl.read_parquet(output_path)

        df_tickers = self.client.fetch_tickers(exclude_delisted=exclude_delisted)
        df_tickers.write_parquet(output_path)
        logger.info(f"Saved {len(df_tickers)} tickers to {output_path}")

        return df_tickers

    def fetch_and_save_sf1(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Fetch and save SF1 fundamentals, partitioned by quarter (YYYYQN format)."""
        logger.info(f"Fetching and saving SF1 data from {start_date} to {end_date}...")

        df_sf1 = self.client.fetch_sf1(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )

        if len(df_sf1) == 0:
            logger.warning("No SF1 data fetched. Skipping save.")
            return

        df_sf1 = df_sf1.with_columns(
            pl.col("calendardate").cast(pl.Date)
        )

        df_sf1 = df_sf1.with_columns([
            pl.col("calendardate").dt.year().alias("year"),
            pl.col("calendardate").dt.quarter().alias("quarter")
        ])

        df_sf1 = df_sf1.with_columns(
            (pl.col("year").cast(pl.Utf8) + "Q" + pl.col("quarter").cast(pl.Utf8)).alias("quarter_id")
        )

        quarters = df_sf1["quarter_id"].unique().sort()
        logger.info(f"Partitioning SF1 data across {len(quarters)} quarters: {quarters.to_list()}")

        for quarter_id in quarters:
            df_quarter = df_sf1.filter(pl.col("quarter_id") == quarter_id)
            output_path = Config.SF1_DIR / f"sf1_{quarter_id}.parquet"

            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path} already exists. Skipping (use --overwrite to replace).")
                continue

            df_quarter.write_parquet(output_path)
            logger.info(f"Saved {len(df_quarter)} SF1 records to {output_path}")

    def fetch_and_save_sep(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[list[str]] = None,
        overwrite: bool = False,
    ) -> None:
        """Fetch and save SEP price data, partitioned by year."""
        logger.info(f"Fetching and saving SEP data from {start_date} to {end_date}...")

        df_sep = self.client.fetch_sep(
            start_date=start_date,
            end_date=end_date,
            tickers=tickers,
        )

        if len(df_sep) == 0:
            logger.warning("No SEP data fetched. Skipping save.")
            return

        # Pandas datetime64[ns] gets converted to Polars Datetime, so cast to Date
        df_sep = df_sep.with_columns(
            pl.col("date").cast(pl.Date)
        )

        df_sep = df_sep.with_columns(
            pl.col("date").dt.year().alias("year")
        )

        years = df_sep["year"].unique().sort()
        logger.info(f"Partitioning SEP data across {len(years)} years: {years.to_list()}")

        for year in years:
            df_year = df_sep.filter(pl.col("year") == year)
            output_path = Config.SEP_DIR / f"sep_{year}.parquet"

            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path} already exists. Skipping (use --overwrite to replace).")
                continue

            df_year.write_parquet(output_path)
            logger.info(f"Saved {len(df_year)} SEP records to {output_path}")

    def fetch_all(
        self,
        start_date: str,
        end_date: str,
        exclude_delisted: bool = True,
        overwrite: bool = False,
    ) -> None:
        """Fetch all data (TICKERS, SF1, SEP) for the given date range."""
        logger.info(f"Starting full data fetch for {start_date} to {end_date}")

        df_tickers = self.fetch_and_save_tickers(
            exclude_delisted=exclude_delisted,
            overwrite=overwrite
        )
        active_tickers = df_tickers["ticker"].to_list()
        logger.info(f"Working with {len(active_tickers)} active tickers")

        self.fetch_and_save_sf1(
            start_date=start_date,
            end_date=end_date,
            tickers=active_tickers,
            overwrite=overwrite,
        )

        self.fetch_and_save_sep(
            start_date=start_date,
            end_date=end_date,
            tickers=active_tickers,
            overwrite=overwrite,
        )

        logger.info("Full data fetch completed successfully")
