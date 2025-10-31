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
        """
        Fetch and save SF1 fundamentals, partitioned by quarter (YYYYQN format).

        Intelligently fetches only missing quarters. If existing quarters found, only fetches
        from the most recent quarter forward. Use --overwrite to re-fetch all quarters.
        """
        from datetime import datetime, timedelta
        import re

        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        adjusted_end_dt = end_dt - timedelta(days=Config.REPORTING_DELAY_DAYS)
        adjusted_end_date = adjusted_end_dt.strftime("%Y-%m-%d")

        logger.info(f"Fetching and saving SF1 data for analysis period {start_date} to {end_date}...")
        logger.info(f"Applying {Config.REPORTING_DELAY_DAYS}-day reporting delay: adjusted end = {adjusted_end_date}")

        # Determine fetch start date by checking existing quarters
        fetch_start_date = "1990-01-01"

        if not overwrite:
            # Check for existing sf1_*.parquet files
            existing_files = sorted(Config.SF1_DIR.glob("sf1_*.parquet"))
            if existing_files:
                # Extract quarter IDs from filenames
                quarter_pattern = re.compile(r'sf1_(\d{4}Q\d)\.parquet')
                quarters = []
                for f in existing_files:
                    match = quarter_pattern.match(f.name)
                    if match:
                        quarters.append(match.group(1))

                if quarters:
                    most_recent_quarter = sorted(quarters)[-1]
                    year, quarter = most_recent_quarter.split('Q')
                    # Start from the quarter after the most recent one we have
                    fetch_start_year = int(year)
                    fetch_start_quarter = int(quarter)

                    # Move to next quarter
                    if fetch_start_quarter == 4:
                        fetch_start_year += 1
                        fetch_start_quarter = 1
                    else:
                        fetch_start_quarter += 1

                    # Calculate start date for next quarter
                    fetch_start_month = (fetch_start_quarter - 1) * 3 + 1
                    fetch_start_date = f"{fetch_start_year}-{fetch_start_month:02d}-01"

                    logger.info(f"Found existing quarters up to {most_recent_quarter}")

                    # Check if there are any new quarters to fetch
                    if fetch_start_date > adjusted_end_date:
                        logger.info(f"No new quarters available (would need data after {adjusted_end_date})")
                        logger.info("All available quarters already downloaded. Skipping SF1 fetch.")
                        return
                    else:
                        logger.info(f"Fetching only new quarters from {fetch_start_date} onwards")

        logger.info(f"Fetching quarters with calendardate: {fetch_start_date} to {adjusted_end_date}")

        df_sf1 = self.client.fetch_sf1(
            start_date=fetch_start_date,
            end_date=adjusted_end_date,
            tickers=tickers,
        )

        if len(df_sf1) == 0:
            logger.warning("No SF1 data fetched. Skipping save.")
            return

        # Pandas datetime64[ns] gets converted to Polars Datetime, so cast to Date
        df_sf1 = df_sf1.with_columns([
            pl.col("calendardate").cast(pl.Date),
            pl.col("datekey").cast(pl.Date)
        ])

        # Partition by reporting period (datekey), not filing date (calendardate)
        df_sf1 = df_sf1.with_columns([
            pl.col("datekey").dt.year().alias("year"),
            pl.col("datekey").dt.quarter().alias("quarter")
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
        """Fetch and save SEP price data, partitioned by day."""
        from datetime import datetime, timedelta

        logger.info(f"Fetching and saving SEP data from {start_date} to {end_date}...")

        if not overwrite:
            # Check which dates already exist locally
            start_dt = datetime.strptime(start_date, "%Y-%m-%d")
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")

            missing_dates = []
            current_dt = start_dt
            while current_dt <= end_dt:
                date_str = current_dt.strftime("%Y-%m-%d")
                output_path = Config.SEP_DIR / f"sep_{date_str}.parquet"
                if not output_path.exists():
                    missing_dates.append(current_dt)
                current_dt += timedelta(days=1)

            if not missing_dates:
                logger.info(f"All SEP data for {start_date} to {end_date} already exists. Skipping fetch.")
                return

            # Group missing dates into contiguous ranges
            date_ranges = []
            range_start = missing_dates[0]
            range_end = missing_dates[0]

            for i in range(1, len(missing_dates)):
                if (missing_dates[i] - missing_dates[i-1]).days == 1:
                    # Contiguous, extend current range
                    range_end = missing_dates[i]
                else:
                    # Gap found, save current range and start new one
                    date_ranges.append((range_start, range_end))
                    range_start = missing_dates[i]
                    range_end = missing_dates[i]

            # Add final range
            date_ranges.append((range_start, range_end))

            logger.info(f"Found {len(missing_dates)} days missing in {len(date_ranges)} contiguous range(s)")

            # Fetch each range separately
            all_data = []
            for range_start, range_end in date_ranges:
                range_start_str = range_start.strftime("%Y-%m-%d")
                range_end_str = range_end.strftime("%Y-%m-%d")
                logger.info(f"Fetching range: {range_start_str} to {range_end_str}")

                df_range = self.client.fetch_sep(
                    start_date=range_start_str,
                    end_date=range_end_str,
                    tickers=tickers,
                )

                if len(df_range) > 0:
                    all_data.append(df_range)

            if not all_data:
                logger.warning("No SEP data fetched for any missing ranges. Skipping save.")
                return

            # Concatenate all fetched data
            df_sep = pl.concat(all_data)
        else:
            # Overwrite mode: fetch entire range as before
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

        # Convert date to string format YYYY-MM-DD for partitioning
        df_sep = df_sep.with_columns(
            pl.col("date").dt.strftime("%Y-%m-%d").alias("date_str")
        )

        dates = df_sep["date_str"].unique().sort()
        logger.info(f"Partitioning SEP data across {len(dates)} days")

        for date_str in dates:
            df_day = df_sep.filter(pl.col("date_str") == date_str)
            output_path = Config.SEP_DIR / f"sep_{date_str}.parquet"

            if output_path.exists() and not overwrite:
                logger.info(f"File {output_path} already exists. Skipping (use --overwrite to replace).")
                continue

            df_day.write_parquet(output_path)
            logger.info(f"Saved {len(df_day)} SEP records to {output_path}")

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
