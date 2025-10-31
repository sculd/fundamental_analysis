"""Sharadar API client for fetching financial data from NASDAQ Data Link."""

import time
from typing import Optional
import polars as pl
import nasdaqdatalink

from fundamental_analysis.utils.config import Config
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


class SharadarClient:
    """Client for fetching data from Sharadar via NASDAQ Data Link."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Sharadar client.

        Args:
            api_key: NASDAQ Data Link API key. If None, uses Config.NASDAQ_API_KEY
        """
        self.api_key = api_key or Config.NASDAQ_API_KEY
        if not self.api_key:
            raise ValueError("NASDAQ API key is required")

        nasdaqdatalink.ApiConfig.api_key = self.api_key
        logger.info("SharadarClient initialized")

    def fetch_tickers(self, exclude_delisted: bool = True) -> pl.DataFrame:
        """
        Fetch ticker metadata from SHARADAR/TICKERS table.

        Args:
            exclude_delisted: If True, filter out delisted companies

        Returns:
            Polars DataFrame with ticker metadata
        """
        logger.info("Fetching TICKERS table...")

        try:
            # Fetch the entire TICKERS table
            df_pandas = nasdaqdatalink.get_table(
                Config.SHARADAR_TICKERS,
                paginate=True
            )

            # Convert to Polars
            df = pl.from_pandas(df_pandas)

            logger.info(f"Fetched {len(df)} tickers")

            # Filter out delisted if requested
            if exclude_delisted:
                original_count = len(df)
                df = df.filter(pl.col("isdelisted") == "N")
                logger.info(
                    f"Filtered delisted companies: {original_count} -> {len(df)} tickers"
                )

            return df

        except Exception as e:
            logger.error(f"Error fetching TICKERS: {e}")
            raise

    def fetch_sf1(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[list[str]] = None,
        dimension: str = Config.SF1_DIMENSION,
    ) -> pl.DataFrame:
        """
        Fetch fundamental data from SHARADAR/SF1 table.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Optional list of tickers to fetch. If None, fetches all.
            dimension: SF1 dimension (MRQ, MRY, ARQ, ARY). Default: MRQ

        Returns:
            Polars DataFrame with fundamental data
        """
        logger.info(
            f"Fetching SF1 data for {start_date} to {end_date} "
            f"(dimension: {dimension})"
        )

        try:
            # Build query parameters
            query_params = {
                "calendardate.gte": start_date,
                "calendardate.lte": end_date,
                "dimension": dimension,
                "paginate": True,
            }

            # Add ticker filter if specified
            if tickers:
                logger.info(f"Filtering for {len(tickers)} tickers")
                # NASDAQ Data Link supports ticker filtering via ticker parameter
                # But for many tickers, we might need to fetch all and filter
                # For now, let's fetch all and filter in memory
                pass

            # Fetch data
            df_pandas = nasdaqdatalink.get_table(
                Config.SHARADAR_SF1,
                **query_params
            )

            # Convert to Polars
            df = pl.from_pandas(df_pandas)

            logger.info(f"Fetched {len(df)} SF1 records")

            # Filter by tickers if specified
            if tickers:
                df = df.filter(pl.col("ticker").is_in(tickers))
                logger.info(f"Filtered to {len(df)} records for specified tickers")

            return df

        except Exception as e:
            logger.error(f"Error fetching SF1 data: {e}")
            raise

    def fetch_sep(
        self,
        start_date: str,
        end_date: str,
        tickers: Optional[list[str]] = None,
    ) -> pl.DataFrame:
        """
        Fetch price data from SHARADAR/SEP table.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            tickers: Optional list of tickers to fetch. If None, fetches all.

        Returns:
            Polars DataFrame with price data
        """
        logger.info(f"Fetching SEP price data for {start_date} to {end_date}")

        try:
            # Build query parameters
            query_params = {
                "date.gte": start_date,
                "date.lte": end_date,
                "paginate": True,
            }

            # Fetch data
            df_pandas = nasdaqdatalink.get_table(
                Config.SHARADAR_SEP,
                **query_params
            )

            # Convert to Polars
            df = pl.from_pandas(df_pandas)

            logger.info(f"Fetched {len(df)} SEP price records")

            # Filter by tickers if specified
            if tickers:
                df = df.filter(pl.col("ticker").is_in(tickers))
                logger.info(f"Filtered to {len(df)} records for specified tickers")

            return df

        except Exception as e:
            logger.error(f"Error fetching SEP data: {e}")
            raise

    def fetch_with_retry(
        self, fetch_func, max_retries: int = Config.MAX_RETRIES, **kwargs
    ):
        """
        Execute a fetch function with retry logic.

        Args:
            fetch_func: Function to execute
            max_retries: Maximum number of retry attempts
            **kwargs: Arguments to pass to fetch_func

        Returns:
            Result from fetch_func
        """
        for attempt in range(max_retries):
            try:
                return fetch_func(**kwargs)
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = Config.RETRY_DELAY * (2 ** attempt)
                    logger.warning(
                        f"Attempt {attempt + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"All {max_retries} attempts failed")
                    raise
