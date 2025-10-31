"""Configuration management for fundamental analysis."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Configuration class for the application."""

    # API Configuration
    NASDAQ_API_KEY = os.getenv("NASDAQ_DATA_API_KEY")

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    RESULTS_DIR = PROJECT_ROOT / "results"

    # Raw data subdirectories
    SF1_DIR = RAW_DATA_DIR / "sf1"
    SEP_DIR = RAW_DATA_DIR / "sep"
    TICKERS_DIR = RAW_DATA_DIR / "tickers"

    # Sharadar table codes
    SHARADAR_SF1 = "SHARADAR/SF1"
    SHARADAR_SEP = "SHARADAR/SEP"
    SHARADAR_TICKERS = "SHARADAR/TICKERS"

    # SF1 dimension (MRQ = Most Recent Quarterly)
    SF1_DIMENSION = "MRQ"

    # Point-in-time reporting delay
    # Average days after quarter end before financial reports are filed
    REPORTING_DELAY_DAYS = 45  # Conservative estimate (can be 45-90 days)

    # API settings
    BULK_DOWNLOAD_URL = "https://data.nasdaq.com/api/v3/datatables"
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

    @classmethod
    def validate(cls):
        """Validate that required configuration is present."""
        if not cls.NASDAQ_API_KEY:
            raise ValueError(
                "NASDAQ_DATA_API_KEY not found in environment variables. "
                "Please set it in your .env file."
            )

        # Create data directories if they don't exist
        for directory in [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.RESULTS_DIR,
            cls.SF1_DIR,
            cls.SEP_DIR,
            cls.TICKERS_DIR,
        ]:
            directory.mkdir(parents=True, exist_ok=True)
