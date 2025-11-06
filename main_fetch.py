"""CLI entry point for fetching Sharadar data."""

import argparse
import sys

from fundamental_analysis.data_acquisition.data_fetcher import DataFetcher
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch fundamental data from Sharadar (NASDAQ Data Link)"
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format (e.g., 2023-12-31)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing files)",
    )

    return parser.parse_args()


def main():
    """Main entry point for data fetching."""
    args = parse_args()

    logger.info("=" * 60)
    logger.info("Sharadar Data Fetch")
    logger.info("=" * 60)
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Overwrite Existing: {args.overwrite}")
    logger.info("=" * 60)

    try:
        fetcher = DataFetcher()
        fetcher.fetch_all(
            end_date=args.end_date,
            overwrite=args.overwrite,
        )

        logger.info("=" * 60)
        logger.info("Data fetch completed successfully!")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Data fetch failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
