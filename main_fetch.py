"""CLI entry point for fetching Sharadar data."""

import argparse
import sys
from datetime import datetime

from fundamental_analysis.data_acquisition.data_fetcher import DataFetcher
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fetch fundamental data from Sharadar (NASDAQ Data Link)"
    )

    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format (e.g., 2018-01-01)",
    )

    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format (e.g., 2023-12-31)",
    )

    parser.add_argument(
        "--include-delisted",
        action="store_true",
        help="Include delisted companies (default: exclude delisted)",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip existing files)",
    )

    return parser.parse_args()


def validate_date(date_str: str) -> bool:
    """Validate date string is in YYYY-MM-DD format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def main():
    """Main entry point for data fetching."""
    args = parse_args()

    if not validate_date(args.start_date):
        logger.error(f"Invalid start date format: {args.start_date}. Use YYYY-MM-DD")
        sys.exit(1)

    if not validate_date(args.end_date):
        logger.error(f"Invalid end date format: {args.end_date}. Use YYYY-MM-DD")
        sys.exit(1)

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    if start_dt >= end_dt:
        logger.error("start-date must be before end-date")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("Sharadar Data Fetch")
    logger.info("=" * 60)
    logger.info(f"Start Date: {args.start_date}")
    logger.info(f"End Date: {args.end_date}")
    logger.info(f"Exclude Delisted: {not args.include_delisted}")
    logger.info(f"Overwrite Existing: {args.overwrite}")
    logger.info("=" * 60)

    try:
        fetcher = DataFetcher()
        fetcher.fetch_all(
            start_date=args.start_date,
            end_date=args.end_date,
            exclude_delisted=not args.include_delisted,
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
