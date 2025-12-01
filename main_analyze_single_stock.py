"""Analyze a single stock's fundamental metrics at a point in time."""

import argparse
from datetime import datetime, timedelta

import polars as pl

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.metrics import calculate_all_metrics
from fundamental_analysis.scoring.common import ScoreOption
from fundamental_analysis.scoring.deepdive.single_stock import print_single_stock_analysis
from fundamental_analysis.scoring.percentile_score import calculate_metric_percentiles
from fundamental_analysis.segmentation.sector import add_sector_segmentation
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def analyze_stock(ticker: str, as_of_date: str, window_days: int = 180):
    """Analyze a single stock's metrics at a point in time."""
    reader = DataReader()

    # Parse as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")

    # Load data: need window_days before as_of_date for rolling stats
    start_date = (as_of_dt - timedelta(days=window_days + 365)).strftime("%Y-%m-%d")

    print(f"\nAnalyzing {ticker} as of {as_of_date}")
    print(f"Loading data from {start_date} to {as_of_date}...")

    # Load SF1 data
    df = reader.read_sf1(start_date=start_date, end_date=as_of_date)
    if df is None or len(df) == 0:
        print(f"No data found for date range {start_date} to {as_of_date}")
        return

    # Check if ticker exists
    if ticker not in df["ticker"].unique().to_list():
        print(f"Ticker {ticker} not found in data")
        return

    # Calculate metrics first (this drops non-essential columns)
    df = calculate_all_metrics(df)

    # Load tickers metadata for sector info and join
    tickers_df = reader.read_tickers(snapshot_date=as_of_date)
    df = df.join(
        tickers_df.select(["ticker", "sector"]),
        on="ticker",
        how="left"
    )

    # Add segmentation
    df = add_sector_segmentation(df)

    # Calculate percentiles
    option = ScoreOption(window_days=window_days)
    df = calculate_metric_percentiles(df, option=option)

    # Filter to ticker and get most recent record as of as_of_date
    df_ticker = df.filter(
        (pl.col("ticker") == ticker) &
        (pl.col("datekey") <= as_of_dt.date())
    ).sort("datekey", descending=True).head(1)

    if len(df_ticker) == 0:
        print(f"No data found for {ticker} as of {as_of_date}")
        return

    row = df_ticker.to_dicts()[0]

    # Print analysis
    print_single_stock_analysis(row, ticker)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single stock's fundamental metrics at a point in time."
    )
    parser.add_argument(
        "--ticker",
        type=str,
        required=True,
        help="Stock ticker symbol (e.g., AAPL, MSFT)"
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="Rolling window size in days for percentile calculation (default: 180)"
    )

    args = parser.parse_args()

    analyze_stock(
        ticker=args.ticker.upper(),
        as_of_date=args.as_of_date,
        window_days=args.window_days
    )


if __name__ == "__main__":
    main()
