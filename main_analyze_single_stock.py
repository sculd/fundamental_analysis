"""Analyze a single stock's fundamental metrics at a point in time."""

import argparse
from datetime import datetime, timedelta

import anthropic
import polars as pl
from joblib import Memory

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.utils.config import Config
from fundamental_analysis.metrics import calculate_all_metrics
from fundamental_analysis.metrics.price_metrics.price_metric import calculate_price_metrics
from fundamental_analysis.scoring.common import ScoreOption
from fundamental_analysis.scoring.deepdive.single_stock import (
    format_single_stock_analysis, format_price_chart, print_single_stock_analysis)
from fundamental_analysis.scoring.percentile_score import \
    calculate_metric_percentiles
from fundamental_analysis.segmentation.sector import add_sector_segmentation
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)

# Cache for LLM responses
llm_cache = Memory(Config.DATA_DIR / "llm_cache", verbose=0)


@llm_cache.cache
def get_llm_analysis(ticker: str, as_of_date_str: str, metrics_str: str) -> str:
    """Get Claude's analysis of the stock using Anthropic API."""

    client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var

    prompt = f"""Let's talk about the stock whose ticker is {ticker}. Is it a sound company? I want you to provide a succinct description, point out the critical points.
To help the analysis, these are the metrics as of {as_of_date_str} I calculated using sf1 data:

{metrics_str}"""

    message = client.messages.create(
        model="claude-opus-4-20250514",
        max_tokens=4096,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        messages=[{"role": "user", "content": prompt}],
    )

    # Extract text from response (may include web search results)
    text_parts = []
    for block in message.content:
        if block.type == "text":
            text_parts.append(block.text)
    return "\n".join(text_parts)


def analyze_stock(ticker: str, as_of_date: str, window_days: int = 180, use_llm: bool = False):
    """Analyze a single stock's metrics at a point in time."""
    reader = DataReader()

    # Parse as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")

    # Load data: need window_days before as_of_date for rolling stats
    start_date = (as_of_dt - timedelta(days=window_days + 365)).strftime("%Y-%m-%d")
    # For price metrics, need 5 years of history
    price_start_date = (as_of_dt - timedelta(days=365 * 5 + 30)).strftime("%Y-%m-%d")

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

    # Load SEP price data for price metrics (separate from SF1 metrics)
    try:
        df_sep = reader.read_sep(start_date=price_start_date, end_date=as_of_date)
    except FileNotFoundError:
        df_sep = None

    # Calculate SF1-based metrics (no price metrics here)
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

    # Calculate price metrics at as-of-date (not SF1 release date)
    df_ticker_price = None
    if df_sep is not None:
        df_ticker_sep = df_sep.filter(
            (pl.col("ticker") == ticker) &
            (pl.col("date") <= as_of_dt.date())
        )
        if len(df_ticker_sep) > 0:
            df_ticker_price = df_ticker_sep.select(["date", "closeadj"])
            # Calculate price metrics and get the as-of-date row
            df_price_metrics = calculate_price_metrics(df_ticker_sep)
            price_row = df_price_metrics.sort("date", descending=True).head(1).to_dicts()
            if price_row:
                # Merge price metrics into row dict
                row.update(price_row[0])

    # Format and print analysis
    metrics_str = format_single_stock_analysis(row, ticker)
    print(metrics_str)

    # Print price chart if SEP data available
    if df_ticker_price is not None and len(df_ticker_price) > 0:
        chart = format_price_chart(df_ticker_price, ticker)
        if chart:
            print("\n" + chart)

    # Get LLM analysis if requested
    if use_llm:
        print("\n" + "=" * 70)
        print("  Claude's Analysis")
        print("=" * 70 + "\n")
        llm_response = get_llm_analysis(ticker, as_of_date, metrics_str)
        print(llm_response)


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
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Get Claude's analysis of the stock (requires ANTHROPIC_API_KEY)"
    )

    args = parser.parse_args()

    analyze_stock(
        ticker=args.ticker.upper(),
        as_of_date=args.as_of_date,
        window_days=args.window_days,
        use_llm=args.llm,
    )


if __name__ == "__main__":
    main()
