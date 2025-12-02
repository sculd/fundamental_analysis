"""Find stocks with outliers in a specific metric."""

import argparse
from datetime import datetime, timedelta

import polars as pl

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.metrics import calculate_all_metrics
from fundamental_analysis.scoring.common import ALL_METRICS, ScoreOption
from fundamental_analysis.scoring.deepdive.metric_based_selection import \
    get_stocks_with_metric_outlier
from fundamental_analysis.scoring.deepdive.single_stock import \
    print_single_stock_analysis
from fundamental_analysis.scoring.melt import melt_and_classify_metrics
from fundamental_analysis.scoring.percentile_score import \
    calculate_metric_percentiles
from fundamental_analysis.segmentation.sector import add_sector_segmentation
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)

# Valid metric names for CLI
VALID_METRICS = [m[0] for m in ALL_METRICS]


def main():
    parser = argparse.ArgumentParser(
        description="Find stocks with outliers in a specific metric."
    )
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=VALID_METRICS,
        help=f"Metric to filter on. Choices: {', '.join(VALID_METRICS)}"
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="favorable",
        choices=["favorable", "unfavorable"],
        help="Filter direction: favorable (good outliers) or unfavorable (bad outliers)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Percentile threshold for outlier detection (default: 90.0, meaning top/bottom 10%%)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of top stocks to display (default: 20)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="Rolling window size in days for percentile calculation (default: 180)"
    )
    parser.add_argument(
        "--drill-down",
        type=str,
        default=None,
        help="Ticker to drill down into (show all metrics for this stock)"
    )

    args = parser.parse_args()

    run_metric_based_selection(
        metric=args.metric,
        as_of_date=args.as_of_date,
        direction=args.direction,
        percentile_threshold=args.threshold,
        top_n=args.top_n,
        window_days=args.window_days,
        drill_down_ticker=args.drill_down,
    )


def run_metric_based_selection(
    metric: str,
    as_of_date: str,
    direction: str = "favorable",
    percentile_threshold: float = 90.0,
    top_n: int = 20,
    window_days: int = 180,
    drill_down_ticker: str | None = None,
):
    """Run metric-based stock selection."""
    reader = DataReader()

    # Parse as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")

    # Load data: need window_days before as_of_date for rolling stats
    start_date = (as_of_dt - timedelta(days=window_days + 365)).strftime("%Y-%m-%d")

    # Get metric direction for display
    metric_config = {name: dir for name, dir in ALL_METRICS}
    metric_direction = metric_config.get(metric, "lower")

    print(f"\n{'='*70}")
    print(f"  Metric-Based Stock Selection")
    print(f"  As of: {as_of_date} | Metric: {metric} | Direction: {direction}")
    print(f"  Threshold: {percentile_threshold}% | Window: {window_days} days")
    print(f"{'='*70}")

    print(f"\nLoading data from {start_date} to {as_of_date}...")

    # Load SF1 data
    df = reader.read_sf1(start_date=start_date, end_date=as_of_date)
    if df is None or len(df) == 0:
        print(f"No data found for date range {start_date} to {as_of_date}")
        return

    # Calculate metrics
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

    # Filter to most recent date per ticker
    df_recent = df.sort("ticker", "datekey").group_by("ticker").tail(1)
    print(f"Evaluating {len(df_recent):,} stocks (most recent data per ticker)")

    # Get stocks with metric outlier
    result = get_stocks_with_metric_outlier(
        df_recent,
        metric_name=metric,
        option=option,
        percentile_threshold=percentile_threshold,
        direction=direction,
        min_stocks=top_n,
    )

    # Display results
    print(f"\n{'-'*70}")
    if direction == "favorable":
        if metric_direction == "lower":
            print(f"Stocks with LOW {metric} (favorable - cheap/safe)")
        else:
            print(f"Stocks with HIGH {metric} (favorable - good performance)")
    else:
        if metric_direction == "lower":
            print(f"Stocks with HIGH {metric} (unfavorable - expensive/risky)")
        else:
            print(f"Stocks with LOW {metric} (unfavorable - poor performance)")
    print(f"{'-'*70}\n")

    if len(result) > 0:
        display_cols = ["ticker", "segment", "raw_value", "percentile", "is_outlier"]
        print(result.select(display_cols).head(top_n))

        # Drill down if requested
        if drill_down_ticker:
            _drill_down(df_recent, drill_down_ticker, percentile_threshold)
        else:
            # Suggest drill down for top stock
            top_ticker = result[0, "ticker"]
            print(f"\nTip: Use --drill-down {top_ticker} to see all metrics for the top stock")
    else:
        print("No stocks found matching criteria.")


def _drill_down(df: pl.DataFrame, ticker: str, percentile_threshold: float):
    """Show all metrics for a specific ticker."""
    print(f"\n{'='*70}")
    print(f"  Drill-down: All metrics for {ticker}")
    print(f"{'='*70}")

    # Get the row for this ticker
    df_ticker = df.filter(pl.col("ticker") == ticker)

    if len(df_ticker) == 0:
        print(f"Ticker {ticker} not found in data")
        return

    row = df_ticker.to_dicts()[0]
    print_single_stock_analysis(row, ticker, percentile_threshold)


if __name__ == "__main__":
    main()
