"""Find stocks with multiple favorable/unfavorable metric outliers."""

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.utils.config import Config
from fundamental_analysis.metrics import calculate_all_metrics
from fundamental_analysis.scoring.common import ScoreOption
from fundamental_analysis.scoring.deepdive.count_based_selection import \
    calculate_signal_counts
from fundamental_analysis.segmentation.sector import add_sector_segmentation
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Find stocks with multiple favorable/unfavorable metric outliers."
    )
    parser.add_argument(
        "--as-of-date",
        type=str,
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Analysis date in YYYY-MM-DD format (default: today)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=90.0,
        help="Percentile threshold for outlier detection (default: 90.0, meaning top/bottom 10%%)"
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=1,
        help="Minimum total signal count to include (default: 1)"
    )
    parser.add_argument(
        "--max-signals",
        type=int,
        default=None,
        help="Maximum total signal count to include (default: no limit)"
    )
    parser.add_argument(
        "--sort-by",
        type=str,
        default="net_signal",
        choices=["net_signal", "favorable_count", "unfavorable_count", "total_signal_count"],
        help="Column to sort by (default: net_signal)"
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort in ascending order (default: descending)"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of stocks to display (default: 20)"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=180,
        help="Rolling window size in days for percentile calculation (default: 180)"
    )

    args = parser.parse_args()

    run_count_based_selection(
        as_of_date=args.as_of_date,
        percentile_threshold=args.threshold,
        min_signals=args.min_signals,
        max_signals=args.max_signals,
        sort_by=args.sort_by,
        ascending=args.ascending,
        top_n=args.top_n,
        window_days=args.window_days,
    )


def load_verdicts() -> pl.DataFrame | None:
    """Load verdicts CSV and aggregate comments per ticker."""
    verdicts_path = Config.PROJECT_ROOT / "verdicts.csv"
    if not verdicts_path.exists():
        return None

    df = pl.read_csv(verdicts_path)
    if len(df) == 0:
        return None

    # Format each row as "comment (date)" and aggregate by ticker
    return df.with_columns(
        pl.concat_str([
            pl.col("comment"),
            pl.lit(" ("),
            pl.col("date"),
            pl.lit(")")
        ]).alias("verdict_entry")
    ).group_by("ticker").agg(
        pl.col("verdict_entry").str.concat("\n").alias("verdict")
    )


def run_count_based_selection(
    as_of_date: str,
    percentile_threshold: float = 90.0,
    min_signals: int = 1,
    max_signals: int | None = None,
    sort_by: str = "net_signal",
    ascending: bool = False,
    top_n: int = 20,
    window_days: int = 180,
):
    """Run count-based stock selection."""
    reader = DataReader()

    # Parse as_of_date
    as_of_dt = datetime.strptime(as_of_date, "%Y-%m-%d")

    # Load data: need window_days before as_of_date for rolling stats
    start_date = (as_of_dt - timedelta(days=window_days + 365)).strftime("%Y-%m-%d")

    # Build signal range string for display
    signal_range = f">= {min_signals}"
    if max_signals is not None:
        signal_range = f"{min_signals}-{max_signals}"

    print(f"\n{'='*70}")
    print(f"  Count-Based Stock Selection")
    print(f"  As of: {as_of_date} | Threshold: {percentile_threshold}%")
    print(f"  Signals: {signal_range} | Sort by: {sort_by}")
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

    # Calculate signal counts
    option = ScoreOption(window_days=window_days)
    df = calculate_signal_counts(
        df,
        option=option,
        percentile_threshold=percentile_threshold,
        min_total_signal_count=min_signals,
    )

    # Filter by max signals if specified
    if max_signals is not None:
        df = df.filter(pl.col("total_signal_count") <= max_signals)

    # Filter to most recent date per ticker
    df_recent = df.sort("ticker", "datekey").group_by("ticker").tail(1)
    print(f"Found {len(df_recent):,} stocks with {signal_range} signal(s)")

    # Sort by specified column
    df_sorted = df_recent.sort(sort_by, descending=not ascending)

    # Display results
    print(f"\n{'-'*70}")
    print(f"Top {top_n} stocks by {sort_by} ({'ascending' if ascending else 'descending'})")
    print(f"{'-'*70}\n")

    if len(df_sorted) > 0:
        display_cols = [
            "ticker", "segment",
            "favorable_count", "unfavorable_count",
            "total_signal_count", "net_signal",
            "metrics_available",
        ]
        df_display = df_sorted.select(display_cols).head(top_n)

        # Join verdicts if available
        verdicts_df = load_verdicts()
        if verdicts_df is not None:
            df_display = df_display.join(verdicts_df, on="ticker", how="left")
        else:
            df_display = df_display.with_columns(pl.lit(None).alias("verdict"))

        with pl.Config(tbl_rows=top_n, fmt_str_lengths=60):
            print(df_display)

        # Suggest using single stock analysis for drill-down
        top_ticker = df_sorted[0, "ticker"]
        print(f"\nTip: Run `python main_analyze_single_stock.py --ticker {top_ticker} --as-of-date {as_of_date}` for details")
    else:
        print("No stocks found matching criteria.")


if __name__ == "__main__":
    main()
