"""Analyze a single stock's fundamental metrics at a point in time."""

import argparse
from datetime import datetime, timedelta

import polars as pl

from fundamental_analysis.data_acquisition.data_reader import DataReader
from fundamental_analysis.metrics import calculate_all_metrics
from fundamental_analysis.scoring.melt import melt_and_classify_metrics
from fundamental_analysis.scoring.z_score import ALL_METRICS, ZScoreOption, calculate_metric_z_scores
from fundamental_analysis.segmentation.sector import add_sector_segmentation
from fundamental_analysis.utils.config import Config
from fundamental_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)

# Metric descriptions for human-friendly output
METRIC_DESCRIPTIONS = {
    "pe_ratio": "Price-to-Earnings ratio (lower = cheaper)",
    "pb_ratio": "Price-to-Book ratio (lower = cheaper)",
    "ps_ratio": "Price-to-Sales ratio (lower = cheaper)",
    "pc_ratio": "Price-to-Cash ratio (lower = cheaper)",
    "ev_ebitda_ratio": "EV/EBITDA ratio (lower = cheaper)",
    "roe_calculated": "Return on Equity (higher = better profitability)",
    "roic_calculated": "Return on Invested Capital (higher = better profitability)",
    "current_ratio": "Current Ratio (higher = better liquidity)",
    "interest_coverage": "Interest Coverage (higher = better debt serviceability)",
    "debt_to_equity": "Debt-to-Equity (lower = less leveraged)",
    "debt_to_assets": "Debt-to-Assets (lower = less leveraged)",
}


def format_value(value, metric_name: str) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    if metric_name in ("roe_calculated", "roic_calculated", "debt_to_assets"):
        return f"{value:.1%}"
    return f"{value:.2f}"


def format_zscore(zscore) -> str:
    """Format z-score with direction indicator."""
    if zscore is None:
        return "N/A"
    if abs(zscore) >= 2.0:
        indicator = "**"
    elif abs(zscore) >= 1.5:
        indicator = "*"
    else:
        indicator = ""
    return f"{zscore:+.2f}{indicator}"


def get_outlier_label(zscore, direction: str) -> str:
    """Get outlier label based on z-score and metric direction."""
    if zscore is None or abs(zscore) < 2.0:
        return ""

    if direction == "lower":
        if zscore < -2.0:
            return "[FAVORABLE]"
        else:
            return "[UNFAVORABLE]"
    else:  # higher
        if zscore > 2.0:
            return "[FAVORABLE]"
        else:
            return "[UNFAVORABLE]"


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

    # Calculate z-scores
    option = ZScoreOption(window_days=window_days)
    df = calculate_metric_z_scores(df, option=option)

    # Filter to ticker and get most recent record as of as_of_date
    df_ticker = df.filter(
        (pl.col("ticker") == ticker) &
        (pl.col("datekey") <= as_of_dt.date())
    ).sort("datekey", descending=True).head(1)

    if len(df_ticker) == 0:
        print(f"No data found for {ticker} as of {as_of_date}")
        return

    row = df_ticker.to_dicts()[0]

    # Display header
    print("\n" + "=" * 70)
    print(f"  {ticker} - Fundamental Analysis")
    print(f"  As of: {row['datekey']} | Segment: {row.get('segment', 'N/A')}")
    print("=" * 70)

    # Group metrics by category
    categories = {
        "Valuation": ["pe_ratio", "pb_ratio", "ps_ratio", "pc_ratio", "ev_ebitda_ratio"],
        "Profitability": ["roe_calculated", "roic_calculated"],
        "Liquidity": ["current_ratio", "interest_coverage"],
        "Leverage": ["debt_to_equity", "debt_to_assets"],
    }

    metric_directions = {name: direction for name, direction in ALL_METRICS}

    for category, metrics in categories.items():
        print(f"\n{category}:")
        print("-" * 70)

        for metric in metrics:
            value = row.get(metric)
            zscore = row.get(f"{metric}_zscore")
            mean = row.get(f"{metric}_mean")
            std = row.get(f"{metric}_std")
            direction = metric_directions.get(metric, "lower")

            desc = METRIC_DESCRIPTIONS.get(metric, metric)
            value_str = format_value(value, metric)
            zscore_str = format_zscore(zscore)
            outlier_label = get_outlier_label(zscore, direction)

            # Format segment stats
            if mean is not None and std is not None:
                mean_str = format_value(mean, metric)
                std_str = format_value(std, metric)
                stats_str = f"(segment: mean={mean_str}, std={std_str})"
            else:
                stats_str = ""

            print(f"  {desc}")
            print(f"    Value: {value_str}  |  Z-score: {zscore_str}  {stats_str} {outlier_label}")

    print("\n" + "-" * 70)
    print(f"Z-scores calculated using {window_days}-day rolling window within segment")
    print("Legend: * = notable (|z| >= 1.5), ** = outlier (|z| >= 2.0)")
    print("[FAVORABLE] = outlier in good direction, [UNFAVORABLE] = outlier in bad direction")
    print("=" * 70)


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
        help="Rolling window size in days for z-score calculation (default: 180)"
    )

    args = parser.parse_args()

    analyze_stock(
        ticker=args.ticker.upper(),
        as_of_date=args.as_of_date,
        window_days=args.window_days
    )


if __name__ == "__main__":
    main()
