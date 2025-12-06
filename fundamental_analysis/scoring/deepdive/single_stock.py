"""Single stock analysis display functions."""

import plotext as plt
import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS

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

# Metrics grouped by category
METRIC_CATEGORIES = {
    "Valuation": ["pe_ratio", "pb_ratio", "ps_ratio", "pc_ratio", "ev_ebitda_ratio"],
    "Profitability": ["roe_calculated", "roic_calculated"],
    "Liquidity": ["current_ratio", "interest_coverage"],
    "Leverage": ["debt_to_equity", "debt_to_assets"],
}

# Price metrics (from SEP data) - these don't have percentile rankings
# Format: (metric_name, description, base_price_col or None)
PRICE_METRICS = [
    ("return_1y", "1-Year Return", "price_1y_ago"),
    ("return_5y_or_longest", "5-Year (or Longest) Return", "price_5y_ago"),
    ("max_drawdown_1y", "Max Drawdown (1Y)", None),
    ("max_drawdown_5y", "Max Drawdown (5Y)", None),
    ("pct_from_high_5y", "% From 5Y High", "high_5y"),
    ("pct_from_low_5y", "% From 5Y Low", "low_5y"),
    ("volatility_1y", "Volatility (1Y, annualized)", None),
    ("pct_from_sma_200", "% From 200-Day SMA", "sma_200"),
]

PERCENTILE_THRESHOLD = 90.0


PERCENT_METRICS = {
    "roe_calculated", "roic_calculated", "debt_to_assets",
    "return_1y", "return_5y_or_longest",
    "max_drawdown_1y", "max_drawdown_5y",
    "pct_from_high_5y", "pct_from_low_5y",
    "volatility_1y", "pct_from_sma_200",
}


def format_value(value, metric_name: str) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    if metric_name in PERCENT_METRICS:
        return f"{value:.1%}"
    if metric_name == "return_period_days":
        return f"{int(value)} days"
    return f"{value:.2f}"


def format_percentile(percentile, threshold: float = PERCENTILE_THRESHOLD) -> str:
    """Format percentile with indicator. threshold=90 means top/bottom 10% are outliers."""
    if percentile is None:
        return "N/A"
    outlier_cutoff = 100 - threshold  # e.g., 10 when threshold=90
    if percentile <= outlier_cutoff or percentile >= threshold:
        indicator = "**"
    elif percentile <= 20 or percentile >= 80:
        indicator = "*"
    else:
        indicator = ""
    return f"{percentile:.1f}%{indicator}"


def format_growth(qoq: float | None, yoy: float | None) -> str:
    """Format QoQ and YoY growth as string. Returns empty string if both are None."""
    parts = []
    if qoq is not None:
        parts.append(f"QoQ: {qoq:+.1%}")
    if yoy is not None:
        parts.append(f"YoY: {yoy:+.1%}")
    return f" ({', '.join(parts)})" if parts else ""


def get_outlier_label(percentile, direction: str, threshold: float = PERCENTILE_THRESHOLD) -> str:
    """Get outlier label based on percentile and metric direction.

    threshold=90 means top/bottom 10% are outliers.
    """
    if percentile is None:
        return ""

    outlier_cutoff = 100 - threshold  # e.g., 10 when threshold=90

    if direction == "lower":
        # Lower is better (valuation, leverage)
        if percentile <= outlier_cutoff:
            return "[FAVORABLE]"
        elif percentile >= threshold:
            return "[UNFAVORABLE]"
    else:  # higher
        # Higher is better (profitability, liquidity)
        if percentile >= threshold:
            return "[FAVORABLE]"
        elif percentile <= outlier_cutoff:
            return "[UNFAVORABLE]"
    return ""


def format_single_stock_analysis(
    row: dict,
    ticker: str,
    percentile_threshold: float = PERCENTILE_THRESHOLD,
) -> str:
    """
    Format analysis for a single stock as a string.

    Parameters
    ----------
    row : dict
        Dictionary containing stock data with metric values and percentile columns.
        Expected columns: metric values, {metric}_percentile, {metric}_population,
        {metric}_median, {metric}_p10, {metric}_p90, datekey, segment
    ticker : str
        Stock ticker symbol
    percentile_threshold : float, default 90.0
        Threshold for outlier detection. 90 means top/bottom 10% are outliers.

    Returns
    -------
    str
        Formatted analysis string
    """
    metric_directions = {name: direction for name, direction in ALL_METRICS}
    lines = []

    # Header
    lines.append("=" * 70)
    lines.append(f"  {ticker} - Fundamental Analysis")
    lines.append(f"  As of: {row.get('datekey', 'N/A')} | Segment: {row.get('segment', 'N/A')}")

    # Market cap info
    marketcap = row.get("marketcap")

    if marketcap is not None:
        # Format market cap with appropriate suffix (T for trillions, B for billions, M for millions)
        if marketcap >= 1e12:
            mc_str = f"${marketcap / 1e12:.2f}T"
        elif marketcap >= 1e9:
            mc_str = f"${marketcap / 1e9:.2f}B"
        else:
            mc_str = f"${marketcap / 1e6:.0f}M"

        marketcap_category = row.get("marketcap_category", "N/A")
        growth_str = format_growth(
            row.get("marketcap_growth_qoq"),
            row.get("marketcap_growth_yoy")
        )
        lines.append(f"  Market Cap: {mc_str} [{marketcap_category}]{growth_str}")

    lines.append("=" * 70)

    for category, metrics in METRIC_CATEGORIES.items():
        lines.append(f"\n{category}:")
        lines.append("-" * 70)

        for metric in metrics:
            value = row.get(metric)
            percentile = row.get(f"{metric}_percentile")
            population = row.get(f"{metric}_population")
            median = row.get(f"{metric}_median")
            p10 = row.get(f"{metric}_p10")
            p90 = row.get(f"{metric}_p90")
            direction = metric_directions.get(metric, "lower")

            desc = METRIC_DESCRIPTIONS.get(metric, metric)
            value_str = format_value(value, metric)
            value_str += format_growth(
                row.get(f"{metric}_growth_qoq"),
                row.get(f"{metric}_growth_yoy")
            )

            percentile_str = format_percentile(percentile, percentile_threshold)
            median_str = format_value(median, metric)
            p10_str = format_value(p10, metric)
            p90_str = format_value(p90, metric)
            outlier_label = get_outlier_label(percentile, direction, percentile_threshold)

            # Format stats info
            if population is not None:
                stats_str = f"(p10={p10_str}, med={median_str}, p90={p90_str}, n={population})"
            else:
                stats_str = ""

            lines.append(f"  {desc}")
            lines.append(f"    Value: {value_str}  |  Pctl: {percentile_str} {stats_str} {outlier_label}")

    # Price metrics section (if available)
    has_price_metrics = any(row.get(m[0]) is not None for m in PRICE_METRICS)
    if has_price_metrics:
        current_price = row.get("closeadj")
        current_price_str = f"${current_price:.2f}" if current_price else "N/A"

        lines.append(f"\nPrice History (Current: {current_price_str}):")
        lines.append("-" * 70)

        for metric_name, desc, base_col in PRICE_METRICS:
            value = row.get(metric_name)
            if value is None:
                continue

            value_str = format_value(value, metric_name)
            base_price = row.get(base_col) if base_col else None

            # Build suffix: (base -> current, period) or (base -> current) or empty
            if base_price is not None:
                period_desc = ""
                if metric_name == "return_5y_or_longest":
                    period_days = row.get("return_period_days")
                    if period_days is not None:
                        period_desc = f", {format_value(period_days, 'return_period_days')}"

                suffix = f" (${base_price:.2f} -> {current_price_str}{period_desc})"
            else:
                suffix = ""

            lines.append(f"  {desc}: {value_str}{suffix}")

    outlier_cutoff = 100 - percentile_threshold  # e.g., 10 when threshold=90
    lines.append("\n" + "-" * 70)
    lines.append(f"Legend: * = notable (<=20% or >=80%), ** = outlier (<={outlier_cutoff}% or >={percentile_threshold}%)")
    lines.append("[FAVORABLE] = outlier in good direction, [UNFAVORABLE] = outlier in bad direction")
    lines.append("=" * 70)

    return "\n".join(lines)


def format_price_chart(
    df_price: pl.DataFrame,
    ticker: str,
    df_baseline: pl.DataFrame | None = None,
    baseline_ticker: str = "SPY",
    width: int = 70,
    height: int = 15,
) -> str:
    """
    Format ASCII price chart from SEP data with optional baseline comparison.

    Parameters
    ----------
    df_price : pl.DataFrame
        Price data with columns: date, closeadj (filtered to single ticker)
    ticker : str
        Stock ticker symbol for title
    df_baseline : pl.DataFrame | None
        Optional baseline price data (e.g., SPY) with columns: date, closeadj.
        Baseline is rescaled so its starting point matches the ticker's starting price.
    baseline_ticker : str
        Baseline ticker symbol for legend (default: SPY)
    width, height : int
        Chart dimensions in characters
    """
    if df_price is None or len(df_price) == 0:
        return ""

    df = df_price.sort("date")
    dates = df["date"].to_list()
    prices = df["closeadj"].to_list()

    # Process baseline if provided - rescale to start at same price as ticker
    baseline_prices_scaled = None
    if df_baseline is not None and len(df_baseline) > 0:
        df_base = df_baseline.sort("date")
        # Join on dates to align
        df_joined = df.select(["date"]).join(
            df_base.select(["date", "closeadj"]),
            on="date",
            how="left"
        )
        baseline_prices = df_joined["closeadj"].to_list()
        if baseline_prices and baseline_prices[0] is not None:
            # Rescale baseline so it starts at the same price as ticker
            scale_factor = prices[0] / baseline_prices[0]
            baseline_prices_scaled = [
                p * scale_factor if p is not None else None
                for p in baseline_prices
            ]

    # Downsample if too many points (take every nth point to fit width)
    if len(prices) > width:
        step = len(prices) // width
        prices = prices[::step]
        dates = dates[::step]
        if baseline_prices_scaled:
            baseline_prices_scaled = baseline_prices_scaled[::step]

    plt.clear_figure()
    plt.plot_size(width, height)

    if baseline_prices_scaled:
        plt.title(f"{ticker} vs {baseline_ticker}")
        plt.plot(prices, label=ticker, marker="braille")
        plt.plot(baseline_prices_scaled, label=baseline_ticker, marker="braille")
    else:
        plt.title(f"{ticker} Price History")
        plt.plot(prices, marker="braille")

    plt.theme("clear")

    # Add date labels (start, middle, end)
    if len(dates) >= 3:
        plt.xticks(
            [0, len(dates) // 2, len(dates) - 1],
            [str(dates[0]), str(dates[len(dates) // 2]), str(dates[-1])]
        )

    # Capture output to string
    return plt.build()


def print_single_stock_analysis(
    row: dict,
    ticker: str,
    percentile_threshold: float = PERCENTILE_THRESHOLD,
    df_price: pl.DataFrame | None = None,
) -> None:
    """Print formatted analysis for a single stock."""
    print(format_single_stock_analysis(row, ticker, percentile_threshold))

    # Print price chart if data available
    if df_price is not None and len(df_price) > 0:
        chart = format_price_chart(df_price, ticker)
        if chart:
            print("\n" + chart)
