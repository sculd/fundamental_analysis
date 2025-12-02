"""Single stock analysis display functions."""

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

PERCENTILE_THRESHOLD = 90.0


def format_value(value, metric_name: str) -> str:
    """Format metric value for display."""
    if value is None:
        return "N/A"
    if metric_name in ("roe_calculated", "roic_calculated", "debt_to_assets"):
        return f"{value:.1%}"
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

    outlier_cutoff = 100 - percentile_threshold  # e.g., 10 when threshold=90
    lines.append("\n" + "-" * 70)
    lines.append(f"Legend: * = notable (<=20% or >=80%), ** = outlier (<={outlier_cutoff}% or >={percentile_threshold}%)")
    lines.append("[FAVORABLE] = outlier in good direction, [UNFAVORABLE] = outlier in bad direction")
    lines.append("=" * 70)

    return "\n".join(lines)


def print_single_stock_analysis(
    row: dict,
    ticker: str,
    percentile_threshold: float = PERCENTILE_THRESHOLD,
) -> None:
    """Print formatted analysis for a single stock."""
    print(format_single_stock_analysis(row, ticker, percentile_threshold))
