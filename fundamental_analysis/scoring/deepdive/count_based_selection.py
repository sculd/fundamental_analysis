"""Count-based scoring: count metrics beyond percentile threshold."""

import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS, ScoreOption
from fundamental_analysis.scoring.percentile_score import \
    calculate_metric_percentiles


def calculate_signal_counts(
    df: pl.DataFrame,
    option: ScoreOption | None = None,
    percentile_threshold: float = 90.0,
    min_total_signal_count: int = 0,
) -> pl.DataFrame:
    """
    Calculate percentiles and bidirectional outlier counts for all fundamental metrics.

    Convenience function that combines percentile calculation (calculate_metric_percentiles)
    with outlier counting logic. Treats all metrics equally, counting outliers in
    both favorable and unfavorable directions.

    Each metric has a defined "good" direction (threshold=90 means top/bottom 10%):
    - Valuation ratios: lower is better (cheap) - favorable if <= (100 - threshold)
    - Profitability: higher is better - favorable if >= threshold
    - Liquidity: higher is better - favorable if >= threshold
    - Leverage: lower is better - favorable if <= (100 - threshold)

    Example:
        df = calculate_signal_counts(df, percentile_threshold=90.0, min_total_signal_count=1)

        Result:
        | ticker | favorable_count | unfavorable_count | total_signal_count | net_signal |
        |--------|-----------------|-------------------|--------------------| -----------|
        | AAPL   | 3               | 1                 | 4                  | 2          |
        | NVDA   | 5               | 0                 | 5                  | 5          |

    Parameters
    ----------
    df : pl.DataFrame
        Input data with fundamental metrics
    option : ScoreOption | None, default None
        Configuration for percentile calculation. If None, uses default values.
    percentile_threshold : float, default 90.0
        Percentile threshold for outlier detection (0-100).
        90 means top/bottom 10% are outliers.
    min_total_signal_count : int, default 0
        Minimum total_signal_count to include in results. Rows with fewer signals are filtered out.

    Returns
    -------
    pl.DataFrame
        Original dataframe with added columns:
        - favorable_count: number of metrics beyond threshold in "good" direction
        - unfavorable_count: number of metrics beyond threshold in "bad" direction
        - total_signal_count: favorable + unfavorable (magnitude of extremeness)
        - net_signal: favorable - unfavorable (overall direction)
        - metrics_available: number of metrics with valid percentiles
    """
    if option is None:
        option = ScoreOption()

    # Calculate percentiles for all standard metrics
    df = calculate_metric_percentiles(df, option=option)

    # Build conditions for favorable and unfavorable outliers
    favorable_conditions = []
    unfavorable_conditions = []
    available_conditions = []

    # percentile_threshold=90 means top/bottom 10% are outliers
    outlier_cutoff = 100 - percentile_threshold  # e.g., 10 when threshold=90

    for metric_name, direction in ALL_METRICS:
        percentile_col = f"{metric_name}_percentile"

        # Determine comparison based on metric direction
        if direction == "lower":
            # Lower is better (valuation, leverage)
            # Favorable: low percentile (bottom X%) = <= outlier_cutoff
            # Unfavorable: high percentile (top X%) = >= threshold
            favorable_condition = pl.col(percentile_col) <= outlier_cutoff
            unfavorable_condition = pl.col(percentile_col) >= percentile_threshold
        else:  # direction == "higher"
            # Higher is better (profitability, liquidity)
            # Favorable: high percentile (top X%) = >= threshold
            # Unfavorable: low percentile (bottom X%) = <= outlier_cutoff
            favorable_condition = pl.col(percentile_col) >= percentile_threshold
            unfavorable_condition = pl.col(percentile_col) <= outlier_cutoff

        # Add favorable condition with null check
        favorable_conditions.append(
            pl.when(
                pl.col(percentile_col).is_not_null() &
                favorable_condition
            )
            .then(1)
            .otherwise(0)
        )

        # Add unfavorable condition with null check
        unfavorable_conditions.append(
            pl.when(
                pl.col(percentile_col).is_not_null() &
                unfavorable_condition
            )
            .then(1)
            .otherwise(0)
        )

        # Count available metrics
        available_conditions.append(
            pl.when(pl.col(percentile_col).is_not_null())
            .then(1)
            .otherwise(0)
        )

    # Add count columns
    df = df.with_columns([
        pl.sum_horizontal(favorable_conditions).alias("favorable_count"),
        pl.sum_horizontal(unfavorable_conditions).alias("unfavorable_count"),
        pl.sum_horizontal(available_conditions).alias("metrics_available"),
    ])

    # Add derived columns (total and net signals)
    df = df.with_columns([
        # Total signal magnitude (how extreme/interesting is this stock?)
        (pl.col("favorable_count") + pl.col("unfavorable_count")).alias("total_signal_count"),

        # Net signal (overall good vs bad)
        (pl.col("favorable_count") - pl.col("unfavorable_count")).alias("net_signal"),
    ])

    # Filter by minimum total signal count
    if min_total_signal_count > 0:
        df = df.filter(pl.col("total_signal_count") >= min_total_signal_count)

    return df
