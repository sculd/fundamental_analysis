"""Melt wide-format scored data to long format for analysis."""

from dataclasses import dataclass

import polars as pl

from fundamental_analysis.scoring.common import ALL_METRICS


@dataclass
class OutlierSummaryFilter:
    """Filter configuration for outlier summary.

    Parameters
    ----------
    outlier_only : bool, default True
        If True, filter to only rows where is_outlier=True
    direction_filter : str | None, default None
        If provided, filter to only this direction ("favorable" or "unfavorable")
    """

    outlier_only: bool = True
    direction_filter: str | None = None

    def __post_init__(self):
        """Validate filter configuration."""
        if self.direction_filter is not None:
            if self.direction_filter not in ("favorable", "unfavorable"):
                raise ValueError(
                    f"direction_filter must be 'favorable' or 'unfavorable', "
                    f"got '{self.direction_filter}'"
                )


def melt_and_classify_metrics(
    df: pl.DataFrame,
    tickers: list[str] | None = None,
    percentile_threshold: float = 10.0,
    filter_config: OutlierSummaryFilter | None = None,
) -> pl.DataFrame:
    """
    Reshape scored data from wide to long format and classify outliers.

    Transforms each stock-quarter row into multiple rows (one per metric),
    classifying which metrics are outliers based on percentile ranks.

    Example transformation:
        Input (wide format):
        | ticker | pe_ratio | pe_ratio_percentile | pb_ratio | pb_ratio_percentile |
        |--------|----------|---------------------|----------|---------------------|
        | AAPL   | 10       | 5.2                 | 3.2      | 45.0                |

        Output (long format):
        | ticker | metric_name | raw_value | percentile | is_outlier | outlier_direction |
        |--------|-------------|-----------|------------|------------|-------------------|
        | AAPL   | pe_ratio    | 10        | 5.2        | True       | favorable         |
        | AAPL   | pb_ratio    | 3.2       | 45.0       | False      | None              |

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with percentiles for fundamental metrics. Typically output from
        calculate_metric_percentiles() or calculate_signal_counts().
    tickers : list[str] | None, default None
        If provided, filter to only these tickers
    percentile_threshold : float, default 10.0
        Percentile threshold for outlier detection (0-100).
        For "lower is better" metrics: favorable if <= threshold
        For "higher is better" metrics: favorable if >= (100 - threshold)
    filter_config : OutlierSummaryFilter | None, default None
        If provided, filter results based on outlier status and/or direction.
        If None, return all metrics without filtering.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with one row per stock-metric combination.
        Includes metric values, percentiles, population size, and outlier flags.
        Filtered according to filter_config if provided.
    """
    # Filter to specific tickers if requested
    if tickers is not None:
        df = df.filter(pl.col("ticker").is_in(tickers))

    # Build list of rows for long format
    rows = []

    for stock_row in df.iter_rows(named=True):
        for metric_name, metric_direction in ALL_METRICS:
            # Get values for this metric
            raw_value = stock_row.get(metric_name)
            percentile = stock_row.get(f"{metric_name}_percentile")
            population = stock_row.get(f"{metric_name}_population")

            # Determine if this is an outlier and in which direction
            is_outlier = False
            outlier_direction = None

            if percentile is not None:
                if metric_direction == "lower":
                    # Lower is better (valuation, leverage)
                    # Favorable: low percentile (bottom X%)
                    # Unfavorable: high percentile (top X%)
                    if percentile <= percentile_threshold:
                        is_outlier = True
                        outlier_direction = "favorable"
                    elif percentile >= (100 - percentile_threshold):
                        is_outlier = True
                        outlier_direction = "unfavorable"
                else:  # metric_direction == "higher"
                    # Higher is better (profitability, liquidity)
                    # Favorable: high percentile (top X%)
                    # Unfavorable: low percentile (bottom X%)
                    if percentile >= (100 - percentile_threshold):
                        is_outlier = True
                        outlier_direction = "favorable"
                    elif percentile <= percentile_threshold:
                        is_outlier = True
                        outlier_direction = "unfavorable"

            rows.append({
                "ticker": stock_row["ticker"],
                "datekey": stock_row["datekey"],
                "segment": stock_row.get("segment"),
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "raw_value": raw_value,
                "percentile": percentile,
                "population": population,
                "is_outlier": is_outlier,
                "outlier_direction": outlier_direction,
            })

    result = pl.DataFrame(rows)

    # Apply filtering if requested
    if filter_config is not None:
        if filter_config.outlier_only:
            result = result.filter(pl.col("is_outlier") == True)

        if filter_config.direction_filter is not None:
            result = result.filter(pl.col("outlier_direction") == filter_config.direction_filter)

    return result
