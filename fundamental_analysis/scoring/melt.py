"""Melt wide-format scored data to long format for analysis."""

from dataclasses import dataclass

import polars as pl

from .z_score import ALL_METRICS


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
    sigma_threshold: float = 2.0,
    filter_config: OutlierSummaryFilter | None = None,
) -> pl.DataFrame:
    """
    Reshape scored data from wide to long format and classify outliers.

    Transforms each stock-quarter row into multiple rows (one per metric),
    classifying which metrics are outliers with their values and statistics.

    Example transformation:
        Input (wide format):
        | ticker | pe_ratio | pe_ratio_zscore | pb_ratio | pb_ratio_zscore |
        |--------|----------|-----------------|----------|-----------------|
        | AAPL   | 10       | -2.5            | 3.2      | 0.8             |

        Output (long format):
        | ticker | metric_name | raw_value | zscore | is_outlier | outlier_direction |
        |--------|-------------|-----------|--------|------------|-------------------|
        | AAPL   | pe_ratio    | 10        | -2.5   | True       | favorable         |
        | AAPL   | pb_ratio    | 3.2       | 0.8    | False      | None              |

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with z-scores for fundamental metrics. Typically output from
        calculate_metric_z_scores() or calculate_signal_counts().
    tickers : list[str] | None, default None
        If provided, filter to only these tickers
    sigma_threshold : float, default 2.0
        Z-score threshold used to determine outliers
    filter_config : OutlierSummaryFilter | None, default None
        If provided, filter results based on outlier status and/or direction.
        If None, return all metrics without filtering.

    Returns
    -------
    pl.DataFrame
        Long-format DataFrame with one row per stock-metric combination.
        Includes metric values, z-scores, segment statistics, and outlier flags.
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
            zscore = stock_row.get(f"{metric_name}_zscore")
            segment_mean = stock_row.get(f"{metric_name}_mean")
            segment_std = stock_row.get(f"{metric_name}_std")

            # Determine if this is an outlier and in which direction
            is_outlier = False
            outlier_direction = None

            if zscore is not None and abs(zscore) >= sigma_threshold:
                is_outlier = True

                if metric_direction == "lower":
                    # Lower is better (valuation, leverage)
                    if zscore < -sigma_threshold:
                        outlier_direction = "favorable"
                    elif zscore > sigma_threshold:
                        outlier_direction = "unfavorable"
                else:  # metric_direction == "higher"
                    # Higher is better (profitability, liquidity)
                    if zscore > sigma_threshold:
                        outlier_direction = "favorable"
                    elif zscore < -sigma_threshold:
                        outlier_direction = "unfavorable"

            rows.append({
                "ticker": stock_row["ticker"],
                "datekey": stock_row["datekey"],
                "segment": stock_row.get("segment"),
                "metric_name": metric_name,
                "metric_direction": metric_direction,
                "raw_value": raw_value,
                "zscore": zscore,
                "segment_mean": segment_mean,
                "segment_std": segment_std,
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
