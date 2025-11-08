"""Financial health and leverage ratio calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change


def _debt_to_equity_expr() -> pl.Expr:
    """Debt-to-Equity ratio: total debt / total equity."""
    return pl.when(pl.col("equity") != 0).then(
        pl.col("debt") / pl.col("equity")
    ).otherwise(None)


def _debt_to_assets_expr() -> pl.Expr:
    """Debt-to-Assets ratio: total debt / total assets."""
    return pl.when(pl.col("assets") != 0).then(
        pl.col("debt") / pl.col("assets")
    ).otherwise(None)


def _current_ratio_expr() -> pl.Expr:
    """Current Ratio: current assets / current liabilities."""
    return pl.when(pl.col("liabilitiesc") != 0).then(
        pl.col("assetsc") / pl.col("liabilitiesc")
    ).otherwise(None)


def _interest_coverage_expr() -> pl.Expr:
    """Interest Coverage ratio: EBIT / interest expense."""
    return pl.when(pl.col("intexp") != 0).then(
        pl.col("ebit") / pl.col("intexp")
    ).otherwise(None)


def get_financial_health_expressions() -> list[pl.Expr]:
    """
    Return list of financial health metric expressions for composing with other metrics.

    Use this for efficient batch calculation with other metric types.

    Includes snapshot values and temporal features (growth QoQ/YoY).
    """
    return [
        # Snapshot values
        _debt_to_equity_expr().alias("debt_to_equity"),
        _current_ratio_expr().alias("current_ratio"),
        _debt_to_assets_expr().alias("debt_to_assets"),
        _interest_coverage_expr().alias("interest_coverage"),
        # Temporal features (growth - percentage change)
        temporal_change(_debt_to_equity_expr(), 1, check_sign_crossing=True).alias("debt_to_equity_growth_qoq"),
        temporal_change(_debt_to_equity_expr(), 4, check_sign_crossing=True).alias("debt_to_equity_growth_yoy"),
        temporal_change(_current_ratio_expr(), 1).alias("current_ratio_growth_qoq"),
        temporal_change(_current_ratio_expr(), 4).alias("current_ratio_growth_yoy"),
        temporal_change(_debt_to_assets_expr(), 1).alias("debt_to_assets_growth_qoq"),
        temporal_change(_debt_to_assets_expr(), 4).alias("debt_to_assets_growth_yoy"),
        temporal_change(_interest_coverage_expr(), 1, check_sign_crossing=True).alias("interest_coverage_growth_qoq"),
        temporal_change(_interest_coverage_expr(), 4, check_sign_crossing=True).alias("interest_coverage_growth_yoy"),
    ]
