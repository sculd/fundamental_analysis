"""Financial health and leverage ratio calculations."""

import polars as pl


def _debt_to_equity_expr() -> pl.Expr:
    """Debt-to-Equity ratio: total debt / total equity."""
    return pl.when(pl.col("equity") != 0).then(
        pl.col("debt") / pl.col("equity")
    ).otherwise(None)


def _debt_to_equity_growth_expr(shift: int) -> pl.Expr:
    """
    Debt-to-Equity growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Returns null when previous value is 0 or when crossing zero to avoid misleading values.
    """
    current = _debt_to_equity_expr()
    previous = _debt_to_equity_expr().shift(shift).over("ticker", order_by="reportperiod")

    return pl.when((previous != 0) & (current * previous > 0)).then(
        (current - previous) / previous.abs()
    ).otherwise(None)


def _debt_to_assets_expr() -> pl.Expr:
    """Debt-to-Assets ratio: total debt / total assets."""
    return pl.when(pl.col("assets") != 0).then(
        pl.col("debt") / pl.col("assets")
    ).otherwise(None)


def _debt_to_assets_growth_expr(shift: int) -> pl.Expr:
    """
    Debt-to-Assets growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Returns null when previous value is 0.
    """
    current = _debt_to_assets_expr()
    previous = _debt_to_assets_expr().shift(shift).over("ticker", order_by="reportperiod")

    return pl.when(previous != 0).then(
        (current - previous) / previous
    ).otherwise(None)


def _current_ratio_expr() -> pl.Expr:
    """Current Ratio: current assets / current liabilities."""
    return pl.when(pl.col("liabilitiesc") != 0).then(
        pl.col("assetsc") / pl.col("liabilitiesc")
    ).otherwise(None)


def _current_ratio_growth_expr(shift: int) -> pl.Expr:
    """
    Current Ratio growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Returns null when previous value is 0.
    """
    current = _current_ratio_expr()
    previous = _current_ratio_expr().shift(shift).over("ticker", order_by="reportperiod")

    return pl.when(previous != 0).then(
        (current - previous) / previous
    ).otherwise(None)


def _interest_coverage_expr() -> pl.Expr:
    """Interest Coverage ratio: EBIT / interest expense."""
    return pl.when(pl.col("intexp") != 0).then(
        pl.col("ebit") / pl.col("intexp")
    ).otherwise(None)


def _interest_coverage_growth_expr(shift: int) -> pl.Expr:
    """
    Interest Coverage growth rate (percentage change). Use shift=1 for QoQ, shift=4 for YoY.

    Returns null when previous value is 0 or when crossing zero to avoid misleading values.
    """
    current = _interest_coverage_expr()
    previous = _interest_coverage_expr().shift(shift).over("ticker", order_by="reportperiod")

    return pl.when((previous != 0) & (current * previous > 0)).then(
        (current - previous) / previous.abs()
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
        _debt_to_equity_growth_expr(shift=1).alias("debt_to_equity_growth_qoq"),
        _debt_to_equity_growth_expr(shift=4).alias("debt_to_equity_growth_yoy"),
        _current_ratio_growth_expr(shift=1).alias("current_ratio_growth_qoq"),
        _current_ratio_growth_expr(shift=4).alias("current_ratio_growth_yoy"),
        _debt_to_assets_growth_expr(shift=1).alias("debt_to_assets_growth_qoq"),
        _debt_to_assets_growth_expr(shift=4).alias("debt_to_assets_growth_yoy"),
        _interest_coverage_growth_expr(shift=1).alias("interest_coverage_growth_qoq"),
        _interest_coverage_growth_expr(shift=4).alias("interest_coverage_growth_yoy"),
    ]
