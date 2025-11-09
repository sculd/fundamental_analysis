"""Profitability and efficiency metric calculations."""

import polars as pl

from fundamental_analysis.metrics.temporal_utils import temporal_change


def _roe_expr() -> pl.Expr:
    """
    Return on Equity (ROE): net income / shareholders' equity.

    Measures how efficiently a company generates profit from shareholders' equity.
    Higher values indicate better profitability.

    Note: SF1 provides pre-calculated 'roe' column, but we calculate it here
    for transparency and consistency with other metrics.
    """
    return pl.when(pl.col("equity") != 0).then(
        pl.col("netinccmn") / pl.col("equity")
    ).otherwise(None)


def _roic_expr() -> pl.Expr:
    """
    Return on Invested Capital (ROIC): NOPAT / invested capital.

    NOPAT (Net Operating Profit After Tax) = EBIT * (1 - effective_tax_rate)
    Effective tax rate = taxexp / ebt (earnings before tax)
    Invested Capital = debt + equity - cash

    Measures how efficiently a company generates returns from all capital invested
    (both debt and equity). Higher values indicate better capital efficiency.

    Note: SF1 provides pre-calculated 'roic' column, but we calculate it here
    for transparency and consistency with other metrics.
    """
    # Calculate effective tax rate: taxexp / ebt
    # When ebt is 0 or negative, use 0 as tax rate
    effective_tax_rate = pl.when(pl.col("ebt") > 0).then(
        pl.col("taxexp") / pl.col("ebt")
    ).otherwise(0.0)

    # Calculate NOPAT: EBIT * (1 - tax_rate)
    nopat = pl.col("ebit") * (1 - effective_tax_rate)

    # Calculate Invested Capital: debt + equity - cash
    invested_capital = pl.col("debt") + pl.col("equity") - pl.col("cashneq")

    # ROIC = NOPAT / Invested Capital
    return pl.when(invested_capital != 0).then(
        nopat / invested_capital
    ).otherwise(None)


def get_profitability_snapshot_expressions() -> list[pl.Expr]:
    """
    Return snapshot profitability metric expressions (non-temporal).

    Includes:
    - Return on Equity (ROE)
    - Return on Invested Capital (ROIC)
    """
    return [
        _roe_expr().alias("roe_calculated"),
        _roic_expr().alias("roic_calculated"),
    ]


def get_profitability_growth_expressions() -> list[pl.Expr]:
    """
    Return temporal growth expressions for profitability metrics.

    Includes growth (QoQ and YoY) for ROE and ROIC.
    """
    return [
        temporal_change(_roe_expr(), 1, check_sign_crossing=True).alias("roe_calculated_growth_qoq"),
        temporal_change(_roe_expr(), 4, check_sign_crossing=True).alias("roe_calculated_growth_yoy"),
        temporal_change(_roic_expr(), 1, check_sign_crossing=True).alias("roic_calculated_growth_qoq"),
        temporal_change(_roic_expr(), 4, check_sign_crossing=True).alias("roic_calculated_growth_yoy"),
    ]


def get_profitability_expressions() -> list[pl.Expr]:
    """
    Return all profitability metric expressions (snapshot + growth).

    Use this for efficient batch calculation with other metric types.
    """
    return get_profitability_snapshot_expressions() + get_profitability_growth_expressions()
