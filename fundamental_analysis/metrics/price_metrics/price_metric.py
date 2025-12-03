"""Price-based metrics calculated from SEP (daily price) data."""

import polars as pl

# Trading days per year (approximate)
TRADING_DAYS_PER_YEAR = 252


def calculate_price_metrics(df_sep: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate price-based metrics for each ticker and date.

    Uses adjusted close (closeadj) for accurate returns accounting for splits/dividends.

    Parameters
    ----------
    df_sep : pl.DataFrame
        SEP price data with columns: ticker, date, closeadj, high, low

    Returns
    -------
    pl.DataFrame
        DataFrame with columns:
        - ticker, date, closeadj (original)
        - return_1y: 1-year return
        - return_5y_or_longest: 5-year return (or longest available)
        - return_period_days: number of days used for 5y/longest return
        - max_drawdown_1y: max drawdown in past 1 year
        - max_drawdown_5y: max drawdown in past 5 years (or longest)
        - pct_from_high_5y: % change from 5-year high (negative = below high)
        - pct_from_low_5y: % change from 5-year low (positive = above low)
        - volatility_1y: annualized volatility (std dev of daily returns)
        - pct_from_sma_200: % difference from 200-day SMA
    """
    # Ensure sorted by ticker and date
    df = df_sep.select(["ticker", "date", "closeadj"]).sort("ticker", "date")

    # Calculate daily returns
    df = df.with_columns([
        (pl.col("closeadj") / pl.col("closeadj").shift(1).over("ticker") - 1)
        .alias("daily_return")
    ])

    # Define window sizes
    days_1y = TRADING_DAYS_PER_YEAR
    days_5y = TRADING_DAYS_PER_YEAR * 5
    days_200 = 200

    # 1-year return with base price
    df = df.with_columns([
        pl.col("closeadj").shift(days_1y).over("ticker").alias("_price_1y_ago"),
        (pl.col("closeadj") / pl.col("closeadj").shift(days_1y).over("ticker") - 1)
        .alias("return_1y")
    ])

    # 5-year (or longest) return
    # First, calculate the row number within each ticker to know how much history we have
    df = df.with_columns([
        pl.col("ticker").cum_count().over("ticker").alias("_row_num")
    ])

    # Price from 5 years ago (or earliest available)
    df = df.with_columns([
        # Get the price from 5y ago, or earliest if less history
        pl.when(pl.col("_row_num") >= days_5y)
        .then(pl.col("closeadj").shift(days_5y).over("ticker"))
        .otherwise(pl.col("closeadj").first().over("ticker"))
        .alias("_price_5y_ago"),

        # Track how many days back we're comparing
        pl.when(pl.col("_row_num") >= days_5y)
        .then(pl.lit(days_5y))
        .otherwise(pl.col("_row_num"))
        .alias("return_period_days"),
    ])

    df = df.with_columns([
        (pl.col("closeadj") / pl.col("_price_5y_ago") - 1).alias("return_5y_or_longest")
    ])

    # Rolling high/low for 5 years (or available history)
    df = df.with_columns([
        pl.col("closeadj").rolling_max(window_size=days_5y, min_samples=1).over("ticker")
        .alias("_high_5y"),
        pl.col("closeadj").rolling_min(window_size=days_5y, min_samples=1).over("ticker")
        .alias("_low_5y"),
    ])

    # Pct from high/low
    df = df.with_columns([
        (pl.col("closeadj") / pl.col("_high_5y") - 1).alias("pct_from_high_5y"),
        (pl.col("closeadj") / pl.col("_low_5y") - 1).alias("pct_from_low_5y"),
    ])

    # Max drawdown calculation
    # Drawdown = (current price - running max) / running max
    df = df.with_columns([
        # Running max for 1y window
        pl.col("closeadj").rolling_max(window_size=days_1y, min_samples=1).over("ticker")
        .alias("_running_max_1y"),
        # Running max for 5y window
        pl.col("closeadj").rolling_max(window_size=days_5y, min_samples=1).over("ticker")
        .alias("_running_max_5y"),
    ])

    df = df.with_columns([
        # Drawdown at each point
        ((pl.col("closeadj") - pl.col("_running_max_1y")) / pl.col("_running_max_1y"))
        .alias("_drawdown_1y"),
        ((pl.col("closeadj") - pl.col("_running_max_5y")) / pl.col("_running_max_5y"))
        .alias("_drawdown_5y"),
    ])

    # Max drawdown is the minimum (most negative) drawdown in the period
    df = df.with_columns([
        pl.col("_drawdown_1y").rolling_min(window_size=days_1y, min_samples=1).over("ticker")
        .alias("max_drawdown_1y"),
        pl.col("_drawdown_5y").rolling_min(window_size=days_5y, min_samples=1).over("ticker")
        .alias("max_drawdown_5y"),
    ])

    # Volatility (annualized std dev of daily returns)
    df = df.with_columns([
        (pl.col("daily_return").rolling_std(window_size=days_1y, min_samples=20).over("ticker")
         * (TRADING_DAYS_PER_YEAR ** 0.5))
        .alias("volatility_1y")
    ])

    # 200-day SMA and pct from it
    df = df.with_columns([
        pl.col("closeadj").rolling_mean(window_size=days_200, min_samples=days_200).over("ticker")
        .alias("_sma_200")
    ])

    df = df.with_columns([
        (pl.col("closeadj") / pl.col("_sma_200") - 1).alias("pct_from_sma_200")
    ])

    # Select final columns, renaming underscore-prefixed temps to output names
    return df.select([
        "ticker", "date", "closeadj",
        pl.col("_price_1y_ago").alias("price_1y_ago"), "return_1y",
        pl.col("_price_5y_ago").alias("price_5y_ago"), "return_5y_or_longest", "return_period_days",
        "max_drawdown_1y", "max_drawdown_5y",
        pl.col("_high_5y").alias("high_5y"), "pct_from_high_5y",
        pl.col("_low_5y").alias("low_5y"), "pct_from_low_5y",
        "volatility_1y",
        pl.col("_sma_200").alias("sma_200"), "pct_from_sma_200",
    ])
