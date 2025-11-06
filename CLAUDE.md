# CLAUDE.md

Guidance to Claude Code (claude.ai/code).

## Project Overview

A Python-based fundamental analysis system for screening stocks based on financial metrics. The primary goal is to identify companies whose fundamental metrics have statistical outlier values, meaning they are potentially over/undervalued companies (e.g., price-to-cash, P/E, P/B) against industry peers.
The system should be capable of running backtest.

**Core Methodology:**
- Fetch historical fundamental data from Sharadar (NASDAQ Data Link)
- Perform point-in-time analysis: analyze at a given date using only data available within a rolling window before that date. 
- Calculate key fundamental ratios for all companies at a given date
  - Only use fundamental ratios (not absolute values) to normalize the metrics.
- Segment companies by industry (GICS sectors) and market cap and potentially by other dimensions.
  - The segmentation logic needs to be abstracted so that the expansion can be made.
  - The segmentation is defined as function that translates a stock to a string (segment).
  - A number of segmentation will be allowed, and each segmentation is registered with registry logic.
    - Here registry means annotation with str argument, e.g. @segmentation("sector_cap"), example:
```python
  @segmentation("sector_cap")
  def sector_market_cap_segmentation(ticker_data):
      # assuming ticker_data has `sector` and `cap_bucket` attributes.
      return f"{ticker_data.sector}_{ticker_data.cap_bucket}"
```
- Compute statistical distributions (mean, std dev) within each segment.
- Get Outliers within segmentation 
  - There might be a number of ways to get the outliers. Each outlier finding logic needs to be registered with registry logic (python annotation with str arg).
  - Outliers may be companies with value off > 2 sigma from mean
  - Outliers may be companies with value > 2 times mean
- Calculate forward returns from analysis date for backtesting performance
- Use Polars instead of Pandas for analysis calculations for multithreading speed-up.
- Using pandas for lightweight task such as data fetching etc. is fine.
- Leave __init__.py mostly empty unless necessary. Avoid using __all__ in the __init__.py.
- I do not want the code be verbose. Refrain from adding non-informative, trivial comment. 
  - It is the best if the meaning of variable can be understood from its name without redundant comment.
  - Do not make the doc string to be absolutely formal. Follow the format, but do not add args, return, unless necessary.
    - If the args and return values are obvious, omit it in the doc string.

**Key Design Principles:**
- **No Look-Ahead Bias**: Only use data that would have been available at the analysis date
- **Historical Analysis**: Support running screening at any historical date
- **Forward Returns**: Calculate future performance (e.g., 1M, 3M, 6M, 1Y returns) from analysis date
- **Backtesting**: Enable evaluation of screening strategy performance over time

## Architecture

**Data Pipeline:**
1. **Data Acquisition Layer**: API clients for fetching historical fundamental data and price data
   - Fundamental data: Balance sheets, income statements, cash flow (SF1 table)
   - Price data: Historical prices for forward return calculation (SEP table)
   - Metadata: Ticker information, sector classifications (TICKERS table)
2. **Point-in-Time Processing**: Ensure only data available at analysis date is used
3. **Data Processing Layer**: polars-based transformations to calculate ratios and metrics
4. **Segmentation Engine**: Group companies by industry classification and size
5. **Statistical Analysis**: Calculate distributions and identify outliers within segments
6. **Forward Returns Calculation**: Compute future returns (1M, 3M, 6M, 1Y) from analysis date
7. **Screening/Reporting**: Output undervalued candidates with supporting metrics and forward returns

**Key Metrics to Calculate:**

Positive (higher is better) metrics
- Return on Equity (ROE)
- Return on Invested Capital (ROIC)
- Earnings per Share (EPS) Growth Rate
- Revenue Growth Rate
- Current Ratio

Negative (lower is better) metrics
- Price-to-Cash ratio
- Price-to-Book (P/B)
- Price-to-Earnings (P/E)
- Debt-to-Equity
- EV/EBITDA Ratio

Note:
- **EV/EBITDA Ratio** *(low = good, high = bad; sector-dependent)*
- **Current Ratio** *(too high can indicate inefficient asset use)*

## Project Structure

```
workfolder/
├── fundamental_analysis/
│   ├── __init__.py
│   ├── data_acquisition/            # API clients and data fetching
│   │   ├── __init__.py
│   │   ├── sharadar_client.py       # Sharadar/NASDAQ Data Link integration
│   │   └── data_fetcher.py          # Unified interface for data sources
│   ├── metrics/                     # Financial ratio calculations
│   │   ├── __init__.py
│   │   ├── fundamental_ratios.py    # P/E, P/B, P/C, EV/EBITDA calculations
│   │   ├── financial_health.py      # Debt ratios, current ratio, etc.
│   │   ├── profitability.py         # ROE, ROIC, ROA, margins
│   │   └── growth_metrics.py        # EPS growth rate, revenue growth rate
│   ├── segmentation/                # Industry and size classification
│   │   ├── __init__.py
│   │   ├── registry.py              # segmentation registry
│   │   ├── industry_classifier.py   # GICS sector mapping
│   │   └── market_cap_classifier.py # Size bucket assignment
│   ├── screening/                   # Statistical analysis and outlier detection
│   │   ├── __init__.py
│   │   ├── registry.py              # screening registry
│   │   ├── sigma.py                 # Sigma-based outlier identification
│   │   ├── value_to_mean.py         # value to mean ratio based outlier identification
│   │   └── screener.py              # Main screening logic
│   ├── backtesting/                 # Backtesting and performance evaluation
│   │   ├── __init__.py
│   │   ├── forward_returns.py       # Calculate forward returns from analysis date
│   │   ├── point_in_time.py         # Ensure no look-ahead bias in data
│   │   └── performance_evaluator.py # Evaluate strategy performance over time
│   └── utils/                       # Helper functions
│       ├── __init__.py
│       ├── config.py                # Configuration management
│       └── logger.py                # Logging setup
├── tests/
│   ├── __init__.py
│   ├── test_metrics.py              # Unit tests for ratio calculations
│   ├── test_segmentation.py         # Tests for classification logic
│   ├── test_screening.py            # Tests for outlier detection
│   └── fixtures/                    # Sample data for testing
│       └── sample_financials.json
├── data/                            # Local data storage (gitignored)
│   ├── raw/                         # Raw API responses (Parquet format)
│   └── processed/                   # Calculated metrics (Parquet format)
├── notebooks/                       # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
├── results/                         # Screening outputs (gitignored)
│   └── screened_stocks.csv
├── main.py                          # Entry point for running analysis
├── main_fetch.py                    # Entry point for fetching
├── .env.example                     # Example environment variables
├── .gitignore
├── requirements.txt                 # Python dependencies
├── README.md
└── CLAUDE.md                        # This file
```

**Key Components:**
- `data_acquisition/`: Handles all external API calls and data fetching
- `metrics/`: Pure calculation functions for financial ratios
- `segmentation/`: Groups companies by industry and market cap
- `screening/`: Statistical analysis to find outliers
- `data/`: Local cache for API responses stored as Parquet files (add to .gitignore)
- `results/`: Output CSV files with screening results

**Understanding SF1 Date Fields:**

The SF1 table contains three critical date fields that serve different purposes:
- **`reportperiod`**: The actual fiscal quarter end date (varies by company based on their fiscal calendar)
- **`calendardate`**: Normalized to standard calendar quarters (Mar 31, Jun 30, Sep 30, Dec 31) for easier cross-company comparison
- **`datekey`**: The date when the data became available in Sharadar (critical for point-in-time correctness)

**Why Quarterly Partitioning Was Not Chosen for SF1:**

Despite fundamentals being quarterly in nature, we use snapshot-based storage instead of quarterly partitioning due to:
1. **Irregular reporting delays**: The gap between `reportperiod` and `datekey` is highly variable:
   - Median: 44 days
   - P75: 58 days
   - P90: 89 days
   - Some companies: >100 days (late filings, restatements)
2. **Fiscal vs. calendar mismatch**: `reportperiod` reflects each company's fiscal calendar, while `calendardate` is normalized
3. **Point-in-time complexity**: Quarterly partitions would mix data with different availability dates (`datekey`), making point-in-time analysis difficult

**Data Storage Format:**

All local data is stored in **Parquet format** for efficient storage and fast I/O with Polars:

- **Snapshot-based storage for SF1 (fundamentals)**:
  - SF1 fundamentals: `data/raw/sf1/sf1_snapshot_YYYY-MM-DD.parquet` (e.g., `sf1_snapshot_2023-12-31.parquet`)
  - Each file contains ALL historical data from 1990-01-01 to the snapshot date (adjusted for reporting delay)
  - Filters by `datekey` to ensure point-in-time correctness (only data available by that date)
  - Applies REPORTING_DELAY_DAYS (45 days) to account for median filing lag
  - Use `--overwrite` to regenerate snapshots, otherwise skips existing files

- **Monthly-partitioned storage for SEP (price data)**:
  - SEP prices: `data/raw/sep/sep_YYYY-MM.parquet` (e.g., `sep_2023-01.parquet`, `sep_2023-02.parquet`)
  - Contains OHLCV data (Open, High, Low, Close, Volume) plus adjusted close prices
  - Partitioned by month from 1990-01-01 onwards, excluding the incomplete current month
  - Each file contains all tickers for that specific month (~6,000 tickers × ~20 trading days)
  - Monthly partitioning balances file size and number of files (~420 files for full history)

- **Dated TICKERS metadata**: `data/raw/tickers/tickers_YYYY-MM-DD.parquet`
  - Filename includes the snapshot date for consistency with other data types
  - Example: `tickers_2023-12-31.parquet` (17,331 tickers, 699KB)

- **Processed metrics**: `data/processed/*.parquet`

- **Overwrite behavior**: Use `--overwrite` flag to re-download existing files, otherwise skips existing snapshots/months/dates

## Development Commands

### Setup
> Currently do not worry about setting up venv for the project.

### Running Analysis
```bash
#  Fetch the fundamental data up to a specific date
#  This creates a snapshot of ALL historical data from 1990-01-01 to end-date
#  By default, skips existing files (incremental updates)
python main_fetch.py --end-date 2023-12-31

#  Force re-download and overwrite existing files
python main_fetch.py --end-date 2023-12-31 --overwrite

# Run screening analysis at a specific historical date
python main.py --analysis-date 2020-01-15

# Run backtesting over a date range
# The backtest does the analysis per rebalance frequency and collect the results together.
python main.py --backtest --start-date 2018-01-01 --end-date 2023-12-31 --rebalance-frequency 1M

# Run specific screening strategy with forward returns
python main.py --strategy price_to_cash --sigma 2.0 --analysis-date 2020-01-15 --forward-periods 1M,3M,6M,1Y

# Analyze specific sector with backtesting
python main.py --sector Technology --market-cap large --backtest --start-date 2020-01-01 --end-date 2023-12-31
```
The `--strategy` flag is required for `main.py`. `all` value is default for `--sector` and `--market-cap`.

**Time Period Format:**
- Use suffix notation for time periods (e.g., `1M`, `3M`, `6M`, `1Y`)
- Format: `<number><unit>` where unit is:
  - `M` = months (e.g., `10M` = 10 months)
  - `Y` = years (e.g., `1Y` = 1 year)
- Examples: `--forward-periods 1M,3M,6M,1Y` or `--rebalance-frequency 1M`

### Testing
> Currently do not worry about testing.

## Data Sources

**Primary Provider**: Sharadar via NASDAQ Data Link (https://data.nasdaq.com/databases/SFA)
- Comprehensive historical fundamental data for US stocks
- High-quality, standardized fundamental metrics
- Access via `nasdaq-data-link` Python library (preferred) or REST API
- Requires API key (set as `NASDAQ_DATA_API_KEY` environment variable)
- **Key Tables Used**:
  - `SF1` (Core Fundamental Data): Quarterly/annual fundamentals with standardized metrics
    - **Dimension**: Use **ARQ (As Reported Quarterly)** for analysis
  - `TICKERS`: Ticker metadata including industry classification and market cap
    - Fetches **all tickers including delisted companies** for point-in-time correctness
    - Rationale: Delisted stocks must be included if analysis date is before delisting occurred
    - Point-in-time filtering happens during analysis, not during data fetch
  - `SEP` (Sharadar Equity Prices): Daily OHLCV price data for calculating forward returns
    - **Requires separate SEP subscription** (in addition to SFA)
    - Columns: ticker, date, open, high, low, close, volume, closeadj (dividend-adjusted), closeunadj
    - Used for: calculating forward returns, price-based ratios, backtesting performance
  - `DAILY` (Daily metrics): Contains daily fundamental ratios (P/E, P/B, market cap, enterprise value)
    - Part of SFA subscription, no separate subscription needed

**Development Approach**:
- Start with a small test set of tickers (~10-50 stocks) for faster iteration and debugging
- Test point-in-time logic thoroughly with known historical dates
- Validate forward returns calculations against known benchmarks
- Scale to full universe after validating the pipeline

## Important Considerations

**Point-in-Time Data Integrity**:
- **Critical for backtesting**: Only use data available at the analysis date to avoid look-ahead bias
- **Sharadar's ARQ dimension provides true point-in-time dates**:
  - `reportperiod`: Actual fiscal quarter end (varies by company)
  - `calendardate`: Normalized calendar quarter end (Mar 31, Jun 30, Sep 30, Dec 31)
  - `datekey`: When data became available in Sharadar (critical for point-in-time correctness)
  - Gap between `reportperiod` and `datekey` reflects real-world filing delays (median: 44 days, some >100 days)
- **Restatement limitation**: Sharadar only keeps current/restated values, not original as-filed numbers
  - Historical data reflects the most recent restatements/corrections
  - This introduces slight optimism bias in backtesting (restated numbers are more accurate)
- **Our implementation**:
  - Fetch all data where `calendardate <= (end_date - REPORTING_DELAY_DAYS)`
  - Filter by `datekey <= (end_date - REPORTING_DELAY_DAYS)` to ensure point-in-time correctness
  - Apply 45-day reporting delay to account for median filing lag
  - Store as snapshot files for efficient point-in-time analysis
- Price data must be from the analysis date or before for ratio calculations

**Industry Classification**: Use standard classification systems (GICS, SIC, or NAICS) for proper peer comparison. Different industries have different normal ranges for ratios (e.g., tech companies typically have higher P/E ratios).

**Market Cap Segmentation**: Common buckets are:
- Mega cap: > $200B
- Large cap: $10B - $200B
- Mid cap: $2B - $10B
- Small cap: $300M - $2B
- Micro cap: < $300M

**Statistical Considerations**:
- Use robust statistics (median, IQR) alongside mean/stdev for skewed distributions
- Financial ratios often have outliers and non-normal distributions
- Consider winsorizing extreme values before calculating statistics
- Be cautious with companies that have negative denominators (e.g., negative earnings for P/E)

**Data Quality**:
- Handle missing data appropriately (not all companies report all metrics)
- Account for different reporting periods and currencies
- Validate data freshness (fundamental data is typically quarterly/annual)
