# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python-based fundamental analysis system for screening stocks based on financial metrics. The primary goal is to identify potentially undervalued companies by comparing fundamental ratios (e.g., price-to-cash, P/E, P/B) against industry peers, using statistical outlier detection.

**Core Methodology:**
- Fetch fundamental data from financial APIs (Tiingo or Alpha Vantage)
- Calculate key fundamental ratios for all companies
- Segment companies by industry (GICS sectors) and market cap
- Compute statistical distributions (mean, std dev) within each segment
- Identify outliers (e.g., companies > 2 sigma from segment mean)

## Architecture

**Data Pipeline:**
1. **Data Acquisition Layer**: API clients for fetching fundamental data (balance sheets, income statements, cash flow)
2. **Data Processing Layer**: pandas-based transformations to calculate ratios and metrics
3. **Segmentation Engine**: Group companies by industry classification and size
4. **Statistical Analysis**: Calculate distributions and identify outliers within segments
5. **Screening/Reporting**: Output undervalued candidates with supporting metrics

**Key Metrics to Calculate:**
- Price-to-Cash ratio
- Price-to-Book (P/B)
- Price-to-Earnings (P/E)
- Debt-to-Equity
- Current Ratio
- Return on Equity (ROE)

## Project Structure

```
workfolder/
├── fundamental_analysis/
│   ├── __init__.py
│   ├── data_acquisition/            # API clients and data fetching
│   │   ├── __init__.py
│   │   ├── tiingo_client.py         # Tiingo API integration
│   │   ├── alpha_vantage_client.py  # Alpha Vantage API (alternative)
│   │   └── data_fetcher.py          # Unified interface for data sources
│   ├── metrics/                     # Financial ratio calculations
│   │   ├── __init__.py
│   │   ├── fundamental_ratios.py    # P/E, P/B, P/C calculations
│   │   ├── financial_health.py      # Debt ratios, current ratio, etc.
│   │   └── profitability.py         # ROE, ROA, margins
│   ├── segmentation/                # Industry and size classification
│   │   ├── __init__.py
│   │   ├── industry_classifier.py   # GICS sector mapping
│   │   └── market_cap_classifier.py # Size bucket assignment
│   ├── screening/                   # Statistical analysis and outlier detection
│   │   ├── __init__.py
│   │   ├── outlier_detector.py      # Sigma-based outlier identification
│   │   └── screener.py              # Main screening logic
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
│   ├── raw/                         # Raw API responses
│   └── processed/                   # Calculated metrics
├── notebooks/                       # Jupyter notebooks for exploration
│   └── exploratory_analysis.ipynb
├── results/                         # Screening outputs (gitignored)
│   └── screened_stocks.csv
├── main.py                          # Entry point for running analysis
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
- `data/`: Local cache for API responses (add to .gitignore)
- `results/`: Output CSV files with screening results

## Development Commands

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running Analysis
```bash
# Run full screening analysis
python main.py

# Run specific screening strategy
python main.py --strategy price_to_cash --sigma 2.0

# Analyze specific sector
python main.py --sector Technology --market-cap large
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_metrics.py

# Run with coverage
pytest --cov=fundamental_analysis tests/
```

## Data Sources

**Primary Provider**: Tiingo API (https://api.tiingo.com/)
- Free tier available with reasonable rate limits
- Provides fundamental data endpoints
- Requires API key (set as `TIINGO_API_KEY` environment variable)

## Important Considerations

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
