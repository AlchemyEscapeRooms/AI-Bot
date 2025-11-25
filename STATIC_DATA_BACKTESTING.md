# Static Data Backtesting

This document describes the static data backtesting feature for the AI Trading Bot.

## Overview

Static data backtesting allows you to run backtests using pre-downloaded historical market data instead of fetching from external APIs. This provides several advantages:

- **Faster Backtesting**: No API calls means faster execution
- **Reproducible Results**: Same data every time ensures consistent results
- **Offline Capability**: Run backtests without internet connection
- **Cost Savings**: No API rate limits or costs
- **Benchmarking**: Consistent data for comparing different strategies

## Table of Contents

1. [Data Structure](#data-structure)
2. [Generating Static Data](#generating-static-data)
3. [Using Static Data for Backtesting](#using-static-data-for-backtesting)
4. [Command Line Interface](#command-line-interface)
5. [Python API](#python-api)
6. [Examples](#examples)

## Data Structure

Static data is stored in CSV files under the `static_data/` directory:

```
static_data/
├── daily/
│   ├── AAPL_1d.csv
│   ├── AAPL_1d_metadata.json
│   ├── SPY_1d.csv
│   ├── SPY_1d_metadata.json
│   └── ...
├── hourly/
│   └── (hourly data files)
└── data_summary.json
```

### CSV Format

Each CSV file contains OHLCV data with the following columns:

- `date`: Trading date (YYYY-MM-DD)
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume
- `dividends`: Dividend payments
- `stock splits`: Stock split ratios
- `symbol`: Stock symbol

### Metadata Files

Each symbol has a corresponding metadata JSON file with information about:

- Symbol name
- Date range
- Number of bars
- Download/generation timestamp
- Data type (real or synthetic)

## Generating Static Data

### Option 1: Generate Sample Data (Recommended for Testing)

Use the sample data generator to create realistic synthetic data:

```bash
python3 generate_sample_data.py
```

This generates 3 years of daily data for 36+ symbols including:
- Major indices (SPY, QQQ, IWM, DIA)
- Tech stocks (AAPL, MSFT, GOOGL, NVDA, TSLA, etc.)
- Financial stocks (JPM, BAC, GS, etc.)
- Healthcare, Energy, Industrial, and more

### Option 2: Download Real Data

If you have yfinance installed, use the real data downloader:

```bash
python3 download_static_data_simple.py
```

This downloads actual historical data from Yahoo Finance.

## Using Static Data for Backtesting

### Prerequisites

Install required dependencies:

```bash
pip install pandas numpy
```

For downloading real data, also install:

```bash
pip install yfinance
```

### Quick Start

1. Generate sample data:
   ```bash
   python3 generate_sample_data.py
   ```

2. List available symbols:
   ```bash
   python3 main.py list-static
   ```

3. Run a backtest:
   ```bash
   python3 main.py backtest --symbol SPY --static --strategy momentum
   ```

## Command Line Interface

### List Available Symbols

```bash
python3 main.py list-static
```

Shows all symbols available in the static data directory.

### Run Single Strategy Backtest

```bash
python3 main.py backtest \
    --symbol AAPL \
    --static \
    --strategy momentum \
    --start-date 2023-01-01 \
    --end-date 2023-12-31
```

**Options:**
- `--symbol`: Stock symbol to backtest (required)
- `--static`: Use static data mode (required for static backtesting)
- `--strategy`: Strategy name (momentum, mean_reversion, rsi, macd, etc.)
- `--start-date`: Start date in YYYY-MM-DD format
- `--end-date`: End date in YYYY-MM-DD format
- `--output`: Save results to file (JSON format)

### Compare All Strategies

```bash
python3 main.py backtest \
    --symbol SPY \
    --static \
    --compare \
    --start-date 2023-01-01
```

This compares all available strategies on the same symbol and data.

### Save Results to File

```bash
python3 main.py backtest \
    --symbol AAPL \
    --static \
    --strategy rsi \
    --output backtest_results.json
```

## Python API

### Using StaticDataLoader

```python
from data.static_data_loader import StaticDataLoader

# Initialize loader
loader = StaticDataLoader(data_dir='static_data')

# List available symbols
symbols = loader.list_available_symbols()
print(f"Available: {symbols}")

# Load data for a symbol
df = loader.get_historical_data('AAPL', start_date='2023-01-01')
print(df.head())

# Get metadata
metadata = loader.get_metadata('AAPL')
print(metadata)

# Validate data quality
validation = loader.validate_data('AAPL')
print(validation)
```

### Using StaticBacktestRunner

```python
from backtesting.static_backtest_runner import StaticBacktestRunner

# Initialize runner
runner = StaticBacktestRunner(
    data_dir='static_data',
    initial_capital=100000.0,
    commission=0.001,
    slippage=0.0005
)

# Run single backtest
result = runner.run_single_backtest(
    symbol='AAPL',
    strategy_name='momentum',
    start_date='2023-01-01',
    end_date='2023-12-31'
)

print(f"Total Return: {result['total_return']:.2f}%")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")
print(f"Win Rate: {result['win_rate']:.2f}%")

# Compare strategies
comparison = runner.run_strategy_comparison(
    symbol='SPY',
    strategies=['momentum', 'mean_reversion', 'rsi']
)

print(comparison[['strategy_name', 'total_return', 'sharpe_ratio']])

# Run across multiple symbols
multi_results = runner.run_multi_symbol_backtest(
    symbols=['AAPL', 'MSFT', 'GOOGL'],
    strategy_name='momentum'
)

print(multi_results[['symbol', 'total_return', 'sharpe_ratio']])
```

### Batch Backtesting

```python
# Run batch backtest
config = {
    'symbols': ['SPY', 'QQQ', 'AAPL', 'MSFT'],
    'strategies': ['momentum', 'mean_reversion', 'rsi'],
    'start_date': '2023-01-01',
    'end_date': '2023-12-31',
    'max_symbols': 10
}

batch_results = runner.run_batch_backtest(config)

print(f"Total backtests: {batch_results['total_backtests']}")
print(f"Average return: {batch_results['avg_return']:.2f}%")
print(f"Average Sharpe: {batch_results['avg_sharpe']:.2f}")

# Access detailed results
results_df = batch_results['results']
top_performers = results_df.nlargest(5, 'total_return')
print(top_performers[['symbol', 'strategy_name', 'total_return']])
```

## Examples

### Example 1: Basic Backtest

```bash
# Generate sample data
python3 generate_sample_data.py

# Run momentum strategy on SPY
python3 main.py backtest --symbol SPY --static --strategy momentum

# Output:
# ================================================================================
# BACKTEST RESULTS - SPY
# ================================================================================
# Strategy: momentum
# Period: 2022-11-24 to 2025-11-21
#
# Performance Metrics:
#   Total Return: 15.23%
#   Sharpe Ratio: 1.45
#   Max Drawdown: -8.34%
#
# Trading Statistics:
#   Total Trades: 42
#   Win Rate: 57.14%
#   Profit Factor: 1.82
# ================================================================================
```

### Example 2: Strategy Comparison

```bash
# Compare all strategies on AAPL
python3 main.py backtest --symbol AAPL --static --compare

# Output shows ranking of all strategies by performance
```

### Example 3: Multi-Symbol Analysis

```python
from backtesting.static_backtest_runner import StaticBacktestRunner

runner = StaticBacktestRunner()

# Get all available symbols
symbols = runner.list_available_symbols()

# Run momentum strategy on all tech stocks
tech_symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']
results = runner.run_multi_symbol_backtest(
    symbols=tech_symbols,
    strategy_name='momentum',
    start_date='2023-01-01'
)

# Sort by Sharpe ratio
best_performers = results.nlargest(3, 'sharpe_ratio')
print(best_performers[['symbol', 'total_return', 'sharpe_ratio', 'max_drawdown']])
```

### Example 4: Custom Strategy Testing

```python
# Define custom strategy parameters
custom_params = {
    'lookback': 30,
    'threshold': 0.03,
    'position_size': 0.15
}

# Run with custom parameters
result = runner.run_single_backtest(
    symbol='NVDA',
    strategy_name='momentum',
    strategy_params=custom_params,
    start_date='2023-01-01'
)

print(f"Custom Strategy Return: {result['total_return']:.2f}%")
```

## Available Strategies

The following strategies are available for backtesting:

1. **momentum** - Momentum-based trading
2. **mean_reversion** - Mean reversion using Bollinger Bands
3. **trend_following** - Moving average crossovers
4. **breakout** - Price channel breakouts
5. **rsi** - RSI-based overbought/oversold
6. **macd** - MACD crossover signals
7. **pairs_trading** - Statistical arbitrage
8. **ml_hybrid** - Multi-indicator hybrid strategy

Each strategy has configurable parameters that can be customized.

## Performance Metrics

All backtests report comprehensive metrics:

### Return Metrics
- Total Return (%)
- Net Profit ($)
- Profit Factor

### Risk-Adjusted Metrics
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown (%)

### Trading Statistics
- Total Trades
- Winning Trades / Losing Trades
- Win Rate (%)
- Average Win / Average Loss
- Largest Win / Largest Loss
- Average Holding Period

## Data Quality

### Validation

The static data loader includes validation features:

```python
validation = loader.validate_data('AAPL')

if validation['valid']:
    print(f"✓ Data is valid")
    print(f"  Total bars: {validation['total_bars']}")
    print(f"  Date range: {validation['date_range']}")
else:
    print(f"✗ Data validation failed: {validation['error']}")
```

### Consistency Checks

Automatic checks include:
- OHLC price consistency (high >= low, etc.)
- No negative prices
- No missing required columns
- Date continuity

## Troubleshooting

### No Data Available Error

```
ERROR: No data available for symbol XYZ
```

**Solution**: Run `python3 main.py list-static` to see available symbols, or generate data for the symbol.

### Module Not Found Error

```
ModuleNotFoundError: No module named 'pandas'
```

**Solution**: Install dependencies:
```bash
pip install pandas numpy
```

### Empty DataFrame Warning

```
WARNING: Empty data file for symbol
```

**Solution**: Regenerate the data file using the sample data generator.

## Best Practices

1. **Generate Fresh Data Periodically**: Regenerate sample data or download new real data regularly
2. **Validate Data**: Always validate data before running large batch backtests
3. **Use Appropriate Date Ranges**: Ensure start/end dates fall within available data
4. **Save Important Results**: Use `--output` flag to save backtest results
5. **Compare Strategies**: Use `--compare` to evaluate multiple strategies
6. **Check Data Quality**: Review metadata files to understand data characteristics

## Advanced Usage

### Creating Custom Data Sets

You can create custom static data by:

1. Modifying `generate_sample_data.py` to adjust parameters
2. Adding new symbols to the symbol list
3. Changing volatility, drift, or other generation parameters
4. Creating data for different time periods

### Integrating with Live Trading

Static backtesting is ideal for:
- Strategy development
- Parameter optimization
- Performance validation

Once validated with static data, strategies can be deployed to paper trading:

```bash
# First validate with static data
python3 main.py backtest --symbol SPY --static --strategy momentum

# Then test in paper trading
python3 main.py trade --mode paper --personality momentum_trader
```

## File Reference

### Core Files

- `static_data/` - Data storage directory
- `data/static_data_loader.py` - Static data loader module
- `backtesting/static_backtest_runner.py` - Backtesting runner
- `generate_sample_data.py` - Sample data generator
- `download_static_data_simple.py` - Real data downloader
- `STATIC_DATA_BACKTESTING.md` - This documentation

### Generated Files

- `static_data/data_summary.json` - Overall data summary
- `static_data/daily/*.csv` - Daily OHLCV data
- `static_data/daily/*_metadata.json` - Per-symbol metadata

## Support

For issues or questions:
1. Check this documentation
2. Verify data exists: `python3 main.py list-static`
3. Validate data quality using the Python API
4. Review example code in this document

## License

This feature is part of the AI Trading Bot and follows the same MIT License.

---

**Note**: Static data backtesting is designed for development and testing. Always validate strategies thoroughly before using real capital.
