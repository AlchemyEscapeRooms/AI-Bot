# Static Data for Backtesting

This directory contains pre-downloaded historical market data for backtesting purposes.

## Directory Structure

```
static_data/
├── README.md (this file)
├── data_summary.json (metadata about all data)
├── daily/ (daily OHLCV data)
│   ├── AAPL_1d.csv
│   ├── AAPL_1d_metadata.json
│   ├── SPY_1d.csv
│   └── ...
└── hourly/ (hourly data - optional)
```

## Data Format

Each CSV file contains:
- `date`: Trading date
- `open`, `high`, `low`, `close`: OHLC prices
- `volume`: Trading volume
- `dividends`: Dividend payments
- `stock splits`: Stock split ratios
- `symbol`: Stock symbol

## Available Symbols

Run this command to see all available symbols:

```bash
python3 main.py list-static
```

## Current Data

- **Total Symbols**: 36+ stocks and ETFs
- **Data Period**: ~3 years (782 trading days)
- **Data Type**: Synthetic (realistic simulated data)
- **Generated**: See `data_summary.json` for timestamp

## Included Symbols

### Indices/ETFs
SPY, QQQ, IWM, DIA, VTI, VOO

### Technology
AAPL, MSFT, GOOGL, AMZN, META, NVDA, TSLA, AMD, INTC, NFLX, CRM, ADBE, ORCL, CSCO, AVGO

### Financial
JPM, BAC, WFC, GS, MS, C, BLK, SCHW

### Healthcare
JNJ, UNH, PFE, ABBV, TMO, DHR, CVS

### Consumer
WMT, HD, COST, NKE, MCD, SBUX, TGT

### Energy
XOM, CVX, COP, SLB, EOG

### Industrial
BA, CAT, GE, UPS, HON

### Communication
DIS, CMCSA, T, VZ, TMUS

### Other Sectors
Materials, Utilities, Real Estate

## Usage

### Quick Test

```bash
# Run a backtest using static data
python3 main.py backtest --symbol SPY --static --strategy momentum
```

### List All Symbols

```bash
python3 main.py list-static
```

### Compare Strategies

```bash
python3 main.py backtest --symbol AAPL --static --compare
```

## Regenerating Data

To regenerate the data:

```bash
# Generate synthetic data (no dependencies needed)
python3 generate_sample_data.py

# Or download real data (requires yfinance)
python3 download_static_data_simple.py
```

## Data Quality

All data includes:
- ✓ Realistic price movements
- ✓ Appropriate volatility levels
- ✓ Volume patterns
- ✓ OHLC consistency checks
- ✓ Metadata for validation

## For More Information

See the main documentation: `STATIC_DATA_BACKTESTING.md`
