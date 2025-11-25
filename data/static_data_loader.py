"""
Static Data Loader - Loads pre-downloaded historical data for backtesting.

This module provides functionality to load static/cached historical market data
from CSV files instead of fetching from external APIs. This is useful for:
- Reproducible backtesting
- Faster backtesting (no API calls)
- Offline backtesting
- Consistent data across multiple backtests
"""

import pandas as pd
import json
from typing import List, Optional, Dict, Any
from datetime import datetime
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class StaticDataLoader:
    """Loads historical market data from static CSV files."""

    def __init__(self, data_dir: str = 'static_data'):
        """
        Initialize the static data loader.

        Args:
            data_dir: Directory containing static data files
        """
        self.data_dir = Path(data_dir)

        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        self.daily_dir = self.data_dir / 'daily'
        self.hourly_dir = self.data_dir / 'hourly'

        # Load summary if available
        self.summary = self._load_summary()

        logger.info(f"StaticDataLoader initialized with data from: {self.data_dir}")

    def _load_summary(self) -> Dict[str, Any]:
        """Load data summary file if it exists."""
        summary_file = self.data_dir / 'data_summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                return json.load(f)

        return {}

    def list_available_symbols(self, interval: str = '1d') -> List[str]:
        """
        Get list of available symbols.

        Args:
            interval: Data interval ('1d' for daily, '1h' for hourly)

        Returns:
            List of symbol names
        """
        data_dir = self.daily_dir if interval == '1d' else self.hourly_dir

        if not data_dir.exists():
            logger.warning(f"Directory not found: {data_dir}")
            return []

        # Find all CSV files
        csv_files = list(data_dir.glob(f"*_{interval}.csv"))
        symbols = [f.stem.replace(f"_{interval}", "") for f in csv_files]

        logger.info(f"Found {len(symbols)} symbols for interval {interval}")

        return sorted(symbols)

    def get_historical_data(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """
        Load historical data for a symbol from CSV file.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'SPY')
            start_date: Start date in 'YYYY-MM-DD' format (optional)
            end_date: End date in 'YYYY-MM-DD' format (optional)
            interval: Data interval ('1d' for daily)

        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Determine file path
            data_dir = self.daily_dir if interval == '1d' else self.hourly_dir
            filepath = data_dir / f"{symbol}_{interval}.csv"

            if not filepath.exists():
                logger.warning(f"Data file not found for {symbol}: {filepath}")
                return pd.DataFrame()

            # Load CSV
            df = pd.read_csv(filepath)

            if df.empty:
                logger.warning(f"Empty data file for {symbol}")
                return pd.DataFrame()

            # Convert date column to datetime
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)

            # Filter by date range if specified
            if start_date:
                start_dt = pd.to_datetime(start_date)
                df = df[df.index >= start_dt]

            if end_date:
                end_dt = pd.to_datetime(end_date)
                df = df[df.index <= end_dt]

            logger.info(f"Loaded {len(df)} bars for {symbol} from {filepath}")

            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            return pd.DataFrame()

    def get_metadata(self, symbol: str, interval: str = '1d') -> Dict[str, Any]:
        """
        Get metadata for a symbol's data file.

        Args:
            symbol: Stock symbol
            interval: Data interval

        Returns:
            Dictionary with metadata
        """
        try:
            data_dir = self.daily_dir if interval == '1d' else self.hourly_dir
            metadata_file = data_dir / f"{symbol}_{interval}_metadata.json"

            if not metadata_file.exists():
                return {}

            with open(metadata_file, 'r') as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading metadata for {symbol}: {e}")
            return {}

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        interval: str = '1d'
    ) -> Dict[str, pd.DataFrame]:
        """
        Load data for multiple symbols.

        Args:
            symbols: List of stock symbols
            start_date: Start date (optional)
            end_date: End date (optional)
            interval: Data interval

        Returns:
            Dictionary mapping symbols to DataFrames
        """
        logger.info(f"Loading data for {len(symbols)} symbols")

        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date, interval)
            if not df.empty:
                data[symbol] = df

        logger.info(f"Successfully loaded {len(data)} symbols")

        return data

    def get_date_range(self, symbol: str, interval: str = '1d') -> tuple:
        """
        Get the date range available for a symbol.

        Args:
            symbol: Stock symbol
            interval: Data interval

        Returns:
            Tuple of (start_date, end_date) as strings, or (None, None) if not found
        """
        metadata = self.get_metadata(symbol, interval)

        if metadata:
            return (metadata.get('start_date'), metadata.get('end_date'))

        # Fallback: load data to get range
        df = self.get_historical_data(symbol, interval=interval)
        if not df.empty:
            return (str(df.index.min()), str(df.index.max()))

        return (None, None)

    def validate_data(self, symbol: str, interval: str = '1d') -> Dict[str, Any]:
        """
        Validate data quality for a symbol.

        Args:
            symbol: Stock symbol
            interval: Data interval

        Returns:
            Dictionary with validation results
        """
        df = self.get_historical_data(symbol, interval=interval)

        if df.empty:
            return {
                'valid': False,
                'error': 'No data available'
            }

        validation = {
            'valid': True,
            'total_bars': len(df),
            'missing_values': df.isnull().sum().to_dict(),
            'date_range': (str(df.index.min()), str(df.index.max())),
            'columns': list(df.columns)
        }

        # Check for gaps
        if len(df) > 1:
            date_diffs = df.index.to_series().diff()
            max_gap = date_diffs.max()
            validation['max_gap_days'] = max_gap.days if pd.notna(max_gap) else None

        # Check for negative prices
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in df.columns:
                if (df[col] <= 0).any():
                    validation['valid'] = False
                    validation['error'] = f'Negative or zero prices found in {col}'
                    break

        # Check OHLC consistency
        if all(col in df.columns for col in price_cols):
            ohlc_valid = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ).all()

            if not ohlc_valid:
                validation['warnings'] = validation.get('warnings', [])
                validation['warnings'].append('OHLC consistency issues detected')

        return validation

    def get_data_info(self) -> Dict[str, Any]:
        """
        Get information about all available data.

        Returns:
            Dictionary with data information
        """
        info = {
            'data_directory': str(self.data_dir),
            'summary': self.summary,
            'available_intervals': []
        }

        # Check for daily data
        if self.daily_dir.exists():
            daily_symbols = self.list_available_symbols('1d')
            info['available_intervals'].append({
                'interval': '1d',
                'count': len(daily_symbols),
                'symbols': daily_symbols[:10]  # First 10 as sample
            })

        # Check for hourly data
        if self.hourly_dir.exists():
            hourly_symbols = self.list_available_symbols('1h')
            info['available_intervals'].append({
                'interval': '1h',
                'count': len(hourly_symbols),
                'symbols': hourly_symbols[:10]
            })

        return info


# Convenience function for compatibility with MarketDataCollector
class StaticMarketDataCollector:
    """
    Drop-in replacement for MarketDataCollector that uses static data.

    This class provides the same interface as MarketDataCollector but loads
    data from static files instead of fetching from APIs.
    """

    def __init__(self, data_dir: str = 'static_data'):
        """Initialize with static data directory."""
        self.loader = StaticDataLoader(data_dir)
        self.data_source = 'static_files'
        logger.info("Using static data for backtesting")

    def get_historical_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Get historical data - compatible with MarketDataCollector interface."""
        return self.loader.get_historical_data(symbol, start_date, end_date, interval)

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get data for multiple symbols - compatible with MarketDataCollector."""
        return self.loader.get_multiple_symbols(symbols, start_date, end_date)

    def list_available_symbols(self) -> List[str]:
        """List all available symbols."""
        return self.loader.list_available_symbols()


if __name__ == '__main__':
    # Test the loader
    print("Testing StaticDataLoader...")
    print("=" * 80)

    loader = StaticDataLoader()

    # List available symbols
    symbols = loader.list_available_symbols()
    print(f"\nAvailable symbols: {len(symbols)}")
    print(f"Sample: {symbols[:10]}")

    # Load a sample symbol
    if symbols:
        test_symbol = symbols[0]
        print(f"\nLoading data for {test_symbol}...")

        df = loader.get_historical_data(test_symbol)
        print(f"Loaded {len(df)} bars")
        print(f"\nFirst few rows:")
        print(df.head())

        # Get metadata
        metadata = loader.get_metadata(test_symbol)
        print(f"\nMetadata:")
        print(json.dumps(metadata, indent=2))

        # Validate data
        validation = loader.validate_data(test_symbol)
        print(f"\nValidation:")
        print(json.dumps(validation, indent=2, default=str))

    # Get overall info
    info = loader.get_data_info()
    print(f"\nData Info:")
    print(json.dumps(info, indent=2))

    print("\n" + "=" * 80)
    print("StaticDataLoader test complete!")
