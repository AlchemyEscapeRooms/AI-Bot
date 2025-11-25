#!/usr/bin/env python3
"""
Download static historical stock data for backtesting.
Simplified version without external dependencies beyond yfinance.
"""

import sys
import os
import json
from datetime import datetime, timedelta
from pathlib import Path

# Try to import required libraries
try:
    import pandas as pd
    print("✓ pandas imported")
except ImportError:
    print("ERROR: pandas not installed. Please run: pip install pandas")
    sys.exit(1)

try:
    import yfinance as yf
    print("✓ yfinance imported")
except ImportError:
    print("ERROR: yfinance not installed. Please run: pip install yfinance")
    sys.exit(1)


class StaticDataDownloader:
    """Downloads and saves historical market data for backtesting."""

    def __init__(self, data_dir='static_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

        # Create subdirectories
        (self.data_dir / 'daily').mkdir(exist_ok=True)
        (self.data_dir / 'hourly').mkdir(exist_ok=True)

    def download_symbol_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Download historical data for a single symbol."""

        try:
            print(f"Downloading {symbol} from {start_date} to {end_date}...")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                print(f"  WARNING: No data available for {symbol}")
                return pd.DataFrame()

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Add symbol column
            df['symbol'] = symbol

            # Reset index to make date a column
            df.reset_index(inplace=True)
            if 'Date' in df.columns:
                df.rename(columns={'Date': 'date'}, inplace=True)
            elif 'Datetime' in df.columns:
                df.rename(columns={'Datetime': 'date'}, inplace=True)

            print(f"  ✓ Downloaded {len(df)} bars for {symbol}")

            return df

        except Exception as e:
            print(f"  ERROR downloading {symbol}: {e}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str = '1d'):
        """Save dataframe to CSV file."""

        if df.empty:
            print(f"  WARNING: No data to save for {symbol}")
            return

        # Determine subdirectory
        subdir = 'daily' if interval == '1d' else 'hourly'

        # Save to CSV
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / subdir / filename

        df.to_csv(filepath, index=False)
        print(f"  ✓ Saved {len(df)} bars to {filepath}")

        # Also save metadata
        metadata = {
            'symbol': symbol,
            'interval': interval,
            'start_date': str(df['date'].min()),
            'end_date': str(df['date'].max()),
            'total_bars': len(df),
            'columns': list(df.columns),
            'downloaded_at': datetime.now().isoformat()
        }

        metadata_file = self.data_dir / subdir / f"{symbol}_{interval}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def download_multiple_symbols(
        self,
        symbols: list,
        years: int = 3,
        interval: str = '1d'
    ):
        """Download data for multiple symbols."""

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)

        print(f"\nDownloading {len(symbols)} symbols from {start_date.date()} to {end_date.date()}\n")

        results = {
            'successful': [],
            'failed': []
        }

        for i, symbol in enumerate(symbols, 1):
            print(f"\n[{i}/{len(symbols)}] {symbol}")
            df = self.download_symbol_data(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                interval
            )

            if not df.empty:
                self.save_data(df, symbol, interval)
                results['successful'].append(symbol)
            else:
                results['failed'].append(symbol)

        # Save download summary
        summary = {
            'download_date': datetime.now().isoformat(),
            'total_symbols': len(symbols),
            'successful': len(results['successful']),
            'failed': len(results['failed']),
            'symbols_successful': results['successful'],
            'symbols_failed': results['failed'],
            'period_years': years,
            'interval': interval
        }

        summary_file = self.data_dir / 'download_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*80}")
        print(f"Download Summary:")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Summary saved to {summary_file}")
        print(f"{'='*80}")

        return results


def main():
    """Main function to download historical data."""

    # List of symbols to download
    # Mix of indices, tech stocks, financial, energy, healthcare, consumer
    symbols = [
        # Indices/ETFs
        'SPY', 'QQQ', 'IWM', 'DIA', 'VTI', 'VOO',

        # Tech - Large Cap
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'NFLX',

        # Tech - Mid Cap
        'CRM', 'ADBE', 'ORCL', 'CSCO', 'AVGO',

        # Financial
        'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK', 'SCHW',

        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'DHR', 'CVS',

        # Consumer
        'WMT', 'HD', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT',

        # Energy
        'XOM', 'CVX', 'COP', 'SLB', 'EOG',

        # Industrial
        'BA', 'CAT', 'GE', 'UPS', 'HON',

        # Communication
        'DIS', 'CMCSA', 'T', 'VZ', 'TMUS',

        # Materials
        'LIN', 'APD', 'FCX', 'NEM',

        # Utilities
        'NEE', 'DUK', 'SO', 'D',

        # Real Estate
        'AMT', 'PLD', 'CCI', 'EQIX'
    ]

    print("=" * 80)
    print("STATIC DATA DOWNLOADER FOR BACKTESTING")
    print("=" * 80)
    print(f"Total symbols to download: {len(symbols)}")
    print(f"Period: Last 3 years")
    print(f"Interval: Daily")
    print("=" * 80)

    downloader = StaticDataDownloader()

    # Download daily data
    print("\nSTARTING DOWNLOAD...")
    results = downloader.download_multiple_symbols(symbols, years=3, interval='1d')

    print("\n" + "=" * 80)
    print("DOWNLOAD COMPLETE!")
    print("=" * 80)
    print(f"Successfully downloaded: {len(results['successful'])} symbols")
    print(f"Failed downloads: {len(results['failed'])} symbols")

    if results['failed']:
        print(f"\nFailed symbols: {', '.join(results['failed'])}")

    print(f"\nData saved to: {downloader.data_dir.absolute()}/daily/")
    print("=" * 80)


if __name__ == '__main__':
    main()
