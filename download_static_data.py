#!/usr/bin/env python3
"""
Download static historical stock data for backtesting.
This script downloads 2-3 years of historical data for multiple symbols.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import os
import sys
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from utils.logger import get_logger

logger = get_logger(__name__)


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
            logger.info(f"Downloading {symbol} from {start_date} to {end_date}")

            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date, interval=interval)

            if df.empty:
                logger.warning(f"No data available for {symbol}")
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

            logger.info(f"Downloaded {len(df)} bars for {symbol}")

            return df

        except Exception as e:
            logger.error(f"Error downloading {symbol}: {e}")
            return pd.DataFrame()

    def save_data(self, df: pd.DataFrame, symbol: str, interval: str = '1d'):
        """Save dataframe to CSV file."""

        if df.empty:
            logger.warning(f"No data to save for {symbol}")
            return

        # Determine subdirectory
        subdir = 'daily' if interval == '1d' else 'hourly'

        # Save to CSV
        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / subdir / filename

        df.to_csv(filepath, index=False)
        logger.info(f"Saved {len(df)} bars to {filepath}")

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

        logger.info(f"Downloading {len(symbols)} symbols from {start_date.date()} to {end_date.date()}")

        results = {
            'successful': [],
            'failed': []
        }

        for symbol in symbols:
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

        logger.info(f"\nDownload Summary:")
        logger.info(f"  Successful: {len(results['successful'])}")
        logger.info(f"  Failed: {len(results['failed'])}")
        logger.info(f"  Summary saved to {summary_file}")

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

    logger.info("=" * 80)
    logger.info("STATIC DATA DOWNLOADER")
    logger.info("=" * 80)
    logger.info(f"Downloading data for {len(symbols)} symbols")
    logger.info(f"Period: Last 3 years")
    logger.info("=" * 80)

    downloader = StaticDataDownloader()

    # Download daily data
    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOADING DAILY DATA")
    logger.info("=" * 80)
    results = downloader.download_multiple_symbols(symbols, years=3, interval='1d')

    logger.info("\n" + "=" * 80)
    logger.info("DOWNLOAD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Successfully downloaded: {len(results['successful'])} symbols")
    logger.info(f"Failed downloads: {len(results['failed'])} symbols")

    if results['failed']:
        logger.warning(f"Failed symbols: {', '.join(results['failed'])}")

    logger.info(f"\nData saved to: {downloader.data_dir}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
