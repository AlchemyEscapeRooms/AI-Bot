#!/usr/bin/env python3
"""
Generate sample static historical stock data for backtesting.
This creates realistic-looking price data for testing purposes.
"""

import csv
import json
import random
from datetime import datetime, timedelta
from pathlib import Path
import math


class SampleDataGenerator:
    """Generates realistic sample stock data for backtesting."""

    def __init__(self, data_dir='static_data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        (self.data_dir / 'daily').mkdir(exist_ok=True)

    def generate_price_series(
        self,
        symbol: str,
        start_price: float,
        days: int,
        volatility: float = 0.02,
        drift: float = 0.0001
    ):
        """Generate realistic price series using geometric Brownian motion."""

        prices = []
        current_price = start_price

        for i in range(days):
            # Random daily return
            daily_return = random.gauss(drift, volatility)
            current_price = current_price * (1 + daily_return)

            # Ensure price stays positive
            current_price = max(current_price, 0.01)

            prices.append(current_price)

        return prices

    def generate_ohlcv_data(
        self,
        symbol: str,
        start_date: datetime,
        days: int,
        start_price: float,
        avg_volume: int
    ):
        """Generate OHLCV data for a symbol."""

        prices = self.generate_price_series(symbol, start_price, days)

        data = []
        current_date = start_date

        for close_price in prices:
            # Generate OHLC from close
            intraday_volatility = 0.01
            open_price = close_price * (1 + random.gauss(0, intraday_volatility))
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, intraday_volatility)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, intraday_volatility)))

            # Generate volume with some randomness
            volume = int(avg_volume * (1 + random.gauss(0, 0.3)))
            volume = max(volume, avg_volume // 2)  # Ensure minimum volume

            # Dividends and stock splits (mostly 0)
            dividends = 0.0
            stock_splits = 0.0

            # Occasional dividends (1% chance per day)
            if random.random() < 0.01:
                dividends = close_price * random.uniform(0.001, 0.005)

            row = {
                'date': current_date.strftime('%Y-%m-%d'),
                'open': round(open_price, 2),
                'high': round(high_price, 2),
                'low': round(low_price, 2),
                'close': round(close_price, 2),
                'volume': volume,
                'dividends': round(dividends, 4),
                'stock splits': stock_splits,
                'symbol': symbol
            }

            data.append(row)
            current_date += timedelta(days=1)

            # Skip weekends
            while current_date.weekday() >= 5:
                current_date += timedelta(days=1)

        return data

    def save_data(self, data: list, symbol: str, interval: str = '1d'):
        """Save data to CSV file."""

        filename = f"{symbol}_{interval}.csv"
        filepath = self.data_dir / 'daily' / filename

        # Write CSV
        with open(filepath, 'w', newline='') as f:
            if not data:
                return

            writer = csv.DictWriter(f, fieldnames=data[0].keys())
            writer.writeheader()
            writer.writerows(data)

        print(f"  ✓ Generated {len(data)} bars for {symbol} -> {filepath}")

        # Save metadata
        metadata = {
            'symbol': symbol,
            'interval': interval,
            'start_date': data[0]['date'],
            'end_date': data[-1]['date'],
            'total_bars': len(data),
            'columns': list(data[0].keys()),
            'generated_at': datetime.now().isoformat(),
            'data_type': 'synthetic'
        }

        metadata_file = self.data_dir / 'daily' / f"{symbol}_{interval}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def generate_multiple_symbols(self, years: int = 3):
        """Generate data for multiple symbols."""

        # Symbol configurations: (symbol, start_price, avg_volume, volatility, drift)
        symbols_config = [
            # Indices/ETFs - Lower volatility, positive drift
            ('SPY', 350, 80000000, 0.012, 0.0003),
            ('QQQ', 300, 45000000, 0.015, 0.0004),
            ('IWM', 180, 35000000, 0.018, 0.0002),
            ('DIA', 330, 5000000, 0.011, 0.0002),
            ('VTI', 200, 4000000, 0.012, 0.0003),

            # Tech - Large Cap - Higher volatility, positive drift
            ('AAPL', 140, 70000000, 0.02, 0.0005),
            ('MSFT', 320, 25000000, 0.018, 0.0004),
            ('GOOGL', 120, 20000000, 0.022, 0.0004),
            ('AMZN', 130, 50000000, 0.025, 0.0005),
            ('META', 280, 18000000, 0.028, 0.0003),
            ('NVDA', 300, 45000000, 0.035, 0.0008),
            ('TSLA', 200, 100000000, 0.045, 0.0006),
            ('AMD', 90, 60000000, 0.032, 0.0005),
            ('INTC', 45, 35000000, 0.020, 0.0001),
            ('NFLX', 400, 8000000, 0.030, 0.0003),

            # Tech - Mid Cap
            ('CRM', 210, 6000000, 0.025, 0.0003),
            ('ADBE', 480, 3000000, 0.022, 0.0004),
            ('ORCL', 90, 8000000, 0.016, 0.0002),
            ('CSCO', 50, 20000000, 0.014, 0.0001),
            ('AVGO', 550, 2000000, 0.020, 0.0004),

            # Financial
            ('JPM', 140, 12000000, 0.018, 0.0002),
            ('BAC', 35, 50000000, 0.020, 0.0002),
            ('WFC', 45, 25000000, 0.021, 0.0001),
            ('GS', 350, 2500000, 0.022, 0.0002),
            ('MS', 85, 8000000, 0.020, 0.0002),

            # Healthcare
            ('JNJ', 160, 7000000, 0.012, 0.0002),
            ('UNH', 450, 3000000, 0.016, 0.0004),
            ('PFE', 40, 35000000, 0.018, 0.0001),

            # Consumer
            ('WMT', 145, 8000000, 0.013, 0.0002),
            ('HD', 310, 4000000, 0.016, 0.0003),
            ('COST', 480, 2000000, 0.015, 0.0003),
            ('NKE', 110, 7000000, 0.020, 0.0002),

            # Energy
            ('XOM', 100, 20000000, 0.025, 0.0001),
            ('CVX', 155, 8000000, 0.024, 0.0001),

            # Industrial
            ('BA', 210, 5000000, 0.028, 0.0002),
            ('CAT', 230, 3000000, 0.020, 0.0002),
        ]

        # Calculate days
        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365)
        days = (end_date - start_date).days

        # Adjust for weekends (roughly 5/7 of total days)
        trading_days = int(days * 5 / 7)

        print(f"Generating {len(symbols_config)} symbols for {years} years ({trading_days} trading days)")
        print("=" * 80)

        results = {'successful': [], 'failed': []}

        for i, (symbol, start_price, avg_volume, volatility, drift) in enumerate(symbols_config, 1):
            try:
                print(f"[{i}/{len(symbols_config)}] Generating {symbol}...")

                data = self.generate_ohlcv_data(
                    symbol,
                    start_date,
                    trading_days,
                    start_price,
                    avg_volume
                )

                self.save_data(data, symbol)
                results['successful'].append(symbol)

            except Exception as e:
                print(f"  ERROR generating {symbol}: {e}")
                results['failed'].append(symbol)

        # Save summary
        summary = {
            'generation_date': datetime.now().isoformat(),
            'total_symbols': len(symbols_config),
            'successful': len(results['successful']),
            'failed': len(results['failed']),
            'symbols_successful': results['successful'],
            'symbols_failed': results['failed'],
            'period_years': years,
            'trading_days': trading_days,
            'data_type': 'synthetic'
        }

        summary_file = self.data_dir / 'data_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 80)
        print("DATA GENERATION COMPLETE!")
        print("=" * 80)
        print(f"Successfully generated: {len(results['successful'])} symbols")
        print(f"Failed: {len(results['failed'])} symbols")
        print(f"Summary saved to: {summary_file}")
        print(f"Data saved to: {self.data_dir.absolute()}/daily/")
        print("=" * 80)

        return results


def main():
    """Main function to generate sample data."""

    print("=" * 80)
    print("SAMPLE STATIC DATA GENERATOR FOR BACKTESTING")
    print("=" * 80)
    print("This generates realistic synthetic stock data for backtesting.")
    print("Data uses geometric Brownian motion with configurable parameters.")
    print("=" * 80)
    print()

    generator = SampleDataGenerator()
    generator.generate_multiple_symbols(years=3)


if __name__ == '__main__':
    main()
