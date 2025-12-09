"""
Simple RSI-Only Backtest
One stock, one factor, clear logic

RSI Strategy:
- BUY when RSI < 30 (oversold)
- SELL when RSI > 70 (overbought)
- Hold otherwise
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

# Load API keys
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch hourly data from Alpaca."""
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=datetime.strptime(start_date, '%Y-%m-%d'),
        end=datetime.strptime(end_date, '%Y-%m-%d')
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')

    return df

def run_backtest(symbol: str, start_date: str, end_date: str, initial_capital: float = 100000):
    """
    Run simple RSI backtest.

    Rules:
    - BUY when RSI < 30 (oversold) and not holding
    - SELL when RSI > 70 (overbought) and holding
    """
    print("=" * 60)
    print(f"SIMPLE RSI BACKTEST")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 60)
    print()
    print("Strategy Rules:")
    print("  BUY  when RSI < 30 (oversold)")
    print("  SELL when RSI > 70 (overbought)")
    print()

    # Fetch data
    print("Fetching data...")
    df = fetch_data(symbol, start_date, end_date)
    print(f"Got {len(df)} hourly bars")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print()

    # Calculate RSI
    df['rsi'] = calculate_rsi(df['close'], period=14)

    # Trading simulation
    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []
    equity_curve = []

    # Track signals
    buy_signals = 0
    sell_signals = 0

    print("Walking through data...")
    print("-" * 60)

    for i in range(14, len(df)):  # Start after RSI warmup
        row = df.iloc[i]
        price = row['close']
        rsi = row['rsi']
        timestamp = df.index[i]

        # Track equity
        equity = cash + (shares * price)
        equity_curve.append({
            'timestamp': timestamp,
            'equity': equity,
            'cash': cash,
            'shares': shares,
            'price': price,
            'rsi': rsi
        })

        # Check for signals
        if rsi < 30 and shares == 0:
            # BUY signal - oversold
            buy_signals += 1
            shares_to_buy = int(cash * 0.95 / price)  # Use 95% of cash
            if shares_to_buy > 0:
                cost = shares_to_buy * price
                cash -= cost
                shares = shares_to_buy
                position_price = price

                print(f"BUY  @ {timestamp}")
                print(f"     RSI: {rsi:.1f} (oversold)")
                print(f"     Price: ${price:.2f}")
                print(f"     Shares: {shares_to_buy}")
                print(f"     Cost: ${cost:,.2f}")
                print()

                trades.append({
                    'type': 'BUY',
                    'timestamp': timestamp,
                    'price': price,
                    'shares': shares_to_buy,
                    'rsi': rsi
                })

        elif rsi > 70 and shares > 0:
            # SELL signal - overbought
            sell_signals += 1
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            pnl_pct = (price - position_price) / position_price * 100
            cash += proceeds

            print(f"SELL @ {timestamp}")
            print(f"     RSI: {rsi:.1f} (overbought)")
            print(f"     Price: ${price:.2f}")
            print(f"     Shares: {shares}")
            print(f"     Proceeds: ${proceeds:,.2f}")
            print(f"     P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
            print()

            trades.append({
                'type': 'SELL',
                'timestamp': timestamp,
                'price': price,
                'shares': shares,
                'rsi': rsi,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })

            shares = 0
            position_price = 0

    # Final equity (liquidate if holding)
    final_price = df.iloc[-1]['close']
    if shares > 0:
        print(f"END OF PERIOD - Liquidating {shares} shares @ ${final_price:.2f}")
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = (final_price - position_price) / position_price * 100
        cash += proceeds

        trades.append({
            'type': 'LIQUIDATE',
            'timestamp': df.index[-1],
            'price': final_price,
            'shares': shares,
            'rsi': df.iloc[-1]['rsi'],
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })
        shares = 0
        print(f"     P&L: ${pnl:,.2f} ({pnl_pct:+.2f}%)")
        print()

    final_equity = cash
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Calculate stats
    winning_trades = [t for t in trades if t['type'] in ['SELL', 'LIQUIDATE'] and t.get('pnl', 0) > 0]
    losing_trades = [t for t in trades if t['type'] in ['SELL', 'LIQUIDATE'] and t.get('pnl', 0) <= 0]

    total_pnl = sum(t.get('pnl', 0) for t in trades if t['type'] in ['SELL', 'LIQUIDATE'])

    # RSI distribution
    rsi_values = df['rsi'].dropna()
    oversold_bars = (rsi_values < 30).sum()
    overbought_bars = (rsi_values > 70).sum()
    neutral_bars = len(rsi_values) - oversold_bars - overbought_bars

    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print()
    print("RSI Distribution:")
    print(f"  Oversold (RSI < 30):   {oversold_bars} bars ({oversold_bars/len(rsi_values)*100:.1f}%)")
    print(f"  Neutral (30-70):       {neutral_bars} bars ({neutral_bars/len(rsi_values)*100:.1f}%)")
    print(f"  Overbought (RSI > 70): {overbought_bars} bars ({overbought_bars/len(rsi_values)*100:.1f}%)")
    print()
    print("Signals Generated:")
    print(f"  Buy signals:  {buy_signals}")
    print(f"  Sell signals: {sell_signals}")
    print()
    print("Trading Results:")
    print(f"  Total Trades: {len([t for t in trades if t['type'] == 'BUY'])}")
    print(f"  Winning: {len(winning_trades)}")
    print(f"  Losing:  {len(losing_trades)}")
    if winning_trades or losing_trades:
        print(f"  Win Rate: {len(winning_trades)/(len(winning_trades)+len(losing_trades))*100:.1f}%")
    print()
    print("Financial Results:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Final Equity:    ${final_equity:,.2f}")
    print(f"  Total P&L:       ${total_pnl:,.2f}")
    print(f"  Return:          {total_return:+.2f}%")
    print()

    # Show all trades
    if trades:
        print("Trade Log:")
        print("-" * 60)
        for t in trades:
            if t['type'] == 'BUY':
                print(f"  {t['timestamp']} | BUY  | ${t['price']:.2f} | RSI={t['rsi']:.1f}")
            else:
                print(f"  {t['timestamp']} | {t['type']:4} | ${t['price']:.2f} | RSI={t['rsi']:.1f} | P&L: ${t['pnl']:+,.2f}")

    return {
        'initial_capital': initial_capital,
        'final_equity': final_equity,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'trades': trades,
        'equity_curve': equity_curve,
        'rsi_distribution': {
            'oversold': oversold_bars,
            'neutral': neutral_bars,
            'overbought': overbought_bars
        }
    }

if __name__ == '__main__':
    # Run backtest on AAPL for 3 months
    result = run_backtest(
        symbol='AAPL',
        start_date='2024-09-01',
        end_date='2024-12-01',
        initial_capital=100000
    )
