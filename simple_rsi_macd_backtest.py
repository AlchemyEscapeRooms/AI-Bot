"""
RSI + MACD Backtest (Soft MACD Filter)
One stock, two factors, clear logic

Strategy:
- RSI tells us when to buy (dip) and sell (hot)
- MACD acts as a "danger warning" - only skip if MACD is VERY bearish

Rules:
- BUY when RSI < 35 (dip) - UNLESS MACD is deeply negative (danger zone)
- SELL when RSI > 65 (take profits)
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line

def fetch_data(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
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

def run_backtest(symbol: str, start_date: str, end_date: str,
                 initial_capital: float = 100000,
                 rsi_buy: float = 35,
                 rsi_sell: float = 65,
                 macd_danger_zone: float = -0.2):
    """
    RSI + Soft MACD filter backtest.

    Only skip buys when MACD is DEEPLY negative (danger zone).
    """
    print("=" * 70)
    print(f"RSI + SOFT MACD FILTER BACKTEST")
    print(f"Symbol: {symbol}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print("=" * 70)
    print()
    print("Strategy (in plain English):")
    print(f"  BUY when: RSI < {rsi_buy} (stock dipped)")
    print(f"            BUT skip if MACD-Signal < {macd_danger_zone} (danger zone)")
    print(f"  SELL when: RSI > {rsi_sell} (stock running hot)")
    print()
    print("The idea: Only avoid buying when MACD shows STRONG downtrend.")
    print("          Small dips in an unclear trend are OK to buy.")
    print()

    # Fetch and prepare data
    df = fetch_data(symbol, start_date, end_date)
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_diff'] = df['macd'] - df['macd_signal']

    print(f"Got {len(df)} hourly bars")
    print()

    # Trading simulation
    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []

    bought_count = 0
    skipped_danger = 0

    print("Trading Activity:")
    print("-" * 70)

    for i in range(26, len(df)):
        row = df.iloc[i]
        price = row['close']
        rsi = row['rsi']
        macd_diff = row['macd_diff']
        timestamp = df.index[i]

        # BUY LOGIC
        if rsi < rsi_buy and shares == 0:
            if macd_diff < macd_danger_zone:
                # Danger zone - skip this trade
                skipped_danger += 1
            else:
                # Safe to buy
                bought_count += 1
                shares_to_buy = int(cash * 0.95 / price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * price
                    cash -= cost
                    shares = shares_to_buy
                    position_price = price

                    status = "UPTREND" if macd_diff > 0 else f"mild down ({macd_diff:.2f})"
                    print(f"BUY  {str(timestamp)[:10]} | ${price:.2f} | RSI={rsi:.0f} | MACD={macd_diff:+.3f} ({status})")

                    trades.append({
                        'type': 'BUY',
                        'timestamp': timestamp,
                        'price': price,
                        'rsi': rsi,
                        'macd_diff': macd_diff
                    })

        # SELL LOGIC
        if rsi > rsi_sell and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            pnl_pct = (price - position_price) / position_price * 100
            cash += proceeds

            result = "WIN" if pnl > 0 else "LOSS"
            print(f"SELL {str(timestamp)[:10]} | ${price:.2f} | RSI={rsi:.0f} | P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%) {result}")

            trades.append({
                'type': 'SELL',
                'timestamp': timestamp,
                'price': price,
                'rsi': rsi,
                'macd_diff': macd_diff,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })

            shares = 0
            position_price = 0

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = (final_price - position_price) / position_price * 100
        cash += proceeds

        result = "WIN" if pnl > 0 else "LOSS"
        print(f"END  {str(df.index[-1])[:10]} | ${final_price:.2f} | Liquidate | P&L: ${pnl:+,.0f} ({pnl_pct:+.1f}%) {result}")

        trades.append({
            'type': 'LIQUIDATE',
            'timestamp': df.index[-1],
            'price': final_price,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

    # Calculate results
    final_equity = cash
    total_return = (final_equity - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['type'] in ['SELL', 'LIQUIDATE']]
    winners = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losers = [t for t in sell_trades if t.get('pnl', 0) <= 0]
    total_pnl = sum(t.get('pnl', 0) for t in sell_trades)

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("What happened:")
    print(f"  RSI dips found: {bought_count + skipped_danger}")
    print(f"  - Bought: {bought_count} (MACD was OK)")
    print(f"  - Skipped: {skipped_danger} (MACD in danger zone < {macd_danger_zone})")
    print()
    print("Trade Results:")
    print(f"  Completed trades: {len(sell_trades)}")
    print(f"  Winners: {len(winners)}")
    print(f"  Losers: {len(losers)}")
    if sell_trades:
        print(f"  Win rate: {len(winners)/len(sell_trades)*100:.0f}%")
    print()
    print("Money:")
    print(f"  Started with: ${initial_capital:,.2f}")
    print(f"  Ended with:   ${final_equity:,.2f}")
    print(f"  Total P&L:    ${total_pnl:+,.2f}")
    print(f"  Return:       {total_return:+.2f}%")
    print()

    # Comparison
    print("=" * 70)
    print("COMPARISON TO OTHER STRATEGIES")
    print("=" * 70)
    print()
    print("Strategy                          | Trades | Win Rate | Return")
    print("-" * 70)
    print("RSI Only (no filter)              |   14   |   86%    | +8.15%")
    print("RSI + Strict MACD (must be up)    |    2   |  100%    | +12.62%")
    print("RSI + Loose MACD (35/65)          |    3   |  100%    | +2.4%")
    print(f"RSI + Soft MACD (danger < {macd_danger_zone})   |   {len(sell_trades):2}   |  {len(winners)/len(sell_trades)*100 if sell_trades else 0:3.0f}%    | {total_return:+.2f}%")
    print()

    return {
        'total_pnl': total_pnl,
        'total_return': total_return,
        'trades': trades,
        'bought': bought_count,
        'skipped': skipped_danger
    }

if __name__ == '__main__':
    # Test with soft MACD filter - only skip when MACD is very negative
    print("\n" + "="*70)
    print("TEST 1: Danger zone = -0.2")
    print("="*70 + "\n")

    result1 = run_backtest(
        symbol='AAPL',
        start_date='2024-09-01',
        end_date='2024-12-01',
        initial_capital=100000,
        rsi_buy=35,
        rsi_sell=65,
        macd_danger_zone=-0.2  # Only skip if MACD is strongly bearish
    )

    print("\n" + "="*70)
    print("TEST 2: Danger zone = -0.3 (even more lenient)")
    print("="*70 + "\n")

    result2 = run_backtest(
        symbol='AAPL',
        start_date='2024-09-01',
        end_date='2024-12-01',
        initial_capital=100000,
        rsi_buy=35,
        rsi_sell=65,
        macd_danger_zone=-0.3  # Even more lenient
    )

    print("\n" + "="*70)
    print("TEST 3: Danger zone = -0.15 (stricter)")
    print("="*70 + "\n")

    result3 = run_backtest(
        symbol='AAPL',
        start_date='2024-09-01',
        end_date='2024-12-01',
        initial_capital=100000,
        rsi_buy=35,
        rsi_sell=65,
        macd_danger_zone=-0.15  # A bit stricter
    )
