"""
Deep test of RSI 30/70 strategy on TSLA
"""
import os
import pandas as pd
from datetime import datetime
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
    return 100 - (100 / (1 + rs))

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

def run_rsi_backtest(symbol: str, start_date: str, end_date: str,
                     rsi_buy: float = 30, rsi_sell: float = 70,
                     initial_capital: float = 100000):
    """Run RSI backtest and return detailed results"""

    df = fetch_data(symbol, start_date, end_date)
    df['rsi'] = calculate_rsi(df['close'])
    df = df.dropna()

    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []

    for i in range(len(df)):
        row = df.iloc[i]
        price = row['close']
        rsi = row['rsi']
        timestamp = df.index[i]

        # Buy
        if rsi < rsi_buy and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({
                    'type': 'BUY',
                    'date': timestamp,
                    'price': price,
                    'rsi': rsi,
                    'shares': shares
                })

        # Sell
        elif rsi > rsi_sell and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            pnl_pct = (price - position_price) / position_price * 100
            cash += proceeds
            trades.append({
                'type': 'SELL',
                'date': timestamp,
                'price': price,
                'rsi': rsi,
                'shares': shares,
                'pnl': pnl,
                'pnl_pct': pnl_pct
            })
            shares = 0

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        pnl_pct = (final_price - position_price) / position_price * 100
        cash += proceeds
        trades.append({
            'type': 'LIQUIDATE',
            'date': df.index[-1],
            'price': final_price,
            'rsi': df.iloc[-1]['rsi'],
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct
        })

    final_equity = cash
    total_return = (final_equity - initial_capital) / initial_capital * 100

    sell_trades = [t for t in trades if t['type'] in ['SELL', 'LIQUIDATE']]
    winners = [t for t in sell_trades if t.get('pnl', 0) > 0]
    losers = [t for t in sell_trades if t.get('pnl', 0) <= 0]
    total_pnl = sum(t.get('pnl', 0) for t in sell_trades)

    return {
        'symbol': symbol,
        'period': f"{start_date} to {end_date}",
        'bars': len(df),
        'trades': len(sell_trades),
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        'total_pnl': total_pnl,
        'return_pct': total_return,
        'trade_list': trades
    }

def main():
    print("=" * 70)
    print("TSLA - RSI 30/70 STRATEGY DEEP TEST")
    print("=" * 70)
    print()

    # Test 1: Recent 3 months
    print("TEST 1: Sep-Dec 2024 (3 months)")
    print("-" * 70)
    result = run_rsi_backtest('TSLA', '2024-09-01', '2024-12-01')

    print(f"Period: {result['period']}")
    print(f"Bars: {result['bars']}")
    print(f"Trades: {result['trades']}")
    print(f"Winners: {result['winners']} | Losers: {result['losers']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Total P&L: ${result['total_pnl']:+,.2f}")
    print(f"Return: {result['return_pct']:+.2f}%")
    print()
    print("Trade Log:")
    for t in result['trade_list']:
        if t['type'] == 'BUY':
            print(f"  {str(t['date'])[:10]} | BUY  ${t['price']:.2f} | RSI={t['rsi']:.1f}")
        else:
            result_str = "WIN" if t['pnl'] > 0 else "LOSS"
            print(f"  {str(t['date'])[:10]} | {t['type']:<4} ${t['price']:.2f} | RSI={t['rsi']:.1f} | ${t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%) {result_str}")

    print()
    print()

    # Test 2: Different time periods
    print("TEST 2: Different Time Periods")
    print("-" * 70)
    print()

    periods = [
        ('2024-09-01', '2024-12-01', '3 months (Sep-Dec 2024)'),
        ('2024-06-01', '2024-12-01', '6 months'),
        ('2024-01-01', '2024-12-01', '12 months (YTD 2024)'),
        ('2023-01-01', '2024-01-01', '2023 full year'),
        ('2023-01-01', '2024-12-01', '2 years (2023-2024)'),
    ]

    print(f"{'Period':<28} | {'Trades':>6} | {'Win %':>6} | {'Return':>10}")
    print("-" * 60)

    all_results = []
    for start, end, label in periods:
        try:
            result = run_rsi_backtest('TSLA', start, end)
            result['label'] = label
            all_results.append(result)
            print(f"{label:<28} | {result['trades']:>6} | {result['win_rate']:>5.0f}% | {result['return_pct']:>+9.2f}%")
        except Exception as e:
            print(f"{label:<28} | Error: {e}")

    print()
    print()

    # Show the 2-year detailed trades
    print("TEST 3: 2-Year Trade Details (2023-2024)")
    print("-" * 70)

    result = run_rsi_backtest('TSLA', '2023-01-01', '2024-12-01')

    print(f"Total Trades: {result['trades']}")
    print(f"Winners: {result['winners']} | Losers: {result['losers']}")
    print(f"Win Rate: {result['win_rate']:.1f}%")
    print(f"Total P&L: ${result['total_pnl']:+,.2f}")
    print(f"Return: {result['return_pct']:+.2f}%")
    print()

    # Show biggest winners and losers
    sell_trades = [t for t in result['trade_list'] if t['type'] in ['SELL', 'LIQUIDATE']]
    sell_trades_sorted = sorted(sell_trades, key=lambda x: x['pnl'], reverse=True)

    print("Top 5 Winners:")
    for t in sell_trades_sorted[:5]:
        print(f"  {str(t['date'])[:10]} | ${t['price']:.2f} | ${t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%)")

    print()
    print("Top 5 Losers:")
    for t in sell_trades_sorted[-5:]:
        print(f"  {str(t['date'])[:10]} | ${t['price']:.2f} | ${t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%)")

    print()
    print("=" * 70)
    print("TSLA vs AAPL COMPARISON")
    print("=" * 70)
    print()
    print("Over 2 years (2023-2024):")
    print()

    aapl = run_rsi_backtest('AAPL', '2023-01-01', '2024-12-01')
    tsla = run_rsi_backtest('TSLA', '2023-01-01', '2024-12-01')

    print(f"{'Stock':<8} | {'Trades':>6} | {'Win %':>6} | {'Return':>10} | Avg per trade")
    print("-" * 60)
    print(f"{'AAPL':<8} | {aapl['trades']:>6} | {aapl['win_rate']:>5.0f}% | {aapl['return_pct']:>+9.2f}% | {aapl['return_pct']/aapl['trades']:+.2f}%")
    print(f"{'TSLA':<8} | {tsla['trades']:>6} | {tsla['win_rate']:>5.0f}% | {tsla['return_pct']:>+9.2f}% | {tsla['return_pct']/tsla['trades']:+.2f}%")

if __name__ == '__main__':
    main()
