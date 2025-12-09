"""
Deep test of RSI 30/70 strategy
1. Confirm the +8.15% result
2. Test on longer time periods
"""
import os
import pandas as pd
import numpy as np
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
                     initial_capital: float = 100000, verbose: bool = True):
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

    # Calculate stats
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
    print("RSI 30/70 STRATEGY - DEEP TEST")
    print("=" * 70)
    print()

    # Test 1: Confirm original result (Sep-Dec 2024)
    print("TEST 1: Confirm original +8.15% result")
    print("-" * 70)
    result = run_rsi_backtest('AAPL', '2024-09-01', '2024-12-01', rsi_buy=30, rsi_sell=70)

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
            print(f"  {str(t['date'])[:10]} | {t['type']:<4} ${t['price']:.2f} | RSI={t['rsi']:.1f} | P&L: ${t['pnl']:+,.0f} ({t['pnl_pct']:+.1f}%)")

    print()
    print()

    # Test 2: Longer time periods
    print("TEST 2: Different Time Periods")
    print("-" * 70)
    print()

    periods = [
        ('2024-09-01', '2024-12-01', '3 months (original)'),
        ('2024-06-01', '2024-12-01', '6 months'),
        ('2024-01-01', '2024-12-01', '12 months (YTD 2024)'),
        ('2023-01-01', '2024-01-01', '2023 full year'),
        ('2023-01-01', '2024-12-01', '2 years (2023-2024)'),
    ]

    print(f"{'Period':<25} | {'Months':>6} | {'Trades':>6} | {'Win %':>6} | {'Return':>8}")
    print("-" * 70)

    all_results = []
    for start, end, label in periods:
        try:
            result = run_rsi_backtest('AAPL', start, end, rsi_buy=30, rsi_sell=70, verbose=False)
            result['label'] = label
            all_results.append(result)
            print(f"{label:<25} | {result['bars']//160:>6} | {result['trades']:>6} | {result['win_rate']:>5.0f}% | {result['return_pct']:>+7.2f}%")
        except Exception as e:
            print(f"{label:<25} | Error: {e}")

    print()
    print()

    # Test 3: Different stocks
    print("TEST 3: Different Stocks (same period: Sep-Dec 2024)")
    print("-" * 70)
    print()

    stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'SPY', 'QQQ']

    print(f"{'Stock':<8} | {'Trades':>6} | {'Win %':>6} | {'Return':>8}")
    print("-" * 50)

    stock_results = []
    for stock in stocks:
        try:
            result = run_rsi_backtest(stock, '2024-09-01', '2024-12-01', rsi_buy=30, rsi_sell=70, verbose=False)
            stock_results.append(result)
            print(f"{stock:<8} | {result['trades']:>6} | {result['win_rate']:>5.0f}% | {result['return_pct']:>+7.2f}%")
        except Exception as e:
            print(f"{stock:<8} | Error: {e}")

    print()

    # Summary
    if stock_results:
        avg_return = sum(r['return_pct'] for r in stock_results) / len(stock_results)
        avg_win_rate = sum(r['win_rate'] for r in stock_results) / len(stock_results)
        total_trades = sum(r['trades'] for r in stock_results)

        print(f"Average across {len(stock_results)} stocks:")
        print(f"  Total trades: {total_trades}")
        print(f"  Avg win rate: {avg_win_rate:.1f}%")
        print(f"  Avg return: {avg_return:+.2f}%")

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()

    # Find best and worst
    if all_results:
        best_period = max(all_results, key=lambda x: x['return_pct'])
        worst_period = min(all_results, key=lambda x: x['return_pct'])
        print(f"Best period: {best_period['label']} with {best_period['return_pct']:+.2f}%")
        print(f"Worst period: {worst_period['label']} with {worst_period['return_pct']:+.2f}%")

    if stock_results:
        best_stock = max(stock_results, key=lambda x: x['return_pct'])
        worst_stock = min(stock_results, key=lambda x: x['return_pct'])
        print(f"Best stock: {best_stock['symbol']} with {best_stock['return_pct']:+.2f}%")
        print(f"Worst stock: {worst_stock['symbol']} with {worst_stock['return_pct']:+.2f}%")

if __name__ == '__main__':
    main()
