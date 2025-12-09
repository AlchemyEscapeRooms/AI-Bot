"""
Test different filters with RSI to see which works best
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

def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()

def calculate_bollinger(prices: pd.Series, period: int = 20, std_dev: float = 2):
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    # Position: 0 = at lower band, 1 = at upper band
    position = (prices - lower) / (upper - lower)
    return upper, lower, position

def calculate_volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    avg_volume = volume.rolling(window=period).mean()
    return volume / avg_volume

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

def simulate_trades(df, buy_signals, rsi_sell=65):
    """Run a simple backtest given buy signals"""
    cash = 100000
    shares = 0
    position_price = 0
    trades = []

    for i in range(len(df)):
        price = df.iloc[i]['close']
        rsi = df.iloc[i]['rsi']
        timestamp = df.index[i]

        # Buy if signal is True and we don't have shares
        if i < len(buy_signals) and buy_signals.iloc[i] and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({'type': 'BUY', 'price': price, 'time': timestamp})

        # Sell if RSI > threshold
        if rsi > rsi_sell and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            cash += proceeds
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'time': timestamp})
            shares = 0

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        cash += proceeds
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl, 'time': df.index[-1]})

    # Calculate stats
    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t['pnl'] > 0]
    total_pnl = sum(t['pnl'] for t in sell_trades)

    return {
        'trades': len(sell_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        'total_pnl': total_pnl,
        'return_pct': total_pnl / 100000 * 100
    }

def run_all_tests(symbol='AAPL', start_date='2024-09-01', end_date='2024-12-01'):
    """Test different filters"""

    print("=" * 70)
    print(f"TESTING DIFFERENT FILTERS WITH RSI")
    print(f"Symbol: {symbol} | Period: {start_date} to {end_date}")
    print("=" * 70)
    print()

    # Fetch data
    df = fetch_data(symbol, start_date, end_date)

    # Calculate all indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df['sma50'] = calculate_sma(df['close'], 50)
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])

    # MACD for comparison
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    # Price momentum (% change over last 5 bars)
    df['momentum_5'] = df['close'].pct_change(5) * 100

    # Drop NaN rows
    df = df.dropna()

    print(f"Got {len(df)} bars of data")
    print()

    # Base condition: RSI < 35
    rsi_buy = df['rsi'] < 35

    results = []

    # 1. RSI Only (baseline)
    result = simulate_trades(df, rsi_buy)
    result['name'] = "RSI Only (no filter)"
    result['description'] = "Buy when RSI < 35"
    results.append(result)

    # 2. RSI + MACD (strict - uptrend only)
    macd_up = df['macd_diff'] > 0
    result = simulate_trades(df, rsi_buy & macd_up)
    result['name'] = "RSI + MACD Uptrend"
    result['description'] = "RSI < 35 AND MACD > Signal"
    results.append(result)

    # 3. RSI + MACD (soft filter)
    macd_not_danger = df['macd_diff'] > -0.2
    result = simulate_trades(df, rsi_buy & macd_not_danger)
    result['name'] = "RSI + MACD Soft"
    result['description'] = "RSI < 35 AND MACD > -0.2"
    results.append(result)

    # 4. RSI + Price above SMA20
    above_sma20 = df['close'] > df['sma20']
    result = simulate_trades(df, rsi_buy & above_sma20)
    result['name'] = "RSI + Above SMA20"
    result['description'] = "RSI < 35 AND Price > 20-bar avg"
    results.append(result)

    # 5. RSI + Price above SMA50
    above_sma50 = df['close'] > df['sma50']
    result = simulate_trades(df, rsi_buy & above_sma50)
    result['name'] = "RSI + Above SMA50"
    result['description'] = "RSI < 35 AND Price > 50-bar avg"
    results.append(result)

    # 6. RSI + Bollinger Band (price near lower band)
    bb_low = df['bb_position'] < 0.2  # Within 20% of lower band
    result = simulate_trades(df, rsi_buy & bb_low)
    result['name'] = "RSI + Bollinger Low"
    result['description'] = "RSI < 35 AND near lower Bollinger band"
    results.append(result)

    # 7. RSI + High Volume (buying interest)
    high_volume = df['volume_ratio'] > 1.5  # 50% above average
    result = simulate_trades(df, rsi_buy & high_volume)
    result['name'] = "RSI + High Volume"
    result['description'] = "RSI < 35 AND volume 50%+ above avg"
    results.append(result)

    # 8. RSI + Volume above average
    volume_above_avg = df['volume_ratio'] > 1.0
    result = simulate_trades(df, rsi_buy & volume_above_avg)
    result['name'] = "RSI + Volume > Avg"
    result['description'] = "RSI < 35 AND volume above average"
    results.append(result)

    # 9. RSI + Momentum slowing (drop is slowing down)
    momentum_recovering = df['momentum_5'] > -2  # Not dropping too fast
    result = simulate_trades(df, rsi_buy & momentum_recovering)
    result['name'] = "RSI + Momentum OK"
    result['description'] = "RSI < 35 AND not dropping >2% over 5 bars"
    results.append(result)

    # 10. RSI + Price not too far below SMA20
    not_too_low = df['close'] > df['sma20'] * 0.97  # Within 3% of SMA20
    result = simulate_trades(df, rsi_buy & not_too_low)
    result['name'] = "RSI + Near SMA20"
    result['description'] = "RSI < 35 AND within 3% of 20-bar avg"
    results.append(result)

    # 11. Combo: RSI + Volume + Not crashing
    combo1 = rsi_buy & volume_above_avg & momentum_recovering
    result = simulate_trades(df, combo1)
    result['name'] = "RSI + Volume + Momentum"
    result['description'] = "RSI < 35 AND volume up AND not crashing"
    results.append(result)

    # 12. RSI + Bollinger + Volume
    combo2 = rsi_buy & bb_low & volume_above_avg
    result = simulate_trades(df, combo2)
    result['name'] = "RSI + BB + Volume"
    result['description'] = "RSI < 35 AND near BB low AND volume up"
    results.append(result)

    # Print results
    print("=" * 70)
    print("RESULTS - RANKED BY RETURN")
    print("=" * 70)
    print()
    print(f"{'Strategy':<28} | {'Trades':>6} | {'Win %':>6} | {'Return':>8} | Description")
    print("-" * 90)

    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)

    for r in results:
        print(f"{r['name']:<28} | {r['trades']:>6} | {r['win_rate']:>5.0f}% | {r['return_pct']:>+7.2f}% | {r['description']}")

    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()

    best = results[0]
    worst = results[-1]

    print(f"BEST: {best['name']}")
    print(f"  {best['trades']} trades, {best['win_rate']:.0f}% win rate, {best['return_pct']:+.2f}% return")
    print(f"  Strategy: {best['description']}")
    print()
    print(f"WORST: {worst['name']}")
    print(f"  {worst['trades']} trades, {worst['win_rate']:.0f}% win rate, {worst['return_pct']:+.2f}% return")
    print(f"  Strategy: {worst['description']}")
    print()

    # Find the sweet spot
    print("LOOKING FOR THE SWEET SPOT:")
    print("(Good return + reasonable number of trades + high win rate)")
    print()
    for r in results:
        if r['trades'] >= 5 and r['win_rate'] >= 70 and r['return_pct'] > 3:
            print(f"  * {r['name']}: {r['trades']} trades, {r['win_rate']:.0f}% wins, {r['return_pct']:+.2f}%")

    return results

if __name__ == '__main__':
    run_all_tests()
