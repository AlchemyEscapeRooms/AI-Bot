"""
Test each indicator on its own - no RSI
See which single indicator works best
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

def simulate_trades_with_sell_condition(df, buy_condition, sell_condition):
    """Run backtest with custom buy and sell conditions"""
    cash = 100000
    shares = 0
    position_price = 0
    trades = []

    for i in range(len(df)):
        price = df.iloc[i]['close']
        timestamp = df.index[i]

        # Buy
        if buy_condition.iloc[i] and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({'type': 'BUY', 'price': price, 'time': timestamp})

        # Sell
        elif shares > 0 and sell_condition.iloc[i]:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            cash += proceeds
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'time': timestamp})
            shares = 0

    # Liquidate
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        cash += proceeds
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl})

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

def run_single_indicator_tests(symbol='AAPL', start_date='2024-09-01', end_date='2024-12-01'):
    """Test each indicator on its own"""

    print("=" * 70)
    print(f"TESTING SINGLE INDICATORS (no combinations)")
    print(f"Symbol: {symbol} | Period: {start_date} to {end_date}")
    print("=" * 70)
    print()

    df = fetch_data(symbol, start_date, end_date)

    # Calculate all indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df['sma50'] = calculate_sma(df['close'], 50)
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])

    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['macd_diff'] = df['macd'] - df['macd_signal']

    df['momentum_5'] = df['close'].pct_change(5) * 100

    df = df.dropna()

    print(f"Got {len(df)} bars of data")
    print()

    results = []

    # 1. RSI: Buy < 30, Sell > 70
    buy = df['rsi'] < 30
    sell = df['rsi'] > 70
    result = simulate_trades_with_sell_condition(df, buy, sell)
    result['name'] = "RSI (30/70)"
    result['buy'] = "RSI < 30"
    result['sell'] = "RSI > 70"
    results.append(result)

    # 2. RSI: Buy < 35, Sell > 65
    buy = df['rsi'] < 35
    sell = df['rsi'] > 65
    result = simulate_trades_with_sell_condition(df, buy, sell)
    result['name'] = "RSI (35/65)"
    result['buy'] = "RSI < 35"
    result['sell'] = "RSI > 65"
    results.append(result)

    # 3. MACD: Buy when crosses up, sell when crosses down
    macd_cross_up = (df['macd_diff'] > 0) & (df['macd_diff'].shift(1) <= 0)
    macd_cross_down = (df['macd_diff'] < 0) & (df['macd_diff'].shift(1) >= 0)
    result = simulate_trades_with_sell_condition(df, macd_cross_up, macd_cross_down)
    result['name'] = "MACD Crossover"
    result['buy'] = "MACD crosses above signal"
    result['sell'] = "MACD crosses below signal"
    results.append(result)

    # 4. MACD: Buy when above signal, sell when below
    macd_above = df['macd_diff'] > 0
    macd_below = df['macd_diff'] < 0
    result = simulate_trades_with_sell_condition(df, macd_above & ~macd_above.shift(1).fillna(False), macd_below)
    result['name'] = "MACD Above/Below"
    result['buy'] = "MACD goes above signal"
    result['sell'] = "MACD goes below signal"
    results.append(result)

    # 5. Bollinger: Buy at lower band, sell at upper
    bb_buy = df['bb_position'] < 0.1
    bb_sell = df['bb_position'] > 0.9
    result = simulate_trades_with_sell_condition(df, bb_buy, bb_sell)
    result['name'] = "Bollinger (10%/90%)"
    result['buy'] = "Price at lower 10% of BB"
    result['sell'] = "Price at upper 10% of BB"
    results.append(result)

    # 6. Bollinger: Buy at lower band, sell at middle
    bb_buy = df['bb_position'] < 0.2
    bb_sell = df['bb_position'] > 0.5
    result = simulate_trades_with_sell_condition(df, bb_buy, bb_sell)
    result['name'] = "Bollinger (20%/50%)"
    result['buy'] = "Price at lower 20% of BB"
    result['sell'] = "Price at middle of BB"
    results.append(result)

    # 7. Volume spike: Buy on high volume dip, sell on high volume rise
    high_vol = df['volume_ratio'] > 1.5
    price_down = df['close'] < df['close'].shift(1)
    price_up = df['close'] > df['close'].shift(1)
    result = simulate_trades_with_sell_condition(df, high_vol & price_down, high_vol & price_up)
    result['name'] = "Volume Spike"
    result['buy'] = "High volume + price down"
    result['sell'] = "High volume + price up"
    results.append(result)

    # 8. SMA20 Crossover
    above_sma = df['close'] > df['sma20']
    cross_above = above_sma & ~above_sma.shift(1).fillna(False)
    cross_below = ~above_sma & above_sma.shift(1).fillna(True)
    result = simulate_trades_with_sell_condition(df, cross_above, cross_below)
    result['name'] = "SMA20 Cross"
    result['buy'] = "Price crosses above SMA20"
    result['sell'] = "Price crosses below SMA20"
    results.append(result)

    # 9. Momentum: Buy when momentum turns positive, sell when negative
    mom_positive = df['momentum_5'] > 0
    mom_turns_up = mom_positive & ~mom_positive.shift(1).fillna(False)
    mom_turns_down = ~mom_positive & mom_positive.shift(1).fillna(True)
    result = simulate_trades_with_sell_condition(df, mom_turns_up, mom_turns_down)
    result['name'] = "Momentum (5-bar)"
    result['buy'] = "5-bar momentum turns positive"
    result['sell'] = "5-bar momentum turns negative"
    results.append(result)

    # 10. Mean Reversion: Buy 2% below SMA20, sell at SMA20
    below_2pct = df['close'] < df['sma20'] * 0.98
    at_sma = df['close'] >= df['sma20']
    result = simulate_trades_with_sell_condition(df, below_2pct, at_sma)
    result['name'] = "Mean Reversion 2%"
    result['buy'] = "Price 2%+ below SMA20"
    result['sell'] = "Price returns to SMA20"
    results.append(result)

    # 11. Mean Reversion: Buy 3% below SMA20
    below_3pct = df['close'] < df['sma20'] * 0.97
    result = simulate_trades_with_sell_condition(df, below_3pct, at_sma)
    result['name'] = "Mean Reversion 3%"
    result['buy'] = "Price 3%+ below SMA20"
    result['sell'] = "Price returns to SMA20"
    results.append(result)

    # Print results
    print("=" * 70)
    print("RESULTS - RANKED BY RETURN")
    print("=" * 70)
    print()
    print(f"{'Strategy':<22} | {'Trades':>6} | {'Win %':>6} | {'Return':>8} | Buy Signal")
    print("-" * 85)

    results.sort(key=lambda x: x['return_pct'], reverse=True)

    for r in results:
        print(f"{r['name']:<22} | {r['trades']:>6} | {r['win_rate']:>5.0f}% | {r['return_pct']:>+7.2f}% | {r['buy']}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    best = results[0]
    print(f"BEST SINGLE INDICATOR: {best['name']}")
    print(f"  Buy when: {best['buy']}")
    print(f"  Sell when: {best['sell']}")
    print(f"  Results: {best['trades']} trades, {best['win_rate']:.0f}% win rate, {best['return_pct']:+.2f}% return")

    return results

if __name__ == '__main__':
    run_single_indicator_tests()
