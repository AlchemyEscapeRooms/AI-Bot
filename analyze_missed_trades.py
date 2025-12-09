"""
Analyze ALL trading opportunities - both taken and missed
See what would have happened if we said YES to every signal
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

def analyze_all_opportunities(symbol: str, start_date: str, end_date: str, rsi_buy: float = 35, rsi_sell: float = 65):
    """
    Find every RSI buy signal and simulate what would have happened.
    """
    print("=" * 70)
    print(f"ANALYZING ALL TRADING OPPORTUNITIES")
    print(f"Symbol: {symbol} | Period: {start_date} to {end_date}")
    print(f"Looking for: RSI < {rsi_buy} (buy signals)")
    print("=" * 70)
    print()

    # Fetch and prepare data
    df = fetch_data(symbol, start_date, end_date)
    df['rsi'] = calculate_rsi(df['close'], period=14)
    df['macd'], df['macd_signal'] = calculate_macd(df['close'])
    df['macd_diff'] = df['macd'] - df['macd_signal']
    df['uptrend'] = df['macd'] > df['macd_signal']

    print(f"Got {len(df)} hourly bars")
    print()

    # Find all buy opportunities (RSI < threshold)
    opportunities = []

    for i in range(26, len(df) - 20):  # Leave room to find exit
        row = df.iloc[i]
        rsi = row['rsi']

        if rsi < rsi_buy:
            buy_price = row['close']
            buy_time = df.index[i]
            macd_diff = row['macd_diff']
            uptrend = row['uptrend']

            # Find the next sell signal (RSI > rsi_sell)
            sell_price = None
            sell_time = None
            sell_rsi = None
            bars_held = 0

            for j in range(i + 1, min(i + 100, len(df))):  # Look up to 100 bars ahead
                future_row = df.iloc[j]
                if future_row['rsi'] > rsi_sell:
                    sell_price = future_row['close']
                    sell_time = df.index[j]
                    sell_rsi = future_row['rsi']
                    bars_held = j - i
                    break

            # If no sell signal found, use price 20 bars later as reference
            if sell_price is None:
                if i + 20 < len(df):
                    sell_price = df.iloc[i + 20]['close']
                    sell_time = df.index[i + 20]
                    sell_rsi = df.iloc[i + 20]['rsi']
                    bars_held = 20
                else:
                    continue

            pnl_pct = (sell_price - buy_price) / buy_price * 100

            opportunities.append({
                'buy_time': buy_time,
                'buy_price': buy_price,
                'rsi_at_buy': rsi,
                'macd_diff': macd_diff,
                'uptrend': uptrend,
                'sell_time': sell_time,
                'sell_price': sell_price,
                'rsi_at_sell': sell_rsi,
                'bars_held': bars_held,
                'pnl_pct': pnl_pct,
                'would_profit': pnl_pct > 0
            })

            # Skip ahead to avoid counting the same dip multiple times
            # (only count first bar of each dip)
            while i + 1 < len(df) - 20 and df.iloc[i + 1]['rsi'] < rsi_buy:
                i += 1

    # Convert to dataframe for analysis
    opp_df = pd.DataFrame(opportunities)

    if len(opp_df) == 0:
        print("No opportunities found!")
        return

    # Split by uptrend/downtrend
    uptrend_opps = opp_df[opp_df['uptrend'] == True]
    downtrend_opps = opp_df[opp_df['uptrend'] == False]

    print("=" * 70)
    print("SUMMARY: What if we bought EVERY dip?")
    print("=" * 70)
    print()
    print(f"Total dips found (RSI < {rsi_buy}): {len(opp_df)}")
    print()

    # Uptrend opportunities
    print("-" * 70)
    print("UPTREND DIPS (MACD above signal - we WOULD have bought)")
    print("-" * 70)
    if len(uptrend_opps) > 0:
        winners = uptrend_opps[uptrend_opps['would_profit'] == True]
        losers = uptrend_opps[uptrend_opps['would_profit'] == False]
        print(f"  Total: {len(uptrend_opps)}")
        print(f"  Winners: {len(winners)} ({len(winners)/len(uptrend_opps)*100:.0f}%)")
        print(f"  Losers: {len(losers)} ({len(losers)/len(uptrend_opps)*100:.0f}%)")
        print(f"  Avg profit: {uptrend_opps['pnl_pct'].mean():+.2f}%")
        print(f"  Best trade: {uptrend_opps['pnl_pct'].max():+.2f}%")
        print(f"  Worst trade: {uptrend_opps['pnl_pct'].min():+.2f}%")
    else:
        print("  None found")
    print()

    # Downtrend opportunities
    print("-" * 70)
    print("DOWNTREND DIPS (MACD below signal - we SKIPPED these)")
    print("-" * 70)
    if len(downtrend_opps) > 0:
        winners = downtrend_opps[downtrend_opps['would_profit'] == True]
        losers = downtrend_opps[downtrend_opps['would_profit'] == False]
        print(f"  Total: {len(downtrend_opps)}")
        print(f"  Winners: {len(winners)} ({len(winners)/len(downtrend_opps)*100:.0f}%)")
        print(f"  Losers: {len(losers)} ({len(losers)/len(downtrend_opps)*100:.0f}%)")
        print(f"  Avg profit: {downtrend_opps['pnl_pct'].mean():+.2f}%")
        print(f"  Best trade: {downtrend_opps['pnl_pct'].max():+.2f}%")
        print(f"  Worst trade: {downtrend_opps['pnl_pct'].min():+.2f}%")
    else:
        print("  None found")
    print()

    # Show ALL opportunities sorted by profit
    print("=" * 70)
    print("ALL OPPORTUNITIES RANKED BY PROFIT")
    print("=" * 70)
    print()
    print(f"{'Date':<12} {'Trend':<10} {'RSI':<6} {'MACD Diff':<10} {'P&L':<8} {'Result'}")
    print("-" * 70)

    for _, row in opp_df.sort_values('pnl_pct', ascending=False).iterrows():
        date_str = str(row['buy_time'])[:10]
        trend = "UPTREND" if row['uptrend'] else "DOWNTREND"
        rsi = row['rsi_at_buy']
        macd = row['macd_diff']
        pnl = row['pnl_pct']
        result = "WIN" if row['would_profit'] else "LOSS"
        action = "BOUGHT" if row['uptrend'] else "SKIPPED"

        print(f"{date_str:<12} {trend:<10} {rsi:<6.1f} {macd:<+10.4f} {pnl:<+8.2f}% {result:<6} ({action})")

    # Find the best and worst
    print()
    print("=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    print()

    best = opp_df.loc[opp_df['pnl_pct'].idxmax()]
    worst = opp_df.loc[opp_df['pnl_pct'].idxmin()]

    print("BEST OPPORTUNITY:")
    print(f"  Date: {str(best['buy_time'])[:10]}")
    print(f"  Buy price: ${best['buy_price']:.2f} -> Sell: ${best['sell_price']:.2f}")
    print(f"  Profit: {best['pnl_pct']:+.2f}%")
    print(f"  RSI at buy: {best['rsi_at_buy']:.1f}")
    print(f"  MACD - Signal: {best['macd_diff']:+.4f}")
    print(f"  Trend: {'UPTREND' if best['uptrend'] else 'DOWNTREND'}")
    print(f"  We {'BOUGHT' if best['uptrend'] else 'SKIPPED'} this one")
    print()

    print("WORST OPPORTUNITY:")
    print(f"  Date: {str(worst['buy_time'])[:10]}")
    print(f"  Buy price: ${worst['buy_price']:.2f} -> Sell: ${worst['sell_price']:.2f}")
    print(f"  Profit: {worst['pnl_pct']:+.2f}%")
    print(f"  RSI at buy: {worst['rsi_at_buy']:.1f}")
    print(f"  MACD - Signal: {worst['macd_diff']:+.4f}")
    print(f"  Trend: {'UPTREND' if worst['uptrend'] else 'DOWNTREND'}")
    print(f"  We {'BOUGHT' if worst['uptrend'] else 'SKIPPED'} this one")
    print()

    # Did MACD filter actually help?
    print("=" * 70)
    print("DID THE MACD FILTER HELP?")
    print("=" * 70)
    print()

    uptrend_avg = uptrend_opps['pnl_pct'].mean() if len(uptrend_opps) > 0 else 0
    downtrend_avg = downtrend_opps['pnl_pct'].mean() if len(downtrend_opps) > 0 else 0

    print(f"Avg profit when MACD bullish (we bought):   {uptrend_avg:+.2f}%")
    print(f"Avg profit when MACD bearish (we skipped): {downtrend_avg:+.2f}%")
    print()

    if uptrend_avg > downtrend_avg:
        print(f"VERDICT: MACD filter HELPED! We avoided trades that averaged {downtrend_avg:+.2f}%")
        print(f"         and took trades that averaged {uptrend_avg:+.2f}%")
    else:
        print(f"VERDICT: MACD filter HURT! We skipped trades that averaged {downtrend_avg:+.2f}%")
        print(f"         and took trades that averaged only {uptrend_avg:+.2f}%")

    # Missed profits
    print()
    if len(downtrend_opps) > 0:
        missed_winners = downtrend_opps[downtrend_opps['would_profit'] == True]
        if len(missed_winners) > 0:
            print(f"MISSED PROFITS: We skipped {len(missed_winners)} winning trades!")
            print(f"  Total missed profit: {missed_winners['pnl_pct'].sum():.2f}%")
            print()
            print("Top 5 missed winners:")
            for _, row in missed_winners.nlargest(5, 'pnl_pct').iterrows():
                print(f"  {str(row['buy_time'])[:10]} | RSI={row['rsi_at_buy']:.1f} | MACD={row['macd_diff']:+.4f} | +{row['pnl_pct']:.2f}%")

    # Avoided losses
    print()
    if len(downtrend_opps) > 0:
        avoided_losers = downtrend_opps[downtrend_opps['would_profit'] == False]
        if len(avoided_losers) > 0:
            print(f"AVOIDED LOSSES: We skipped {len(avoided_losers)} losing trades!")
            print(f"  Total avoided loss: {avoided_losers['pnl_pct'].sum():.2f}%")
            print()
            print("Top 5 avoided losers:")
            for _, row in avoided_losers.nsmallest(5, 'pnl_pct').iterrows():
                print(f"  {str(row['buy_time'])[:10]} | RSI={row['rsi_at_buy']:.1f} | MACD={row['macd_diff']:+.4f} | {row['pnl_pct']:.2f}%")

    return opp_df

if __name__ == '__main__':
    result = analyze_all_opportunities(
        symbol='AAPL',
        start_date='2024-09-01',
        end_date='2024-12-01',
        rsi_buy=35,
        rsi_sell=65
    )
