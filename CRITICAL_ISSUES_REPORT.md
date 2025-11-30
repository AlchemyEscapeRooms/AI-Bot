# CRITICAL ISSUES REPORT - AlpacaBot Trading System
## Date: November 29, 2025
## Status: 🔴 **NOT FUNCTIONAL - NO LIVE TRADES BEING EXECUTED**

---

## EXECUTIVE SUMMARY

After thorough examination of the logs and code, I identified why **ZERO live/paper trades have been executed**. The bot appears to run, shows "Running trading cycle" every minute, but never actually executes trades. Here's why:

---

## CRITICAL BUG #1: Strategies Only Trigger on Exact Crossover Moment

### Problem
The trading strategies (trend_following, momentum, etc.) are designed for **backtesting**, not live trading. They only generate signals at the **exact moment** of a crossover event.

### Code Evidence (strategies.py lines 191-192, 219-220):
```python
# Bullish crossover - only triggers when crossover happens RIGHT NOW
if prev_short <= prev_long and current_short > current_long:
    if symbol not in engine.open_positions:  # AND symbol not held
        signals.append({...})

# Bearish crossover - only triggers on exact crossover moment
elif prev_short >= prev_long and current_short < current_long:
    if symbol in engine.open_positions:  # AND must hold position
        signals.append({...})
```

### Why This Fails in Live Trading
- MA crossovers are **RARE events** - maybe 2-5 times per YEAR per symbol
- The bot checks every minute "did a crossover happen RIGHT NOW?"
- Answer is almost always NO
- Result: **No signals generated → No trades executed**

### Evidence from Logs
```
2025-11-28 11:23:37 | Running trading cycle at 11:23:37...
2025-11-28 11:23:55 | yfinance: Retrieved 249 bars for NFLX
2025-11-28 11:23:57 | yfinance: Retrieved 249 bars for BAC
[...data retrieved but NO trade signals, NO executions...]
2025-11-28 11:25:08 | Running trading cycle at 11:25:08...
```

50+ trading cycles ran on Nov 28 - **ZERO trades executed**.

---

## CRITICAL BUG #2: No Trade Logging Shows ZERO Live Activity

### Evidence
- **All logged trades have "BT-" prefix** = BACKTEST only
- **Zero "PP-" (paper) or "LV-" (live) trades exist**
- Database shows backtest trades from 2023, but no recent live activity

```
Trade logged: BT-20251125215634-0001 | BUY 519.57 TSLA
Trade logged: BT-20251125215634-0002 | SELL 519.57 TSLA
[All are BT = backtest, none are PP/LV]
```

---

## CRITICAL BUG #3: Alpaca API Connection Failing (DNS Error)

### Problem
Every API call to Alpaca fails with DNS resolution error:
```
Alpaca error for ARKK: [Errno 11001] getaddrinfo failed
```

### Impact
- Bot falls back to yfinance (which works)
- But **cannot execute actual trades** via Alpaca without fixing this
- Network/firewall issue on the Windows machine

---

## CRITICAL BUG #4: Silent Failures - No Error Logging When No Signals

### Problem
When strategies return empty signals, the bot logs NOTHING:

```python
if trades_this_cycle > 0:
    logger.info(f"Trading cycle complete: {trades_this_cycle} trades executed")
else:
    logger.debug("Trading cycle complete: no signals triggered")  # DEBUG level = not shown!
```

The "no signals" message is at DEBUG level, so you never see why nothing happened.

---

## ROOT CAUSE ANALYSIS

The core issue is a **fundamental architecture mismatch**:

| Aspect | Backtesting Mode | Live Trading Mode |
|--------|------------------|-------------------|
| Signal Generation | Runs through ALL historical data | Checks only current bar |
| Crossover Detection | Finds every crossover in history | Only finds if crossover is TODAY |
| Trade Frequency | Many trades over 2 years | Almost never trades |
| Position Management | Simulated positions | Empty positions always |

The strategies were designed to iterate through historical data and find patterns. In live mode, they only see "is there a signal right now?" which almost never triggers.

---

## RECOMMENDATIONS FOR IMMEDIATE FIXES

### Fix #1: Add Entry Signal Logic (Most Important)
Instead of waiting for exact crossovers, add **trend confirmation signals**:

```python
def trend_following_strategy_live(data, engine, params):
    signals = []
    
    short_ma = data['close'].rolling(20).mean()
    long_ma = data['close'].rolling(50).mean()
    
    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    current_price = data['close'].iloc[-1]
    
    # NEW: Trade when in established trend, not just crossover
    if current_short > current_long * 1.01:  # Short MA 1% above long = bullish trend
        if symbol not in engine.open_positions:
            # BUY in confirmed uptrend
            signals.append({...})
    
    elif current_short < current_long * 0.99:  # Short MA 1% below = bearish trend
        if symbol in engine.open_positions:
            # SELL in confirmed downtrend
            signals.append({...})
    
    return signals
```

### Fix #2: Fix Network/DNS for Alpaca
```bash
# Test connectivity
ping data.alpaca.markets
nslookup data.alpaca.markets

# Add to hosts file if DNS fails:
# 35.186.168.24 data.alpaca.markets
```

### Fix #3: Add Proper Logging for Debugging
```python
# In run_trading_cycle, change:
logger.debug("no signals triggered")  
# To:
logger.info(f"Cycle complete: 0 signals (checked {len(symbols)} symbols)")
```

### Fix #4: Consider Different Strategy Approach
For live trading, consider:
- RSI-based signals (oversold < 30 = buy, overbought > 70 = sell)
- Price vs SMA signals (price > SMA = uptrend, buy dips)
- Momentum signals (sustained positive momentum = stay long)

---

## DECISION QUALITY ANALYSIS

### Backtest Trade Decisions (from logs)
Looking at the backtest trades, the reasoning is clear but profits are mixed:

| Trade | Action | Reason | Entry | Exit | P&L |
|-------|--------|--------|-------|------|-----|
| TSLA-1 | BUY→SELL | momentum +60.2% → -2% | $173.22 | $187.71 | +8.4% ✅ |
| TSLA-2 | BUY→SELL | momentum +20% → -2% | $207.46 | $184.31 | -11.2% ❌ |
| TSLA-3 | BUY→SELL | momentum trigger | $173.86 | $267.43 | +53.8% ✅ |

**Observation**: The momentum strategy works well in backtests but has false signals. The 2% threshold for selling is very aggressive and causes premature exits.

### Recommendation for Higher Profits
1. **Wider exit thresholds**: Change momentum exit from -2% to -5%
2. **Add trailing stops**: Lock in profits as price rises
3. **Confirmation signals**: Don't buy on momentum alone - combine with RSI/MACD

---

## PRIORITY ACTION ITEMS

| Priority | Issue | Impact | Fix Effort |
|----------|-------|--------|------------|
| 🔴 P0 | Strategies don't generate live signals | NO TRADING | Medium |
| 🔴 P0 | Alpaca DNS failing | NO ORDER EXECUTION | Low |
| 🟠 P1 | Silent failures in logs | CAN'T DEBUG | Low |
| 🟠 P1 | Strategy logic too strict | RARE TRADES | Medium |
| 🟡 P2 | Momentum exit too aggressive | LOWER PROFITS | Low |

---

## CONCLUSION

**The bot is architecturally broken for live trading.** It was built and tested for backtesting where strategies iterate through historical data. In live mode, it only checks "is there a signal right this second?" which almost never triggers.

The fix requires redesigning the signal generation logic to:
1. Recognize **established trends** (not just crossover moments)
2. Generate **entry signals** when conditions are favorable
3. Use **trailing stops** instead of waiting for reverse crossovers

Until these fixes are made, the bot will continue to run cycles and generate zero trades.
