"""
Adaptive Strategy Trading - Backtest with Full Logging
=======================================================

Before each trade decision:
1. Run all 7 strategies on recent data (calibration window)
2. Pick the best performing strategy
3. Use THAT strategy's signal to decide buy/sell/hold
4. Log everything - every test, every decision, every hypothetical

Daily synopsis report explains everything in plain English.

Author: Claude AI
Date: December 2024
"""

import os
import sys
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict
from pathlib import Path

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


# =============================================================================
# INDICATOR CALCULATIONS
# =============================================================================

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices: pd.Series):
    ema12 = prices.ewm(span=12).mean()
    ema26 = prices.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    diff = macd - signal
    return macd, signal, diff


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


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    return prices.rolling(window=period).mean()


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add all indicators to dataframe."""
    df = df.copy()
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])
    df['sma20'] = calculate_sma(df['close'], 20)
    return df


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

STRATEGIES = {
    "RSI 30/70": {
        "buy": lambda df, i: df.iloc[i]['rsi'] < 30,
        "sell": lambda df, i: df.iloc[i]['rsi'] > 70,
        "buy_desc": "RSI < 30",
        "sell_desc": "RSI > 70"
    },
    "RSI 40/70": {
        "buy": lambda df, i: df.iloc[i]['rsi'] < 40,
        "sell": lambda df, i: df.iloc[i]['rsi'] > 70,
        "buy_desc": "RSI < 40",
        "sell_desc": "RSI > 70"
    },
    "MACD Cross": {
        "buy": lambda df, i: i >= 1 and df.iloc[i]['macd_diff'] > 0 and df.iloc[i-1]['macd_diff'] <= 0,
        "sell": lambda df, i: i >= 1 and df.iloc[i]['macd_diff'] < 0 and df.iloc[i-1]['macd_diff'] >= 0,
        "buy_desc": "MACD crosses above signal",
        "sell_desc": "MACD crosses below signal"
    },
    "Bollinger 20/80": {
        "buy": lambda df, i: df.iloc[i]['bb_position'] < 0.2,
        "sell": lambda df, i: df.iloc[i]['bb_position'] > 0.8,
        "buy_desc": "Price at lower 20% of Bollinger bands",
        "sell_desc": "Price at upper 20% of Bollinger bands"
    },
    "Bollinger Mean Rev": {
        "buy": lambda df, i: df.iloc[i]['bb_position'] < 0.2,
        "sell": lambda df, i: df.iloc[i]['bb_position'] > 0.5,
        "buy_desc": "Price at lower 20% of Bollinger bands",
        "sell_desc": "Price returns to middle of bands"
    },
    "Volume Spike": {
        "buy": lambda df, i: i >= 1 and df.iloc[i]['volume_ratio'] > 1.5 and df.iloc[i]['close'] < df.iloc[i-1]['close'],
        "sell": lambda df, i: i >= 1 and df.iloc[i]['volume_ratio'] > 1.5 and df.iloc[i]['close'] > df.iloc[i-1]['close'],
        "buy_desc": "High volume (1.5x avg) + price drop",
        "sell_desc": "High volume (1.5x avg) + price rise"
    },
    "Mean Rev 2%": {
        "buy": lambda df, i: df.iloc[i]['close'] < df.iloc[i]['sma20'] * 0.98,
        "sell": lambda df, i: df.iloc[i]['close'] >= df.iloc[i]['sma20'],
        "buy_desc": "Price 2%+ below 20-bar average",
        "sell_desc": "Price returns to 20-bar average"
    }
}


# =============================================================================
# LOGGING STRUCTURES
# =============================================================================

@dataclass
class CalibrationResult:
    """Result of testing one strategy."""
    strategy_name: str
    trades: int
    winners: int
    win_rate: float
    return_pct: float
    buy_rule: str
    sell_rule: str


@dataclass
class CalibrationLog:
    """Log of a full calibration (testing all strategies)."""
    timestamp: datetime
    symbol: str
    calibration_window_days: int
    results: List[CalibrationResult]
    best_strategy: str
    best_return: float
    reason: str  # Why this strategy was picked


@dataclass
class TradeDecision:
    """A single trade decision (YES or NO)."""
    timestamp: datetime
    symbol: str
    price: float
    action: str  # 'BUY', 'SELL', 'HOLD'
    decided: str  # 'YES' or 'NO'
    strategy_used: str
    strategy_signal: bool  # Did the strategy say to trade?
    reason: str  # Why we made this decision

    # Indicator values at decision time
    rsi: float
    macd_diff: float
    bb_position: float
    volume_ratio: float

    # For actual trades
    shares: int = 0
    cost_basis: float = 0
    pnl: float = 0
    pnl_pct: float = 0
    result: str = ""  # 'WIN', 'LOSS', or ''

    # For hypotheticals (when we said NO)
    hypothetical_entry: float = 0
    hypothetical_exit: float = 0
    hypothetical_pnl_pct: float = 0
    hypothetical_result: str = ""  # 'WOULD WIN', 'WOULD LOSE', ''


@dataclass
class DailyLog:
    """All activity for one day."""
    date: str
    symbols: List[str]
    calibrations: List[CalibrationLog]
    decisions: List[TradeDecision]

    # Summary stats
    trades_executed: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0

    missed_opportunities: int = 0  # Times we said NO but would have won
    avoided_losses: int = 0  # Times we said NO and would have lost


# =============================================================================
# STRATEGY CALIBRATION
# =============================================================================

def run_strategy_backtest(df: pd.DataFrame, strategy_name: str,
                          initial_capital: float = 10000) -> CalibrationResult:
    """Backtest a single strategy on the data."""
    strategy = STRATEGIES[strategy_name]

    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []

    for i in range(len(df)):
        price = df.iloc[i]['close']

        # Buy
        if strategy['buy'](df, i) and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({'type': 'BUY', 'price': price})

        # Sell
        elif strategy['sell'](df, i) and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            cash += proceeds
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl})
            shares = 0

    # Liquidate at end
    if shares > 0:
        final_price = df.iloc[-1]['close']
        proceeds = shares * final_price
        pnl = proceeds - (shares * position_price)
        cash += proceeds
        trades.append({'type': 'SELL', 'price': final_price, 'pnl': pnl})

    sell_trades = [t for t in trades if t['type'] == 'SELL']
    winners = [t for t in sell_trades if t.get('pnl', 0) > 0]

    return CalibrationResult(
        strategy_name=strategy_name,
        trades=len(sell_trades),
        winners=len(winners),
        win_rate=len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        return_pct=(cash - initial_capital) / initial_capital * 100,
        buy_rule=strategy['buy_desc'],
        sell_rule=strategy['sell_desc']
    )


def calibrate_for_symbol(df_calibration: pd.DataFrame, symbol: str,
                         timestamp: datetime) -> CalibrationLog:
    """Run all strategies and pick the best one."""
    results = []

    for strategy_name in STRATEGIES:
        result = run_strategy_backtest(df_calibration, strategy_name)
        results.append(result)

    # Sort by return
    results.sort(key=lambda x: x.return_pct, reverse=True)
    best = results[0]

    # Build reason
    if best.return_pct > 0:
        reason = f"{best.strategy_name} had the best return (+{best.return_pct:.1f}%) with {best.win_rate:.0f}% win rate over {len(df_calibration)} bars"
    else:
        reason = f"{best.strategy_name} had the least loss ({best.return_pct:.1f}%) - all strategies were negative"

    return CalibrationLog(
        timestamp=timestamp,
        symbol=symbol,
        calibration_window_days=90,
        results=results,
        best_strategy=best.strategy_name,
        best_return=best.return_pct,
        reason=reason
    )


# =============================================================================
# MAIN BACKTEST ENGINE
# =============================================================================

class AdaptiveStrategyBacktest:
    """
    Backtest the Adaptive Strategy system with full logging.

    Before each potential trade:
    1. Calibrate (test all 7 strategies on recent data)
    2. Use the best strategy's signal
    3. Log everything
    """

    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        calibration_window_days: int = 90,
        initial_capital: float = 100000,
        position_size_pct: float = 0.10,
        max_positions: int = 5
    ):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.calibration_window_days = calibration_window_days
        self.initial_capital = initial_capital
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        self.max_sell_pct = 0.85  # Can sell up to 85% of position at a time
        self.trailing_stop_pct = 0.02  # Only sell if price drops 2% from peak
        self.max_loss_to_sell = 0.10  # Don't sell if we'd lose more than 10% - hold for recovery

        self.client = StockHistoricalDataClient(API_KEY, API_SECRET)

        # State
        self.cash = initial_capital
        self.positions: Dict[str, dict] = {}  # symbol -> {shares, entry_price, entry_time, peak_price}

        # Logs
        self.calibration_logs: List[CalibrationLog] = []
        self.decisions: List[TradeDecision] = []
        self.daily_logs: List[DailyLog] = []

        # Data cache
        self.data_cache: Dict[str, pd.DataFrame] = {}

        # Calibration cache - don't recalibrate within same day
        self.calibration_cache: Dict[str, tuple] = {}  # symbol -> (date, best_strategy, calibration)

        # Output directory
        self.log_dir = Path("logs/adaptive_strategy")
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def fetch_all_data(self):
        """Fetch all historical data upfront."""
        print(f"Fetching data for {len(self.symbols)} symbols...")

        # Need extra data before start_date for calibration
        cal_start = datetime.strptime(self.start_date, '%Y-%m-%d') - timedelta(days=self.calibration_window_days + 30)
        end = datetime.strptime(self.end_date, '%Y-%m-%d')

        for symbol in self.symbols:
            try:
                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame.Hour,
                    start=cal_start,
                    end=end
                )

                bars = self.client.get_stock_bars(request)
                df = bars.df

                if isinstance(df.index, pd.MultiIndex):
                    df = df.xs(symbol, level='symbol')

                df = add_all_indicators(df)
                df = df.dropna()

                self.data_cache[symbol] = df
                print(f"  {symbol}: {len(df)} bars")

            except Exception as e:
                print(f"  {symbol}: Error - {e}")

        print()

    def get_calibration_data(self, symbol: str, current_time) -> Optional[pd.DataFrame]:
        """Get data for calibration (90 days before current_time)."""
        df = self.data_cache.get(symbol)
        if df is None:
            return None

        cal_start = current_time - pd.Timedelta(days=self.calibration_window_days)
        mask = (df.index >= cal_start) & (df.index < current_time)
        return df[mask]

    def calculate_hypothetical(self, df: pd.DataFrame, entry_idx: int,
                               strategy_name: str) -> tuple:
        """
        Calculate what would have happened if we had traded.
        Returns (exit_price, pnl_pct, result)
        """
        strategy = STRATEGIES[strategy_name]
        entry_price = df.iloc[entry_idx]['close']

        # Find when we would have sold
        for j in range(entry_idx + 1, len(df)):
            if strategy['sell'](df, j):
                exit_price = df.iloc[j]['close']
                pnl_pct = (exit_price - entry_price) / entry_price * 100
                result = 'WOULD WIN' if pnl_pct > 0 else 'WOULD LOSE'
                return exit_price, pnl_pct, result

        # Never hit sell signal - use last price
        exit_price = df.iloc[-1]['close']
        pnl_pct = (exit_price - entry_price) / entry_price * 100
        result = 'WOULD WIN' if pnl_pct > 0 else 'WOULD LOSE'
        return exit_price, pnl_pct, result

    def run_backtest(self):
        """Run the full backtest."""
        print("="*70)
        print("ADAPTIVE STRATEGY BACKTEST")
        print("="*70)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Calibration Window: {self.calibration_window_days} days")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print()

        # Fetch data
        self.fetch_all_data()

        # Get all unique timestamps across all symbols
        all_timestamps = set()
        for symbol, df in self.data_cache.items():
            start = pd.Timestamp(self.start_date, tz='UTC')
            end = pd.Timestamp(self.end_date, tz='UTC') + pd.Timedelta(days=1)
            mask = (df.index >= start) & (df.index <= end)
            all_timestamps.update(df[mask].index.tolist())

        all_timestamps = sorted(all_timestamps)
        print(f"Processing {len(all_timestamps)} time periods...")
        print()

        current_date = None
        daily_decisions = []
        daily_calibrations = []

        progress_interval = max(1, len(all_timestamps) // 20)  # Show progress 20 times

        for ti, timestamp in enumerate(all_timestamps):
            if ti % progress_interval == 0:
                print(f"  Progress: {ti}/{len(all_timestamps)} ({ti/len(all_timestamps)*100:.0f}%)")

            ts_date = timestamp.date()

            # New day - save previous day's log
            if current_date and ts_date != current_date:
                self._save_daily_log(current_date, daily_calibrations, daily_decisions)
                daily_decisions = []
                daily_calibrations = []

            current_date = ts_date

            # Process each symbol at this timestamp
            for symbol in self.symbols:
                df = self.data_cache.get(symbol)
                if df is None:
                    continue

                if timestamp not in df.index:
                    continue

                idx = df.index.get_loc(timestamp)
                if idx < 1:
                    continue

                row = df.iloc[idx]
                price = row['close']

                # Check if any indicator is near a tradeable threshold
                # Only calibrate if we might actually trade
                should_calibrate = False
                has_position = symbol in self.positions

                if not has_position:
                    # Might buy - check if any buy indicators are near threshold
                    if (row['rsi'] < 45 or row['bb_position'] < 0.35 or
                        row['volume_ratio'] > 1.3 or
                        row['close'] < row['sma20'] * 0.99):
                        should_calibrate = True
                else:
                    # Might sell - check if any sell indicators are near threshold
                    if (row['rsi'] > 60 or row['bb_position'] > 0.65 or
                        row['volume_ratio'] > 1.3):
                        should_calibrate = True

                if not should_calibrate:
                    continue

                # Check calibration cache - only recalibrate once per day per symbol
                cache_key = symbol
                ts_date_str = str(ts_date)
                cached = self.calibration_cache.get(cache_key)

                if cached and cached[0] == ts_date_str:
                    # Use cached calibration
                    best_strategy = cached[1]
                    calibration = cached[2]
                else:
                    # Get calibration data
                    cal_data = self.get_calibration_data(symbol, timestamp)
                    if cal_data is None or len(cal_data) < 50:
                        continue

                    # Calibrate before decision
                    calibration = calibrate_for_symbol(cal_data, symbol, timestamp)
                    best_strategy = calibration.best_strategy

                    # Cache it
                    self.calibration_cache[cache_key] = (ts_date_str, best_strategy, calibration)

                    daily_calibrations.append(calibration)
                    self.calibration_logs.append(calibration)

                strategy = STRATEGIES[best_strategy]

                # Check signals
                has_position = symbol in self.positions
                buy_signal = strategy['buy'](df, idx)
                sell_signal = strategy['sell'](df, idx)

                # Make decision
                decision = TradeDecision(
                    timestamp=timestamp,
                    symbol=symbol,
                    price=price,
                    action='HOLD',
                    decided='NO',
                    strategy_used=best_strategy,
                    strategy_signal=False,
                    reason='',
                    rsi=row['rsi'],
                    macd_diff=row['macd_diff'],
                    bb_position=row['bb_position'],
                    volume_ratio=row['volume_ratio']
                )

                if not has_position:
                    # Consider buying
                    if buy_signal:
                        decision.strategy_signal = True
                        decision.action = 'BUY'

                        # Check if we can buy
                        if len(self.positions) < self.max_positions:
                            # Execute buy
                            position_value = self.cash * self.position_size_pct
                            shares = int(position_value / price)

                            if shares > 0 and self.cash >= shares * price:
                                self.cash -= shares * price
                                self.positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'entry_time': timestamp,
                                    'peak_price': price  # Track highest price
                                }

                                decision.decided = 'YES'
                                decision.shares = shares
                                decision.cost_basis = shares * price
                                decision.reason = f"BUY signal from {best_strategy}: {strategy['buy_desc']}"
                            else:
                                decision.decided = 'NO'
                                decision.reason = "Insufficient cash"
                        else:
                            decision.decided = 'NO'
                            decision.reason = f"Max positions ({self.max_positions}) reached"

                            # Calculate hypothetical
                            exit_price, hypo_pnl, hypo_result = self.calculate_hypothetical(
                                df, idx, best_strategy
                            )
                            decision.hypothetical_entry = price
                            decision.hypothetical_exit = exit_price
                            decision.hypothetical_pnl_pct = hypo_pnl
                            decision.hypothetical_result = hypo_result
                    else:
                        decision.action = 'HOLD'
                        decision.reason = f"No buy signal from {best_strategy} ({strategy['buy_desc']} not met)"

                        # Check if we WOULD have made money if we bought anyway
                        # Only log these for "close calls" where indicators are near threshold
                        if row['rsi'] < 45 or row['bb_position'] < 0.3:
                            exit_price, hypo_pnl, hypo_result = self.calculate_hypothetical(
                                df, idx, best_strategy
                            )
                            decision.hypothetical_entry = price
                            decision.hypothetical_exit = exit_price
                            decision.hypothetical_pnl_pct = hypo_pnl
                            decision.hypothetical_result = hypo_result

                else:
                    # Consider selling
                    pos = self.positions[symbol]

                    # Update peak price if current price is higher
                    if price > pos['peak_price']:
                        pos['peak_price'] = price
                        self.positions[symbol] = pos

                    # Check trailing stop - only sell if price dropped 2% from peak
                    peak = pos['peak_price']
                    entry = pos['entry_price']
                    drop_from_peak = (peak - price) / peak
                    trailing_stop_hit = drop_from_peak >= self.trailing_stop_pct

                    # Check if we'd be locking in too big a loss
                    current_loss_pct = (entry - price) / entry
                    would_lose_too_much = current_loss_pct > self.max_loss_to_sell

                    if sell_signal and trailing_stop_hit and not would_lose_too_much:
                        decision.strategy_signal = True
                        decision.action = 'SELL'
                        decision.decided = 'YES'

                        # Execute PARTIAL sell (up to 85% of position)
                        total_shares = pos['shares']
                        shares_to_sell = int(total_shares * self.max_sell_pct)

                        if shares_to_sell < 1:
                            shares_to_sell = total_shares  # Sell all if position is tiny

                        entry_price = pos['entry_price']

                        proceeds = shares_to_sell * price
                        pnl = proceeds - (shares_to_sell * entry_price)
                        pnl_pct = (price - entry_price) / entry_price * 100

                        self.cash += proceeds

                        # Update or remove position
                        remaining_shares = total_shares - shares_to_sell
                        if remaining_shares > 0:
                            self.positions[symbol]['shares'] = remaining_shares
                            decision.reason = f"PARTIAL SELL ({shares_to_sell}/{total_shares} shares, keeping {remaining_shares}) - {best_strategy}: {strategy['sell_desc']}"
                        else:
                            del self.positions[symbol]
                            decision.reason = f"FULL SELL - {best_strategy}: {strategy['sell_desc']}"

                        decision.shares = shares_to_sell
                        decision.pnl = pnl
                        decision.pnl_pct = pnl_pct
                        decision.result = 'WIN' if pnl > 0 else 'LOSS'
                    elif sell_signal and would_lose_too_much:
                        decision.action = 'HOLD'
                        decision.reason = f"Sell signal but holding to avoid locking in {current_loss_pct*100:.1f}% loss (max allowed: {self.max_loss_to_sell*100:.0f}%) - waiting for recovery"
                    elif sell_signal and not trailing_stop_hit:
                        decision.action = 'HOLD'
                        decision.reason = f"Sell signal but trailing stop not hit (peak ${peak:.2f}, current ${price:.2f}, drop {drop_from_peak*100:.1f}% < {self.trailing_stop_pct*100:.0f}% required)"
                    else:
                        decision.action = 'HOLD'
                        decision.reason = f"No sell signal from {best_strategy} ({strategy['sell_desc']} not met)"

                self.decisions.append(decision)
                daily_decisions.append(decision)

        # Save last day
        if current_date:
            self._save_daily_log(current_date, daily_calibrations, daily_decisions)

        # Final liquidation
        self._liquidate_all()

        # Generate reports
        self._generate_summary_report()
        self._save_all_logs()

    def _save_daily_log(self, date, calibrations, decisions):
        """Save a daily log."""
        trades = [d for d in decisions if d.decided == 'YES']
        wins = [d for d in trades if d.result == 'WIN']
        losses = [d for d in trades if d.result == 'LOSS']

        missed = [d for d in decisions if d.decided == 'NO' and d.hypothetical_result == 'WOULD WIN']
        avoided = [d for d in decisions if d.decided == 'NO' and d.hypothetical_result == 'WOULD LOSE']

        daily_log = DailyLog(
            date=str(date),
            symbols=self.symbols,
            calibrations=calibrations,
            decisions=decisions,
            trades_executed=len(trades),
            winning_trades=len(wins),
            losing_trades=len(losses),
            total_pnl=sum(d.pnl for d in trades),
            missed_opportunities=len(missed),
            avoided_losses=len(avoided)
        )

        self.daily_logs.append(daily_log)

    def _liquidate_all(self):
        """Liquidate all remaining positions."""
        for symbol, pos in list(self.positions.items()):
            df = self.data_cache.get(symbol)
            if df is None:
                continue

            final_price = df.iloc[-1]['close']
            shares = pos['shares']
            entry_price = pos['entry_price']

            proceeds = shares * final_price
            pnl = proceeds - (shares * entry_price)

            self.cash += proceeds

            decision = TradeDecision(
                timestamp=df.index[-1],
                symbol=symbol,
                price=final_price,
                action='LIQUIDATE',
                decided='YES',
                strategy_used='End of backtest',
                strategy_signal=True,
                reason='Liquidating at end of backtest period',
                rsi=df.iloc[-1]['rsi'],
                macd_diff=df.iloc[-1]['macd_diff'],
                bb_position=df.iloc[-1]['bb_position'],
                volume_ratio=df.iloc[-1]['volume_ratio'],
                shares=shares,
                pnl=pnl,
                pnl_pct=(final_price - entry_price) / entry_price * 100,
                result='WIN' if pnl > 0 else 'LOSS'
            )
            self.decisions.append(decision)

        self.positions = {}

    def _generate_summary_report(self):
        """Generate the summary report."""
        print("\n" + "="*70)
        print("BACKTEST RESULTS")
        print("="*70)

        # Calculate stats
        all_trades = [d for d in self.decisions if d.decided == 'YES' and d.action in ['BUY', 'SELL', 'LIQUIDATE']]
        sell_trades = [d for d in all_trades if d.action in ['SELL', 'LIQUIDATE']]
        wins = [d for d in sell_trades if d.result == 'WIN']
        losses = [d for d in sell_trades if d.result == 'LOSS']

        total_pnl = sum(d.pnl for d in sell_trades)
        final_equity = self.cash
        total_return = (final_equity - self.initial_capital) / self.initial_capital * 100

        # Hypotheticals
        nos_with_hypo = [d for d in self.decisions if d.decided == 'NO' and d.hypothetical_result]
        missed_wins = [d for d in nos_with_hypo if d.hypothetical_result == 'WOULD WIN']
        avoided_losses = [d for d in nos_with_hypo if d.hypothetical_result == 'WOULD LOSE']

        print(f"\nPERFORMANCE SUMMARY")
        print("-"*70)
        print(f"Starting Capital:     ${self.initial_capital:>12,.2f}")
        print(f"Ending Capital:       ${final_equity:>12,.2f}")
        print(f"Total P&L:            ${total_pnl:>+12,.2f}")
        print(f"Total Return:         {total_return:>+12.2f}%")
        print()
        print(f"Total Trades:         {len(sell_trades):>12}")
        print(f"  Winners:            {len(wins):>12}")
        print(f"  Losers:             {len(losses):>12}")
        print(f"  Win Rate:           {len(wins)/len(sell_trades)*100 if sell_trades else 0:>11.1f}%")
        print()
        print(f"DECISIONS WE SAID NO:")
        print(f"  Would have won:     {len(missed_wins):>12} (missed opportunities)")
        print(f"  Would have lost:    {len(avoided_losses):>12} (good calls)")

        if missed_wins:
            avg_missed = sum(d.hypothetical_pnl_pct for d in missed_wins) / len(missed_wins)
            print(f"  Avg missed gain:    {avg_missed:>+11.2f}%")

        if avoided_losses:
            avg_avoided = sum(d.hypothetical_pnl_pct for d in avoided_losses) / len(avoided_losses)
            print(f"  Avg avoided loss:   {avg_avoided:>+11.2f}%")

        # Strategy usage
        print(f"\nSTRATEGY USAGE:")
        print("-"*70)
        strategy_counts = {}
        for cal in self.calibration_logs:
            s = cal.best_strategy
            strategy_counts[s] = strategy_counts.get(s, 0) + 1

        for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
            print(f"  {strategy:<25} {count:>6} times selected")

        print("\n" + "="*70)

    def _save_all_logs(self):
        """Save all logs to files."""
        # Save detailed decision log
        decisions_data = []
        for d in self.decisions:
            decisions_data.append({
                'timestamp': str(d.timestamp),
                'symbol': d.symbol,
                'price': d.price,
                'action': d.action,
                'decided': d.decided,
                'strategy': d.strategy_used,
                'reason': d.reason,
                'rsi': round(d.rsi, 2),
                'macd_diff': round(d.macd_diff, 4),
                'bb_position': round(d.bb_position, 3),
                'volume_ratio': round(d.volume_ratio, 2),
                'shares': d.shares,
                'pnl': round(d.pnl, 2),
                'pnl_pct': round(d.pnl_pct, 2),
                'result': d.result,
                'hypothetical_pnl_pct': round(d.hypothetical_pnl_pct, 2),
                'hypothetical_result': d.hypothetical_result
            })

        with open(self.log_dir / 'decisions.json', 'w') as f:
            json.dump(decisions_data, f, indent=2)

        # Save calibration logs
        calibrations_data = []
        for c in self.calibration_logs:
            calibrations_data.append({
                'timestamp': str(c.timestamp),
                'symbol': c.symbol,
                'best_strategy': c.best_strategy,
                'best_return': round(c.best_return, 2),
                'reason': c.reason,
                'all_results': [
                    {
                        'strategy': r.strategy_name,
                        'trades': r.trades,
                        'win_rate': round(r.win_rate, 1),
                        'return_pct': round(r.return_pct, 2)
                    }
                    for r in c.results
                ]
            })

        with open(self.log_dir / 'calibrations.json', 'w') as f:
            json.dump(calibrations_data, f, indent=2)

        print(f"\nLogs saved to: {self.log_dir}")

    def generate_daily_synopsis(self, date: str) -> str:
        """Generate a plain-English synopsis for a specific day."""
        daily_log = None
        for dl in self.daily_logs:
            if dl.date == date:
                daily_log = dl
                break

        if not daily_log:
            return f"No data for {date}"

        report = []
        report.append("="*70)
        report.append(f"DAILY SYNOPSIS - {date}")
        report.append("="*70)
        report.append("")

        # What happened today
        report.append("WHAT HAPPENED TODAY:")
        report.append("-"*70)

        if daily_log.trades_executed == 0:
            report.append("We didn't make any trades today.")
        else:
            report.append(f"We made {daily_log.trades_executed} trade(s):")

            for d in daily_log.decisions:
                if d.decided == 'YES' and d.action in ['BUY', 'SELL']:
                    if d.action == 'BUY':
                        report.append(f"  - BOUGHT {d.shares} shares of {d.symbol} at ${d.price:.2f}")
                        report.append(f"    Reason: {d.reason}")
                    else:
                        report.append(f"  - SOLD {d.shares} shares of {d.symbol} at ${d.price:.2f}")
                        report.append(f"    Result: {d.result} (${d.pnl:+,.2f}, {d.pnl_pct:+.1f}%)")

        report.append("")
        report.append(f"Today's P&L: ${daily_log.total_pnl:+,.2f}")
        report.append(f"Wins: {daily_log.winning_trades} | Losses: {daily_log.losing_trades}")

        # What we said no to
        report.append("")
        report.append("WHAT WE SAID NO TO (and what would have happened):")
        report.append("-"*70)

        nos = [d for d in daily_log.decisions if d.decided == 'NO' and d.hypothetical_result]

        if not nos:
            report.append("No notable 'close calls' today.")
        else:
            for d in nos:
                report.append(f"  {d.symbol} at ${d.price:.2f}:")
                report.append(f"    Strategy said: {d.reason}")
                report.append(f"    If we had bought anyway: {d.hypothetical_result} ({d.hypothetical_pnl_pct:+.1f}%)")
                report.append("")

        report.append("")
        report.append("SUMMARY:")
        report.append("-"*70)
        report.append(f"  Missed opportunities (said NO, would have won): {daily_log.missed_opportunities}")
        report.append(f"  Good calls (said NO, would have lost): {daily_log.avoided_losses}")

        if daily_log.missed_opportunities > daily_log.avoided_losses:
            report.append("")
            report.append("  >> We were too conservative today - missed more winners than losers.")
        elif daily_log.avoided_losses > daily_log.missed_opportunities:
            report.append("")
            report.append("  >> We were appropriately cautious today - avoided more losers than winners.")

        report.append("")
        report.append("="*70)

        return "\n".join(report)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run backtest on portfolio stocks."""
    # Portfolio stocks - start with fewer for faster test
    symbols = ['AAPL', 'TSLA', 'NVDA', 'SPY']

    backtest = AdaptiveStrategyBacktest(
        symbols=symbols,
        start_date='2024-10-01',  # Shorter period for faster test
        end_date='2024-11-15',
        calibration_window_days=60,  # Shorter calibration window
        initial_capital=100000,
        position_size_pct=0.25,  # 25% per position = 100% with 4 max positions
        max_positions=4
    )

    backtest.run_backtest()

    # Generate a sample daily synopsis
    print("\n")
    if backtest.daily_logs:
        print(backtest.generate_daily_synopsis(backtest.daily_logs[len(backtest.daily_logs)//2].date))


if __name__ == '__main__':
    main()
