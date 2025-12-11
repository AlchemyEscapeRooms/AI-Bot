"""
Simple Trader - Strategy Selector Based Trading
================================================

Before trading each stock:
1. Run strategy selector to find best indicator
2. Use ONLY that indicator's buy/sell thresholds
3. Execute when threshold is hit

No weighted scoring. No AI. No complex combinations.
One stock, one indicator, clear rules.

Author: Claude AI
Date: December 2024
"""

import os
import sys
import time as time_module
import threading
from datetime import datetime, timedelta, time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except ImportError:
    from backports.zoneinfo import ZoneInfo  # Fallback for older Python

import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from dotenv import load_dotenv
load_dotenv()

# Import trade logger for logging all trades
from utils.trade_logger import get_trade_logger, TradeReason

sys.stdout.reconfigure(encoding='utf-8')

API_KEY = os.environ.get('ALPACA_API_KEY')
API_SECRET = os.environ.get('ALPACA_SECRET_KEY')


# =============================================================================
# PORTFOLIO SYMBOL FETCHING
# =============================================================================

def get_portfolio_symbols(paper: bool = True) -> List[str]:
    """
    Get symbols from current Alpaca portfolio positions.

    This ensures the bot only monitors/trades stocks you actually own.

    Args:
        paper: If True, use paper trading account. If False, use live account.

    Returns:
        List of stock symbols currently in the portfolio.
    """
    try:
        client = TradingClient(API_KEY, API_SECRET, paper=paper)
        positions = client.get_all_positions()
        symbols = [pos.symbol for pos in positions]

        if not symbols:
            print("[WARNING] No positions found in Alpaca portfolio!")
            print("          The bot needs stocks in your portfolio to monitor.")

        return symbols
    except Exception as e:
        print(f"[ERROR] Failed to get portfolio symbols: {e}")
        return []


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


# =============================================================================
# STRATEGY DEFINITIONS
# =============================================================================

@dataclass
class Strategy:
    """A trading strategy with clear buy/sell rules."""
    name: str
    buy_rule: str
    sell_rule: str

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        """Check if buy condition is met at index i."""
        raise NotImplementedError

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        """Check if sell condition is met at index i."""
        raise NotImplementedError


class RSI_30_70(Strategy):
    def __init__(self):
        super().__init__("RSI 30/70", "RSI < 30", "RSI > 70")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 30

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 70


class RSI_40_70(Strategy):
    def __init__(self):
        super().__init__("RSI 40/70", "RSI < 40", "RSI > 70")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] < 40

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['rsi'] > 70


class MACDCross(Strategy):
    def __init__(self):
        super().__init__("MACD Cross", "MACD crosses above signal", "MACD crosses below signal")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        return df.iloc[i]['macd_diff'] > 0 and df.iloc[i-1]['macd_diff'] <= 0

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        return df.iloc[i]['macd_diff'] < 0 and df.iloc[i-1]['macd_diff'] >= 0


class Bollinger_20_80(Strategy):
    def __init__(self):
        super().__init__("Bollinger 20/80", "Price at lower 20% of bands", "Price at upper 20% of bands")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] < 0.2

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] > 0.8


class BollingerMeanRev(Strategy):
    def __init__(self):
        super().__init__("Bollinger Mean Rev", "Price at lower 20% of bands", "Price returns to middle")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] < 0.2

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['bb_position'] > 0.5


class VolumeSpike(Strategy):
    def __init__(self):
        super().__init__("Volume Spike", "High volume + price drop", "High volume + price rise")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        high_vol = df.iloc[i]['volume_ratio'] > 1.5
        price_down = df.iloc[i]['close'] < df.iloc[i-1]['close']
        return high_vol and price_down

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        if i < 1:
            return False
        high_vol = df.iloc[i]['volume_ratio'] > 1.5
        price_up = df.iloc[i]['close'] > df.iloc[i-1]['close']
        return high_vol and price_up


class MeanRev2Pct(Strategy):
    def __init__(self):
        super().__init__("Mean Rev 2%", "Price 2%+ below SMA20", "Price returns to SMA20")

    def check_buy(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] < df.iloc[i]['sma20'] * 0.98

    def check_sell(self, df: pd.DataFrame, i: int) -> bool:
        return df.iloc[i]['close'] >= df.iloc[i]['sma20']


# Map strategy names to classes
STRATEGIES = {
    "RSI 30/70": RSI_30_70,
    "RSI 40/70": RSI_40_70,
    "MACD Cross": MACDCross,
    "Bollinger 20/80": Bollinger_20_80,
    "Bollinger Mean Rev": BollingerMeanRev,
    "Volume Spike": VolumeSpike,
    "Mean Rev 2%": MeanRev2Pct,
}


# =============================================================================
# STRATEGY SELECTOR
# =============================================================================

def backtest_strategy(df: pd.DataFrame, strategy: Strategy, initial_capital: float = 10000,
                       symbol: str = 'UNKNOWN', log_trades: bool = False) -> dict:
    """Run backtest with given strategy."""
    cash = initial_capital
    shares = 0
    position_price = 0
    trades = []
    trade_logger = None

    if log_trades:
        try:
            trade_logger = get_trade_logger()
        except:
            pass

    for i in range(len(df)):
        price = df.iloc[i]['close']
        timestamp = df.index[i] if hasattr(df.index[i], 'to_pydatetime') else datetime.now()

        # Buy
        if strategy.check_buy(df, i) and shares == 0:
            shares = int(cash * 0.95 / price)
            if shares > 0:
                cash -= shares * price
                position_price = price
                trades.append({'type': 'BUY', 'price': price, 'shares': shares, 'timestamp': timestamp})

                # Log backtest trade
                if trade_logger:
                    try:
                        reason = TradeReason(
                            primary_signal=strategy.name,
                            signal_value=0,
                            threshold=0,
                            direction='buy_signal',
                            explanation=f"BACKTEST BUY: {strategy.buy_rule}"
                        )
                        trade_logger.log_trade(
                            symbol=symbol,
                            action='BUY',
                            quantity=shares,
                            price=price,
                            strategy_name=strategy.name,
                            strategy_params={'buy_rule': strategy.buy_rule, 'sell_rule': strategy.sell_rule},
                            reason=reason,
                            mode='backtest',
                            portfolio_value_before=cash + shares * price,
                            timestamp=timestamp if isinstance(timestamp, datetime) else None
                        )
                    except:
                        pass

        # Sell
        elif strategy.check_sell(df, i) and shares > 0:
            proceeds = shares * price
            pnl = proceeds - (shares * position_price)
            cash += proceeds
            trades.append({'type': 'SELL', 'price': price, 'pnl': pnl, 'shares': shares, 'timestamp': timestamp})

            # Log backtest trade
            if trade_logger:
                try:
                    reason = TradeReason(
                        primary_signal=strategy.name,
                        signal_value=0,
                        threshold=0,
                        direction='sell_signal',
                        explanation=f"BACKTEST SELL: {strategy.sell_rule}"
                    )
                    trade_logger.log_trade(
                        symbol=symbol,
                        action='SELL',
                        quantity=shares,
                        price=price,
                        strategy_name=strategy.name,
                        strategy_params={'buy_rule': strategy.buy_rule, 'sell_rule': strategy.sell_rule},
                        reason=reason,
                        mode='backtest',
                        portfolio_value_before=cash,
                        realized_pnl=pnl,
                        timestamp=timestamp if isinstance(timestamp, datetime) else None
                    )
                except:
                    pass

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
    total_pnl = sum(t.get('pnl', 0) for t in sell_trades)

    return {
        'name': strategy.name,
        'trades': len(sell_trades),
        'winners': len(winners),
        'win_rate': len(winners) / len(sell_trades) * 100 if sell_trades else 0,
        'return_pct': (cash - initial_capital) / initial_capital * 100
    }


def select_best_strategy(symbol: str, lookback_days: int = 90) -> dict:
    """
    Test all strategies on a stock and return the best one.
    """
    # Fetch data
    client = StockHistoricalDataClient(API_KEY, API_SECRET)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame.Hour,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level='symbol')

    # Calculate all indicators
    df['rsi'] = calculate_rsi(df['close'])
    df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
    df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
    df['volume_ratio'] = calculate_volume_ratio(df['volume'])
    df['sma20'] = calculate_sma(df['close'], 20)
    df = df.dropna()

    # Test all strategies
    results = []
    for name, strategy_class in STRATEGIES.items():
        strategy = strategy_class()
        result = backtest_strategy(df, strategy)
        results.append(result)

    # Sort by return
    results.sort(key=lambda x: x['return_pct'], reverse=True)
    best = results[0]

    return {
        'symbol': symbol,
        'best_strategy': best['name'],
        'best_return': best['return_pct'],
        'best_win_rate': best['win_rate'],
        'best_trades': best['trades'],
        'all_results': results
    }


# =============================================================================
# SIMPLE TRADER
# =============================================================================

@dataclass
class Position:
    """Current position in a stock."""
    symbol: str
    shares: int
    entry_price: float
    entry_time: datetime
    strategy_name: str


@dataclass
class StockConfig:
    """Configuration for trading a single stock."""
    symbol: str
    strategy: Strategy
    last_calibration: datetime = None


class SimpleTrader:
    """
    Simple trading bot that:
    1. Calibrates best strategy per stock
    2. Monitors indicators
    3. Buys/sells when thresholds are hit
    """

    def __init__(
        self,
        symbols: List[str],
        paper: bool = True,
        calibration_days: int = 90,
        recalibrate_hours: int = 24,
        position_size_pct: float = 0.10,  # 10% of portfolio per position
        max_positions: int = 5
    ):
        self.symbols = symbols
        self.paper = paper
        self.calibration_days = calibration_days
        self.recalibrate_hours = recalibrate_hours
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions

        # Alpaca clients
        self.trading_client = TradingClient(API_KEY, API_SECRET, paper=paper)
        self.data_client = StockHistoricalDataClient(API_KEY, API_SECRET)

        # Stock configs (strategy per stock)
        self.stock_configs: Dict[str, StockConfig] = {}

        # Current positions
        self.positions: Dict[str, Position] = {}

        # State
        self.running = False
        self._stop_event = threading.Event()

        # Stats
        self.stats = {
            'trades_executed': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0
        }

        # Data directory for saving state
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

        print(f"SimpleTrader initialized for {len(symbols)} symbols ({'PAPER' if paper else 'LIVE'})")

    def calibrate_all(self):
        """Calibrate best strategy for all stocks."""
        print("\n" + "="*60)
        print("CALIBRATING STRATEGIES")
        print("="*60 + "\n")

        for symbol in self.symbols:
            try:
                result = select_best_strategy(symbol, self.calibration_days)

                strategy_class = STRATEGIES.get(result['best_strategy'])
                if strategy_class:
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.now()
                    )

                    print(f"{symbol}: {result['best_strategy']} "
                          f"(+{result['best_return']:.1f}%, {result['best_win_rate']:.0f}% win rate)")

            except Exception as e:
                print(f"{symbol}: Error - {e}")

        print("\n" + "="*60)
        self._save_state()

    def calibrate_if_needed(self, symbol: str):
        """Recalibrate a stock if enough time has passed."""
        config = self.stock_configs.get(symbol)

        if config is None or config.last_calibration is None:
            needs_calibration = True
        else:
            hours_since = (datetime.now() - config.last_calibration).total_seconds() / 3600
            needs_calibration = hours_since >= self.recalibrate_hours

        if needs_calibration:
            try:
                result = select_best_strategy(symbol, self.calibration_days)
                strategy_class = STRATEGIES.get(result['best_strategy'])
                if strategy_class:
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.now()
                    )
                    print(f"[RECALIBRATED] {symbol}: Now using {result['best_strategy']}")
            except Exception as e:
                print(f"[ERROR] Calibration failed for {symbol}: {e}")

    def get_account(self) -> dict:
        """Get account info."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power)
            }
        except Exception as e:
            print(f"Error getting account: {e}")
            return {'equity': 0, 'cash': 0, 'buying_power': 0}

    def get_current_positions(self) -> Dict[str, dict]:
        """Get current positions from Alpaca."""
        try:
            positions = self.trading_client.get_all_positions()
            return {
                pos.symbol: {
                    'shares': int(float(pos.qty)),
                    'avg_cost': float(pos.avg_entry_price),
                    'current_price': float(pos.current_price),
                    'unrealized_pnl': float(pos.unrealized_pl)
                }
                for pos in positions
            }
        except Exception as e:
            print(f"Error getting positions: {e}")
            return {}

    def fetch_latest_data(self, symbol: str, bars: int = 50) -> Optional[pd.DataFrame]:
        """Fetch latest bar data for a symbol."""
        try:
            # Use a date range to get recent data (works even when market is closed)
            end = datetime.now()
            start = end - timedelta(days=7)  # Get last 7 days of hourly data

            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Hour,
                start=start,
                end=end
            )

            result = self.data_client.get_stock_bars(request)
            df = result.df

            if df.empty:
                print(f"No data returned for {symbol}")
                return None

            if isinstance(df.index, pd.MultiIndex):
                df = df.xs(symbol, level='symbol')

            if df.empty or 'close' not in df.columns:
                print(f"No valid data for {symbol}")
                return None

            # Calculate indicators
            df['rsi'] = calculate_rsi(df['close'])
            df['macd'], df['macd_signal'], df['macd_diff'] = calculate_macd(df['close'])
            df['bb_upper'], df['bb_lower'], df['bb_position'] = calculate_bollinger(df['close'])
            df['volume_ratio'] = calculate_volume_ratio(df['volume'])
            df['sma20'] = calculate_sma(df['close'], 20)

            return df.dropna()

        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def execute_buy(self, symbol: str, strategy_name: str) -> bool:
        """Execute a buy order."""
        try:
            account = self.get_account()
            equity = account['equity']

            # Check if we have room for more positions
            current_positions = self.get_current_positions()
            if len(current_positions) >= self.max_positions:
                print(f"[SKIP] Max positions ({self.max_positions}) reached")
                return False

            # Already have position?
            if symbol in current_positions:
                print(f"[SKIP] Already have position in {symbol}")
                return False

            # Calculate position size
            position_value = equity * self.position_size_pct

            # Get current price
            quote_request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(quote_request)

            if symbol not in quotes:
                return False

            price = quotes[symbol].ask_price or quotes[symbol].bid_price
            shares = int(position_value / price)

            if shares <= 0:
                return False

            # Execute order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Track position
            self.positions[symbol] = Position(
                symbol=symbol,
                shares=shares,
                entry_price=price,
                entry_time=datetime.now(),
                strategy_name=strategy_name
            )

            print(f"[BUY] {shares} {symbol} @ ${price:.2f} using {strategy_name}")
            self.stats['trades_executed'] += 1
            self._save_state()

            # Log the trade
            try:
                trade_logger = get_trade_logger()
                config = self.stock_configs.get(symbol)
                strategy = config.strategy if config else None

                reason = TradeReason(
                    primary_signal=strategy_name,
                    signal_value=0,
                    threshold=0,
                    direction='buy_signal',
                    explanation=f"BUY: {strategy.buy_rule if strategy else 'Manual'}"
                )

                trade_logger.log_trade(
                    symbol=symbol,
                    action='BUY',
                    quantity=shares,
                    price=price,
                    strategy_name=strategy_name,
                    strategy_params={'buy_rule': strategy.buy_rule if strategy else '', 'sell_rule': strategy.sell_rule if strategy else ''},
                    reason=reason,
                    mode='paper' if self.paper else 'live',
                    portfolio_value_before=equity
                )
            except Exception as log_err:
                print(f"[WARN] Trade logging failed: {log_err}")

            return True

        except Exception as e:
            print(f"[ERROR] Buy failed for {symbol}: {e}")
            return False

    def execute_sell(self, symbol: str) -> bool:
        """Execute a sell order."""
        try:
            current_positions = self.get_current_positions()

            if symbol not in current_positions:
                print(f"[SKIP] No position in {symbol} to sell")
                return False

            pos = current_positions[symbol]
            shares = pos['shares']

            # Execute order
            order = MarketOrderRequest(
                symbol=symbol,
                qty=shares,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )

            result = self.trading_client.submit_order(order)

            # Calculate P&L
            pnl = pos['unrealized_pnl']
            self.stats['total_pnl'] += pnl

            if pnl > 0:
                self.stats['winning_trades'] += 1
                result_str = f"WIN +${pnl:.2f}"
            else:
                self.stats['losing_trades'] += 1
                result_str = f"LOSS ${pnl:.2f}"

            # Remove from tracking
            if symbol in self.positions:
                del self.positions[symbol]

            print(f"[SELL] {shares} {symbol} @ ${pos['current_price']:.2f} - {result_str}")
            self.stats['trades_executed'] += 1
            self._save_state()

            # Log the trade
            try:
                trade_logger = get_trade_logger()
                config = self.stock_configs.get(symbol)
                strategy = config.strategy if config else None
                strategy_name = strategy.name if strategy else 'Unknown'

                reason = TradeReason(
                    primary_signal=strategy_name,
                    signal_value=0,
                    threshold=0,
                    direction='sell_signal',
                    explanation=f"SELL: {strategy.sell_rule if strategy else 'Manual'} - {result_str}"
                )

                account = self.get_account()
                trade_logger.log_trade(
                    symbol=symbol,
                    action='SELL',
                    quantity=shares,
                    price=pos['current_price'],
                    strategy_name=strategy_name,
                    strategy_params={'buy_rule': strategy.buy_rule if strategy else '', 'sell_rule': strategy.sell_rule if strategy else ''},
                    reason=reason,
                    mode='paper' if self.paper else 'live',
                    portfolio_value_before=account['equity'],
                    realized_pnl=pnl
                )
            except Exception as log_err:
                print(f"[WARN] Trade logging failed: {log_err}")

            return True

        except Exception as e:
            print(f"[ERROR] Sell failed for {symbol}: {e}")
            return False

    def check_signals(self):
        """Check all stocks for buy/sell signals."""
        current_positions = self.get_current_positions()

        for symbol in self.symbols:
            # Make sure we have a strategy
            self.calibrate_if_needed(symbol)

            config = self.stock_configs.get(symbol)
            if not config:
                continue

            # Get latest data
            df = self.fetch_latest_data(symbol)
            if df is None or len(df) < 2:
                continue

            strategy = config.strategy
            i = len(df) - 1  # Latest bar

            # Check for signals
            has_position = symbol in current_positions

            if not has_position:
                # Check for buy signal
                if strategy.check_buy(df, i):
                    print(f"\n[SIGNAL] {symbol} - BUY triggered by {strategy.name}")
                    self.execute_buy(symbol, strategy.name)
            else:
                # Check for sell signal
                if strategy.check_sell(df, i):
                    print(f"\n[SIGNAL] {symbol} - SELL triggered by {strategy.name}")
                    self.execute_sell(symbol)

    def _is_market_hours(self) -> bool:
        """Check if market is open (Eastern Time)."""
        et = ZoneInfo('America/New_York')
        now = datetime.now(et)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours (9:30 AM - 4:00 PM ET)
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= now.time() <= market_close

    def _save_state(self):
        """Save current state to file."""
        state = {
            'positions': {
                s: {
                    'shares': p.shares,
                    'entry_price': p.entry_price,
                    'entry_time': p.entry_time.isoformat(),
                    'strategy_name': p.strategy_name
                }
                for s, p in self.positions.items()
            },
            'stock_configs': {
                s: {
                    'strategy': c.strategy.name,
                    'last_calibration': c.last_calibration.isoformat() if c.last_calibration else None
                }
                for s, c in self.stock_configs.items()
            },
            'stats': self.stats
        }

        with open(self.data_dir / "simple_trader_state.json", 'w') as f:
            json.dump(state, f, indent=2)

    def _load_state(self):
        """Load state from file."""
        state_file = self.data_dir / "simple_trader_state.json"
        if not state_file.exists():
            return

        try:
            with open(state_file, 'r') as f:
                state = json.load(f)

            # Restore stock configs
            for symbol, config in state.get('stock_configs', {}).items():
                strategy_class = STRATEGIES.get(config['strategy'])
                if strategy_class:
                    self.stock_configs[symbol] = StockConfig(
                        symbol=symbol,
                        strategy=strategy_class(),
                        last_calibration=datetime.fromisoformat(config['last_calibration']) if config['last_calibration'] else None
                    )

            # Restore stats
            self.stats = state.get('stats', self.stats)

            print(f"Loaded state: {len(self.stock_configs)} stock configs")

        except Exception as e:
            print(f"Error loading state: {e}")

    def run(self, check_interval_seconds: int = 60):
        """Run the trading bot."""
        print("\n" + "="*60)
        print("SIMPLE TRADER - STARTING")
        print("="*60)

        # Load previous state
        self._load_state()

        # Initial calibration for any uncalibrated stocks
        uncalibrated = [s for s in self.symbols if s not in self.stock_configs]
        if uncalibrated:
            print(f"\nCalibrating {len(uncalibrated)} stocks...")
            for symbol in uncalibrated:
                self.calibrate_if_needed(symbol)

        # Print current strategies
        print("\nCurrent strategies:")
        for symbol in self.symbols:
            config = self.stock_configs.get(symbol)
            if config:
                print(f"  {symbol}: {config.strategy.name}")

        self.running = True
        self._stop_event.clear()

        print(f"\nMonitoring {len(self.symbols)} stocks...")
        print(f"Checking every {check_interval_seconds} seconds during market hours")
        print("Press Ctrl+C to stop\n")

        try:
            while not self._stop_event.is_set():
                if self._is_market_hours():
                    self.check_signals()
                else:
                    # Print status once per minute when market is closed
                    now = datetime.now()
                    if now.second < check_interval_seconds:
                        print(f"[{now.strftime('%H:%M')}] Market closed - waiting...")

                time_module.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\nShutting down...")

        self.running = False
        self._save_state()
        print("Simple Trader stopped.")

    def stop(self):
        """Stop the trading bot."""
        self._stop_event.set()

    def get_status(self) -> dict:
        """Get current status."""
        account = self.get_account()
        positions = self.get_current_positions()

        return {
            'running': self.running,
            'paper': self.paper,
            'symbols': self.symbols,
            'strategies': {
                s: c.strategy.name
                for s, c in self.stock_configs.items()
            },
            'positions': positions,
            'account': account,
            'stats': self.stats,
            'market_open': self._is_market_hours()
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    # Get symbols from Alpaca portfolio
    paper = True  # Start with paper trading
    symbols = get_portfolio_symbols(paper=paper)

    if not symbols:
        print("\n[ERROR] No positions found in your Alpaca portfolio!")
        print("        Please buy some stocks first, then run the bot.")
        print("        The bot monitors and trades stocks you already own.")
        return

    print(f"\n[INFO] Found {len(symbols)} stocks in portfolio: {', '.join(symbols)}")

    # Create trader
    trader = SimpleTrader(
        symbols=symbols,
        paper=paper,
        calibration_days=90,
        recalibrate_hours=24,
        position_size_pct=0.10,
        max_positions=5
    )

    # Calibrate all stocks first
    trader.calibrate_all()

    # Run
    trader.run(check_interval_seconds=60)


if __name__ == '__main__':
    main()
