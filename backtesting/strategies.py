"""Pre-built trading strategies for backtesting."""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import talib as ta


def momentum_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Momentum-based trading strategy."""

    signals = []

    if len(data) < params.get('lookback', 20):
        return signals

    # Get parameters
    lookback = params.get('lookback', 20)
    threshold = params.get('threshold', 0.02)

    # Calculate momentum
    current_price = data['close'].iloc[-1]
    past_price = data['close'].iloc[-lookback]
    momentum = (current_price - past_price) / past_price

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Calculate additional indicators for context
    sma_20 = data['close'].rolling(window=20).mean().iloc[-1] if len(data) >= 20 else current_price
    volatility = data['close'].pct_change().tail(20).std() if len(data) >= 20 else 0

    # Generate signals
    if momentum > threshold and symbol not in engine.open_positions:
        # Buy signal
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'momentum',
                'signal_value': momentum,
                'threshold': threshold,
                'direction': 'above',
                'supporting_indicators': {
                    'lookback_price': past_price,
                    'sma_20': sma_20,
                    'volatility': volatility
                },
                'explanation': f"BUY: {lookback}-day momentum ({momentum:.2%}) exceeded threshold ({threshold:.2%}). "
                              f"Price rose from ${past_price:.2f} to ${current_price:.2f}."
            }
        })

    elif momentum < -threshold and symbol in engine.open_positions:
        # Sell signal
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'momentum',
                'signal_value': momentum,
                'threshold': -threshold,
                'direction': 'below',
                'supporting_indicators': {
                    'lookback_price': past_price,
                    'sma_20': sma_20,
                    'volatility': volatility
                },
                'explanation': f"SELL: {lookback}-day momentum ({momentum:.2%}) fell below threshold ({-threshold:.2%}). "
                              f"Price dropped from ${past_price:.2f} to ${current_price:.2f}."
            }
        })

    return signals


def mean_reversion_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Mean reversion strategy using Bollinger Bands."""

    signals = []

    if len(data) < params.get('period', 20):
        return signals

    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2)

    # Calculate Bollinger Bands
    sma = data['close'].rolling(window=period).mean()
    std = data['close'].rolling(window=period).std()
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)

    current_price = data['close'].iloc[-1]
    current_upper = upper_band.iloc[-1]
    current_lower = lower_band.iloc[-1]
    current_sma = sma.iloc[-1]
    current_std = std.iloc[-1]

    # Calculate z-score (distance from mean in std devs)
    z_score = (current_price - current_sma) / current_std if current_std > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when price touches lower band
    if current_price <= current_lower and symbol not in engine.open_positions:
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'bollinger_lower_band',
                'signal_value': current_price,
                'threshold': current_lower,
                'direction': 'below',
                'supporting_indicators': {
                    'sma': current_sma,
                    'upper_band': current_upper,
                    'lower_band': current_lower,
                    'z_score': z_score,
                    'std_dev_multiplier': std_dev
                },
                'explanation': f"BUY: Price ${current_price:.2f} touched lower Bollinger Band ${current_lower:.2f}. "
                              f"Z-score: {z_score:.2f} (oversold). Expecting mean reversion to SMA ${current_sma:.2f}."
            }
        })

    # Sell when price reaches SMA or upper band
    elif symbol in engine.open_positions:
        if current_price >= current_sma:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'bollinger_mean_reversion',
                    'signal_value': current_price,
                    'threshold': current_sma,
                    'direction': 'above',
                    'supporting_indicators': {
                        'sma': current_sma,
                        'upper_band': current_upper,
                        'lower_band': current_lower,
                        'z_score': z_score
                    },
                    'explanation': f"SELL: Price ${current_price:.2f} reverted to SMA ${current_sma:.2f}. "
                                  f"Mean reversion target reached. Z-score: {z_score:.2f}."
                }
            })

    return signals


def trend_following_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Trend following using moving average crossovers."""

    signals = []

    short_period = params.get('short_period', 20)
    long_period = params.get('long_period', 50)

    if len(data) < long_period:
        return signals

    # Calculate moving averages
    short_ma = data['close'].rolling(window=short_period).mean()
    long_ma = data['close'].rolling(window=long_period).mean()

    # Get current and previous values
    current_short = short_ma.iloc[-1]
    current_long = long_ma.iloc[-1]
    prev_short = short_ma.iloc[-2]
    prev_long = long_ma.iloc[-2]

    current_price = data['close'].iloc[-1]
    ma_diff = current_short - current_long
    ma_diff_pct = (ma_diff / current_long) * 100 if current_long > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Bullish crossover
    if prev_short <= prev_long and current_short > current_long:
        if symbol not in engine.open_positions:
            position_size = engine.capital * params.get('position_size', 0.1)
            quantity = position_size / current_price

            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'ma_crossover_bullish',
                    'signal_value': ma_diff,
                    'threshold': 0,
                    'direction': 'above',
                    'supporting_indicators': {
                        f'sma_{short_period}': current_short,
                        f'sma_{long_period}': current_long,
                        'prev_short_ma': prev_short,
                        'prev_long_ma': prev_long,
                        'ma_spread_pct': ma_diff_pct
                    },
                    'explanation': f"BUY: Golden Cross detected. {short_period}-day SMA (${current_short:.2f}) "
                                  f"crossed above {long_period}-day SMA (${current_long:.2f}). Bullish trend confirmed."
                }
            })

    # Bearish crossover
    elif prev_short >= prev_long and current_short < current_long:
        if symbol in engine.open_positions:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'ma_crossover_bearish',
                    'signal_value': ma_diff,
                    'threshold': 0,
                    'direction': 'below',
                    'supporting_indicators': {
                        f'sma_{short_period}': current_short,
                        f'sma_{long_period}': current_long,
                        'prev_short_ma': prev_short,
                        'prev_long_ma': prev_long,
                        'ma_spread_pct': ma_diff_pct
                    },
                    'explanation': f"SELL: Death Cross detected. {short_period}-day SMA (${current_short:.2f}) "
                                  f"crossed below {long_period}-day SMA (${current_long:.2f}). Bearish trend confirmed."
                }
            })

    return signals


def breakout_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Breakout strategy based on price channels."""

    signals = []

    lookback = params.get('lookback', 20)

    if len(data) < lookback:
        return signals

    # Calculate high/low channels
    high_channel = data['high'].rolling(window=lookback).max()
    low_channel = data['low'].rolling(window=lookback).min()

    current_price = data['close'].iloc[-1]
    current_high_channel = high_channel.iloc[-2]  # Previous period to avoid looking ahead
    current_low_channel = low_channel.iloc[-2]
    channel_width = current_high_channel - current_low_channel
    channel_width_pct = (channel_width / current_low_channel) * 100 if current_low_channel > 0 else 0

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Breakout above resistance
    if current_price > current_high_channel and symbol not in engine.open_positions:
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price
        breakout_strength = ((current_price - current_high_channel) / current_high_channel) * 100

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'channel_breakout_up',
                'signal_value': current_price,
                'threshold': current_high_channel,
                'direction': 'above',
                'supporting_indicators': {
                    'resistance_level': current_high_channel,
                    'support_level': current_low_channel,
                    'channel_width': channel_width,
                    'channel_width_pct': channel_width_pct,
                    'breakout_strength_pct': breakout_strength,
                    'lookback_period': lookback
                },
                'explanation': f"BUY: Price ${current_price:.2f} broke above {lookback}-day resistance ${current_high_channel:.2f}. "
                              f"Breakout strength: {breakout_strength:.2f}%. Channel width: {channel_width_pct:.1f}%."
            }
        })

    # Break below support
    elif current_price < current_low_channel and symbol in engine.open_positions:
        breakdown_strength = ((current_low_channel - current_price) / current_low_channel) * 100

        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'channel_breakdown',
                'signal_value': current_price,
                'threshold': current_low_channel,
                'direction': 'below',
                'supporting_indicators': {
                    'resistance_level': current_high_channel,
                    'support_level': current_low_channel,
                    'channel_width': channel_width,
                    'breakdown_strength_pct': breakdown_strength
                },
                'explanation': f"SELL: Price ${current_price:.2f} broke below {lookback}-day support ${current_low_channel:.2f}. "
                              f"Breakdown strength: {breakdown_strength:.2f}%. Exiting to prevent further losses."
            }
        })

    return signals


def rsi_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """RSI-based strategy for overbought/oversold conditions."""

    signals = []

    period = params.get('period', 14)
    oversold = params.get('oversold', 30)
    overbought = params.get('overbought', 70)

    if len(data) < period + 1:
        return signals

    # Calculate RSI
    rsi = ta.RSI(data['close'].values, timeperiod=period)

    current_rsi = rsi[-1]
    prev_rsi = rsi[-2] if len(rsi) > 1 else current_rsi
    current_price = data['close'].iloc[-1]

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when oversold
    if current_rsi < oversold and symbol not in engine.open_positions:
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'rsi_oversold',
                'signal_value': current_rsi,
                'threshold': oversold,
                'direction': 'below',
                'supporting_indicators': {
                    'rsi_period': period,
                    'prev_rsi': prev_rsi,
                    'rsi_change': current_rsi - prev_rsi,
                    'oversold_threshold': oversold,
                    'overbought_threshold': overbought
                },
                'explanation': f"BUY: RSI({period}) = {current_rsi:.1f} is below oversold threshold ({oversold}). "
                              f"Asset is oversold, expecting bounce. Previous RSI: {prev_rsi:.1f}."
            }
        })

    # Sell when overbought
    elif current_rsi > overbought and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'rsi_overbought',
                'signal_value': current_rsi,
                'threshold': overbought,
                'direction': 'above',
                'supporting_indicators': {
                    'rsi_period': period,
                    'prev_rsi': prev_rsi,
                    'rsi_change': current_rsi - prev_rsi
                },
                'explanation': f"SELL: RSI({period}) = {current_rsi:.1f} is above overbought threshold ({overbought}). "
                              f"Asset is overbought, expecting pullback. Taking profits."
            }
        })

    return signals


def macd_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """MACD crossover strategy."""

    signals = []

    fast = params.get('fast', 12)
    slow = params.get('slow', 26)
    signal_period = params.get('signal', 9)

    if len(data) < slow + signal_period:
        return signals

    # Calculate MACD
    macd, macd_signal, macd_hist = ta.MACD(
        data['close'].values,
        fastperiod=fast,
        slowperiod=slow,
        signalperiod=signal_period
    )

    current_macd = macd[-1]
    current_signal = macd_signal[-1]
    current_hist = macd_hist[-1]
    prev_macd = macd[-2]
    prev_signal = macd_signal[-2]
    prev_hist = macd_hist[-2]

    current_price = data['close'].iloc[-1]
    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Bullish crossover
    if prev_macd <= prev_signal and current_macd > current_signal:
        if symbol not in engine.open_positions:
            position_size = engine.capital * params.get('position_size', 0.1)
            quantity = position_size / current_price

            signals.append({
                'action': 'buy',
                'symbol': symbol,
                'price': current_price,
                'quantity': quantity,
                'reason': {
                    'primary_signal': 'macd_bullish_crossover',
                    'signal_value': current_macd,
                    'threshold': current_signal,
                    'direction': 'above',
                    'supporting_indicators': {
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_hist,
                        'prev_macd': prev_macd,
                        'prev_signal': prev_signal,
                        'histogram_change': current_hist - prev_hist,
                        'fast_period': fast,
                        'slow_period': slow,
                        'signal_period': signal_period
                    },
                    'explanation': f"BUY: MACD bullish crossover. MACD ({current_macd:.4f}) crossed above "
                                  f"Signal line ({current_signal:.4f}). Histogram: {current_hist:.4f}. "
                                  f"Momentum shifting bullish."
                }
            })

    # Bearish crossover
    elif prev_macd >= prev_signal and current_macd < current_signal:
        if symbol in engine.open_positions:
            signals.append({
                'action': 'sell',
                'symbol': symbol,
                'price': current_price,
                'reason': {
                    'primary_signal': 'macd_bearish_crossover',
                    'signal_value': current_macd,
                    'threshold': current_signal,
                    'direction': 'below',
                    'supporting_indicators': {
                        'macd_line': current_macd,
                        'signal_line': current_signal,
                        'histogram': current_hist,
                        'prev_macd': prev_macd,
                        'prev_signal': prev_signal,
                        'histogram_change': current_hist - prev_hist
                    },
                    'explanation': f"SELL: MACD bearish crossover. MACD ({current_macd:.4f}) crossed below "
                                  f"Signal line ({current_signal:.4f}). Histogram: {current_hist:.4f}. "
                                  f"Momentum shifting bearish."
                }
            })

    return signals


def pairs_trading_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Pairs trading / statistical arbitrage strategy."""

    signals = []

    # This is a simplified version - full implementation would need two correlated assets

    lookback = params.get('lookback', 20)
    entry_z = params.get('entry_z', 2.0)
    exit_z = params.get('exit_z', 0.5)

    if len(data) < lookback:
        return signals

    # Calculate z-score
    sma = data['close'].rolling(window=lookback).mean()
    std = data['close'].rolling(window=lookback).std()
    z_score = (data['close'] - sma) / std

    current_z = z_score.iloc[-1]
    prev_z = z_score.iloc[-2] if len(z_score) > 1 else current_z
    current_price = data['close'].iloc[-1]
    current_sma = sma.iloc[-1]
    current_std = std.iloc[-1]

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy when significantly below mean
    if current_z < -entry_z and symbol not in engine.open_positions:
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'z_score_oversold',
                'signal_value': current_z,
                'threshold': -entry_z,
                'direction': 'below',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma,
                    'std_dev': current_std,
                    'entry_threshold': entry_z,
                    'exit_threshold': exit_z,
                    'lookback': lookback
                },
                'explanation': f"BUY: Z-score ({current_z:.2f}) below -{entry_z} threshold. "
                              f"Price ${current_price:.2f} is {abs(current_z):.1f} std devs below {lookback}-day mean ${current_sma:.2f}. "
                              f"Expecting mean reversion."
            }
        })

    # Sell when reverting to mean
    elif abs(current_z) < exit_z and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'z_score_mean_reversion',
                'signal_value': current_z,
                'threshold': exit_z,
                'direction': 'near_zero',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma
                },
                'explanation': f"SELL: Z-score ({current_z:.2f}) reverted within ±{exit_z} of mean. "
                              f"Mean reversion target achieved. Taking profits."
            }
        })

    # Also sell if goes too far in opposite direction
    elif current_z > entry_z and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'z_score_overbought',
                'signal_value': current_z,
                'threshold': entry_z,
                'direction': 'above',
                'supporting_indicators': {
                    'z_score': current_z,
                    'prev_z_score': prev_z,
                    'sma': current_sma
                },
                'explanation': f"SELL: Z-score ({current_z:.2f}) exceeded +{entry_z} threshold. "
                              f"Price overextended above mean. Exiting position."
            }
        })

    return signals


def ml_hybrid_strategy(data: pd.DataFrame, engine, params: Dict[str, Any]) -> List[Dict]:
    """Hybrid strategy combining ML predictions with technical indicators."""

    signals = []

    # This would integrate with the ML models
    # Simplified version using technical indicators

    if len(data) < 50:
        return signals

    # Multiple confirmation signals
    confirmations = 0
    confirmation_details = []
    target_confirmations = params.get('min_confirmations', 3)

    current_price = data['close'].iloc[-1]
    current_volume = data['volume'].iloc[-1]

    # RSI confirmation
    rsi = ta.RSI(data['close'].values, timeperiod=14)
    current_rsi = rsi[-1]
    if current_rsi < 30:
        confirmations += 1
        confirmation_details.append(f"RSI oversold ({current_rsi:.1f})")
    elif current_rsi > 70:
        confirmations -= 1
        confirmation_details.append(f"RSI overbought ({current_rsi:.1f})")

    # MACD confirmation
    macd, macd_signal, hist = ta.MACD(data['close'].values)
    current_macd = macd[-1]
    current_macd_signal = macd_signal[-1]
    if macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
        confirmations += 1
        confirmation_details.append("MACD bullish crossover")
    elif macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
        confirmations -= 1
        confirmation_details.append("MACD bearish crossover")

    # Moving average confirmation
    sma_20 = data['close'].rolling(window=20).mean().iloc[-1]
    sma_50 = data['close'].rolling(window=50).mean().iloc[-1]
    if sma_20 > sma_50:
        confirmations += 1
        confirmation_details.append(f"SMA20 > SMA50 (bullish trend)")
    else:
        confirmations -= 1
        confirmation_details.append(f"SMA20 < SMA50 (bearish trend)")

    # Volume confirmation
    vol_avg = data['volume'].rolling(window=20).mean().iloc[-1]
    volume_ratio = current_volume / vol_avg if vol_avg > 0 else 1
    if current_volume > vol_avg * 1.5:
        confirmations += 1
        confirmation_details.append(f"High volume ({volume_ratio:.1f}x avg)")

    symbol = data.get('symbol', ['DEFAULT']).iloc[-1] if 'symbol' in data.columns else 'DEFAULT'

    # Buy signal
    if confirmations >= target_confirmations and symbol not in engine.open_positions:
        position_size = engine.capital * params.get('position_size', 0.1)
        quantity = position_size / current_price

        signals.append({
            'action': 'buy',
            'symbol': symbol,
            'price': current_price,
            'quantity': quantity,
            'reason': {
                'primary_signal': 'multi_indicator_bullish',
                'signal_value': confirmations,
                'threshold': target_confirmations,
                'direction': 'above',
                'supporting_indicators': {
                    'confirmations': confirmations,
                    'target_confirmations': target_confirmations,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50,
                    'volume_ratio': volume_ratio
                },
                'confirmations': confirmation_details,
                'explanation': f"BUY: {confirmations} bullish confirmations (threshold: {target_confirmations}). "
                              f"Signals: {', '.join(confirmation_details)}."
            }
        })

    # Sell signal
    elif confirmations <= -target_confirmations and symbol in engine.open_positions:
        signals.append({
            'action': 'sell',
            'symbol': symbol,
            'price': current_price,
            'reason': {
                'primary_signal': 'multi_indicator_bearish',
                'signal_value': confirmations,
                'threshold': -target_confirmations,
                'direction': 'below',
                'supporting_indicators': {
                    'confirmations': confirmations,
                    'target_confirmations': target_confirmations,
                    'rsi': current_rsi,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'sma_20': sma_20,
                    'sma_50': sma_50
                },
                'confirmations': confirmation_details,
                'explanation': f"SELL: {abs(confirmations)} bearish confirmations (threshold: {target_confirmations}). "
                              f"Signals: {', '.join(confirmation_details)}."
            }
        })

    return signals


# Strategy registry
STRATEGY_REGISTRY = {
    'momentum': momentum_strategy,
    'mean_reversion': mean_reversion_strategy,
    'trend_following': trend_following_strategy,
    'breakout': breakout_strategy,
    'rsi': rsi_strategy,
    'macd': macd_strategy,
    'pairs_trading': pairs_trading_strategy,
    'ml_hybrid': ml_hybrid_strategy
}


# Default parameters for each strategy
# position_size: 0.9 = 90% of capital per trade (aggressive but allows multiple positions)
# For single-asset backtests, this captures most of the market movement
DEFAULT_PARAMS = {
    'momentum': {
        'lookback': 20,
        'threshold': 0.02,
        'position_size': 0.9
    },
    'mean_reversion': {
        'period': 20,
        'std_dev': 2,
        'position_size': 0.9
    },
    'trend_following': {
        'short_period': 20,
        'long_period': 50,
        'position_size': 0.9
    },
    'breakout': {
        'lookback': 20,
        'position_size': 0.9
    },
    'rsi': {
        'period': 14,
        'oversold': 30,
        'overbought': 70,
        'position_size': 0.9
    },
    'macd': {
        'fast': 12,
        'slow': 26,
        'signal': 9,
        'position_size': 0.9
    },
    'pairs_trading': {
        'lookback': 20,
        'entry_z': 2.0,
        'exit_z': 0.5,
        'position_size': 0.9
    },
    'ml_hybrid': {
        'min_confirmations': 3,
        'position_size': 0.9
    }
}
