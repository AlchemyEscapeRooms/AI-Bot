"""
The Alchemy Effect - Web API (ALTERNATIVE)
============================================

FastAPI backend that serves the trading dashboard.
Connects to the existing trading bot infrastructure.

Run with: python web_api.py
Then open: http://localhost:8000

Author: Claude AI

==============================================================================
API CONSOLIDATION NOTE
==============================================================================
There are TWO API servers in this project:

1. api_server.py - PRIMARY (Recommended)
   - Port: 8000
   - Features: Full BackgroundTradingService integration, stock management,
               backtest API, learning profiles, trade signals
   - More comprehensive API for all bot functionality

2. web_api.py (THIS FILE) - ALTERNATIVE (Dashboard-focused)
   - Port: 8000 (CONFLICT - cannot run simultaneously with api_server.py!)
   - Features: Simpler API focused on alchemy_dashboard.html
   - Endpoints: /api/status, /api/brain, /api/pnl, /api/trades, /api/bot/control

WARNING: Do NOT run both api_server.py and web_api.py at the same time!
         They both use port 8000.

RECOMMENDATION:
- For full functionality, use api_server.py instead
- This file may be deprecated in future versions
- Consider running: uvicorn api_server:app --port 8000
==============================================================================
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Alpaca for account sync
from alpaca.trading.client import TradingClient

# Import our existing bot components
from core.market_monitor import MarketMonitor, get_market_monitor
from core.trading_bot import TradingBot
from core.personality_profiles import PERSONALITY_PROFILES, get_profile
from portfolio.risk_manager import RiskManager
from utils.trade_logger import get_trade_logger
from utils.database import Database
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="The Alchemy Effect API",
    description="Trading bot dashboard API",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_bot: Optional[TradingBot] = None
websocket_clients: List[WebSocket] = []
_alpaca_client: Optional[TradingClient] = None
_cached_capital: Optional[float] = None
_capital_cache_time: Optional[datetime] = None

def get_alpaca_client() -> Optional[TradingClient]:
    """Get or create Alpaca trading client."""
    global _alpaca_client
    if _alpaca_client is None:
        api_key = os.environ.get('ALPACA_API_KEY') or config.get('alpaca.api_key')
        api_secret = os.environ.get('ALPACA_SECRET_KEY') or config.get('alpaca.api_secret')
        if api_key and api_secret:
            _alpaca_client = TradingClient(api_key, api_secret, paper=True)
    return _alpaca_client

def get_initial_capital() -> float:
    """Get initial capital - from Alpaca if configured as 'auto', otherwise from config."""
    global _cached_capital, _capital_cache_time

    # Check config setting
    config_capital = config.get('trading.initial_capital', 100000)

    # If not "auto", use the configured value
    if config_capital != "auto" and config_capital != "Auto":
        try:
            return float(config_capital)
        except (ValueError, TypeError):
            pass

    # Use cached value if recent (within 5 minutes)
    if _cached_capital is not None and _capital_cache_time is not None:
        if (datetime.now() - _capital_cache_time).total_seconds() < 300:
            return _cached_capital

    # Fetch from Alpaca
    try:
        client = get_alpaca_client()
        if client:
            account = client.get_account()
            _cached_capital = float(account.equity)
            _capital_cache_time = datetime.now()
            logger.info(f"Synced capital from Alpaca: ${_cached_capital:,.2f}")
            return _cached_capital
    except Exception as e:
        logger.error(f"Error fetching Alpaca account: {e}")

    # Fallback
    return _cached_capital if _cached_capital else 100000.0

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BotCommand(BaseModel):
    action: str  # 'start', 'stop', 'pause'

class BacktestRequest(BaseModel):
    symbol: str
    days: int = 365

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_bot() -> Optional[TradingBot]:
    """Get the global trading bot instance."""
    global trading_bot
    return trading_bot

def get_trade_stats(days: int = 1) -> Dict[str, Any]:
    """Get trade statistics for a given period."""
    try:
        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'best_trade': None,
                'worst_trade': None
            }

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        # Only filter if we have valid datetime data
        if len(df) > 0 and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            # Handle timezone-aware vs naive datetime comparison
            if df['timestamp'].dt.tz is not None:
                cutoff = pd.Timestamp(cutoff).tz_localize(df['timestamp'].dt.tz)
            df = df[df['timestamp'] >= cutoff]

        # Calculate stats
        sells = df[df['realized_pnl'].notna()]

        if sells.empty:
            return {
                'total_trades': len(df),
                'wins': 0,
                'losses': 0,
                'total_pnl': 0,
                'win_rate': 0,
                'best_trade': None,
                'worst_trade': None
            }

        wins = sells[sells['realized_pnl'] > 0]
        losses = sells[sells['realized_pnl'] < 0]

        best = sells.loc[sells['realized_pnl'].idxmax()] if not sells.empty else None
        worst = sells.loc[sells['realized_pnl'].idxmin()] if not sells.empty else None

        return {
            'total_trades': len(sells),
            'wins': len(wins),
            'losses': len(losses),
            'total_pnl': float(sells['realized_pnl'].sum()),
            'win_rate': (len(wins) / len(sells) * 100) if len(sells) > 0 else 0,
            'best_trade': {
                'symbol': best['symbol'],
                'pnl': float(best['realized_pnl'])
            } if best is not None else None,
            'worst_trade': {
                'symbol': worst['symbol'],
                'pnl': float(worst['realized_pnl'])
            } if worst is not None else None
        }
    except Exception as e:
        logger.error(f"Error getting trade stats: {e}")
        return {'total_trades': 0, 'total_pnl': 0, 'win_rate': 0}

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    dashboard_path = Path(__file__).parent / "alchemy_dashboard.html"

    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        return HTMLResponse(content="<h1>Dashboard not found. Run the setup first.</h1>")

@app.get("/api/status")
async def get_status():
    """Get overall bot status."""
    bot = get_bot()
    monitor = get_market_monitor()

    is_running = bot is not None and hasattr(bot, 'is_running') and bot.is_running

    return {
        'bot_active': is_running,
        'mode': config.get('trading.mode', 'paper'),
        'personality': config.get('trading.personality', 'ai_adaptive'),
        'timestamp': datetime.now().isoformat()
    }

@app.get("/api/brain")
async def get_brain_status():
    """Get AI brain status - global and per-stock weights."""
    try:
        monitor = get_market_monitor()

        # Get prediction stats
        stats = monitor.prediction_tracker.get_accuracy_stats(days=30)
        signal_perf = monitor.prediction_tracker.get_signal_performance(days=30)

        # Format global weights
        global_weights = []
        signal_names = {
            'momentum_20d': {'name': '20-Day Momentum', 'desc': 'Measures price change over 20 days'},
            'rsi': {'name': 'RSI', 'desc': 'Overbought/oversold indicator'},
            'macd_signal': {'name': 'MACD Signal', 'desc': 'Trend & momentum crossover'},
            'volume_ratio': {'name': 'Volume Ratio', 'desc': 'Current vs average volume'},
            'price_vs_sma20': {'name': 'Price vs SMA', 'desc': 'Price relative to 20-day moving average'},
            'bollinger_position': {'name': 'Bollinger Position', 'desc': 'Price within Bollinger Bands'}
        }

        for signal, weight in sorted(monitor.signal_weights.items(), key=lambda x: x[1], reverse=True):
            info = signal_names.get(signal, {'name': signal, 'desc': ''})

            # Get accuracy for this signal
            accuracy = 0
            uses = 0
            if not signal_perf.empty:
                sig_row = signal_perf[signal_perf['signal'] == signal]
                if not sig_row.empty:
                    accuracy = float(sig_row.iloc[0]['accuracy'])
                    uses = int(sig_row.iloc[0]['uses'])

            global_weights.append({
                'signal': signal,
                'name': info['name'],
                'description': info['desc'],
                'weight': float(weight),
                'accuracy': accuracy,
                'uses': uses,
                'status': 'strong' if weight > 1.1 else ('weak' if weight < 0.9 else 'normal')
            })

        # Get per-stock weights
        stock_weights_df = monitor.prediction_tracker.get_all_stock_weights()
        per_stock = {}

        if not stock_weights_df.empty:
            for symbol in stock_weights_df['symbol'].unique():
                symbol_data = stock_weights_df[stock_weights_df['symbol'] == symbol]
                per_stock[symbol] = []

                for _, row in symbol_data.iterrows():
                    info = signal_names.get(row['signal_name'], {'name': row['signal_name'], 'desc': ''})
                    per_stock[symbol].append({
                        'signal': row['signal_name'],
                        'name': info['name'],
                        'weight': float(row['weight']),
                        'accuracy': float(row['accuracy']) if row['accuracy'] else 0,
                        'uses': int(row['sample_size']) if row['sample_size'] else 0
                    })

        return {
            'overall_accuracy': stats.get('accuracy', 0),
            'total_predictions': stats.get('total_predictions', 0),
            'correct': stats.get('correct', 0),
            'wrong': stats.get('wrong', 0),
            'global_weights': global_weights,
            'per_stock_weights': per_stock,
            'learning_status': 'active'
        }

    except Exception as e:
        logger.error(f"Error getting brain status: {e}")
        return {
            'overall_accuracy': 0,
            'total_predictions': 0,
            'global_weights': [],
            'per_stock_weights': {},
            'error': str(e)
        }

@app.get("/api/pnl")
async def get_pnl():
    """Get P&L data for dashboard."""
    try:
        trade_logger = get_trade_logger()

        # Get all-time stats
        summary = trade_logger.get_trade_summary()

        # Get today's stats
        today_stats = get_trade_stats(days=1)

        # Get historical P&L for sparkline (last 30 days)
        df = trade_logger.get_trades()
        sparkline = []

        if not df.empty:
            import pandas as pd
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df = df.dropna(subset=['timestamp'])
            df = df[df['realized_pnl'].notna()]

            # Group by date and sum P&L - only if we have valid datetime data
            if not df.empty and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['date'] = df['timestamp'].dt.date
                daily = df.groupby('date')['realized_pnl'].sum().tail(30)

                # Calculate cumulative
                cumulative = 0
                for date, pnl in daily.items():
                    cumulative += pnl
                    sparkline.append({
                        'date': str(date),
                        'daily_pnl': float(pnl),
                        'cumulative': float(cumulative)
                    })

        initial_capital = get_initial_capital()
        total_pnl = summary.get('total_realized_pnl', 0)
        current_value = initial_capital + total_pnl

        return {
            'today_pnl': today_stats.get('total_pnl', 0),
            'today_pnl_pct': (today_stats.get('total_pnl', 0) / initial_capital * 100) if initial_capital > 0 else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / initial_capital * 100) if initial_capital > 0 else 0,
            'initial_capital': initial_capital,
            'current_value': current_value,
            'sparkline': sparkline
        }

    except Exception as e:
        logger.error(f"Error getting P&L: {e}")
        return {
            'today_pnl': 0,
            'today_pnl_pct': 0,
            'total_pnl': 0,
            'initial_capital': 100000,
            'current_value': 100000,
            'sparkline': []
        }

@app.get("/api/yesterday")
async def get_yesterday_summary():
    """Get yesterday's trading summary."""
    try:
        stats = get_trade_stats(days=1)

        return {
            'pnl': stats.get('total_pnl', 0),
            'trades': stats.get('total_trades', 0),
            'wins': stats.get('wins', 0),
            'losses': stats.get('losses', 0),
            'win_rate': stats.get('win_rate', 0),
            'best_trade': stats.get('best_trade'),
            'worst_trade': stats.get('worst_trade')
        }

    except Exception as e:
        logger.error(f"Error getting yesterday summary: {e}")
        return {'pnl': 0, 'trades': 0, 'wins': 0, 'losses': 0}

@app.get("/api/trending")
async def get_trending():
    """Get stocks trending upward with bot confidence."""
    try:
        monitor = get_market_monitor()

        trending = []

        # Get recent predictions
        with monitor.prediction_tracker.db.get_connection() as conn:
            import pandas as pd
            df = pd.read_sql_query("""
                SELECT symbol, predicted_direction, confidence, timestamp
                FROM ai_predictions
                WHERE timestamp >= datetime('now', '-1 day')
                ORDER BY timestamp DESC
            """, conn)

        if not df.empty:
            # Get latest prediction per symbol where direction is 'up'
            up_predictions = df[df['predicted_direction'] == 'up']

            for symbol in up_predictions['symbol'].unique():
                symbol_preds = up_predictions[up_predictions['symbol'] == symbol]
                latest = symbol_preds.iloc[0]

                # Get the hybrid weight confidence for this symbol
                weights = monitor.get_weights_for_symbol(symbol)
                avg_weight = sum(weights.values()) / len(weights) if weights else 1.0

                trending.append({
                    'symbol': symbol,
                    'direction': 'up',
                    'confidence': float(latest['confidence']),
                    'brain_confidence': min(100, float(latest['confidence']) * avg_weight),
                    'change_pct': 0  # Would need real-time data
                })

        # Sort by confidence
        trending.sort(key=lambda x: x['confidence'], reverse=True)

        return {'trending': trending[:5]}

    except Exception as e:
        logger.error(f"Error getting trending: {e}")
        return {'trending': []}

@app.get("/api/predicted-profit")
async def get_predicted_profit():
    """Get predicted profit based on open signals."""
    try:
        monitor = get_market_monitor()

        # Get active high-confidence predictions
        active = []

        with monitor.prediction_tracker.db.get_connection() as conn:
            import pandas as pd
            df = pd.read_sql_query("""
                SELECT symbol, predicted_direction, confidence, predicted_change_pct
                FROM ai_predictions
                WHERE resolved = 0
                AND confidence >= 60
                ORDER BY confidence DESC
            """, conn)

        if df.empty:
            return {
                'predicted_low': 0,
                'predicted_high': 0,
                'open_signals': 0,
                'avg_confidence': 0
            }

        # Calculate predicted profit range
        avg_conf = float(df['confidence'].mean())
        num_signals = len(df)

        # Estimate based on confidence and typical trade size
        initial_capital = get_initial_capital()
        position_size = config.get('trading.max_position_size', 0.1)
        typical_position = initial_capital * position_size

        # Conservative and optimistic estimates
        avg_predicted_change = float(df['predicted_change_pct'].mean()) if 'predicted_change_pct' in df.columns else 2.0

        predicted_low = num_signals * typical_position * (avg_predicted_change * 0.5) / 100
        predicted_high = num_signals * typical_position * (avg_predicted_change * 1.5) / 100

        return {
            'predicted_low': round(predicted_low, 2),
            'predicted_high': round(predicted_high, 2),
            'open_signals': num_signals,
            'avg_confidence': round(avg_conf, 1)
        }

    except Exception as e:
        logger.error(f"Error getting predicted profit: {e}")
        return {'predicted_low': 0, 'predicted_high': 0, 'open_signals': 0}

@app.get("/api/signals")
async def get_trade_signals():
    """Get current trade signals - the AI live feed."""
    try:
        monitor = get_market_monitor()

        signals = []

        # Get recent predictions with high confidence
        with monitor.prediction_tracker.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT symbol, predicted_direction, confidence, predicted_change_pct,
                       timestamp, signals
                FROM ai_predictions
                WHERE timestamp >= datetime('now', '-4 hours')
                AND resolved = 0
                AND confidence >= 60
                ORDER BY confidence DESC, timestamp DESC
                LIMIT 20
            """, conn)

        if not df.empty:
            for _, row in df.iterrows():
                # Get stock-specific accuracy if available
                stock_weights = monitor.prediction_tracker.get_stock_weights(row['symbol'])

                signals.append({
                    'symbol': row['symbol'],
                    'action': 'BUY' if row['predicted_direction'] == 'up' else 'SELL',
                    'direction': row['predicted_direction'],
                    'confidence': float(row['confidence']),
                    'predicted_change': float(row['predicted_change_pct']) if row['predicted_change_pct'] else 0,
                    'timestamp': str(row['timestamp']),
                    'has_custom_weights': len(stock_weights) > 0
                })

        return {
            'signals': signals,
            'total': len(signals),
            'last_updated': datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting signals: {e}")
        return {'signals': [], 'total': 0, 'error': str(e)}

@app.get("/api/trades")
async def get_trades(limit: int = 50):
    """Get recent trades."""
    try:
        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {'trades': []}

        trades = []
        for _, row in df.head(limit).iterrows():
            trades.append({
                'id': row['trade_id'],
                'timestamp': str(row['timestamp']),
                'symbol': row['symbol'],
                'action': row['action'],
                'quantity': float(row['quantity']),
                'price': float(row['price']),
                'pnl': float(row['realized_pnl']) if row['realized_pnl'] else None,
                'strategy': row['strategy_name'],
                'reason': row['primary_signal']
            })

        return {'trades': trades}

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {'trades': []}

@app.get("/api/portfolio")
async def get_portfolio():
    """Get Alpaca portfolio positions - these are the stocks the bot monitors."""
    try:
        client = get_alpaca_client()
        if not client:
            return {'positions': [], 'error': 'Alpaca client not configured'}

        account = client.get_account()
        positions = client.get_all_positions()

        portfolio_positions = []
        for pos in positions:
            portfolio_positions.append({
                'symbol': pos.symbol,
                'qty': float(pos.qty),
                'avg_entry_price': float(pos.avg_entry_price),
                'current_price': float(pos.current_price),
                'market_value': float(pos.market_value),
                'unrealized_pl': float(pos.unrealized_pl),
                'unrealized_pl_pct': float(pos.unrealized_plpc) * 100,
                'side': pos.side
            })

        return {
            'equity': float(account.equity),
            'cash': float(account.cash),
            'buying_power': float(account.buying_power),
            'positions': portfolio_positions,
            'total_positions': len(portfolio_positions)
        }

    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return {'positions': [], 'error': str(e)}

@app.get("/api/watchlist")
async def get_watchlist():
    """Get the bot's watchlist - combines Alpaca positions with configured stocks."""
    try:
        watchlist = set()

        # 1. Get Alpaca positions
        client = get_alpaca_client()
        if client:
            try:
                positions = client.get_all_positions()
                for pos in positions:
                    watchlist.add(pos.symbol)
            except Exception as e:
                logger.warning(f"Could not get Alpaca positions: {e}")

        # 2. Add configured stocks as fallback/additions
        configured_stocks = config.get('data.universe.initial_stocks', [])
        if isinstance(configured_stocks, list):
            watchlist.update(configured_stocks)

        # Sort and return
        sorted_watchlist = sorted(list(watchlist))

        return {
            'symbols': sorted_watchlist,
            'total': len(sorted_watchlist),
            'from_positions': len([p for p in positions]) if client else 0,
            'from_config': len(configured_stocks) if configured_stocks else 0
        }

    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        return {'symbols': [], 'error': str(e)}

@app.post("/api/bot/control")
async def control_bot(command: BotCommand):
    """Control the trading bot (start/stop/pause)."""
    global trading_bot

    try:
        if command.action == 'start':
            if trading_bot is None:
                trading_bot = TradingBot(
                    initial_capital=get_initial_capital(),
                    personality='ai_adaptive',
                    mode='paper'
                )
            trading_bot.start()
            return {'success': True, 'status': 'running'}

        elif command.action == 'stop':
            if trading_bot:
                trading_bot.stop()
            return {'success': True, 'status': 'stopped'}

        elif command.action == 'pause':
            if trading_bot:
                trading_bot.pause()
            return {'success': True, 'status': 'paused'}

        else:
            return {'success': False, 'error': f'Unknown action: {command.action}'}

    except Exception as e:
        logger.error(f"Error controlling bot: {e}")
        return {'success': False, 'error': str(e)}

@app.get("/api/brain/details")
async def get_brain_details():
    """Get detailed brain information for the Brain tab."""
    try:
        monitor = get_market_monitor()

        # Get all the brain data
        stats = monitor.prediction_tracker.get_accuracy_stats(days=30)
        signal_perf = monitor.prediction_tracker.get_signal_performance(days=30)
        stock_weights_df = monitor.prediction_tracker.get_all_stock_weights()

        # Signal explanations
        signal_info = {
            'momentum_20d': {
                'name': '20-Day Momentum',
                'description': 'Measures the rate of price change over 20 trading days. Positive momentum suggests upward trend continuation.',
                'interpretation': 'Higher weight = bot trusts momentum signals more for predictions'
            },
            'rsi': {
                'name': 'RSI (Relative Strength Index)',
                'description': 'Oscillator measuring speed and magnitude of price changes. Values above 70 suggest overbought, below 30 oversold.',
                'interpretation': 'Higher weight = bot relies more on overbought/oversold signals'
            },
            'macd_signal': {
                'name': 'MACD Signal Line',
                'description': 'Trend-following momentum indicator showing relationship between two moving averages.',
                'interpretation': 'Higher weight = bot trusts MACD crossovers for entry/exit timing'
            },
            'volume_ratio': {
                'name': 'Volume Ratio',
                'description': 'Compares current volume to average volume. High ratios suggest significant market interest.',
                'interpretation': 'Higher weight = bot values volume confirmation for trades'
            },
            'price_vs_sma20': {
                'name': 'Price vs 20-SMA',
                'description': 'Relationship between current price and 20-day simple moving average.',
                'interpretation': 'Higher weight = bot uses moving average crossovers more heavily'
            },
            'bollinger_position': {
                'name': 'Bollinger Band Position',
                'description': 'Where price sits within Bollinger Bands (volatility measure).',
                'interpretation': 'Higher weight = bot trusts mean-reversion signals from bands'
            }
        }

        # Format detailed weights with full info
        detailed_weights = []
        for signal, weight in monitor.signal_weights.items():
            info = signal_info.get(signal, {})

            # Get accuracy
            acc = 0
            uses = 0
            if not signal_perf.empty:
                sig_row = signal_perf[signal_perf['signal'] == signal]
                if not sig_row.empty:
                    acc = float(sig_row.iloc[0]['accuracy'])
                    uses = int(sig_row.iloc[0]['uses'])

            detailed_weights.append({
                'signal': signal,
                'name': info.get('name', signal),
                'description': info.get('description', ''),
                'interpretation': info.get('interpretation', ''),
                'weight': float(weight),
                'accuracy': acc,
                'uses': uses,
                'status': 'strong' if weight > 1.1 else ('weak' if weight < 0.9 else 'normal')
            })

        # Format per-stock data
        stock_profiles = []
        if not stock_weights_df.empty:
            for symbol in stock_weights_df['symbol'].unique():
                symbol_data = stock_weights_df[stock_weights_df['symbol'] == symbol]
                total_samples = int(symbol_data['sample_size'].sum())

                signals = []
                for _, row in symbol_data.iterrows():
                    signals.append({
                        'signal': row['signal_name'],
                        'name': signal_info.get(row['signal_name'], {}).get('name', row['signal_name']),
                        'weight': float(row['weight']),
                        'accuracy': float(row['accuracy']) if row['accuracy'] else 0,
                        'uses': int(row['sample_size']) if row['sample_size'] else 0
                    })

                stock_profiles.append({
                    'symbol': symbol,
                    'total_predictions': total_samples,
                    'signals': sorted(signals, key=lambda x: x['weight'], reverse=True)
                })

        return {
            'summary': {
                'total_predictions': stats.get('total_predictions', 0),
                'overall_accuracy': stats.get('accuracy', 0),
                'correct': stats.get('correct', 0),
                'wrong': stats.get('wrong', 0)
            },
            'global_weights': sorted(detailed_weights, key=lambda x: x['weight'], reverse=True),
            'stock_profiles': sorted(stock_profiles, key=lambda x: x['total_predictions'], reverse=True),
            'learning_explanation': {
                'how_it_works': 'The bot tracks which signals lead to correct predictions. Signals that perform well get MORE weight (trusted more). Signals that perform poorly get LESS weight.',
                'weight_range': 'Weights range from 0.5 (minimum, rarely trusted) to 2.0 (maximum, highly trusted). Starting value is 1.0.',
                'per_stock_learning': 'Each stock develops its own signal preferences over time. MACD might work great for AAPL but poorly for TSLA - the bot learns this.'
            }
        }

    except Exception as e:
        logger.error(f"Error getting brain details: {e}")
        return {'error': str(e)}

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    websocket_clients.append(websocket)

    logger.info(f"WebSocket client connected. Total: {len(websocket_clients)}")

    try:
        while True:
            # Send status update every 30 seconds
            status = await get_status()
            await websocket.send_json({
                'type': 'status',
                'data': status
            })
            await asyncio.sleep(30)

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

# ============================================================================
# MAIN
# ============================================================================

# Need pandas for some operations
import pandas as pd

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  THE ALCHEMY EFFECT")
    print("  Starting web dashboard...")
    print("=" * 50)
    print("\n  Open http://localhost:8000 in your browser\n")

    uvicorn.run(app, host="0.0.0.0", port=8000)
