"""
The Alchemy Effect - Web API (DEPRECATED - USE api_server.py INSTEAD)
=====================================================================

DEPRECATION NOTICE:
This file is deprecated and will be removed in a future version.
Please use api_server.py instead, which includes all functionality.

Run the primary API server with:
    uvicorn api_server:app --port 8000

This file now runs on port 8001 to avoid conflicts during migration.

==============================================================================
MIGRATION GUIDE
==============================================================================
All endpoints in this file have been consolidated into api_server.py:

Old (web_api.py)          ->  New (api_server.py)
/api/status               ->  /api/service/status
/api/brain                ->  /api/brain
/api/pnl                  ->  /api/pnl
/api/trades               ->  /api/trades
/api/bot/control          ->  /api/service/start, /api/service/stop
/api/portfolio            ->  /api/portfolio
/api/watchlist            ->  /api/learning/stocks

Use api_server.py for new development.
==============================================================================

Author: Claude AI
"""

import asyncio
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# Import our existing bot components
from core.market_monitor import MarketMonitor, get_market_monitor
from core.trading_bot import TradingBot
from core.order_executor import OrderExecutor
from core.personality_profiles import PERSONALITY_PROFILES, get_profile
from portfolio.risk_manager import RiskManager
from utils.trade_logger import get_trade_logger
from utils.database import Database
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# Standalone Alpaca connection for portfolio data (independent of bot)
_alpaca_executor: Optional[OrderExecutor] = None

def get_alpaca_executor() -> Optional[OrderExecutor]:
    """Get or create a standalone Alpaca executor for portfolio queries."""
    global _alpaca_executor
    if _alpaca_executor is None:
        try:
            _alpaca_executor = OrderExecutor(mode="paper")
            logger.info("Standalone Alpaca executor initialized for portfolio queries")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca executor: {e}")
    return _alpaca_executor

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
bot_thread: Optional[threading.Thread] = None
websocket_clients: List[WebSocket] = []

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

def get_trade_stats(days: int = 1, include_backtest: bool = False) -> Dict[str, Any]:
    """Get trade statistics for a given period (paper/live trades only by default)."""
    try:
        trade_logger = get_trade_logger()

        # Get paper and live trades only (exclude backtests by default)
        if include_backtest:
            df = trade_logger.get_trades()
        else:
            df_paper = trade_logger.get_trades(mode='paper')
            df_live = trade_logger.get_trades(mode='live')
            df = pd.concat([df_paper, df_live], ignore_index=True)

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

        # Filter by date - handle timezone-aware vs naive datetimes
        cutoff = datetime.now() - timedelta(days=days)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
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

    # Check is_trading (the actual flag used in TradingBot)
    is_running = bot is not None and hasattr(bot, 'is_trading') and bot.is_trading

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
    """Get P&L data from Alpaca paper/live account."""
    global trading_bot

    try:
        # Use bot's executor if running, otherwise use standalone executor
        executor = None
        if trading_bot and hasattr(trading_bot, 'executor') and trading_bot.executor:
            executor = trading_bot.executor
        else:
            executor = get_alpaca_executor()

        # Try to get real Alpaca account data
        if executor:
            account_info = executor.get_account_info()
            positions = executor.get_positions()

            if account_info and 'portfolio_value' in account_info:
                initial_capital = config.get('trading.initial_capital', 100000)
                portfolio_value = account_info['portfolio_value']
                total_pnl = portfolio_value - initial_capital

                # Calculate unrealized P&L from positions
                unrealized_pnl = sum(p.get('unrealized_pl', 0) for p in positions)

                # Get portfolio history for sparkline
                sparkline = executor.get_portfolio_history(days=30)

                return {
                    'today_pnl': unrealized_pnl,
                    'today_pnl_pct': (unrealized_pnl / initial_capital * 100) if initial_capital > 0 else 0,
                    'total_pnl': total_pnl,
                    'total_pnl_pct': (total_pnl / initial_capital * 100) if initial_capital > 0 else 0,
                    'initial_capital': initial_capital,
                    'current_value': portfolio_value,
                    'cash': account_info.get('cash', 0),
                    'buying_power': account_info.get('buying_power', 0),
                    'positions_count': len(positions),
                    'sparkline': sparkline,
                    'source': 'alpaca'
                }

        # Fallback to trade log data if Alpaca not available
        trade_logger = get_trade_logger()
        df_paper = trade_logger.get_trades(mode='paper')
        df_live = trade_logger.get_trades(mode='live')
        df = pd.concat([df_paper, df_live], ignore_index=True)

        today_stats = get_trade_stats(days=1)
        total_pnl = 0
        sparkline = []

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
            completed_trades = df[df['realized_pnl'].notna()]

            if not completed_trades.empty:
                total_pnl = float(completed_trades['realized_pnl'].sum())

                completed_trades = completed_trades.copy()
                completed_trades['date'] = completed_trades['timestamp'].dt.date
                daily = completed_trades.groupby('date')['realized_pnl'].sum().tail(30)

                cumulative = 0
                for date, pnl in daily.items():
                    cumulative += pnl
                    sparkline.append({
                        'date': str(date),
                        'daily_pnl': float(pnl),
                        'cumulative': float(cumulative)
                    })

        initial_capital = config.get('trading.initial_capital', 100000)
        current_value = initial_capital + total_pnl

        return {
            'today_pnl': today_stats.get('total_pnl', 0),
            'today_pnl_pct': (today_stats.get('total_pnl', 0) / initial_capital * 100) if initial_capital > 0 else 0,
            'total_pnl': total_pnl,
            'total_pnl_pct': (total_pnl / initial_capital * 100) if initial_capital > 0 else 0,
            'initial_capital': initial_capital,
            'current_value': current_value,
            'sparkline': sparkline,
            'source': 'trade_log'
        }

    except Exception as e:
        logger.error(f"Error getting P&L: {e}")
        return {
            'today_pnl': 0,
            'today_pnl_pct': 0,
            'total_pnl': 0,
            'initial_capital': 100000,
            'current_value': 100000,
            'sparkline': [],
            'error': str(e)
        }

@app.get("/api/portfolio")
async def get_portfolio():
    """Get full portfolio from Alpaca paper/live account."""
    global trading_bot

    try:
        # Use bot's executor if running, otherwise use standalone executor
        executor = None
        if trading_bot and hasattr(trading_bot, 'executor') and trading_bot.executor:
            executor = trading_bot.executor
        else:
            executor = get_alpaca_executor()

        if executor:
            account_info = executor.get_account_info()
            positions = executor.get_positions()

            if account_info and 'portfolio_value' in account_info:
                return {
                    'account': account_info,
                    'positions': positions,
                    'positions_count': len(positions),
                    'total_market_value': sum(p.get('market_value', 0) for p in positions),
                    'total_unrealized_pl': sum(p.get('unrealized_pl', 0) for p in positions),
                    'source': 'alpaca'
                }

        return {
            'account': None,
            'positions': [],
            'positions_count': 0,
            'error': 'Alpaca not connected - check API keys'
        }

    except Exception as e:
        logger.error(f"Error getting portfolio: {e}")
        return {
            'account': None,
            'positions': [],
            'error': str(e)
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
        initial_capital = config.get('trading.initial_capital', 100000)
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

@app.get("/api/trades")
async def get_trades(limit: int = 50, mode: str = None):
    """
    Get recent trades.

    Args:
        limit: Max number of trades to return
        mode: Filter by mode ('paper', 'live', 'backtest', or None for all)
              Default behavior: shows paper and live trades, excludes backtests
    """
    try:
        trade_logger = get_trade_logger()

        # By default, exclude backtest trades to show only "real" paper/live trades
        if mode is None:
            # Get paper and live trades only
            df_paper = trade_logger.get_trades(mode='paper')
            df_live = trade_logger.get_trades(mode='live')
            df = pd.concat([df_paper, df_live], ignore_index=True)
            # Sort by timestamp descending
            if not df.empty and 'timestamp' in df.columns:
                df = df.sort_values('timestamp', ascending=False)
        elif mode == 'all':
            df = trade_logger.get_trades()
        else:
            df = trade_logger.get_trades(mode=mode)

        if df.empty:
            return {'trades': [], 'mode_filter': mode or 'paper+live'}

        trades = []
        for _, row in df.head(limit).iterrows():
            trades.append({
                'id': row['trade_id'],
                'timestamp': str(row['timestamp']),
                'symbol': row['symbol'],
                'action': row['action'],
                'quantity': float(row['quantity']),
                'price': float(row['price']),
                'pnl': float(row['realized_pnl']) if pd.notna(row.get('realized_pnl')) else None,
                'strategy': row['strategy_name'],
                'reason': row['primary_signal'],
                'mode': row.get('mode', 'unknown')
            })

        return {'trades': trades, 'mode_filter': mode or 'paper+live', 'total': len(df)}

    except Exception as e:
        logger.error(f"Error getting trades: {e}")
        return {'trades': [], 'error': str(e)}

@app.get("/api/activity")
async def get_activity(limit: int = 50):
    """
    Get live activity feed by reading directly from the log files.
    Parses log lines and formats them into readable messages.
    """
    try:
        activities = []
        log_dir = Path("logs")

        # Get today's log file (primary) - this is where active bot logs go
        today = datetime.now().strftime('%Y-%m-%d')
        today_log = log_dir / f"trading_bot_{today}.log"

        # Read log lines from today's file only
        log_lines = []
        if today_log.exists():
            try:
                with open(today_log, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    # Get last N lines (more than limit to allow filtering)
                    log_lines = lines[-(limit * 5):]
            except Exception as e:
                logger.debug(f"Could not read {today_log}: {e}")

        # Fallback to main log only if no today's log
        if not log_lines:
            main_log = log_dir / "trading_bot.log"
            if main_log.exists():
                try:
                    with open(main_log, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        log_lines = lines[-(limit * 5):]
                except Exception as e:
                    logger.debug(f"Could not read {main_log}: {e}")

        # Parse and format log lines
        for line in reversed(log_lines):  # Newest first
            parsed = parse_log_line(line.strip())
            if parsed:
                activities.append(parsed)
                if len(activities) >= limit:
                    break

        # Determine bot status
        bot = get_bot()
        bot_is_active = (
            (bot is not None and hasattr(bot, 'is_trading') and bot.is_trading) or
            (bot_thread is not None and bot_thread.is_alive())
        )

        return {
            'activities': activities,
            'total': len(activities),
            'bot_active': bot_is_active
        }

    except Exception as e:
        logger.error(f"Error getting activity feed: {e}")
        return {'activities': [], 'total': 0, 'bot_active': False, 'error': str(e)}


def parse_log_line(line: str) -> Optional[Dict]:
    """
    Parse a log line and convert it to a user-friendly activity item.

    Log format: 2025-12-04 14:10:47 | INFO     | module:function:line | Message
    """
    import re

    if not line or len(line) < 20:
        return None

    # Skip certain noisy log lines
    skip_patterns = [
        'Logger initialized',
        'Database initialized',
        'Database tables',
        'oneDNN custom operations',
        'tensorflow',
        'INFO:',  # uvicorn logs
        'HTTP/1.1',
        '127.0.0.1:',
    ]
    for skip in skip_patterns:
        if skip in line:
            return None

    # Parse timestamp and message
    # Format: 2025-12-04 14:10:47 | INFO     | core.trading_bot:run_trading_cycle:513 | Message
    match = re.match(r'^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\|\s*(\w+)\s*\|\s*([^|]+)\|\s*(.+)$', line)

    if not match:
        return None

    timestamp, level, source, message = match.groups()
    message = message.strip()
    source = source.strip()

    # Determine activity type and icon based on message content
    result = format_log_message(message, source, level)

    # Handle None return or tuple with None values
    if result is None or result == (None, None, None):
        return None

    activity_type, icon, formatted_message = result

    if not formatted_message:
        return None

    # Extract symbol if present
    symbol = None
    symbol_match = re.search(r'\b([A-Z]{2,5})\b(?=:|\s|$)', message)
    if symbol_match:
        potential_symbol = symbol_match.group(1)
        # Filter out common non-symbol words
        non_symbols = {'INFO', 'ERROR', 'WARN', 'DEBUG', 'BUY', 'SELL', 'AND', 'FOR', 'THE', 'RSI', 'MACD', 'SMA', 'EMA'}
        if potential_symbol not in non_symbols:
            symbol = potential_symbol

    return {
        'type': activity_type,
        'icon': icon,
        'timestamp': timestamp,
        'symbol': symbol,
        'message': formatted_message,
        'raw': message,
        'level': level
    }


def format_log_message(message: str, source: str, level: str) -> tuple:
    """
    Convert a raw log message into a user-friendly format.
    Returns (activity_type, icon, formatted_message)
    """
    import re

    # Trading cycle messages
    if 'TRADING CYCLE' in message:
        time_match = re.search(r'TRADING CYCLE - (\d+:\d+:\d+)', message)
        time_str = time_match.group(1) if time_match else ''
        return ('cycle', 'cycle', f"🔄 Starting trading cycle at {time_str}")

    if 'Trading cycle complete' in message:
        match = re.search(r'(\d+) symbols analyzed', message)
        count = match.group(1) if match else '?'
        if 'no trades' in message.lower():
            return ('cycle', 'cycle', f"✅ Cycle complete: Analyzed {count} stocks, no trades")
        return ('cycle', 'cycle', f"✅ Cycle complete: Analyzed {count} stocks")

    # Strategy info
    if 'Strategy:' in message:
        strategy = message.replace('Strategy:', '').strip()
        return ('strategy', 'brain', f"🧠 Using strategy: {strategy}")

    if 'Trades today:' in message:
        return ('info', 'info', f"📊 {message}")

    # Stock analysis
    if 'No signal' in message:
        match = re.search(r'(\w+): No signal - (.+)', message)
        if match:
            symbol, reason = match.groups()
            # Simplify the reason
            reason = reason.replace('Indicators not showing a clear buy opportunity', 'No clear opportunity')
            return ('analysis', 'search', f"🔍 {symbol}: {reason}")

    if 'SIGNAL(S) FOUND' in message:
        match = re.search(r'(\w+): (\d+) SIGNAL', message)
        if match:
            symbol, count = match.groups()
            return ('signal', 'alert', f"⚡ {symbol}: Found {count} trading signal(s)!")

    # Buy/Sell signals
    if 'BUY signal' in message or 'SELL signal' in message:
        return ('signal', 'trade', f"📈 {message}")

    # Trade execution
    if 'Trade logged' in message:
        match = re.search(r'(BUY|SELL)\s+([\d.]+)\s+(\w+)\s+@\s+\$([\d.]+)', message)
        if match:
            action, qty, symbol, price = match.groups()
            emoji = '🟢' if action == 'BUY' else '🔴'
            return ('trade', 'trade', f"{emoji} {action} {float(qty):.2f} {symbol} @ ${float(price):.2f}")

    if 'TRADE EXECUTED' in message:
        return ('trade', 'trade', f"✅ {message}")

    if 'TRADE REJECTED' in message:
        return ('trade', 'alert', f"❌ {message}")

    # Predictions
    if 'Prediction added' in message:
        match = re.search(r'(\w+)\s+(up|down|sideways)\s+\(([\d.]+)%', message)
        if match:
            symbol, direction, confidence = match.groups()
            emoji = '📈' if direction == 'up' else '📉' if direction == 'down' else '➡️'
            return ('prediction', 'prediction', f"{emoji} Prediction: {symbol} will go {direction} ({confidence}% confidence)")

    if 'Prediction resolved' in message:
        if 'CORRECT' in message:
            return ('prediction', 'confirmed', f"✅ {message.replace('Prediction resolved:', 'Prediction CORRECT:')}")
        else:
            return ('prediction', 'alert', f"❌ {message.replace('Prediction resolved:', 'Prediction WRONG:')}")

    # Backtest messages
    if 'Running backtest' in message:
        return ('backtest', 'chart', f"📊 {message}")

    if 'Backtest complete' in message:
        match = re.search(r'Final capital: \$([\d,.]+)', message)
        if match:
            capital = match.group(1)
            return ('backtest', 'chart', f"📊 Backtest complete: Final capital ${capital}")

    if 'Total return:' in message or 'Sharpe ratio:' in message or 'Win rate:' in message:
        return ('backtest', 'chart', f"📈 {message}")

    # Strategy evaluation
    if 'Strategy Performance Summary' in message or '===' in message:
        return None  # Skip separator lines

    if 'SELECTED' in message:
        match = re.search(r'(\w+): ([\d.]+)% return', message)
        if match:
            strategy, ret = match.groups()
            return ('strategy', 'brain', f"🏆 Selected strategy: {strategy} ({ret}% return)")

    # Learning messages
    if 'AI Learning' in message:
        return ('learning', 'learning', f"🤖 {message.replace('AI Learning:', '').strip()}")

    if 'signal weights' in message.lower():
        return ('learning', 'learning', f"⚙️ {message}")

    # Data fetching
    if 'Fetching data for' in message:
        match = re.search(r'Fetching data for (\w+)', message)
        if match:
            symbol = match.group(1)
            return ('data', 'data', f"📥 Loading market data for {symbol}")

    if 'Retrieved' in message and 'bars' in message:
        match = re.search(r'Retrieved (\d+) bars for (\w+)', message)
        if match:
            bars, symbol = match.groups()
            return ('data', 'data', f"📊 Loaded {bars} days of {symbol} data")

    # Bot status
    if 'Trading Bot is now running' in message:
        return ('status', 'active', "🚀 Trading bot is now running!")

    if 'Bot stopped' in message or 'Stopping Trading Bot' in message:
        return ('status', 'inactive', "⏹️ Trading bot stopped")

    if 'startup sequence' in message.lower():
        return ('status', 'loading', f"⏳ {message}")

    # Watchlist
    if 'Watchlist has' in message:
        match = re.search(r'Watchlist has (\d+) symbols', message)
        if match:
            count = match.group(1)
            return ('watchlist', 'list', f"📋 Monitoring {count} stocks on watchlist")

    # Screening
    if 'screen' in message.lower() and 'passed' in message.lower():
        return ('screening', 'filter', f"🔎 {message}")

    # Market monitor
    if 'Market monitor' in message or 'Live Monitor' in message:
        return ('monitor', 'monitor', f"👁️ {message}")

    # Daily performance
    if 'DAILY PERFORMANCE' in message or 'Portfolio Value' in message or 'Daily P&L' in message:
        return ('performance', 'chart', f"📊 {message}")

    # Errors and warnings
    if level == 'ERROR':
        return ('error', 'error', f"❌ Error: {message}")

    if level == 'WARNING':
        return ('warning', 'alert', f"⚠️ {message}")

    # Skip certain generic messages
    if any(x in message for x in ['========', '------', 'INFO', 'initialized']):
        return (None, None, None)

    # Default: show as info if it seems meaningful
    if len(message) > 10 and not message.startswith('  '):
        return ('info', 'info', f"ℹ️ {message}")

    return (None, None, None)

# In-memory activity log for real-time updates
activity_log = []
MAX_ACTIVITY_LOG = 100

def add_activity(activity_type: str, icon: str, message_type: str, symbol: str = None, data: dict = None):
    """Add an activity to the in-memory log for immediate display."""
    global activity_log
    activity_log.insert(0, {
        'type': activity_type,
        'icon': icon,
        'timestamp': datetime.now().isoformat(),
        'symbol': symbol,
        'message_type': message_type,
        'data': data or {}
    })
    # Keep log from growing too large
    if len(activity_log) > MAX_ACTIVITY_LOG:
        activity_log = activity_log[:MAX_ACTIVITY_LOG]

@app.post("/api/bot/control")
async def control_bot(command: BotCommand):
    """Control the trading bot (start/stop/pause)."""
    global trading_bot, bot_thread

    try:
        if command.action == 'start':
            # Check if already running
            if trading_bot is not None and hasattr(trading_bot, 'is_trading') and trading_bot.is_trading:
                return {'success': True, 'status': 'already_running', 'message': 'Bot is already running'}

            # Add immediate activity feedback
            add_activity('analysis', 'analysis', 'bot_starting', data={
                'status': 'starting',
                'message': 'Initializing trading bot...'
            })

            # Create new bot if needed
            if trading_bot is None:
                add_activity('learning', 'learning', 'bot_init', data={
                    'message': 'Loading AI models and strategies...'
                })

                trading_bot = TradingBot(
                    initial_capital=config.get('trading.initial_capital', 100000),
                    personality='ai_adaptive',
                    mode='paper'
                )

                add_activity('analysis', 'analysis', 'bot_init', data={
                    'message': 'Bot initialized successfully'
                })

            # Run bot in background thread to avoid blocking the API
            def run_bot():
                try:
                    add_activity('analysis', 'analysis', 'bot_running', data={
                        'status': 'active',
                        'message': 'Bot is now actively trading'
                    })
                    trading_bot.start()
                except Exception as e:
                    logger.error(f"Bot thread error: {e}")
                    add_activity('alert', 'alert', 'bot_error', data={
                        'message': f'Bot error: {str(e)}'
                    })

            bot_thread = threading.Thread(target=run_bot, daemon=True)
            bot_thread.start()

            # Return immediately - don't wait
            return {'success': True, 'status': 'starting', 'message': 'Bot is starting...'}

        elif command.action == 'stop':
            add_activity('analysis', 'analysis', 'bot_stopped', data={
                'message': 'Trading bot stopped by user'
            })
            if trading_bot:
                trading_bot.stop()
                # Wait for thread to finish
                if bot_thread and bot_thread.is_alive():
                    bot_thread.join(timeout=2)
            return {'success': True, 'status': 'stopped', 'message': 'Bot stopped'}

        elif command.action == 'pause':
            add_activity('analysis', 'analysis', 'bot_stopped', data={
                'message': 'Trading bot paused'
            })
            if trading_bot and hasattr(trading_bot, 'pause'):
                trading_bot.pause()
            return {'success': True, 'status': 'paused', 'message': 'Bot paused'}

        else:
            return {'success': False, 'error': f'Unknown action: {command.action}'}

    except Exception as e:
        logger.error(f"Error controlling bot: {e}")
        return {'success': False, 'error': str(e)}

@app.get("/api/reports")
async def get_reports(days: int = 30):
    """Get performance reports for the Reports tab."""
    try:
        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {
                'total_pnl': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'daily_pnl': [],
                'top_performers': [],
                'worst_performers': [],
                'strategy_performance': [],
                'trades': []
            }

        # Filter by date - handle timezone-aware vs naive datetimes
        cutoff = datetime.now() - timedelta(days=days)
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True).dt.tz_localize(None)
        df = df[df['timestamp'] >= cutoff]

        if df.empty:
            return {
                'total_pnl': 0,
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'daily_pnl': [],
                'top_performers': [],
                'worst_performers': [],
                'strategy_performance': [],
                'trades': []
            }

        # Get trades with realized P&L (sells)
        sells = df[df['realized_pnl'].notna()].copy()

        # Calculate stats
        total_pnl = sells['realized_pnl'].sum() if not sells.empty else 0
        total_trades = len(sells)
        wins = sells[sells['realized_pnl'] > 0]
        losses = sells[sells['realized_pnl'] < 0]

        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        avg_win = wins['realized_pnl'].mean() if not wins.empty else 0
        avg_loss = losses['realized_pnl'].mean() if not losses.empty else 0

        gross_profit = wins['realized_pnl'].sum() if not wins.empty else 0
        gross_loss = abs(losses['realized_pnl'].sum()) if not losses.empty else 0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0

        # Calculate max drawdown
        if not sells.empty:
            sells_sorted = sells.sort_values('timestamp')
            cumulative = sells_sorted['realized_pnl'].cumsum()
            peak = cumulative.cummax()
            drawdown = (cumulative - peak)
            max_drawdown = abs(drawdown.min()) if not drawdown.empty else 0
            # Convert to percentage of peak
            max_drawdown_pct = (max_drawdown / peak.max() * 100) if peak.max() > 0 else 0
        else:
            max_drawdown_pct = 0

        # Sharpe ratio (simplified)
        if not sells.empty and len(sells) > 1:
            returns = sells['realized_pnl']
            sharpe_ratio = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0

        # Daily P&L
        daily_pnl = []
        if not sells.empty:
            sells['date'] = sells['timestamp'].dt.date
            daily = sells.groupby('date')['realized_pnl'].sum().reset_index()
            daily_pnl = [{'date': str(row['date']), 'pnl': row['realized_pnl']} for _, row in daily.iterrows()]

        # Top/Worst performers by symbol
        top_performers = []
        worst_performers = []
        if not sells.empty:
            symbol_pnl = sells.groupby('symbol').agg({
                'realized_pnl': 'sum',
                'symbol': 'count'
            }).rename(columns={'symbol': 'trades'}).reset_index()
            symbol_pnl.columns = ['symbol', 'pnl', 'trades']

            top_performers = symbol_pnl.nlargest(5, 'pnl')[['symbol', 'pnl', 'trades']].to_dict('records')
            worst_performers = symbol_pnl.nsmallest(5, 'pnl')[['symbol', 'pnl', 'trades']].to_dict('records')

        # Strategy performance
        strategy_performance = []
        if not sells.empty and 'strategy' in sells.columns:
            strat_group = sells.groupby('strategy').agg({
                'realized_pnl': ['sum', 'mean', 'count'],
            })
            strat_group.columns = ['pnl', 'avg_trade', 'trades']
            strat_group = strat_group.reset_index()

            for _, row in strat_group.iterrows():
                strat_sells = sells[sells['strategy'] == row['strategy']]
                strat_wins = len(strat_sells[strat_sells['realized_pnl'] > 0])
                strat_win_rate = (strat_wins / row['trades'] * 100) if row['trades'] > 0 else 0

                strategy_performance.append({
                    'name': row['strategy'] or 'Unknown',
                    'trades': int(row['trades']),
                    'pnl': float(row['pnl']),
                    'avg_trade': float(row['avg_trade']),
                    'win_rate': strat_win_rate
                })

        # Recent trades
        trades = []
        recent = df.sort_values('timestamp', ascending=False).head(50)
        for _, row in recent.iterrows():
            trades.append({
                'timestamp': row['timestamp'].isoformat(),
                'symbol': row['symbol'],
                'action': row['action'],
                'quantity': float(row.get('quantity', 0)),
                'price': float(row.get('price', 0)),
                'entry_price': float(row.get('entry_price', row.get('price', 0))),
                'exit_price': float(row.get('exit_price', 0)) if pd.notna(row.get('exit_price')) else 0,
                'pnl': float(row.get('realized_pnl', 0)) if pd.notna(row.get('realized_pnl')) else None,
                'strategy': row.get('strategy', '')
            })

        return {
            'total_pnl': float(total_pnl),
            'total_trades': int(total_trades),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'max_drawdown': float(max_drawdown_pct),
            'sharpe_ratio': float(sharpe_ratio),
            'daily_pnl': daily_pnl,
            'top_performers': top_performers,
            'worst_performers': worst_performers,
            'strategy_performance': strategy_performance,
            'trades': trades
        }

    except Exception as e:
        logger.error(f"Error getting reports: {e}")
        return {
            'total_pnl': 0,
            'total_trades': 0,
            'win_rate': 0,
            'profit_factor': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'daily_pnl': [],
            'top_performers': [],
            'worst_performers': [],
            'strategy_performance': [],
            'trades': [],
            'error': str(e)
        }

@app.get("/api/reports/export")
async def export_reports(days: int = 30, format: str = 'csv'):
    """Export reports as CSV."""
    from fastapi.responses import StreamingResponse
    import io

    try:
        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            return {'error': 'No trades to export'}

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df[df['timestamp'] >= cutoff]

        # Create CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return StreamingResponse(
            iter([output.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=trades_{days}d.csv"}
        )

    except Exception as e:
        logger.error(f"Error exporting reports: {e}")
        return {'error': str(e)}

# ============================================================================
# WATCHLIST / SETTINGS ENDPOINTS
# ============================================================================

class WatchlistUpdate(BaseModel):
    symbols: List[str]

class WatchlistAddRemove(BaseModel):
    symbols: List[str]

@app.get("/api/watchlist")
async def get_watchlist():
    """
    Get the current portfolio AND watchlist.

    Returns two distinct lists:
    - portfolio: Stocks you currently OWN (from Alpaca paper/live account)
    - watchlist: Stocks you're WATCHING for buy opportunities (editable)
    """
    global trading_bot

    try:
        portfolio = []

        # Use bot's executor if running, otherwise use standalone executor
        executor = None
        if trading_bot and hasattr(trading_bot, 'executor') and trading_bot.executor:
            executor = trading_bot.executor
        else:
            executor = get_alpaca_executor()

        # Get actual positions from Alpaca
        if executor:
            positions = executor.get_positions()
            portfolio = [{
                'symbol': p['symbol'],
                'qty': p['qty'],
                'avg_price': p['avg_entry_price'],
                'current_price': p['current_price'],
                'market_value': p['market_value'],
                'unrealized_pl': p['unrealized_pl'],
                'unrealized_plpc': p['unrealized_plpc']
            } for p in positions]

        # Get watchlist from config or bot
        if trading_bot is not None and hasattr(trading_bot, 'get_portfolio_and_watchlist'):
            data = trading_bot.get_portfolio_and_watchlist()
            watchlist = data.get('watchlist', [])
        else:
            watchlist = config.get('data.universe.initial_stocks', ['SPY', 'QQQ', 'IWM'])

        return {
            'portfolio': portfolio,
            'watchlist': watchlist,
            'portfolio_count': len(portfolio),
            'watchlist_count': len(watchlist),
            'source': 'alpaca' if portfolio else 'config'
        }

    except Exception as e:
        logger.error(f"Error getting watchlist: {e}")
        return {
            'portfolio': [],
            'watchlist': ['SPY', 'QQQ', 'IWM'],
            'error': str(e)
        }

@app.post("/api/watchlist")
async def update_watchlist(update: WatchlistUpdate):
    """Replace the entire watchlist with new symbols."""
    global trading_bot

    try:
        symbols = [s.strip().upper() for s in update.symbols if s.strip()]

        if not symbols:
            return {'success': False, 'error': 'No valid symbols provided'}

        # Update the config in memory
        config.set('data.universe.initial_stocks', symbols)

        # Also update the trading bot if it's running
        if trading_bot is not None:
            trading_bot._custom_watchlist = symbols
            logger.info(f"Updated watchlist to: {symbols}")

        return {
            'success': True,
            'watchlist': symbols,
            'message': f'Watchlist updated to {len(symbols)} symbols'
        }

    except Exception as e:
        logger.error(f"Error updating watchlist: {e}")
        return {'success': False, 'error': str(e)}

@app.post("/api/watchlist/add")
async def add_to_watchlist(update: WatchlistAddRemove):
    """Add symbols to the watchlist (without removing existing)."""
    global trading_bot

    try:
        symbols = [s.strip().upper() for s in update.symbols if s.strip()]

        if not symbols:
            return {'success': False, 'error': 'No valid symbols provided'}

        added = []

        if trading_bot is not None:
            added = trading_bot.add_to_watchlist(symbols)
        else:
            # Update config directly
            current = config.get('data.universe.initial_stocks', ['SPY', 'QQQ', 'IWM'])
            for s in symbols:
                if s not in current:
                    current.append(s)
                    added.append(s)
            config.set('data.universe.initial_stocks', current)

        return {
            'success': True,
            'added': added,
            'message': f'Added {len(added)} symbol(s) to watchlist'
        }

    except Exception as e:
        logger.error(f"Error adding to watchlist: {e}")
        return {'success': False, 'error': str(e)}

@app.post("/api/watchlist/remove")
async def remove_from_watchlist(update: WatchlistAddRemove):
    """Remove symbols from the watchlist."""
    global trading_bot

    try:
        symbols = [s.strip().upper() for s in update.symbols if s.strip()]

        if not symbols:
            return {'success': False, 'error': 'No valid symbols provided'}

        removed = []

        if trading_bot is not None:
            removed = trading_bot.remove_from_watchlist(symbols)
        else:
            # Update config directly
            current = config.get('data.universe.initial_stocks', ['SPY', 'QQQ', 'IWM'])
            for s in symbols:
                if s in current:
                    current.remove(s)
                    removed.append(s)
            config.set('data.universe.initial_stocks', current)

        return {
            'success': True,
            'removed': removed,
            'message': f'Removed {len(removed)} symbol(s) from watchlist'
        }

    except Exception as e:
        logger.error(f"Error removing from watchlist: {e}")
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
    import warnings
    warnings.warn(
        "web_api.py is deprecated. Use 'uvicorn api_server:app --port 8000' instead.",
        DeprecationWarning,
        stacklevel=2
    )
    print("\n" + "=" * 50)
    print("  THE ALCHEMY EFFECT (DEPRECATED)")
    print("  ⚠️  This server is deprecated!")
    print("  Use api_server.py instead for full functionality.")
    print("=" * 50)
    print("\n  Running on port 8001 (to avoid conflict with api_server.py)")
    print("  Open http://localhost:8001 in your browser\n")

    uvicorn.run(app, host="0.0.0.0", port=8001)
