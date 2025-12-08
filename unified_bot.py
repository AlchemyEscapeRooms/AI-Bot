"""
Unified Trading Bot
====================

The main entry point that combines:
- Background service (always running, learning, monitoring)
- User interaction (morning briefing, commands)
- Trade execution (when enabled)

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    UNIFIED TRADING BOT                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           BACKGROUND SERVICE (always on)             │   │
│   │  ├── Market Data Polling (every 60s)                │   │
│   │  ├── Prediction Engine (every 60m)                  │   │
│   │  ├── Verification Loop (every 5m)                   │   │
│   │  ├── Learning/Weight Updates (continuous)           │   │
│   │  └── Trade Execution (when conditions met)          │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                  │
│                           ▼                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │           USER INTERFACE (on demand)                 │   │
│   │  ├── Morning Briefing                               │   │
│   │  ├── Stock Add/Remove Commands                      │   │
│   │  ├── Status Queries                                 │   │
│   │  ├── Trading Mode Changes                           │   │
│   │  └── End of Day Summary                             │   │
│   └─────────────────────────────────────────────────────┘   │
│                                                              │
└─────────────────────────────────────────────────────────────┘

Author: Claude AI
Date: November 29, 2025
"""

import json
import threading
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional
from pathlib import Path

from core.background_service import (
    BackgroundTradingService,
    ServiceConfig,
    ServiceState,
    TradingMode,
    TradeSignal
)
from core.historical_trainer import HistoricalTrainer
from core.learning_trader import PredictionDatabase, StockLearningProfile
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class UnifiedTradingBot:
    """
    The main trading bot that users interact with.
    
    Combines always-on background processing with user commands.
    """
    
    def __init__(
        self,
        symbols: List[str] = None,
        api_key: str = None,
        api_secret: str = None,
        data_dir: str = "data",
        user_name: str = "Trader"
    ):
        self.api_key = api_key or config.get('alpaca.api_key')
        self.api_secret = api_secret or config.get('alpaca.api_secret')
        self.data_dir = Path(data_dir)
        self.user_name = user_name
        
        # Default symbols
        self.default_symbols = symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "META", "NVDA", "AMD", "SPY", "QQQ"
        ]
        
        # Service config - trading_mode loaded from config.yaml via ServiceConfig defaults
        self.service_config = ServiceConfig(
            symbols=self.default_symbols.copy(),
            # trading_mode is now set from config.yaml via ServiceConfig default
            prediction_interval_minutes=60,
            verification_interval_minutes=5,
        )
        
        # Background service
        self.service = BackgroundTradingService(
            config=self.service_config,
            api_key=self.api_key,
            api_secret=self.api_secret,
            data_dir=str(self.data_dir)
        )
        
        # Register callbacks
        self.service.on_prediction = self._on_prediction
        self.service.on_verification = self._on_verification
        self.service.on_trade_signal = self._on_trade_signal
        self.service.on_trade_executed = self._on_trade_executed
        
        # Event log for user to review
        self.event_log: List[Dict] = []
        self.max_log_size = 1000
        
        # Alert queue for high-priority notifications
        self.alerts: List[Dict] = []
        
        # Session state
        self.session_started = False
        self.briefing_delivered = False
        
        logger.info(f"UnifiedTradingBot initialized for {self.user_name}")
    
    # =========================================================================
    # CALLBACKS FROM BACKGROUND SERVICE
    # =========================================================================
    
    def _on_prediction(self, prediction):
        """Called when a new prediction is made."""
        self._log_event('prediction', {
            'symbol': prediction.symbol,
            'horizon': prediction.horizon.value,
            'direction': prediction.predicted_direction.value,
            'confidence': prediction.confidence,
            'predicted_change': prediction.predicted_change_pct
        })
    
    def _on_verification(self, result: Dict):
        """Called when a prediction is verified."""
        self._log_event('verification', result)
        
        # Alert on significant accuracy changes
        if result.get('was_correct'):
            pass  # Could track streaks
    
    def _on_trade_signal(self, signal: TradeSignal):
        """Called when a trade signal is generated."""
        self._log_event('trade_signal', {
            'symbol': signal.symbol,
            'action': signal.action,
            'confidence': signal.confidence,
            'quantity': signal.suggested_quantity
        })
        
        # High confidence signals get alerts
        if signal.confidence >= 0.75:
            self._add_alert('high_confidence_signal', {
                'symbol': signal.symbol,
                'action': signal.action,
                'confidence': signal.confidence,
                'reasoning': signal.reasoning
            })
    
    def _on_trade_executed(self, signal: TradeSignal):
        """Called when a trade is executed."""
        self._log_event('trade_executed', {
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.suggested_quantity,
            'order_id': signal.order_id
        })
        
        self._add_alert('trade_executed', {
            'symbol': signal.symbol,
            'action': signal.action,
            'quantity': signal.suggested_quantity
        })
    
    def _log_event(self, event_type: str, data: Dict):
        """Log an event."""
        self.event_log.append({
            'timestamp': datetime.now().isoformat(),
            'type': event_type,
            'data': data
        })
        
        # Trim log
        if len(self.event_log) > self.max_log_size:
            self.event_log = self.event_log[-self.max_log_size:]
    
    def _add_alert(self, alert_type: str, data: Dict):
        """Add an alert for user."""
        self.alerts.append({
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'data': data,
            'read': False
        })
    
    # =========================================================================
    # SERVICE CONTROL
    # =========================================================================
    
    def start(self):
        """Start the background service."""
        self.service.start()
        self.session_started = True
        logger.info("Trading bot started")
    
    def stop(self):
        """Stop the background service."""
        self.service.stop()
        logger.info("Trading bot stopped")
    
    def pause(self):
        """Pause predictions and trading."""
        self.service.pause()
    
    def resume(self):
        """Resume predictions and trading."""
        self.service.resume()
    
    # =========================================================================
    # MORNING BRIEFING
    # =========================================================================
    
    def get_morning_briefing(self) -> str:
        """Generate the complete morning briefing."""
        lines = []
        
        now = datetime.now()
        lines.append(f"Good morning, {self.user_name}!")
        lines.append(f"Today is {now.strftime('%A, %B %d, %Y')}.")
        lines.append("")
        
        # Get status from service
        status = self.service.get_status()
        account = status.get('account', {})
        daily = status.get('daily_stats', {})
        
        # Account Summary
        lines.append("=" * 55)
        lines.append("ACCOUNT SUMMARY")
        lines.append("=" * 55)
        lines.append(f"Portfolio Value: ${account.get('equity', 0):,.2f}")
        lines.append(f"Cash Available:  ${account.get('cash', 0):,.2f}")
        lines.append(f"Buying Power:    ${account.get('buying_power', 0):,.2f}")
        lines.append("")
        
        # Yesterday's Learning Performance
        db = PredictionDatabase(str(self.data_dir / "predictions.db"))
        stats = db.get_prediction_stats(days=1)
        
        lines.append("=" * 55)
        lines.append("YESTERDAY'S LEARNING PERFORMANCE")
        lines.append("=" * 55)
        
        total_preds = sum(s.get('total', 0) for s in stats.values())
        correct_preds = sum(s.get('correct', 0) for s in stats.values())
        accuracy = correct_preds / total_preds if total_preds > 0 else 0
        
        lines.append(f"Predictions Made:    {total_preds}")
        lines.append(f"Predictions Correct: {correct_preds}")
        lines.append(f"Overall Accuracy:    {accuracy:.1%}")
        lines.append("")
        
        for horizon, data in stats.items():
            h_acc = data.get('accuracy')
            h_acc_str = f"{h_acc:.1%}" if h_acc else "N/A"
            lines.append(f"  {horizon}: {data.get('total', 0)} preds, {h_acc_str} accuracy")
        lines.append("")
        
        # Stock Readiness
        lines.append("=" * 55)
        lines.append("STOCK READINESS")
        lines.append("=" * 55)
        
        ready_count = 0
        learning_count = 0
        
        for symbol in self.service_config.symbols:
            profile = db.get_stock_profile(symbol)
            
            if profile is None:
                status_icon = "○"
                status_text = "Not trained"
            elif profile.total_predictions >= 100 and profile.overall_accuracy >= 0.55:
                status_icon = "✓"
                status_text = f"Ready ({profile.total_predictions} preds, {profile.overall_accuracy:.1%})"
                ready_count += 1
            else:
                status_icon = "◐"
                status_text = f"Learning ({profile.total_predictions} preds, {profile.overall_accuracy:.1%})"
                learning_count += 1
            
            excluded = " [EXCLUDED]" if symbol in self.service_config.excluded_symbols else ""
            lines.append(f"  {status_icon} {symbol}: {status_text}{excluded}")
        
        lines.append("")
        lines.append(f"Summary: {ready_count} ready, {learning_count} learning")
        lines.append("")
        
        # Current Mode
        lines.append("=" * 55)
        lines.append("CURRENT MODE")
        lines.append("=" * 55)
        mode = self.service_config.trading_mode
        if mode == TradingMode.LEARNING_ONLY:
            lines.append("Mode: LEARNING ONLY (predictions only, no trades)")
        elif mode == TradingMode.PAPER_TRADING:
            lines.append("Mode: PAPER TRADING (simulated trades)")
        else:
            lines.append("Mode: LIVE TRADING (real money)")
        lines.append("")
        
        # Background Service Status
        lines.append("=" * 55)
        lines.append("BACKGROUND SERVICE")
        lines.append("=" * 55)
        svc_status = self.service.get_status()
        lines.append(f"State: {svc_status.get('state', 'unknown')}")
        lines.append(f"Market Open: {'Yes' if svc_status.get('market_open') else 'No'}")
        last_pred = svc_status.get('last_prediction')
        if last_pred:
            lines.append(f"Last Prediction Cycle: {last_pred}")
        lines.append("")
        
        # Pending Alerts
        unread_alerts = [a for a in self.alerts if not a['read']]
        if unread_alerts:
            lines.append("=" * 55)
            lines.append(f"🔔 ALERTS ({len(unread_alerts)} unread)")
            lines.append("=" * 55)
            for alert in unread_alerts[-5:]:
                lines.append(f"  [{alert['type']}] {alert['data']}")
            lines.append("")
        
        self.briefing_delivered = True
        return "\n".join(lines)
    
    def get_stock_prompt(self) -> str:
        """Get the prompt asking user about stock selection."""
        lines = []
        lines.append("=" * 55)
        lines.append("DAILY STOCK CONFIGURATION")
        lines.append("=" * 55)
        lines.append("")
        lines.append(f"{self.user_name}, before we continue:")
        lines.append("")
        lines.append("1. Are there any stocks you want me to SKIP today?")
        lines.append(f"   Currently monitoring: {', '.join(self.service_config.symbols)}")
        if self.service_config.excluded_symbols:
            lines.append(f"   Already excluded: {', '.join(self.service_config.excluded_symbols)}")
        lines.append("")
        lines.append("2. Any NEW stocks to ADD to the watchlist?")
        lines.append("   (I'll analyze historical data to get them ready)")
        lines.append("")
        lines.append("Commands:")
        lines.append("  • 'skip SPY' or 'exclude SPY, QQQ'")
        lines.append("  • 'add ARKQ' or 'watch ARKK, ARKQ'")
        lines.append("  • 'ready' or 'go' to proceed")
        lines.append("")
        
        return "\n".join(lines)
    
    # =========================================================================
    # COMMAND PROCESSING
    # =========================================================================
    
    def process_command(self, command: str) -> str:
        """
        Process a user command.
        
        Returns response string.
        """
        command = command.strip()
        cmd_upper = command.upper()
        
        # Status commands
        if cmd_upper in ['STATUS', 'STATS', 'HOW ARE WE DOING']:
            return self._get_status_report()
        
        # Briefing
        if cmd_upper in ['BRIEFING', 'MORNING', 'BRIEF']:
            return self.get_morning_briefing()
        
        # Pause/Resume
        if cmd_upper == 'PAUSE':
            self.service.pause()
            return "Service PAUSED. Predictions and trading suspended. Say 'resume' to continue."
        
        if cmd_upper == 'RESUME':
            self.service.resume()
            return "Service RESUMED. Predictions and trading active."
        
        # Stop
        if cmd_upper in ['STOP', 'QUIT', 'EXIT']:
            summary = self._get_end_of_day_summary()
            self.stop()
            return summary
        
        # Trading mode
        if 'LEARNING' in cmd_upper and 'MODE' in cmd_upper:
            self.service.set_trading_mode(TradingMode.LEARNING_ONLY)
            return "Switched to LEARNING mode. Bot will predict and learn but NOT trade."
        
        if 'PAPER' in cmd_upper and ('TRADE' in cmd_upper or 'MODE' in cmd_upper):
            self.service.set_trading_mode(TradingMode.PAPER_TRADING)
            return "Switched to PAPER TRADING mode. Bot will execute simulated trades."
        
        if 'LIVE' in cmd_upper and ('TRADE' in cmd_upper or 'MODE' in cmd_upper):
            return self._confirm_live_trading()
        
        if cmd_upper == 'CONFIRM LIVE TRADING':
            self.service.set_trading_mode(TradingMode.LIVE_TRADING)
            return "⚠️ LIVE TRADING MODE ENABLED. Real money will be used."
        
        # Stock management
        if any(word in cmd_upper for word in ['SKIP', 'EXCLUDE', 'OMIT', "DON'T TRADE", 'DONT TRADE']):
            return self._process_exclude_command(command)
        
        if any(word in cmd_upper for word in ['ADD', 'WATCH', 'INCLUDE', 'MONITOR']):
            return self._process_add_command(command)
        
        if cmd_upper in ['READY', 'GO', 'START', 'BEGIN', 'YES']:
            if not self.session_started:
                self.start()
                return "Background service started. I'm now monitoring markets and making predictions."
            return "Already running. Say 'status' to see current stats."
        
        # Alerts
        if cmd_upper in ['ALERTS', 'NOTIFICATIONS']:
            return self._get_alerts()
        
        if cmd_upper in ['CLEAR ALERTS', 'MARK READ']:
            for alert in self.alerts:
                alert['read'] = True
            return "All alerts marked as read."
        
        # Recent activity
        if cmd_upper in ['RECENT', 'ACTIVITY', 'LOG']:
            return self._get_recent_activity()
        
        # Help
        if cmd_upper in ['HELP', '?']:
            return self._get_help()
        
        # Unknown command
        return f"I didn't understand '{command}'. Say 'help' for available commands."
    
    def _process_exclude_command(self, command: str) -> str:
        """Process exclude/skip command."""
        import re
        
        # Extract symbols
        cmd_clean = command.upper()
        for word in ['SKIP', 'EXCLUDE', 'OMIT', "DON'T TRADE", 'DONT TRADE', 'TODAY', 'PLEASE']:
            cmd_clean = cmd_clean.replace(word, '')
        
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd_clean)
        exclude_words = {'AND', 'OR', 'THE', 'FROM', 'MY', 'ANY', 'ALL'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        if not symbols:
            return "I couldn't identify which symbols to exclude. Please specify like: 'skip SPY, QQQ'"
        
        excluded = []
        for symbol in symbols:
            if symbol in self.service_config.symbols:
                self.service.exclude_symbol(symbol)
                excluded.append(symbol)
        
        if excluded:
            return f"Got it. I will NOT trade these today: {', '.join(excluded)}\n(Still monitoring for learning purposes)"
        else:
            return f"Those symbols aren't in the current watchlist: {', '.join(symbols)}"
    
    def _process_add_command(self, command: str) -> str:
        """Process add/watch command."""
        import re
        
        cmd_clean = command.upper()
        for word in ['ADD', 'WATCH', 'INCLUDE', 'MONITOR', 'TO', 'LIST', 'PLEASE', 'THE']:
            cmd_clean = cmd_clean.replace(word, '')
        
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd_clean)
        exclude_words = {'AND', 'OR', 'FROM', 'MY'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        if not symbols:
            return "I couldn't identify which symbols to add. Please specify like: 'add ARKQ, ARKK'"
        
        responses = []
        for symbol in symbols:
            if symbol in self.service_config.symbols:
                responses.append(f"  • {symbol}: Already monitoring")
                continue
            
            responses.append(f"  • {symbol}: Analyzing historical data...")
            
            # This happens in background but we give immediate feedback
            result = self.service.add_symbol(symbol, train_first=True)
            
            if result.get('success'):
                preds = result.get('predictions', 0)
                acc = result.get('accuracy', 0)
                ready = result.get('ready_to_trade', False)
                
                if ready:
                    responses.append(f"    ✓ Ready to trade ({preds} predictions, {acc:.1%} accuracy)")
                else:
                    responses.append(f"    ◐ Learning mode ({preds} predictions, {acc:.1%} accuracy)")
            else:
                responses.append(f"    ✗ Failed: {result.get('reason', 'Unknown error')}")
        
        return "Processing:\n" + "\n".join(responses)
    
    def _confirm_live_trading(self) -> str:
        """Confirm before enabling live trading."""
        lines = []
        lines.append("⚠️  WARNING: LIVE TRADING MODE  ⚠️")
        lines.append("")
        lines.append("This will use REAL MONEY to execute trades.")
        lines.append("The bot will automatically buy and sell based on predictions.")
        lines.append("")
        lines.append("Current safeguards:")
        lines.append(f"  • Max position size: {self.service_config.max_position_pct*100:.0f}%")
        lines.append(f"  • Max daily loss: {self.service_config.max_daily_loss_pct*100:.1f}%")
        lines.append(f"  • Min confidence to trade: {self.service_config.min_confidence_to_trade*100:.0f}%")
        lines.append("")
        lines.append("To confirm, say: 'CONFIRM LIVE TRADING'")
        lines.append("To cancel, say anything else.")
        
        return "\n".join(lines)
    
    def _get_status_report(self) -> str:
        """Get current status report."""
        status = self.service.get_status()
        daily = status.get('daily_stats', {})
        account = status.get('account', {})
        
        lines = []
        lines.append("=" * 55)
        lines.append("CURRENT STATUS")
        lines.append("=" * 55)
        lines.append(f"Service State: {status.get('state', 'unknown')}")
        lines.append(f"Trading Mode: {status.get('trading_mode', 'unknown')}")
        lines.append(f"Market Open: {'Yes' if status.get('market_open') else 'No'}")
        lines.append(f"Active Symbols: {status.get('active_symbols', 0)}")
        lines.append("")
        
        lines.append("Today's Activity:")
        lines.append(f"  Predictions Made: {daily.get('predictions_made', 0)}")
        lines.append(f"  Verified: {daily.get('predictions_verified', 0)}")
        
        accuracy = daily.get('accuracy', 0)
        lines.append(f"  Accuracy: {accuracy:.1%}" if daily.get('predictions_verified', 0) > 0 else "  Accuracy: N/A")
        
        lines.append(f"  Trade Signals: {daily.get('signals_generated', 0)}")
        lines.append(f"  Trades Executed: {daily.get('trades_executed', 0)}")
        lines.append("")
        
        daily_pl = daily.get('daily_pl', 0)
        daily_pl_pct = daily.get('daily_pl_pct', 0)
        pl_sign = '+' if daily_pl >= 0 else ''
        
        lines.append(f"Portfolio: ${account.get('equity', 0):,.2f}")
        lines.append(f"Today's P&L: {pl_sign}${daily_pl:,.2f} ({pl_sign}{daily_pl_pct:.2f}%)")
        
        return "\n".join(lines)
    
    def _get_end_of_day_summary(self) -> str:
        """Generate end of day summary."""
        status = self.service.get_status()
        daily = status.get('daily_stats', {})
        account = status.get('account', {})
        
        lines = []
        lines.append("=" * 55)
        lines.append("END OF DAY SUMMARY")
        lines.append("=" * 55)
        lines.append("")
        
        daily_pl = daily.get('daily_pl', 0)
        daily_pl_pct = daily.get('daily_pl_pct', 0)
        pl_sign = '+' if daily_pl >= 0 else ''
        
        lines.append(f"Final Portfolio Value: ${account.get('equity', 0):,.2f}")
        lines.append(f"Day's P&L: {pl_sign}${daily_pl:,.2f} ({pl_sign}{daily_pl_pct:.2f}%)")
        lines.append("")
        
        lines.append("Learning Performance:")
        lines.append(f"  Predictions Made: {daily.get('predictions_made', 0)}")
        lines.append(f"  Predictions Verified: {daily.get('predictions_verified', 0)}")
        
        accuracy = daily.get('accuracy', 0)
        if daily.get('predictions_verified', 0) > 0:
            lines.append(f"  Accuracy: {accuracy:.1%}")
        
        lines.append("")
        lines.append("Trading Activity:")
        lines.append(f"  Signals Generated: {daily.get('signals_generated', 0)}")
        lines.append(f"  Trades Executed: {daily.get('trades_executed', 0)}")
        lines.append("")
        
        lines.append(f"Session ended. See you tomorrow, {self.user_name}!")
        
        return "\n".join(lines)
    
    def _get_alerts(self) -> str:
        """Get pending alerts."""
        unread = [a for a in self.alerts if not a['read']]
        
        if not unread:
            return "No unread alerts."
        
        lines = []
        lines.append(f"🔔 {len(unread)} Alerts:")
        lines.append("")
        
        for alert in unread[-10:]:
            lines.append(f"[{alert['timestamp'][11:19]}] {alert['type']}")
            lines.append(f"  {alert['data']}")
            lines.append("")
        
        lines.append("Say 'clear alerts' to mark all as read.")
        
        return "\n".join(lines)
    
    def _get_recent_activity(self) -> str:
        """Get recent activity log."""
        recent = self.event_log[-20:]
        
        if not recent:
            return "No recent activity."
        
        lines = []
        lines.append("Recent Activity (last 20 events):")
        lines.append("")
        
        for event in recent:
            time_str = event['timestamp'][11:19]
            event_type = event['type']
            data = event['data']
            
            if event_type == 'prediction':
                lines.append(f"[{time_str}] PRED: {data['symbol']} {data['horizon']} → {data['direction']} ({data['confidence']:.0%})")
            elif event_type == 'verification':
                result = '✓' if data.get('was_correct') else '✗'
                lines.append(f"[{time_str}] VERIFY: {data['symbol']} {result} (actual: {data.get('actual_change_pct', 0):+.2f}%)")
            elif event_type == 'trade_signal':
                lines.append(f"[{time_str}] SIGNAL: {data['action'].upper()} {data['symbol']} ({data['confidence']:.0%})")
            elif event_type == 'trade_executed':
                lines.append(f"[{time_str}] TRADE: {data['action'].upper()} {data['quantity']} {data['symbol']}")
        
        return "\n".join(lines)
    
    def _get_help(self) -> str:
        """Get help text."""
        return """
Available Commands:
==================

STATUS / STATS     - Show current status and today's performance
BRIEFING          - Show morning briefing
ALERTS            - Show pending alerts
RECENT / LOG      - Show recent activity

PAUSE             - Pause predictions and trading
RESUME            - Resume predictions and trading
STOP              - Stop service and show end-of-day summary

SKIP [symbols]    - Exclude symbols from trading today
ADD [symbols]     - Add new symbols to watchlist

LEARNING MODE     - Switch to learning only (no trades)
PAPER TRADING     - Switch to paper trading (simulated)
LIVE TRADING      - Switch to live trading (real money - requires confirmation)

HELP / ?          - Show this help message
"""


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """Main entry point for interactive session."""
    
    # Initialize bot
    bot = UnifiedTradingBot(
        symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"],
        user_name="Clifford"
    )
    
    # Show morning briefing
    print(bot.get_morning_briefing())
    print(bot.get_stock_prompt())
    
    # Main interaction loop
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            response = bot.process_command(user_input)
            print(f"\n{response}")
            
            # Check if we should exit
            if bot.service.state == ServiceState.STOPPED:
                break
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
            if bot.service.state != ServiceState.STOPPED:
                print(bot._get_end_of_day_summary())
                bot.stop()
            break
        except Exception as e:
            print(f"\nError: {e}")
            logger.error(f"Error in main loop: {e}")


if __name__ == "__main__":
    main()
