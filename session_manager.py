"""
Trading Session Manager
========================

Manages the daily trading workflow with human oversight.

Morning Flow:
1. Review overnight news for portfolio stocks
2. Summarize yesterday's P&L
3. Present today's goals and game plan
4. Ask user about stocks to add/remove
5. Fast-train any new stocks on historical data
6. Begin trading cycle

The bot doesn't just run - it collaborates with the trader.

Author: Claude AI
Date: November 29, 2025
"""

import json
from datetime import datetime, timedelta, time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum

# Alpaca imports
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from core.learning_trader import (
    LearningTrader,
    PredictionDatabase,
    MarketDataBatcher,
    StockLearningProfile
)
from core.historical_trainer import HistoricalTrainer
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class SessionState(Enum):
    """Trading session states."""
    NOT_STARTED = "not_started"
    BRIEFING = "briefing"
    AWAITING_INPUT = "awaiting_input"
    PREPARING = "preparing"
    TRADING = "trading"
    PAUSED = "paused"
    CLOSED = "closed"


@dataclass
class PortfolioPosition:
    """A position in the portfolio."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pl: float
    unrealized_pl_pct: float
    today_pl: float
    today_pl_pct: float


@dataclass 
class DailyPerformance:
    """Yesterday's performance summary."""
    date: str
    starting_equity: float
    ending_equity: float
    net_pl: float
    net_pl_pct: float
    trades_made: int
    winning_trades: int
    losing_trades: int
    biggest_winner: Optional[Tuple[str, float]] = None
    biggest_loser: Optional[Tuple[str, float]] = None
    predictions_made: int = 0
    prediction_accuracy: float = 0.0


@dataclass
class TradingGoal:
    """A trading goal for the day."""
    description: str
    target_value: float
    current_value: float = 0.0
    achieved: bool = False


@dataclass
class SessionConfig:
    """Configuration for today's trading session."""
    # Stocks to actively trade
    active_symbols: List[str] = field(default_factory=list)
    
    # Stocks to watch but not trade yet
    watchlist: List[str] = field(default_factory=list)
    
    # Stocks explicitly excluded today
    excluded_symbols: List[str] = field(default_factory=list)
    
    # Risk parameters for today
    max_position_size_pct: float = 0.10  # 10% max per position
    max_daily_loss_pct: float = 0.02     # 2% max daily loss
    max_trades_today: int = 20
    
    # Goals
    goals: List[TradingGoal] = field(default_factory=list)


class NewsAnalyzer:
    """Fetches and summarizes relevant news."""
    
    def __init__(self, api_key: str = None, api_secret: str = None):
        self.api_key = api_key or config.get('alpaca.api_key')
        self.api_secret = api_secret or config.get('alpaca.api_secret')
    
    def get_overnight_news(self, symbols: List[str], hours: int = 16) -> Dict[str, List[Dict]]:
        """
        Get news from overnight/pre-market for given symbols.
        
        Returns dict of symbol -> list of news items
        """
        # TODO: Implement Alpaca news API integration
        # For now, return placeholder
        news = {}
        for symbol in symbols:
            news[symbol] = []
        
        logger.info(f"Fetched news for {len(symbols)} symbols")
        return news
    
    def summarize_news(self, news: Dict[str, List[Dict]]) -> str:
        """Generate a brief summary of overnight news."""
        summary_parts = []
        
        for symbol, items in news.items():
            if items:
                summary_parts.append(f"**{symbol}**: {len(items)} news items")
                # Add headlines
                for item in items[:3]:  # Top 3
                    headline = item.get('headline', 'No headline')
                    summary_parts.append(f"  - {headline}")
        
        if not summary_parts:
            return "No significant overnight news for your portfolio stocks."
        
        return "\n".join(summary_parts)


class TradingSessionManager:
    """
    Manages the complete trading day workflow.
    
    Handles:
    - Morning briefing
    - User interaction for stock selection
    - Fast-training new stocks
    - Running the trading cycle
    - End of day summary
    """
    
    def __init__(
        self,
        default_symbols: List[str] = None,
        api_key: str = None,
        api_secret: str = None,
        data_dir: str = "data",
        paper: bool = True
    ):
        self.api_key = api_key or config.get('alpaca.api_key')
        self.api_secret = api_secret or config.get('alpaca.api_secret')
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default watchlist
        self.default_symbols = default_symbols or [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
            "META", "NVDA", "AMD", "SPY", "QQQ"
        ]
        
        # Alpaca clients
        self.trading_client = TradingClient(
            api_key=self.api_key,
            secret_key=self.api_secret,
            paper=paper
        )
        
        self.data_client = StockHistoricalDataClient(
            api_key=self.api_key,
            secret_key=self.api_secret
        )
        
        # Components
        self.news_analyzer = NewsAnalyzer(self.api_key, self.api_secret)
        self.prediction_db = PredictionDatabase(str(self.data_dir / "predictions.db"))
        
        # Session state
        self.state = SessionState.NOT_STARTED
        self.session_config = SessionConfig()
        self.session_start_time: Optional[datetime] = None
        
        # Learning trader (initialized after user confirms stocks)
        self.trader: Optional[LearningTrader] = None
        
        # Historical trainer for new stocks
        self.historical_trainer: Optional[HistoricalTrainer] = None
        
        # Performance tracking
        self.trades_today: List[Dict] = []
        self.starting_equity: float = 0
        
        logger.info("TradingSessionManager initialized")
    
    # =========================================================================
    # PORTFOLIO & ACCOUNT INFO
    # =========================================================================
    
    def get_account_info(self) -> Dict[str, Any]:
        """Get current account information."""
        try:
            account = self.trading_client.get_account()
            return {
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'daily_pl': float(account.equity) - float(account.last_equity),
                'daily_pl_pct': (float(account.equity) / float(account.last_equity) - 1) * 100 if float(account.last_equity) > 0 else 0
            }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
            return {}
    
    def get_current_positions(self) -> List[PortfolioPosition]:
        """Get all current positions."""
        try:
            positions = self.trading_client.get_all_positions()
            
            result = []
            for pos in positions:
                result.append(PortfolioPosition(
                    symbol=pos.symbol,
                    quantity=float(pos.qty),
                    avg_cost=float(pos.avg_entry_price),
                    current_price=float(pos.current_price),
                    market_value=float(pos.market_value),
                    unrealized_pl=float(pos.unrealized_pl),
                    unrealized_pl_pct=float(pos.unrealized_plpc) * 100,
                    today_pl=float(pos.unrealized_intraday_pl),
                    today_pl_pct=float(pos.unrealized_intraday_plpc) * 100
                ))
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_yesterday_performance(self) -> DailyPerformance:
        """Calculate yesterday's trading performance."""
        try:
            account = self.trading_client.get_account()
            
            # Get yesterday's orders
            yesterday = datetime.now().date() - timedelta(days=1)
            
            request = GetOrdersRequest(
                status=QueryOrderStatus.CLOSED,
                after=datetime.combine(yesterday, time.min),
                until=datetime.combine(yesterday, time.max)
            )
            
            orders = self.trading_client.get_orders(filter=request)
            
            # Calculate stats
            winning = 0
            losing = 0
            biggest_winner = None
            biggest_loser = None
            
            # Get prediction stats from DB
            pred_stats = self.prediction_db.get_prediction_stats(days=1)
            total_preds = sum(s.get('total', 0) for s in pred_stats.values())
            correct_preds = sum(s.get('correct', 0) for s in pred_stats.values())
            
            return DailyPerformance(
                date=yesterday.isoformat(),
                starting_equity=float(account.last_equity),
                ending_equity=float(account.equity),
                net_pl=float(account.equity) - float(account.last_equity),
                net_pl_pct=(float(account.equity) / float(account.last_equity) - 1) * 100 if float(account.last_equity) > 0 else 0,
                trades_made=len(orders),
                winning_trades=winning,
                losing_trades=losing,
                biggest_winner=biggest_winner,
                biggest_loser=biggest_loser,
                predictions_made=total_preds,
                prediction_accuracy=correct_preds / total_preds if total_preds > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Error getting yesterday's performance: {e}")
            return DailyPerformance(
                date=(datetime.now().date() - timedelta(days=1)).isoformat(),
                starting_equity=0, ending_equity=0, net_pl=0, net_pl_pct=0,
                trades_made=0, winning_trades=0, losing_trades=0
            )
    
    # =========================================================================
    # STOCK PROFILE MANAGEMENT
    # =========================================================================
    
    def get_stock_readiness(self, symbol: str) -> Dict[str, Any]:
        """Check if a stock has been trained and is ready to trade."""
        profile = self.prediction_db.get_stock_profile(symbol)
        
        if profile is None:
            return {
                'symbol': symbol,
                'ready': False,
                'trained': False,
                'predictions': 0,
                'accuracy': 0,
                'reason': 'Never trained - needs historical analysis'
            }
        
        min_predictions = 100
        min_accuracy = 0.52
        
        ready = (
            profile.total_predictions >= min_predictions and
            profile.overall_accuracy >= min_accuracy
        )
        
        return {
            'symbol': symbol,
            'ready': ready,
            'trained': True,
            'predictions': profile.total_predictions,
            'accuracy': profile.overall_accuracy,
            'accuracy_1h': profile.accuracy_1h,
            'accuracy_eod': profile.accuracy_eod,
            'accuracy_next_day': profile.accuracy_next_day,
            'reason': 'Ready to trade' if ready else f'Need more predictions or accuracy (have {profile.total_predictions}, {profile.overall_accuracy:.1%})'
        }
    
    def fast_train_stock(self, symbol: str, years: int = 2) -> Dict[str, Any]:
        """
        Fast-train a new stock on historical data.
        
        This runs the historical trainer on just this symbol to get it
        ready for trading quickly.
        """
        logger.info(f"Fast-training {symbol} on {years} years of data...")
        
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=years*365)).strftime("%Y-%m-%d")
        
        # Create trainer for just this symbol
        trainer = HistoricalTrainer(
            symbols=[symbol],
            api_key=self.api_key,
            api_secret=self.api_secret,
            db_path=str(self.data_dir / "predictions.db")
        )
        
        # Train
        results = trainer.train_on_historical(
            start_date=start_date,
            end_date=end_date,
            prediction_interval=1,
            verbose=True
        )
        
        # Get updated profile
        profile = trainer.profiles.get(symbol)
        
        if profile:
            return {
                'symbol': symbol,
                'success': True,
                'predictions': profile.total_predictions,
                'accuracy': profile.overall_accuracy,
                'accuracy_1h': profile.accuracy_1h,
                'accuracy_eod': profile.accuracy_eod,
                'accuracy_next_day': profile.accuracy_next_day,
                'top_features': sorted(
                    profile.feature_weights.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5],
                'ready_to_trade': profile.total_predictions >= 100 and profile.overall_accuracy >= 0.52
            }
        else:
            return {
                'symbol': symbol,
                'success': False,
                'reason': 'Training failed - check data availability'
            }
    
    # =========================================================================
    # MORNING BRIEFING
    # =========================================================================
    
    def generate_morning_briefing(self, user_name: str = "Trader") -> str:
        """
        Generate the complete morning briefing.
        
        Returns formatted string for display/speech.
        """
        self.state = SessionState.BRIEFING
        
        briefing_parts = []
        
        # Header
        now = datetime.now()
        briefing_parts.append(f"Good morning, {user_name}!")
        briefing_parts.append(f"Today is {now.strftime('%A, %B %d, %Y')}.")
        briefing_parts.append("")
        
        # Account Summary
        account = self.get_account_info()
        if account:
            self.starting_equity = account['equity']
            briefing_parts.append("=" * 50)
            briefing_parts.append("ACCOUNT SUMMARY")
            briefing_parts.append("=" * 50)
            briefing_parts.append(f"Portfolio Value: ${account['equity']:,.2f}")
            briefing_parts.append(f"Cash Available: ${account['cash']:,.2f}")
            briefing_parts.append(f"Buying Power: ${account['buying_power']:,.2f}")
            briefing_parts.append("")
        
        # Yesterday's Performance
        yesterday = self.get_yesterday_performance()
        briefing_parts.append("=" * 50)
        briefing_parts.append("YESTERDAY'S PERFORMANCE")
        briefing_parts.append("=" * 50)
        pl_sign = "+" if yesterday.net_pl >= 0 else ""
        briefing_parts.append(f"Net P&L: {pl_sign}${yesterday.net_pl:,.2f} ({pl_sign}{yesterday.net_pl_pct:.2f}%)")
        briefing_parts.append(f"Trades Executed: {yesterday.trades_made}")
        if yesterday.predictions_made > 0:
            briefing_parts.append(f"Predictions Made: {yesterday.predictions_made}")
            briefing_parts.append(f"Prediction Accuracy: {yesterday.prediction_accuracy:.1%}")
        briefing_parts.append("")
        
        # Current Positions
        positions = self.get_current_positions()
        if positions:
            briefing_parts.append("=" * 50)
            briefing_parts.append("CURRENT POSITIONS")
            briefing_parts.append("=" * 50)
            
            for pos in sorted(positions, key=lambda x: abs(x.unrealized_pl), reverse=True):
                pl_sign = "+" if pos.unrealized_pl >= 0 else ""
                briefing_parts.append(
                    f"  {pos.symbol}: {pos.quantity} shares @ ${pos.avg_cost:.2f} "
                    f"| Now: ${pos.current_price:.2f} "
                    f"| P&L: {pl_sign}${pos.unrealized_pl:.2f} ({pl_sign}{pos.unrealized_pl_pct:.1f}%)"
                )
            briefing_parts.append("")
        
        # Stock Readiness
        all_symbols = list(set(self.default_symbols + [p.symbol for p in positions]))
        
        briefing_parts.append("=" * 50)
        briefing_parts.append("STOCK READINESS")
        briefing_parts.append("=" * 50)
        
        ready_stocks = []
        needs_training = []
        
        for symbol in sorted(all_symbols):
            readiness = self.get_stock_readiness(symbol)
            if readiness['ready']:
                ready_stocks.append(f"  ✓ {symbol}: {readiness['predictions']} predictions, {readiness['accuracy']:.1%} accuracy")
            elif readiness['trained']:
                ready_stocks.append(f"  ○ {symbol}: Learning ({readiness['predictions']} preds, {readiness['accuracy']:.1%})")
            else:
                needs_training.append(symbol)
        
        for line in ready_stocks:
            briefing_parts.append(line)
        
        if needs_training:
            briefing_parts.append(f"\n  Needs training: {', '.join(needs_training)}")
        
        briefing_parts.append("")
        
        # Overnight News Summary
        news = self.news_analyzer.get_overnight_news([p.symbol for p in positions])
        news_summary = self.news_analyzer.summarize_news(news)
        
        briefing_parts.append("=" * 50)
        briefing_parts.append("OVERNIGHT NEWS")
        briefing_parts.append("=" * 50)
        briefing_parts.append(news_summary)
        briefing_parts.append("")
        
        # Today's Default Plan
        briefing_parts.append("=" * 50)
        briefing_parts.append("TODAY'S GAME PLAN")
        briefing_parts.append("=" * 50)
        briefing_parts.append(f"• Monitor {len(all_symbols)} stocks")
        briefing_parts.append(f"• Make predictions every hour (3 horizons each)")
        briefing_parts.append(f"• Max position size: {self.session_config.max_position_size_pct*100:.0f}% of portfolio")
        briefing_parts.append(f"• Max daily loss limit: {self.session_config.max_daily_loss_pct*100:.1f}%")
        briefing_parts.append("")
        
        self.state = SessionState.AWAITING_INPUT
        
        return "\n".join(briefing_parts)
    
    def get_user_prompt(self) -> str:
        """Get the prompt to ask user about stock selection."""
        positions = self.get_current_positions()
        position_symbols = [p.symbol for p in positions]
        
        prompt_parts = []
        prompt_parts.append("=" * 50)
        prompt_parts.append("STOCK SELECTION")
        prompt_parts.append("=" * 50)
        prompt_parts.append("")
        prompt_parts.append("Before we begin the trading cycle, I need your input:")
        prompt_parts.append("")
        prompt_parts.append("1. Are there any stocks currently in your portfolio that")
        prompt_parts.append("   you'd like me to EXCLUDE from trading today?")
        prompt_parts.append(f"   Current positions: {', '.join(position_symbols) if position_symbols else 'None'}")
        prompt_parts.append("")
        prompt_parts.append("2. Are there any NEW stocks you'd like me to ADD to")
        prompt_parts.append("   today's watchlist? I'll analyze their historical data")
        prompt_parts.append("   before trading them.")
        prompt_parts.append("")
        prompt_parts.append("Please respond with:")
        prompt_parts.append("  • EXCLUDE: SYMBOL1, SYMBOL2 (stocks to skip today)")
        prompt_parts.append("  • ADD: SYMBOL1, SYMBOL2 (new stocks to watch)")
        prompt_parts.append("  • Or just say 'READY' to proceed with defaults")
        prompt_parts.append("")
        
        return "\n".join(prompt_parts)
    
    def process_user_response(self, response: str) -> Dict[str, Any]:
        """
        Process user's response about stock selection.
        
        Parses commands like:
        - "EXCLUDE: SPY, QQQ"
        - "ADD: ARKQ, ARKK"
        - "READY"
        - Or natural language like "don't trade SPY today and add ARKQ"
        
        Returns dict with parsed commands and any stocks needing training.
        """
        response = response.upper().strip()
        
        result = {
            'exclude': [],
            'add': [],
            'needs_training': [],
            'ready': False,
            'message': ''
        }
        
        # Check for simple READY
        if response == 'READY' or response == 'YES' or response == 'GO':
            result['ready'] = True
            result['message'] = "Proceeding with default configuration."
            return result
        
        # Parse EXCLUDE command
        if 'EXCLUDE:' in response or 'EXCLUDE ' in response or "DON'T TRADE" in response or 'DONT TRADE' in response or 'SKIP' in response or 'OMIT' in response:
            # Extract symbols after EXCLUDE or from natural language
            parts = response.replace('EXCLUDE:', '').replace('EXCLUDE ', '')
            parts = parts.replace("DON'T TRADE", '').replace('DONT TRADE', '')
            parts = parts.replace('SKIP', '').replace('OMIT', '')
            parts = parts.replace('ANY', '').replace('TODAY', '').replace('PLEASE', '')
            
            # Find potential stock symbols (1-5 uppercase letters)
            import re
            potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', parts)
            
            # Filter out common words
            exclude_words = {'AND', 'OR', 'THE', 'ADD', 'TO', 'FROM', 'MY', 'TODAY', 'ALSO'}
            result['exclude'] = [s for s in potential_symbols if s not in exclude_words]
        
        # Parse ADD command
        if 'ADD:' in response or 'ADD ' in response or 'WATCH' in response or 'INCLUDE' in response:
            parts = response.replace('ADD:', '').replace('ADD ', '')
            parts = parts.replace('WATCH', '').replace('INCLUDE', '')
            parts = parts.replace('TO', '').replace('LIST', '').replace('PLEASE', '')
            
            import re
            potential_symbols = re.findall(r'\b[A-Z]{1,5}\b', parts)
            
            exclude_words = {'AND', 'OR', 'THE', 'EXCLUDE', 'TO', 'FROM', 'MY', 'TODAY', 'ALSO', 'THIS'}
            add_symbols = [s for s in potential_symbols if s not in exclude_words and s not in result['exclude']]
            result['add'] = add_symbols
            
            # Check which need training
            for symbol in add_symbols:
                readiness = self.get_stock_readiness(symbol)
                if not readiness['trained']:
                    result['needs_training'].append(symbol)
        
        # Generate response message
        messages = []
        if result['exclude']:
            messages.append(f"I will EXCLUDE these from trading today: {', '.join(result['exclude'])}")
        if result['add']:
            messages.append(f"I will ADD these to the watchlist: {', '.join(result['add'])}")
        if result['needs_training']:
            messages.append(f"These need historical training first: {', '.join(result['needs_training'])}")
            messages.append("I'll analyze their data now...")
        
        result['ready'] = True
        result['message'] = '\n'.join(messages) if messages else "No changes requested."
        
        return result
    
    def prepare_session(self, exclude: List[str] = None, add: List[str] = None) -> str:
        """
        Prepare the trading session based on user input.
        
        - Excludes specified stocks
        - Adds new stocks (training if needed)
        - Initializes the learning trader
        
        Returns status message.
        """
        self.state = SessionState.PREPARING
        
        exclude = exclude or []
        add = add or []
        
        messages = []
        
        # Get current positions
        positions = self.get_current_positions()
        position_symbols = [p.symbol for p in positions]
        
        # Build active symbols list
        active_symbols = list(set(self.default_symbols + position_symbols + add))
        
        # Remove excluded
        active_symbols = [s for s in active_symbols if s not in exclude]
        
        # Store exclusions
        self.session_config.excluded_symbols = exclude
        self.session_config.watchlist = add
        
        # Train any new stocks
        for symbol in add:
            readiness = self.get_stock_readiness(symbol)
            if not readiness['trained']:
                messages.append(f"\nTraining {symbol} on historical data...")
                result = self.fast_train_stock(symbol)
                
                if result['success']:
                    messages.append(f"  ✓ {symbol} trained: {result['predictions']} predictions, {result['accuracy']:.1%} accuracy")
                    if result['ready_to_trade']:
                        messages.append(f"    → Ready to trade!")
                    else:
                        messages.append(f"    → Still learning, will make predictions but trade cautiously")
                else:
                    messages.append(f"  ✗ {symbol} training failed: {result.get('reason', 'Unknown error')}")
                    active_symbols.remove(symbol)
        
        # Store final active list
        self.session_config.active_symbols = active_symbols
        
        # Initialize the learning trader
        self.trader = LearningTrader(
            symbols=active_symbols,
            api_key=self.api_key,
            api_secret=self.api_secret,
            db_path=str(self.data_dir / "predictions.db"),
            learning_mode=True  # Start in learning mode
        )
        
        messages.append(f"\n{'='*50}")
        messages.append("SESSION PREPARED")
        messages.append(f"{'='*50}")
        messages.append(f"Active symbols: {', '.join(active_symbols)}")
        messages.append(f"Excluded today: {', '.join(exclude) if exclude else 'None'}")
        messages.append(f"Newly added: {', '.join(add) if add else 'None'}")
        
        return '\n'.join(messages)
    
    # =========================================================================
    # TRADING CYCLE
    # =========================================================================
    
    def start_trading_cycle(self) -> str:
        """Start the main trading cycle."""
        if self.trader is None:
            return "Error: Session not prepared. Run prepare_session() first."
        
        self.state = SessionState.TRADING
        self.session_start_time = datetime.now()
        
        message = []
        message.append("=" * 50)
        message.append("TRADING CYCLE STARTED")
        message.append("=" * 50)
        message.append(f"Time: {self.session_start_time.strftime('%H:%M:%S')}")
        message.append(f"Monitoring: {len(self.session_config.active_symbols)} stocks")
        message.append("")
        message.append("I will:")
        message.append("  • Make predictions every hour")
        message.append("  • Track 3 horizons: 1hr, EOD, next day")
        message.append("  • Verify predictions and learn from outcomes")
        message.append("  • Alert you to high-confidence opportunities")
        message.append("")
        message.append("Say 'STATUS' anytime for current stats")
        message.append("Say 'PAUSE' to pause trading")
        message.append("Say 'STOP' to end the session")
        
        return '\n'.join(message)
    
    def run_one_cycle(self) -> str:
        """
        Run one prediction cycle and return summary.
        
        This is called every hour (or on demand).
        """
        if self.trader is None:
            return "Error: Trader not initialized."
        
        if self.state != SessionState.TRADING:
            return "Error: Trading cycle not started."
        
        messages = []
        
        # Verify pending predictions first
        verified = self.trader.verify_predictions()
        if verified > 0:
            messages.append(f"Verified {verified} pending predictions")
        
        # Make new predictions
        predictions = self.trader.run_prediction_cycle()
        messages.append(f"Made {len(predictions)} new predictions")
        
        # Find high-confidence opportunities
        high_confidence = [p for p in predictions if p.confidence >= 0.7]
        
        if high_confidence:
            messages.append("\n🔔 HIGH CONFIDENCE SIGNALS:")
            for pred in sorted(high_confidence, key=lambda x: x.confidence, reverse=True)[:5]:
                direction = "📈 UP" if pred.predicted_direction.value == 'up' else "📉 DOWN"
                messages.append(
                    f"  {pred.symbol} [{pred.horizon.value}]: {direction} "
                    f"({pred.predicted_change_pct:+.2f}%) - {pred.confidence:.0%} confident"
                )
        
        # Check portfolio P&L
        account = self.get_account_info()
        if account and self.starting_equity > 0:
            current_pl = account['equity'] - self.starting_equity
            current_pl_pct = (account['equity'] / self.starting_equity - 1) * 100
            
            messages.append(f"\nToday's P&L: {'+'if current_pl >= 0 else ''}${current_pl:,.2f} ({current_pl_pct:+.2f}%)")
        
        return '\n'.join(messages)
    
    def get_status(self) -> str:
        """Get current session status."""
        if self.trader is None:
            return "Session not started."
        
        status = []
        status.append("=" * 50)
        status.append("SESSION STATUS")
        status.append("=" * 50)
        status.append(f"State: {self.state.value}")
        status.append(f"Active symbols: {len(self.session_config.active_symbols)}")
        
        if self.session_start_time:
            elapsed = datetime.now() - self.session_start_time
            status.append(f"Session duration: {elapsed}")
        
        # Prediction stats
        stats = self.prediction_db.get_prediction_stats(days=1)
        total_preds = sum(s.get('total', 0) for s in stats.values())
        correct = sum(s.get('correct', 0) for s in stats.values())
        accuracy = correct / total_preds if total_preds > 0 else 0
        
        status.append(f"\nToday's predictions: {total_preds}")
        status.append(f"Verified accuracy: {accuracy:.1%}")
        
        # Account status
        account = self.get_account_info()
        if account and self.starting_equity > 0:
            current_pl = account['equity'] - self.starting_equity
            current_pl_pct = (account['equity'] / self.starting_equity - 1) * 100
            status.append(f"\nPortfolio: ${account['equity']:,.2f}")
            status.append(f"Today's P&L: {'+'if current_pl >= 0 else ''}${current_pl:,.2f} ({current_pl_pct:+.2f}%)")
        
        # Readiness by stock
        status.append("\nStock readiness:")
        for symbol in self.session_config.active_symbols:
            readiness = self.get_stock_readiness(symbol)
            ready_str = "✓" if readiness['ready'] else "○"
            status.append(f"  {ready_str} {symbol}: {readiness['predictions']} preds, {readiness['accuracy']:.1%}")
        
        return '\n'.join(status)
    
    def pause_trading(self) -> str:
        """Pause the trading cycle."""
        self.state = SessionState.PAUSED
        return "Trading cycle PAUSED. Say 'RESUME' to continue."
    
    def resume_trading(self) -> str:
        """Resume the trading cycle."""
        self.state = SessionState.TRADING
        return "Trading cycle RESUMED."
    
    def end_session(self) -> str:
        """End the trading session and generate summary."""
        self.state = SessionState.CLOSED
        
        summary = []
        summary.append("=" * 50)
        summary.append("END OF SESSION SUMMARY")
        summary.append("=" * 50)
        
        # Final P&L
        account = self.get_account_info()
        if account and self.starting_equity > 0:
            final_pl = account['equity'] - self.starting_equity
            final_pl_pct = (account['equity'] / self.starting_equity - 1) * 100
            summary.append(f"Final Portfolio Value: ${account['equity']:,.2f}")
            summary.append(f"Session P&L: {'+'if final_pl >= 0 else ''}${final_pl:,.2f} ({final_pl_pct:+.2f}%)")
        
        # Prediction stats
        stats = self.prediction_db.get_prediction_stats(days=1)
        summary.append("\nPrediction Performance by Horizon:")
        for horizon, data in stats.items():
            acc = data.get('accuracy')
            acc_str = f"{acc:.1%}" if acc else "N/A"
            summary.append(f"  {horizon}: {data.get('total', 0)} predictions, {acc_str} accuracy")
        
        summary.append("\nSession ended. See you next trading day!")
        
        return '\n'.join(summary)


# =============================================================================
# INTERACTIVE SESSION
# =============================================================================

def run_interactive_session():
    """Run an interactive trading session with the user."""
    
    manager = TradingSessionManager(
        default_symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "SPY", "QQQ"],
        paper=True
    )
    
    # Get user name (could be from config)
    user_name = "Clifford"
    
    # Morning briefing
    print(manager.generate_morning_briefing(user_name))
    print(manager.get_user_prompt())
    
    # Get user input
    while True:
        user_input = input("\nYour response: ").strip()
        
        if not user_input:
            continue
        
        # Process response
        parsed = manager.process_user_response(user_input)
        print(f"\n{parsed['message']}")
        
        if parsed['ready']:
            # Prepare session
            print(manager.prepare_session(
                exclude=parsed['exclude'],
                add=parsed['add']
            ))
            break
    
    # Start trading
    print(manager.start_trading_cycle())
    
    # Main loop
    import time
    last_cycle = datetime.now()
    
    while manager.state in [SessionState.TRADING, SessionState.PAUSED]:
        # Check for user commands
        try:
            import sys
            import select
            
            # Non-blocking input check (Unix only)
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                command = input().strip().upper()
                
                if command == 'STATUS':
                    print(manager.get_status())
                elif command == 'PAUSE':
                    print(manager.pause_trading())
                elif command == 'RESUME':
                    print(manager.resume_trading())
                elif command == 'STOP':
                    print(manager.end_session())
                    break
                elif command == 'CYCLE':
                    print(manager.run_one_cycle())
        except:
            pass
        
        # Run cycle every hour
        if manager.state == SessionState.TRADING:
            if (datetime.now() - last_cycle) >= timedelta(hours=1):
                print(manager.run_one_cycle())
                last_cycle = datetime.now()
        
        time.sleep(60)  # Check every minute


if __name__ == "__main__":
    run_interactive_session()
