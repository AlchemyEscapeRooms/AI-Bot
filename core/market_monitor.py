"""
Live Market Monitor - Watches markets, makes predictions, and learns.

This module provides continuous market monitoring with:
- Real-time price tracking
- Prediction generation and tracking
- Learning from prediction outcomes
- Knowledge base accumulation
"""

import time
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import threading
import pandas as pd
import numpy as np

from data import MarketDataCollector
from ml_models import FeatureEngineer
from utils.logger import get_logger
from utils.database import Database

logger = get_logger(__name__)


@dataclass
class Prediction:
    """A market prediction with tracking info."""

    prediction_id: str
    timestamp: datetime
    symbol: str

    # Prediction details
    predicted_direction: str  # 'up', 'down', 'sideways'
    confidence: float  # 0-100%
    predicted_change_pct: float  # Expected % change
    timeframe: str  # '1h', '4h', '1d', etc.
    target_price: float

    # Signals that drove this prediction
    signals: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""

    # For tracking outcome
    entry_price: float = 0.0
    exit_price: float = 0.0
    actual_change_pct: float = 0.0
    was_correct: Optional[bool] = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        if self.resolved_at:
            data['resolved_at'] = self.resolved_at.isoformat()
        return data


class PredictionTracker:
    """Tracks predictions and their outcomes for learning."""

    def __init__(self):
        self.db = Database()
        self._ensure_tables()
        self.active_predictions: Dict[str, Prediction] = {}
        self.prediction_counter = 0

    def _ensure_tables(self):
        """Create prediction tracking tables."""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()

            # Use ai_predictions table (separate from ML predictions table)
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ai_predictions'")
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                # Create new ai_predictions table for AI learning system
                cursor.execute("""
                    CREATE TABLE ai_predictions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        prediction_id TEXT UNIQUE,
                        timestamp DATETIME NOT NULL,
                        symbol TEXT NOT NULL,
                        predicted_direction TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        predicted_change_pct REAL,
                        timeframe TEXT NOT NULL,
                        target_price REAL,
                        entry_price REAL,
                        exit_price REAL,
                        actual_change_pct REAL,
                        was_correct INTEGER,
                        resolved INTEGER DEFAULT 0,
                        resolved_at DATETIME,
                        signals TEXT,
                        reasoning TEXT
                    )
                """)

            # Learning insights table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME NOT NULL,
                    insight_type TEXT NOT NULL,
                    symbol TEXT,
                    signal_name TEXT,
                    accuracy REAL,
                    sample_size INTEGER,
                    notes TEXT
                )
            """)

            # Create indexes (safe to run multiple times)
            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_predictions_symbol
                    ON ai_predictions(symbol)
                """)
            except:
                pass

            try:
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_ai_predictions_resolved
                    ON ai_predictions(resolved)
                """)
            except:
                pass

            conn.commit()

    def generate_prediction_id(self) -> str:
        """Generate unique prediction ID."""
        self.prediction_counter += 1
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        return f"PRED-{timestamp}-{self.prediction_counter:04d}"

    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    def add_prediction(self, prediction: Prediction):
        """Add a new prediction to track."""
        self.active_predictions[prediction.prediction_id] = prediction

        # Convert signals to serializable format
        signals_serializable = self._convert_to_serializable(prediction.signals)

        # Save to database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ai_predictions (
                    prediction_id, timestamp, symbol, predicted_direction,
                    confidence, predicted_change_pct, timeframe, target_price,
                    entry_price, signals, reasoning, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (
                prediction.prediction_id,
                prediction.timestamp.isoformat(),
                prediction.symbol,
                prediction.predicted_direction,
                float(prediction.confidence),
                float(prediction.predicted_change_pct),
                prediction.timeframe,
                float(prediction.target_price),
                float(prediction.entry_price),
                json.dumps(signals_serializable),
                prediction.reasoning
            ))

        logger.info(f"Prediction added: {prediction.prediction_id} - {prediction.symbol} "
                   f"{prediction.predicted_direction} ({prediction.confidence:.1f}% confidence)")

    def resolve_prediction(self, prediction_id: str, exit_price: float):
        """Resolve a prediction with the actual outcome."""
        if prediction_id not in self.active_predictions:
            return

        prediction = self.active_predictions[prediction_id]
        prediction.exit_price = exit_price
        prediction.actual_change_pct = ((exit_price - prediction.entry_price) / prediction.entry_price) * 100
        prediction.resolved_at = datetime.now()
        prediction.resolved = True

        # Determine if prediction was correct
        if prediction.predicted_direction == 'up':
            prediction.was_correct = prediction.actual_change_pct > 0.1  # Small threshold
        elif prediction.predicted_direction == 'down':
            prediction.was_correct = prediction.actual_change_pct < -0.1
        else:  # sideways
            prediction.was_correct = abs(prediction.actual_change_pct) < 0.5

        # Update database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE ai_predictions SET
                    exit_price = ?,
                    actual_change_pct = ?,
                    was_correct = ?,
                    resolved = 1,
                    resolved_at = ?
                WHERE prediction_id = ?
            """, (
                exit_price,
                prediction.actual_change_pct,
                1 if prediction.was_correct else 0,
                prediction.resolved_at.isoformat(),
                prediction_id
            ))

        # Remove from active
        del self.active_predictions[prediction_id]

        result = "CORRECT" if prediction.was_correct else "WRONG"
        logger.info(f"Prediction resolved: {prediction_id} - {result} "
                   f"(predicted {prediction.predicted_direction}, actual {prediction.actual_change_pct:.2f}%)")

    def get_accuracy_stats(self, symbol: str = None, days: int = 30) -> Dict[str, Any]:
        """Get prediction accuracy statistics."""
        with self.db.get_connection() as conn:
            query = """
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                    AVG(confidence) as avg_confidence,
                    AVG(CASE WHEN was_correct = 1 THEN confidence ELSE NULL END) as avg_correct_confidence,
                    AVG(CASE WHEN was_correct = 0 THEN confidence ELSE NULL END) as avg_wrong_confidence
                FROM ai_predictions
                WHERE resolved = 1
                AND timestamp >= datetime('now', ?)
            """
            params = [f'-{days} days']

            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)

            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()

            if row and row[0] > 0:
                return {
                    'total_predictions': row[0],
                    'correct': row[1],
                    'accuracy': (row[1] / row[0]) * 100 if row[0] > 0 else 0,
                    'avg_confidence': row[2] or 0,
                    'avg_correct_confidence': row[3] or 0,
                    'avg_wrong_confidence': row[4] or 0
                }

            return {'total_predictions': 0, 'accuracy': 0}

    def get_signal_performance(self, days: int = 30) -> pd.DataFrame:
        """Analyze which signals are most accurate."""
        with self.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT prediction_id, signals, was_correct
                FROM ai_predictions
                WHERE resolved = 1
                AND timestamp >= datetime('now', ?)
            """, conn, params=[f'-{days} days'])

        if df.empty:
            return pd.DataFrame()

        # Parse signals and analyze
        signal_stats = {}

        for _, row in df.iterrows():
            try:
                signals = json.loads(row['signals']) if row['signals'] else {}
                was_correct = row['was_correct']

                for signal_name, signal_value in signals.items():
                    if signal_name not in signal_stats:
                        signal_stats[signal_name] = {'correct': 0, 'total': 0, 'values': []}

                    signal_stats[signal_name]['total'] += 1
                    signal_stats[signal_name]['values'].append(signal_value)
                    if was_correct:
                        signal_stats[signal_name]['correct'] += 1
            except:
                continue

        # Convert to DataFrame
        results = []
        for signal_name, stats in signal_stats.items():
            results.append({
                'signal': signal_name,
                'accuracy': (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0,
                'uses': stats['total'],
                'avg_value': np.mean(stats['values']) if stats['values'] else 0
            })

        return pd.DataFrame(results).sort_values('accuracy', ascending=False)


class MarketMonitor:
    """
    Live market monitoring with predictions and learning.

    Watches the market in real-time, generates predictions,
    tracks their accuracy, and learns from outcomes.
    """

    def __init__(self, symbols: List[str] = None):
        self.symbols = symbols or ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']
        self.market_data = MarketDataCollector()
        self.feature_engineer = FeatureEngineer()
        self.prediction_tracker = PredictionTracker()

        self.is_running = False
        self.monitor_thread = None
        self.update_interval = 60  # seconds

        # Current market state
        self.current_prices: Dict[str, float] = {}
        self.price_history: Dict[str, List[Dict]] = {s: [] for s in self.symbols}
        self.last_predictions: Dict[str, Prediction] = {}

        # Learning parameters (adjusted based on performance)
        self.signal_weights = {
            'momentum_20d': 1.0,
            'rsi': 1.0,
            'macd_signal': 1.0,
            'volume_ratio': 1.0,
            'price_vs_sma20': 1.0,
            'bollinger_position': 1.0
        }

        logger.info(f"MarketMonitor initialized for: {', '.join(self.symbols)}")

    def start(self):
        """Start the market monitor."""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Market monitor started")

    def stop(self):
        """Stop the market monitor."""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Market monitor stopped")

    def _monitor_loop(self):
        """Main monitoring loop."""
        cycle_count = 0
        while self.is_running:
            try:
                cycle_count += 1
                self._update_prices()
                self._check_predictions()
                self._generate_predictions()

                # Run learning every 10 cycles (10 minutes)
                if cycle_count % 10 == 0:
                    self._learn_from_history()

                # Log status every 5 cycles (5 minutes)
                if cycle_count % 5 == 0:
                    active_count = len(self.prediction_tracker.active_predictions)
                    stats = self.prediction_tracker.get_accuracy_stats(days=7)
                    logger.info(f"Live Monitor: {active_count} active predictions, "
                               f"{stats['total_predictions']} total, {stats['accuracy']:.1f}% accuracy")

            except Exception as e:
                logger.error(f"Monitor error: {e}")

            time.sleep(self.update_interval)

    def _update_prices(self):
        """Update current prices for all symbols."""
        for symbol in self.symbols:
            try:
                quote = self.market_data.get_real_time_quote(symbol)
                if quote and quote.get('price'):
                    price = quote['price']
                    self.current_prices[symbol] = price

                    # Add to history
                    self.price_history[symbol].append({
                        'timestamp': datetime.now(),
                        'price': price,
                        'volume': quote.get('volume', 0)
                    })

                    # Keep last 1000 entries
                    if len(self.price_history[symbol]) > 1000:
                        self.price_history[symbol] = self.price_history[symbol][-1000:]

            except Exception as e:
                logger.debug(f"Error getting price for {symbol}: {e}")

    def _check_predictions(self):
        """Check and resolve active predictions."""
        to_resolve = []

        for pred_id, prediction in list(self.prediction_tracker.active_predictions.items()):
            # Check if timeframe has elapsed
            if prediction.timeframe == '1h':
                elapsed = (datetime.now() - prediction.timestamp).total_seconds() / 3600
                if elapsed >= 1:
                    to_resolve.append(pred_id)
            elif prediction.timeframe == '4h':
                elapsed = (datetime.now() - prediction.timestamp).total_seconds() / 3600
                if elapsed >= 4:
                    to_resolve.append(pred_id)
            elif prediction.timeframe == '1d':
                elapsed = (datetime.now() - prediction.timestamp).total_seconds() / 86400
                if elapsed >= 1:
                    to_resolve.append(pred_id)
            else:
                # Default: resolve after 1 hour for unknown timeframes (paper, live, etc.)
                elapsed = (datetime.now() - prediction.timestamp).total_seconds() / 3600
                if elapsed >= 1:
                    to_resolve.append(pred_id)

        # Resolve predictions
        for pred_id in to_resolve:
            prediction = self.prediction_tracker.active_predictions[pred_id]
            current_price = self.current_prices.get(prediction.symbol, prediction.entry_price)

            # Log before resolving (resolve_prediction already logs the result)
            logger.info(f"Live Monitor: Resolving prediction {pred_id} for {prediction.symbol} "
                       f"(entry: ${prediction.entry_price:.2f}, current: ${current_price:.2f})")

            self.prediction_tracker.resolve_prediction(pred_id, current_price)

    def _generate_predictions(self):
        """Generate new predictions for each symbol."""
        for symbol in self.symbols:
            try:
                # Skip if we have a recent prediction
                if symbol in self.last_predictions:
                    last_pred = self.last_predictions[symbol]
                    elapsed = (datetime.now() - last_pred.timestamp).total_seconds()
                    if elapsed < 900:  # 15 minutes
                        continue

                # Get data for analysis (use start_date/end_date, not period)
                from datetime import timedelta
                end_date = datetime.now().strftime('%Y-%m-%d')
                start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
                df = self.market_data.get_historical_data(symbol, start_date, end_date)

                if df.empty or len(df) < 20:
                    logger.debug(f"Insufficient data for {symbol}: got {len(df)} rows")
                    continue

                # Generate prediction
                prediction = self._analyze_and_predict(symbol, df)

                if prediction and prediction.confidence >= 60:  # Only track high-confidence predictions
                    self.prediction_tracker.add_prediction(prediction)
                    self.last_predictions[symbol] = prediction
                    logger.info(f"Live Monitor: New prediction for {symbol} - "
                               f"{prediction.predicted_direction.upper()} {prediction.confidence:.1f}% confidence")

            except Exception as e:
                logger.warning(f"Error generating prediction for {symbol}: {e}")

    def _analyze_and_predict(self, symbol: str, df: pd.DataFrame) -> Optional[Prediction]:
        """Analyze market data and generate a prediction."""

        current_price = df['close'].iloc[-1]

        # Calculate signals
        signals = {}

        # 1. Momentum (20-day)
        if len(df) >= 20:
            momentum = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
            signals['momentum_20d'] = momentum

        # 2. RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        signals['rsi'] = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

        # 3. MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        macd = ema12 - ema26
        signal_line = macd.ewm(span=9).mean()
        macd_signal = macd.iloc[-1] - signal_line.iloc[-1]
        signals['macd_signal'] = macd_signal

        # 4. Volume ratio
        if 'volume' in df.columns and len(df) >= 20:
            avg_volume = df['volume'].tail(20).mean()
            current_volume = df['volume'].iloc[-1]
            volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
            signals['volume_ratio'] = volume_ratio

        # 5. Price vs SMA20
        sma20 = df['close'].tail(20).mean()
        price_vs_sma = ((current_price - sma20) / sma20) * 100
        signals['price_vs_sma20'] = price_vs_sma

        # 6. Bollinger Band position
        std20 = df['close'].tail(20).std()
        upper_band = sma20 + (2 * std20)
        lower_band = sma20 - (2 * std20)
        bb_position = (current_price - lower_band) / (upper_band - lower_band) if upper_band != lower_band else 0.5
        signals['bollinger_position'] = bb_position

        # Score the signals with learned weights
        bullish_score = 0
        bearish_score = 0

        # Momentum
        if signals.get('momentum_20d', 0) > 2:
            bullish_score += 20 * self.signal_weights['momentum_20d']
        elif signals.get('momentum_20d', 0) < -2:
            bearish_score += 20 * self.signal_weights['momentum_20d']

        # RSI
        rsi_val = signals.get('rsi', 50)
        if rsi_val < 30:
            bullish_score += 25 * self.signal_weights['rsi']  # Oversold = buy signal
        elif rsi_val > 70:
            bearish_score += 25 * self.signal_weights['rsi']  # Overbought = sell signal

        # MACD
        if signals.get('macd_signal', 0) > 0:
            bullish_score += 15 * self.signal_weights['macd_signal']
        else:
            bearish_score += 15 * self.signal_weights['macd_signal']

        # Volume
        vol_ratio = signals.get('volume_ratio', 1)
        if vol_ratio > 1.5 and signals.get('momentum_20d', 0) > 0:
            bullish_score += 10 * self.signal_weights['volume_ratio']
        elif vol_ratio > 1.5 and signals.get('momentum_20d', 0) < 0:
            bearish_score += 10 * self.signal_weights['volume_ratio']

        # Price vs SMA
        if signals.get('price_vs_sma20', 0) > 2:
            bullish_score += 15 * self.signal_weights['price_vs_sma20']
        elif signals.get('price_vs_sma20', 0) < -2:
            bearish_score += 15 * self.signal_weights['price_vs_sma20']

        # Bollinger
        bb_pos = signals.get('bollinger_position', 0.5)
        if bb_pos < 0.2:
            bullish_score += 15 * self.signal_weights['bollinger_position']  # Near lower band
        elif bb_pos > 0.8:
            bearish_score += 15 * self.signal_weights['bollinger_position']  # Near upper band

        # Determine direction and confidence
        total_score = bullish_score + bearish_score
        if total_score == 0:
            return None

        if bullish_score > bearish_score * 1.2:
            direction = 'up'
            confidence = min(95, 50 + (bullish_score - bearish_score))
            predicted_change = 0.5 + (confidence - 50) / 100
        elif bearish_score > bullish_score * 1.2:
            direction = 'down'
            confidence = min(95, 50 + (bearish_score - bullish_score))
            predicted_change = -(0.5 + (confidence - 50) / 100)
        else:
            direction = 'sideways'
            confidence = 60
            predicted_change = 0

        # Build reasoning
        reasoning_parts = []
        if signals.get('momentum_20d', 0) > 2:
            reasoning_parts.append(f"Strong momentum ({signals['momentum_20d']:.1f}%)")
        elif signals.get('momentum_20d', 0) < -2:
            reasoning_parts.append(f"Weak momentum ({signals['momentum_20d']:.1f}%)")

        if signals.get('rsi', 50) < 30:
            reasoning_parts.append("RSI oversold")
        elif signals.get('rsi', 50) > 70:
            reasoning_parts.append("RSI overbought")

        if signals.get('macd_signal', 0) > 0:
            reasoning_parts.append("MACD bullish")
        else:
            reasoning_parts.append("MACD bearish")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Mixed signals"

        # Create prediction
        prediction = Prediction(
            prediction_id=self.prediction_tracker.generate_prediction_id(),
            timestamp=datetime.now(),
            symbol=symbol,
            predicted_direction=direction,
            confidence=confidence,
            predicted_change_pct=predicted_change,
            timeframe='1h',
            target_price=current_price * (1 + predicted_change / 100),
            entry_price=current_price,
            signals=signals,
            reasoning=reasoning
        )

        return prediction

    def _learn_from_history(self):
        """Analyze prediction history and adjust signal weights."""
        # Get signal performance
        signal_perf = self.prediction_tracker.get_signal_performance(days=7)

        if signal_perf.empty:
            return

        # Adjust weights based on accuracy
        for _, row in signal_perf.iterrows():
            signal_name = row['signal']
            accuracy = row['accuracy']
            uses = row['uses']

            if signal_name in self.signal_weights and uses >= 5:
                # Adjust weight: increase for accurate signals, decrease for inaccurate
                if accuracy > 60:
                    self.signal_weights[signal_name] = min(2.0, self.signal_weights[signal_name] * 1.05)
                elif accuracy < 40:
                    self.signal_weights[signal_name] = max(0.5, self.signal_weights[signal_name] * 0.95)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current monitor status."""
        accuracy = self.prediction_tracker.get_accuracy_stats(days=7)

        return {
            'is_running': self.is_running,
            'symbols': self.symbols,
            'current_prices': self.current_prices.copy(),
            'active_predictions': len(self.prediction_tracker.active_predictions),
            'accuracy_7d': accuracy,
            'signal_weights': self.signal_weights.copy()
        }

    def get_predictions_summary(self) -> pd.DataFrame:
        """Get summary of recent predictions."""
        with self.prediction_tracker.db.get_connection() as conn:
            df = pd.read_sql_query("""
                SELECT
                    prediction_id, timestamp, symbol, predicted_direction,
                    confidence, actual_change_pct, was_correct, resolved
                FROM ai_predictions
                ORDER BY timestamp DESC
                LIMIT 50
            """, conn)
        return df

    def run_single_analysis(self, symbol: str) -> Optional[Prediction]:
        """Run a single analysis for a symbol (for manual use)."""
        try:
            df = self.market_data.get_historical_data(symbol, period='30d')
            if df.empty or len(df) < 20:
                return None

            return self._analyze_and_predict(symbol, df)
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return None


# Global monitor instance
_market_monitor: Optional[MarketMonitor] = None


def get_market_monitor(symbols: List[str] = None) -> MarketMonitor:
    """Get or create the global market monitor."""
    global _market_monitor
    if _market_monitor is None:
        _market_monitor = MarketMonitor(symbols)
    return _market_monitor
