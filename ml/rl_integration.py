"""
Reinforcement Learning Integration Module
==========================================

Bridges the Q-Learning ReinforcementLearningEngine with the existing
signal-weighted trading system.

This module:
1. Converts market data to RL MarketState objects
2. Uses RL engine for action recommendations
3. Feeds trading outcomes back to RL for learning
4. Provides a hybrid approach combining signal weights + RL

Author: Claude AI
Date: December 2025
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

from ml.reinforcement_learner import (
    ReinforcementLearningEngine,
    MarketState,
    TradingAction,
    Experience,
    Prediction as RLPrediction
)
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)


class RLTradingIntegration:
    """
    Integrates Q-Learning RL engine with the trading system.

    Provides:
    - State conversion from market data to RL format
    - Action recommendations using Q-learning
    - Experience-based learning from trade outcomes
    - Hybrid scoring combining signals + RL
    """

    def __init__(
        self,
        enable_rl: bool = True,
        model_path: str = None,
        learning_rate: float = None,
        exploration_rate: float = None
    ):
        """
        Initialize RL integration.

        Args:
            enable_rl: Whether to use RL for decisions (can be disabled)
            model_path: Path to load existing RL model
            learning_rate: Override default learning rate
            exploration_rate: Override default exploration rate
        """
        self.enable_rl = enable_rl

        # Load config values or use defaults
        lr = learning_rate or config.get('rl.learning_rate', 0.001)
        er = exploration_rate or config.get('rl.exploration_rate', 0.15)

        self.rl_engine = ReinforcementLearningEngine(
            learning_rate=lr,
            discount_factor=config.get('rl.discount_factor', 0.95),
            exploration_rate=er,
            min_exploration=config.get('rl.min_exploration', 0.05)
        )

        # Load existing model if available
        if model_path:
            try:
                self.rl_engine.load_model(model_path)
                logger.info(f"Loaded RL model from {model_path}")
            except Exception as e:
                logger.warning(f"Could not load RL model: {e}")
        else:
            # Try to load from default location
            default_path = Path("ml/models/rl_model_latest.pkl")
            if default_path.exists():
                try:
                    self.rl_engine.load_model(str(default_path))
                    logger.info("Loaded RL model from default location")
                except Exception as e:
                    logger.debug(f"No existing RL model loaded: {e}")

        # Track state for learning
        self.current_states: Dict[str, MarketState] = {}
        self.pending_actions: Dict[str, Tuple[TradingAction, MarketState]] = {}

        # Weight for RL vs signal-based decisions (0 = pure signals, 1 = pure RL)
        self.rl_weight = config.get('rl.rl_weight', 0.3)

        logger.info(f"RL Integration initialized (enabled={enable_rl}, weight={self.rl_weight})")

    def create_market_state(
        self,
        symbol: str,
        features: Dict[str, float],
        current_price: float,
        portfolio_info: Dict[str, Any] = None
    ) -> MarketState:
        """
        Convert market features to RL MarketState.

        Args:
            symbol: Stock symbol
            features: Dict of technical indicators and signals
            current_price: Current price
            portfolio_info: Optional portfolio context

        Returns:
            MarketState object for RL engine
        """
        portfolio_info = portfolio_info or {}

        # Extract or calculate required state components
        state = MarketState(
            timestamp=datetime.now(),
            symbol=symbol,
            price=current_price,

            # Price changes (from features or defaults)
            price_change_1h=features.get('price_change_1h', features.get('momentum_score', 0) / 100),
            price_change_1d=features.get('price_change_5', features.get('price_change_10', 0)) / 100,
            price_change_1w=features.get('price_change_20', features.get('momentum_20d', 0)) / 100,

            # Technical indicators
            rsi=features.get('rsi_14', features.get('rsi', 50)),
            macd=features.get('macd_hist', features.get('macd_signal', 0)),
            sma_20=current_price * (1 - features.get('price_vs_sma20', 0) / 100),
            sma_50=current_price * (1 - features.get('price_vs_sma50', 0) / 100),
            sma_200=current_price * (1 - features.get('price_vs_sma200', 0) / 100),
            volatility=features.get('volatility', features.get('atr_pct', 2)) / 100,
            volume_ratio=features.get('volume_ratio', 1.0),

            # Market regime (derive from features)
            regime=self._determine_regime(features),
            regime_confidence=features.get('trend_strength', 50) / 100,

            # Portfolio context
            position_pnl=portfolio_info.get('position_pnl', 0),
            portfolio_cash_pct=portfolio_info.get('cash_pct', 0.5),
            portfolio_risk=portfolio_info.get('risk', 0.02),
            recent_win_rate=portfolio_info.get('win_rate', 0.5),

            # Broader market (use SPY data if available, else defaults)
            spy_change=features.get('spy_change', 0),
            market_breadth=features.get('market_breadth', 0.5),
            vix=features.get('vix', 20)
        )

        # Cache for later learning
        self.current_states[symbol] = state

        return state

    def _determine_regime(self, features: Dict[str, float]) -> str:
        """Determine market regime from features."""
        trend = features.get('trend_strength', 0)
        volatility = features.get('volatility', features.get('atr_pct', 2))
        momentum = features.get('momentum_score', 0)

        if volatility > 3:
            return 'volatile'
        elif trend > 50 and momentum > 0:
            return 'trending_up'
        elif trend > 50 and momentum < 0:
            return 'trending_down'
        else:
            return 'ranging'

    def get_rl_action(
        self,
        symbol: str,
        features: Dict[str, float],
        current_price: float,
        portfolio_info: Dict[str, Any] = None,
        mode: str = 'exploit'
    ) -> TradingAction:
        """
        Get action recommendation from RL engine.

        Args:
            symbol: Stock symbol
            features: Market features
            current_price: Current price
            portfolio_info: Portfolio context
            mode: 'train' (with exploration) or 'exploit' (best action)

        Returns:
            TradingAction with recommended action
        """
        if not self.enable_rl:
            # Return hold action if RL disabled
            return TradingAction('hold', 0.0, 0.5)

        state = self.create_market_state(symbol, features, current_price, portfolio_info)
        action = self.rl_engine.select_action(state, mode=mode)

        # Store for potential learning
        self.pending_actions[symbol] = (action, state)

        return action

    def get_hybrid_score(
        self,
        symbol: str,
        signal_score: float,
        features: Dict[str, float],
        current_price: float,
        portfolio_info: Dict[str, Any] = None
    ) -> Tuple[float, str, Dict[str, Any]]:
        """
        Get hybrid score combining signal weights and RL.

        Args:
            symbol: Stock symbol
            signal_score: Score from signal-weighted system (-1 to 1)
            features: Market features
            current_price: Current price
            portfolio_info: Portfolio context

        Returns:
            Tuple of (final_score, recommended_action, debug_info)
        """
        if not self.enable_rl:
            # Pure signal-based
            action = 'buy' if signal_score > 0.2 else ('sell' if signal_score < -0.2 else 'hold')
            return signal_score, action, {'method': 'signals_only'}

        # Get RL action
        rl_action = self.get_rl_action(symbol, features, current_price, portfolio_info)

        # Convert RL action to score
        if rl_action.action_type == 'buy':
            rl_score = 0.3 + (rl_action.position_size_pct * 2)  # 0.3 to 0.8
        elif rl_action.action_type == 'sell':
            rl_score = -0.5 - (rl_action.confidence * 0.5)  # -0.5 to -1.0
        elif rl_action.action_type in ['increase', 'decrease']:
            rl_score = 0.2 if rl_action.action_type == 'increase' else -0.2
        else:  # hold
            rl_score = 0.0

        # Combine scores
        final_score = (
            signal_score * (1 - self.rl_weight) +
            rl_score * self.rl_weight
        )

        # Determine action
        if final_score > 0.2:
            action = 'buy'
        elif final_score < -0.2:
            action = 'sell'
        else:
            action = 'hold'

        debug_info = {
            'method': 'hybrid',
            'signal_score': signal_score,
            'rl_score': rl_score,
            'rl_action': rl_action.action_type,
            'rl_confidence': rl_action.confidence,
            'rl_weight': self.rl_weight,
            'final_score': final_score
        }

        return final_score, action, debug_info

    def record_trade_outcome(
        self,
        symbol: str,
        entry_price: float,
        exit_price: float,
        was_profitable: bool,
        holding_period_hours: float = 1.0
    ):
        """
        Record trade outcome for RL learning.

        Args:
            symbol: Stock symbol
            entry_price: Entry price
            exit_price: Exit price
            was_profitable: Whether trade was profitable
            holding_period_hours: How long position was held
        """
        if not self.enable_rl:
            return

        if symbol not in self.pending_actions:
            logger.debug(f"No pending action for {symbol}, skipping RL learning")
            return

        action, state = self.pending_actions[symbol]

        # Calculate reward
        pnl_pct = (exit_price - entry_price) / entry_price

        # Reward shaping:
        # - Positive PnL = positive reward (scaled by magnitude)
        # - Negative PnL = negative reward (scaled by magnitude)
        # - Holding bonus for profitable trades (encourages patience)

        base_reward = pnl_pct * 100  # Scale to roughly -10 to +10 range

        # Holding bonus: reward patience on winning trades
        if was_profitable and holding_period_hours > 1:
            holding_bonus = min(0.5, holding_period_hours / 24)
            base_reward += holding_bonus

        # Risk penalty: penalize large losses more than rewarding large gains
        if pnl_pct < -0.02:  # > 2% loss
            base_reward *= 1.5  # Amplify negative signal

        # Create experience
        next_state = self.current_states.get(symbol)
        experience = Experience(
            state=state,
            action=action,
            reward=base_reward,
            next_state=next_state,
            done=True  # Trade completed
        )

        # Learn from experience
        self.rl_engine.learn_from_experience(experience)

        # Clear pending action
        del self.pending_actions[symbol]

        logger.info(f"RL learned from {symbol} trade: reward={base_reward:.2f}, "
                   f"action={action.action_type}")

    def save_model(self, path: str = None):
        """Save RL model to disk."""
        if path is None:
            path = "ml/models/rl_model_latest.pkl"

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.rl_engine.save_model(path)
        logger.info(f"RL model saved to {path}")

    def get_stats(self) -> Dict[str, Any]:
        """Get RL learning statistics."""
        rl_stats = self.rl_engine.get_learning_stats()
        return {
            'enabled': self.enable_rl,
            'rl_weight': self.rl_weight,
            'q_table_size': rl_stats['q_table_size'],
            'learning_episodes': rl_stats['learning_episodes'],
            'exploration_rate': rl_stats['exploration_rate'],
            'total_predictions': rl_stats['total_predictions'],
            'avg_recent_reward': rl_stats['avg_reward_recent'],
            'pending_actions': len(self.pending_actions)
        }


# Singleton instance
_rl_integration: Optional[RLTradingIntegration] = None


def get_rl_integration(enable: bool = True) -> RLTradingIntegration:
    """Get or create the global RL integration instance."""
    global _rl_integration
    if _rl_integration is None:
        _rl_integration = RLTradingIntegration(enable_rl=enable)
    return _rl_integration
