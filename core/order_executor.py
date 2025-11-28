"""Order execution module for live and paper trading with trade logging."""

import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from utils.logger import get_logger
from utils.database import Database
from utils.trade_logger import TradeLogger, TradeReason, get_trade_logger
from config import config

logger = get_logger(__name__)

# Try to import Alpaca
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
    ALPACA_TRADING_AVAILABLE = True
except ImportError:
    ALPACA_TRADING_AVAILABLE = False
    logger.warning("Alpaca trading SDK not available")


class OrderStatus(Enum):
    """Order status enum."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


@dataclass
class OrderResult:
    """Result of an order execution."""
    success: bool
    order_id: str
    symbol: str
    side: str
    quantity: float
    price: float
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    message: str = ""
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class OrderExecutor:
    """Handles order execution for live and paper trading."""

    def __init__(self, mode: str = "paper"):
        """
        Initialize order executor.

        Args:
            mode: "paper" for paper trading, "live" for real trading
        """
        self.mode = mode
        self.db = Database()
        self.trade_logger = get_trade_logger()

        # Paper trading state
        self.paper_orders: Dict[str, Dict] = {}
        self.paper_order_id = 0

        # Alpaca client for live trading
        self.alpaca_client = None
        if mode == "live" and ALPACA_TRADING_AVAILABLE:
            self._init_alpaca_client()

        logger.info(f"OrderExecutor initialized in {mode} mode")

    def _init_alpaca_client(self):
        """Initialize Alpaca trading client."""
        try:
            api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca_key')
            api_secret = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca_secret')

            if api_key and api_secret:
                # Check if paper trading
                paper = config.get('alpaca.paper_trading', True)
                self.alpaca_client = TradingClient(api_key, api_secret, paper=paper)
                logger.info(f"Alpaca trading client initialized (paper={paper})")
            else:
                logger.warning("Alpaca API keys not found for live trading")
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca trading client: {e}")

    def execute_order(
        self,
        symbol: str,
        side: str,  # "buy" or "sell"
        quantity: float,
        order_type: str = "market",
        limit_price: float = None,
        strategy_name: str = None,
        strategy_params: Dict = None,
        reason: TradeReason = None,
        market_snapshot: Dict = None,
        portfolio_value: float = None
    ) -> OrderResult:
        """
        Execute a trading order.

        Args:
            symbol: Stock symbol
            side: "buy" or "sell"
            quantity: Number of shares
            order_type: "market" or "limit"
            limit_price: Price for limit orders
            strategy_name: Name of the strategy that generated the signal
            strategy_params: Parameters of the strategy
            reason: TradeReason object with decision reasoning
            market_snapshot: Current market data snapshot
            portfolio_value: Current portfolio value

        Returns:
            OrderResult with execution details
        """
        logger.info(f"Executing {side} order: {quantity} {symbol} ({order_type})")

        if self.mode == "paper":
            result = self._execute_paper_order(
                symbol, side, quantity, order_type, limit_price
            )
        elif self.mode == "live" and self.alpaca_client:
            result = self._execute_alpaca_order(
                symbol, side, quantity, order_type, limit_price
            )
        else:
            result = OrderResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message="No trading client available"
            )

        # Log the trade with reasoning
        if result.success and self.trade_logger:
            self._log_trade(
                result,
                strategy_name=strategy_name,
                strategy_params=strategy_params,
                reason=reason,
                market_snapshot=market_snapshot,
                portfolio_value=portfolio_value
            )

        return result

    def _execute_paper_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float = None
    ) -> OrderResult:
        """Execute a paper trading order."""
        from data import MarketDataCollector

        self.paper_order_id += 1
        order_id = f"PAPER-{self.paper_order_id:08d}"

        try:
            # Get current price
            market_data = MarketDataCollector()
            quote = market_data.get_real_time_quote(symbol)
            current_price = quote.get('price', 0)

            if current_price <= 0:
                return OrderResult(
                    success=False,
                    order_id=order_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=0,
                    status=OrderStatus.REJECTED,
                    message=f"Could not get price for {symbol}"
                )

            # For limit orders, check if price is acceptable
            if order_type == "limit" and limit_price:
                if side == "buy" and current_price > limit_price:
                    return OrderResult(
                        success=False,
                        order_id=order_id,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=limit_price,
                        status=OrderStatus.PENDING,
                        message="Limit order pending - price above limit"
                    )
                elif side == "sell" and current_price < limit_price:
                    return OrderResult(
                        success=False,
                        order_id=order_id,
                        symbol=symbol,
                        side=side,
                        quantity=quantity,
                        price=limit_price,
                        status=OrderStatus.PENDING,
                        message="Limit order pending - price below limit"
                    )
                fill_price = limit_price
            else:
                # Apply slippage for market orders
                slippage = 0.0005  # 0.05% slippage
                if side == "buy":
                    fill_price = current_price * (1 + slippage)
                else:
                    fill_price = current_price * (1 - slippage)

            # Store paper order
            self.paper_orders[order_id] = {
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'order_type': order_type,
                'limit_price': limit_price,
                'fill_price': fill_price,
                'status': 'filled',
                'timestamp': datetime.now()
            }

            logger.info(f"Paper order filled: {order_id} - {side} {quantity} {symbol} @ ${fill_price:.2f}")

            return OrderResult(
                success=True,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=fill_price,
                status=OrderStatus.FILLED,
                filled_quantity=quantity,
                filled_price=fill_price,
                message="Paper order filled"
            )

        except Exception as e:
            logger.error(f"Paper order failed: {e}")
            return OrderResult(
                success=False,
                order_id=order_id,
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message=str(e)
            )

    def _execute_alpaca_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str,
        limit_price: float = None
    ) -> OrderResult:
        """Execute an order through Alpaca."""
        if not self.alpaca_client:
            return OrderResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message="Alpaca client not initialized"
            )

        try:
            # Convert side to Alpaca enum
            order_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

            # Create order request
            if order_type == "market":
                order_request = MarketOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY
                )
            else:
                order_request = LimitOrderRequest(
                    symbol=symbol,
                    qty=quantity,
                    side=order_side,
                    time_in_force=TimeInForce.DAY,
                    limit_price=limit_price
                )

            # Submit order
            order = self.alpaca_client.submit_order(order_request)

            logger.info(f"Alpaca order submitted: {order.id} - {side} {quantity} {symbol}")

            # Determine status
            status_map = {
                'new': OrderStatus.SUBMITTED,
                'accepted': OrderStatus.SUBMITTED,
                'pending_new': OrderStatus.PENDING,
                'partially_filled': OrderStatus.PARTIALLY_FILLED,
                'filled': OrderStatus.FILLED,
                'canceled': OrderStatus.CANCELLED,
                'rejected': OrderStatus.REJECTED
            }

            order_status = status_map.get(str(order.status).lower(), OrderStatus.PENDING)

            return OrderResult(
                success=order_status in [OrderStatus.SUBMITTED, OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED],
                order_id=str(order.id),
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=float(order.limit_price or order.filled_avg_price or 0),
                status=order_status,
                filled_quantity=float(order.filled_qty or 0),
                filled_price=float(order.filled_avg_price or 0),
                message=f"Order {order.status}"
            )

        except Exception as e:
            logger.error(f"Alpaca order failed: {e}")
            return OrderResult(
                success=False,
                order_id="",
                symbol=symbol,
                side=side,
                quantity=quantity,
                price=0,
                status=OrderStatus.FAILED,
                message=str(e)
            )

    def _log_trade(
        self,
        result: OrderResult,
        strategy_name: str = None,
        strategy_params: Dict = None,
        reason: TradeReason = None,
        market_snapshot: Dict = None,
        portfolio_value: float = None
    ):
        """Log a trade with full reasoning."""
        if not self.trade_logger:
            return

        # Create default reason if not provided
        if reason is None:
            reason = TradeReason(
                primary_signal="manual_trade",
                signal_value=0,
                threshold=0,
                direction="n/a",
                explanation="Trade executed without detailed reasoning"
            )

        try:
            self.trade_logger.log_trade(
                symbol=result.symbol,
                action=result.side.upper(),
                quantity=result.filled_quantity or result.quantity,
                price=result.filled_price or result.price,
                strategy_name=strategy_name or "unknown",
                strategy_params=strategy_params or {},
                reason=reason,
                mode="live" if self.mode == "live" else "paper",
                side="long" if result.side.lower() == "buy" else "short",
                order_id=result.order_id,
                order_type="market",  # Could be passed in
                portfolio_value_before=portfolio_value,
                market_snapshot=market_snapshot,
                timestamp=result.timestamp
            )

            logger.debug(f"Trade logged: {result.order_id}")

        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get status of an order."""
        if self.mode == "paper":
            return self.paper_orders.get(order_id)
        elif self.alpaca_client:
            try:
                order = self.alpaca_client.get_order_by_id(order_id)
                return {
                    'order_id': str(order.id),
                    'symbol': order.symbol,
                    'side': str(order.side),
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty or 0),
                    'status': str(order.status),
                    'filled_price': float(order.filled_avg_price or 0)
                }
            except Exception as e:
                logger.error(f"Failed to get order status: {e}")
                return None
        return None

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an open order."""
        if self.mode == "paper":
            if order_id in self.paper_orders:
                self.paper_orders[order_id]['status'] = 'cancelled'
                return True
            return False
        elif self.alpaca_client:
            try:
                self.alpaca_client.cancel_order_by_id(order_id)
                return True
            except Exception as e:
                logger.error(f"Failed to cancel order: {e}")
                return False
        return False

    def get_open_orders(self) -> List[Dict]:
        """Get all open orders."""
        if self.mode == "paper":
            return [
                order for order in self.paper_orders.values()
                if order['status'] in ['pending', 'submitted']
            ]
        elif self.alpaca_client:
            try:
                orders = self.alpaca_client.get_orders()
                return [
                    {
                        'order_id': str(o.id),
                        'symbol': o.symbol,
                        'side': str(o.side),
                        'quantity': float(o.qty),
                        'status': str(o.status)
                    }
                    for o in orders
                ]
            except Exception as e:
                logger.error(f"Failed to get open orders: {e}")
                return []
        return []

    def get_account_info(self) -> Optional[Dict]:
        """Get account information."""
        if self.mode == "paper":
            return {
                'mode': 'paper',
                'orders_count': len(self.paper_orders)
            }
        elif self.alpaca_client:
            try:
                account = self.alpaca_client.get_account()
                return {
                    'mode': 'live',
                    'buying_power': float(account.buying_power),
                    'cash': float(account.cash),
                    'portfolio_value': float(account.portfolio_value),
                    'equity': float(account.equity)
                }
            except Exception as e:
                logger.error(f"Failed to get account info: {e}")
                return None
        return None
