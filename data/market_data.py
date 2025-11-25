"""Market data collection from various sources."""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import os

# Alpaca API
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from ..utils.logger import get_logger
from ..config import config

logger = get_logger(__name__)


class MarketDataCollector:
    """Collects market data from multiple sources."""

    def __init__(self):
        self.data_source = config.get('data.sources.market_data.primary', 'alpaca')
        self.cache = {}

        # Initialize Alpaca client
        api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca.api_key', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca.secret_key', '')

        if api_key and secret_key and '${' not in api_key:
            self.alpaca_client = StockHistoricalDataClient(api_key, secret_key)
            logger.info("Alpaca client initialized successfully")
        else:
            self.alpaca_client = None
            logger.warning("Alpaca API keys not found - some features may not work")

    def get_historical_data(
        self,
        symbol: str,
        start_date: str = None,
        end_date: str = None,
        interval: str = '1d'
    ) -> pd.DataFrame:
        """Get historical OHLCV data for a symbol using Alpaca API."""

        try:
            logger.info(f"Fetching data for {symbol} from Alpaca")

            if self.alpaca_client is None:
                logger.error("Alpaca client not initialized - check API keys")
                return pd.DataFrame()

            if start_date is None:
                start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')

            # Map interval to Alpaca TimeFrame
            timeframe_map = {
                '1d': TimeFrame.Day,
                '1h': TimeFrame.Hour,
                '15min': TimeFrame.Minute * 15,
                '5min': TimeFrame.Minute * 5,
                '1min': TimeFrame.Minute
            }
            timeframe = timeframe_map.get(interval, TimeFrame.Day)

            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe,
                start=datetime.strptime(start_date, '%Y-%m-%d'),
                end=datetime.strptime(end_date, '%Y-%m-%d')
            )

            # Get data from Alpaca
            bars = self.alpaca_client.get_stock_bars(request_params)

            if symbol not in bars or len(bars[symbol]) == 0:
                logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = bars.df

            # Reset index to get timestamp as column
            df = df.reset_index()

            # Standardize column names
            df.columns = [col.lower() for col in df.columns]

            # Ensure we have the symbol column
            if 'symbol' not in df.columns:
                df['symbol'] = symbol

            # Set timestamp as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)

            logger.info(f"Retrieved {len(df)} bars for {symbol} from Alpaca")

            return df

        except Exception as e:
            logger.error(f"Error fetching data for {symbol} from Alpaca: {e}")
            return pd.DataFrame()

    def get_real_time_quote(self, symbol: str) -> Dict[str, Any]:
        """Get real-time quote for a symbol using Alpaca API."""

        try:
            if self.alpaca_client is None:
                logger.error("Alpaca client not initialized")
                return {}

            # Get latest quote from Alpaca
            request_params = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
            latest_quote = self.alpaca_client.get_stock_latest_quote(request_params)

            if symbol not in latest_quote:
                logger.warning(f"No quote available for {symbol}")
                return {}

            quote_data = latest_quote[symbol]

            quote = {
                'symbol': symbol,
                'price': (quote_data.bid_price + quote_data.ask_price) / 2 if quote_data.bid_price and quote_data.ask_price else 0,
                'bid': quote_data.bid_price or 0,
                'ask': quote_data.ask_price or 0,
                'bid_size': quote_data.bid_size or 0,
                'ask_size': quote_data.ask_size or 0,
                'timestamp': quote_data.timestamp
            }

            return quote

        except Exception as e:
            logger.error(f"Error fetching quote for {symbol} from Alpaca: {e}")
            return {}

    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, pd.DataFrame]:
        """Get historical data for multiple symbols."""

        logger.info(f"Fetching data for {len(symbols)} symbols")

        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date)
            if not df.empty:
                data[symbol] = df

        return data

    def get_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Get fundamental data for a symbol.

        Note: Alpaca doesn't provide fundamental data.
        This would require a separate data provider like Alpha Vantage or Financial Modeling Prep.
        """

        logger.warning(f"Fundamental data not available through Alpaca for {symbol}")
        logger.info("Configure Alpha Vantage or Financial Modeling Prep for fundamentals")

        # Return empty dict - fundamental data requires different API
        return {
            'symbol': symbol,
            'pe_ratio': None,
            'forward_pe': None,
            'pb_ratio': None,
            'market_cap': None,
            'sector': None,
            'industry': None,
            'note': 'Fundamental data requires Alpha Vantage or FMP API'
        }

    def screen_stocks(
        self,
        min_price: float = 5.0,
        min_volume: int = 1000000,
        min_market_cap: float = 1e9
    ) -> List[str]:
        """Screen for stocks meeting criteria."""

        logger.info("Screening stocks...")

        # This is a simplified version
        # In production, you'd use a proper stock screener API

        # Common liquid stocks
        candidates = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD',
            'SPY', 'QQQ', 'IWM', 'DIA',
            'JPM', 'BAC', 'WFC', 'GS',
            'XOM', 'CVX', 'COP',
            'JNJ', 'UNH', 'PFE',
            'WMT', 'HD', 'COST',
            'DIS', 'NFLX', 'CMCSA'
        ]

        filtered = []

        for symbol in candidates:
            try:
                quote = self.get_real_time_quote(symbol)
                fundamentals = self.get_fundamentals(symbol)

                if (quote.get('price', 0) >= min_price and
                    quote.get('volume', 0) >= min_volume and
                    fundamentals.get('market_cap', 0) >= min_market_cap):

                    filtered.append(symbol)

            except Exception as e:
                logger.warning(f"Error screening {symbol}: {e}")
                continue

        logger.info(f"Found {len(filtered)} stocks meeting criteria")

        return filtered

    def calculate_correlations(self, symbols: List[str], period: int = 60) -> pd.DataFrame:
        """Calculate correlation matrix for symbols."""

        logger.info(f"Calculating correlations for {len(symbols)} symbols")

        # Get data
        data = {}
        for symbol in symbols:
            df = self.get_historical_data(symbol)
            if not df.empty and len(df) >= period:
                data[symbol] = df['close'].tail(period)

        # Create correlation matrix
        prices_df = pd.DataFrame(data)
        correlations = prices_df.pct_change().corr()

        return correlations

    def get_market_regime(self, symbol: str = 'SPY') -> str:
        """Determine current market regime."""

        df = self.get_historical_data(symbol)

        if df.empty or len(df) < 50:
            return 'unknown'

        # Calculate indicators
        returns = df['close'].pct_change()
        volatility = returns.std()
        trend = (df['close'].iloc[-1] / df['close'].iloc[-20]) - 1

        # Determine regime
        if trend > 0.05 and volatility < returns.mean():
            return 'bull_trending'
        elif trend < -0.05 and volatility < returns.mean():
            return 'bear_trending'
        elif volatility > returns.std() * 1.5:
            return 'high_volatility'
        else:
            return 'ranging'
