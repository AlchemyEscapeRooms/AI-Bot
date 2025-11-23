"""
Static Backtest Runner - Run backtests using pre-downloaded static data.

This module provides a wrapper for running backtests using static/cached data
instead of fetching from external APIs. This provides:
- Faster backtesting (no API calls)
- Reproducible results (same data every time)
- Offline backtesting capability
- Consistent benchmarking across different runs
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from backtesting.backtest_engine import BacktestEngine
from backtesting.strategy_evaluator import StrategyEvaluator
from backtesting.strategies import STRATEGY_REGISTRY, DEFAULT_PARAMS
from data.static_data_loader import StaticDataLoader

try:
    from utils.logger import get_logger
    logger = get_logger(__name__)
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


class StaticBacktestRunner:
    """Run backtests using static pre-downloaded data."""

    def __init__(
        self,
        data_dir: str = 'static_data',
        initial_capital: float = 100000.0,
        commission: float = 0.001,
        slippage: float = 0.0005
    ):
        """
        Initialize the static backtest runner.

        Args:
            data_dir: Directory containing static data files
            initial_capital: Starting capital for backtests
            commission: Commission rate (0.001 = 0.1%)
            slippage: Slippage rate (0.0005 = 0.05%)
        """
        self.data_loader = StaticDataLoader(data_dir)
        self.backtest_engine = BacktestEngine(
            initial_capital=initial_capital,
            commission=commission,
            slippage=slippage
        )
        self.strategy_evaluator = StrategyEvaluator()

        logger.info("StaticBacktestRunner initialized")
        logger.info(f"Data directory: {data_dir}")
        logger.info(f"Initial capital: ${initial_capital:,.2f}")

    def list_available_symbols(self) -> List[str]:
        """Get list of available symbols for backtesting."""
        return self.data_loader.list_available_symbols()

    def run_single_backtest(
        self,
        symbol: str,
        strategy_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Run a backtest for a single symbol and strategy.

        Args:
            symbol: Stock symbol
            strategy_name: Name of strategy from STRATEGY_REGISTRY
            start_date: Start date (YYYY-MM-DD), optional
            end_date: End date (YYYY-MM-DD), optional
            strategy_params: Strategy parameters, optional

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Running backtest: {symbol} with {strategy_name} strategy")

        # Load data
        df = self.data_loader.get_historical_data(symbol, start_date, end_date)

        if df.empty:
            logger.error(f"No data available for {symbol}")
            return {}

        # Get strategy function
        if strategy_name not in STRATEGY_REGISTRY:
            logger.error(f"Unknown strategy: {strategy_name}")
            logger.info(f"Available strategies: {list(STRATEGY_REGISTRY.keys())}")
            return {}

        strategy_func = STRATEGY_REGISTRY[strategy_name]

        # Get parameters
        params = strategy_params or DEFAULT_PARAMS.get(strategy_name, {})

        # Run backtest
        results = self.backtest_engine.run_backtest(df, strategy_func, params)

        # Add metadata
        results['symbol'] = symbol
        results['strategy_name'] = strategy_name
        results['strategy_params'] = params
        results['start_date'] = str(df.index[0])
        results['end_date'] = str(df.index[-1])
        results['backtest_date'] = datetime.now().isoformat()

        return results

    def run_strategy_comparison(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategies: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Compare multiple strategies on a single symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date, optional
            end_date: End date, optional
            strategies: List of strategy names, or None for all

        Returns:
            DataFrame with comparison results
        """
        logger.info(f"Comparing strategies for {symbol}")

        # Load data
        df = self.data_loader.get_historical_data(symbol, start_date, end_date)

        if df.empty:
            logger.error(f"No data available for {symbol}")
            return pd.DataFrame()

        # Determine which strategies to test
        if strategies is None:
            strategies_to_test = STRATEGY_REGISTRY
        else:
            strategies_to_test = {
                name: STRATEGY_REGISTRY[name]
                for name in strategies
                if name in STRATEGY_REGISTRY
            }

        # Build strategy dict with params
        strategy_dict = {
            name: (func, DEFAULT_PARAMS.get(name, {}))
            for name, func in strategies_to_test.items()
        }

        # Run comparison
        results = self.backtest_engine.compare_strategies(df, strategy_dict)

        return results

    def run_multi_symbol_backtest(
        self,
        symbols: List[str],
        strategy_name: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        strategy_params: Optional[Dict[str, Any]] = None
    ) -> pd.DataFrame:
        """
        Run a single strategy across multiple symbols.

        Args:
            symbols: List of stock symbols
            strategy_name: Strategy name
            start_date: Start date, optional
            end_date: End date, optional
            strategy_params: Strategy parameters, optional

        Returns:
            DataFrame with results for all symbols
        """
        logger.info(f"Running {strategy_name} strategy on {len(symbols)} symbols")

        results = []

        for symbol in symbols:
            logger.info(f"Processing {symbol}...")

            result = self.run_single_backtest(
                symbol,
                strategy_name,
                start_date,
                end_date,
                strategy_params
            )

            if result:
                results.append(result)

        if not results:
            logger.warning("No successful backtests")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(results)

        # Sort by total return
        df = df.sort_values('total_return', ascending=False)

        logger.info(f"Completed backtests for {len(results)} symbols")

        return df

    def find_best_strategy_for_symbol(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> tuple:
        """
        Find the best performing strategy for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date, optional
            end_date: End date, optional

        Returns:
            Tuple of (strategy_name, results_dict)
        """
        logger.info(f"Finding best strategy for {symbol}")

        comparison = self.run_strategy_comparison(symbol, start_date, end_date)

        if comparison.empty:
            return None, {}

        # Get best strategy (highest composite score)
        best = comparison.iloc[0]

        logger.info(f"Best strategy: {best['strategy_name']} (Return: {best['total_return']:.2f}%)")

        return best['strategy_name'], best.to_dict()

    def run_batch_backtest(
        self,
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Run a batch of backtests based on configuration.

        Args:
            config: Configuration dictionary with:
                - symbols: List of symbols (or 'all')
                - strategies: List of strategies (or 'all')
                - start_date: Start date
                - end_date: End date
                - max_symbols: Maximum symbols to process

        Returns:
            Dictionary with batch results
        """
        logger.info("Running batch backtest")

        # Parse config
        symbols = config.get('symbols', 'all')
        if symbols == 'all':
            symbols = self.list_available_symbols()
        elif isinstance(symbols, str):
            symbols = [symbols]

        max_symbols = config.get('max_symbols', len(symbols))
        symbols = symbols[:max_symbols]

        strategies = config.get('strategies', 'all')
        if strategies == 'all':
            strategies = list(STRATEGY_REGISTRY.keys())
        elif isinstance(strategies, str):
            strategies = [strategies]

        start_date = config.get('start_date')
        end_date = config.get('end_date')

        logger.info(f"Testing {len(symbols)} symbols with {len(strategies)} strategies")

        # Run backtests
        all_results = []

        for symbol in symbols:
            for strategy in strategies:
                result = self.run_single_backtest(
                    symbol,
                    strategy,
                    start_date,
                    end_date
                )

                if result:
                    all_results.append(result)

        # Aggregate results
        if not all_results:
            return {}

        results_df = pd.DataFrame(all_results)

        # Calculate summary statistics
        summary = {
            'total_backtests': len(all_results),
            'symbols_tested': len(symbols),
            'strategies_tested': len(strategies),
            'avg_return': results_df['total_return'].mean(),
            'best_return': results_df['total_return'].max(),
            'worst_return': results_df['total_return'].min(),
            'avg_sharpe': results_df['sharpe_ratio'].mean(),
            'avg_win_rate': results_df['win_rate'].mean(),
            'config': config,
            'results': results_df
        }

        logger.info(f"Batch backtest complete: {len(all_results)} backtests")
        logger.info(f"Average return: {summary['avg_return']:.2f}%")
        logger.info(f"Average Sharpe: {summary['avg_sharpe']:.2f}")

        return summary


def main():
    """Example usage of StaticBacktestRunner."""

    print("=" * 80)
    print("STATIC BACKTEST RUNNER - EXAMPLE")
    print("=" * 80)

    # Initialize runner
    runner = StaticBacktestRunner()

    # List available symbols
    symbols = runner.list_available_symbols()
    print(f"\nAvailable symbols: {len(symbols)}")
    print(f"Sample: {symbols[:10]}")

    # Example 1: Run single backtest
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Single Backtest")
    print("=" * 80)

    if symbols:
        result = runner.run_single_backtest(
            symbol=symbols[0],
            strategy_name='momentum',
            start_date='2023-01-01'
        )

        if result:
            print(f"\nBacktest Results for {result['symbol']}:")
            print(f"  Strategy: {result['strategy_name']}")
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Total Trades: {result['total_trades']}")

    # Example 2: Strategy comparison
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Strategy Comparison")
    print("=" * 80)

    if symbols:
        comparison = runner.run_strategy_comparison(
            symbol=symbols[0],
            strategies=['momentum', 'mean_reversion', 'rsi']
        )

        if not comparison.empty:
            print(f"\nStrategy Comparison for {symbols[0]}:")
            print(comparison[['strategy_name', 'total_return', 'sharpe_ratio', 'win_rate']].to_string())

    # Example 3: Multi-symbol backtest
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Multi-Symbol Backtest")
    print("=" * 80)

    if len(symbols) >= 5:
        multi_results = runner.run_multi_symbol_backtest(
            symbols=symbols[:5],
            strategy_name='momentum'
        )

        if not multi_results.empty:
            print("\nTop 3 performers:")
            print(multi_results[['symbol', 'total_return', 'sharpe_ratio', 'win_rate']].head(3).to_string())

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
