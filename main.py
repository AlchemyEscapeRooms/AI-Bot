#!/usr/bin/env python3
"""
AI Trading Bot - Main Entry Point

A multifaceted AI-powered trading bot with machine learning, backtesting,
news sentiment analysis, and self-learning capabilities.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ai_trading_bot.core.trading_bot import TradingBot
from ai_trading_bot.core.personality_profiles import list_profiles
from ai_trading_bot.backtesting import BacktestEngine, StrategyEvaluator
from ai_trading_bot.data import MarketDataCollector
from ai_trading_bot.utils.logger import get_logger
from ai_trading_bot.config import config

# Import static backtest components
try:
    from ai_trading_bot.backtesting.static_backtest_runner import StaticBacktestRunner
    from ai_trading_bot.data.static_data_loader import StaticDataLoader
    STATIC_BACKTEST_AVAILABLE = True
except ImportError:
    STATIC_BACKTEST_AVAILABLE = False

logger = get_logger(__name__)


def run_trading_bot(args):
    """Run the live/paper trading bot."""
    bot = TradingBot(
        initial_capital=args.capital,
        personality=args.personality,
        mode=args.mode
    )

    logger.info(f"Starting trading bot in {args.mode} mode")
    bot.start()


def run_backtest(args):
    """Run backtesting."""

    # Check if using static data
    use_static = getattr(args, 'static', False)

    if use_static:
        if not STATIC_BACKTEST_AVAILABLE:
            logger.error("Static backtest not available. Missing dependencies.")
            return

        logger.info("Starting backtest mode with STATIC DATA")
        run_static_backtest(args)
    else:
        logger.info("Starting backtest mode with LIVE DATA FETCH")
        run_live_backtest(args)


def run_live_backtest(args):
    """Run backtesting with live data fetching."""

    market_data = MarketDataCollector()
    evaluator = StrategyEvaluator()

    # Get historical data
    df = market_data.get_historical_data(
        args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )

    if df.empty:
        logger.error(f"No data available for {args.symbol}")
        return

    # Run evaluation
    results = evaluator.evaluate_all_strategies(df)

    # Print results
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)
    print(results[['strategy_name', 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']].to_string())
    print("=" * 80)

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


def run_static_backtest(args):
    """Run backtesting with static pre-downloaded data."""

    runner = StaticBacktestRunner(data_dir='static_data')

    # Check if symbol is available
    available_symbols = runner.list_available_symbols()

    if args.symbol not in available_symbols:
        logger.error(f"Symbol {args.symbol} not found in static data")
        logger.info(f"Available symbols: {', '.join(available_symbols[:20])}...")
        return

    # Determine what type of backtest to run
    if hasattr(args, 'compare') and args.compare:
        # Compare all strategies
        logger.info(f"Comparing strategies for {args.symbol}")

        results = runner.run_strategy_comparison(
            symbol=args.symbol,
            start_date=args.start_date,
            end_date=args.end_date
        )

        # Print results
        print("\n" + "=" * 80)
        print(f"STRATEGY COMPARISON RESULTS - {args.symbol}")
        print("=" * 80)
        if not results.empty:
            print(results[['strategy_name', 'total_return', 'sharpe_ratio', 'win_rate', 'max_drawdown']].to_string())
        print("=" * 80)

        # Save results
        if args.output:
            results.to_csv(args.output, index=False)
            logger.info(f"Results saved to {args.output}")

    else:
        # Single strategy backtest
        strategy = getattr(args, 'strategy', 'momentum')

        logger.info(f"Running {strategy} strategy on {args.symbol}")

        result = runner.run_single_backtest(
            symbol=args.symbol,
            strategy_name=strategy,
            start_date=args.start_date,
            end_date=args.end_date
        )

        if result:
            # Print results
            print("\n" + "=" * 80)
            print(f"BACKTEST RESULTS - {args.symbol}")
            print("=" * 80)
            print(f"Strategy: {result['strategy_name']}")
            print(f"Period: {result['start_date']} to {result['end_date']}")
            print(f"\nPerformance Metrics:")
            print(f"  Total Return: {result['total_return']:.2f}%")
            print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
            print(f"  Sortino Ratio: {result['sortino_ratio']:.2f}")
            print(f"  Max Drawdown: {result['max_drawdown']:.2f}%")
            print(f"\nTrading Statistics:")
            print(f"  Total Trades: {result['total_trades']}")
            print(f"  Winning Trades: {result['winning_trades']}")
            print(f"  Losing Trades: {result['losing_trades']}")
            print(f"  Win Rate: {result['win_rate']:.2f}%")
            print(f"  Profit Factor: {result['profit_factor']:.2f}")
            print(f"  Average Trade: ${result['avg_trade']:.2f}")
            print(f"  Largest Win: ${result['largest_win']:.2f}")
            print(f"  Largest Loss: ${result['largest_loss']:.2f}")
            print("=" * 80)

            # Save results
            if args.output:
                import json
                with open(args.output, 'w') as f:
                    # Remove non-serializable items
                    save_result = {k: v for k, v in result.items()
                                   if k not in ['equity_curve', 'trades']}
                    json.dump(save_result, f, indent=2)
                logger.info(f"Results saved to {args.output}")


def list_static_symbols(args):
    """List available symbols in static data."""

    if not STATIC_BACKTEST_AVAILABLE:
        print("Static backtest not available. Missing dependencies.")
        return

    loader = StaticDataLoader(data_dir='static_data')

    symbols = loader.list_available_symbols()

    print("\n" + "=" * 80)
    print("AVAILABLE SYMBOLS IN STATIC DATA")
    print("=" * 80)
    print(f"Total symbols: {len(symbols)}\n")

    # Print in columns
    cols = 6
    for i in range(0, len(symbols), cols):
        row = symbols[i:i+cols]
        print("  ".join(f"{s:8}" for s in row))

    print("=" * 80)

    # Get data info
    info = loader.get_data_info()
    print(f"\nData Directory: {info['data_directory']}")
    if info.get('summary'):
        print(f"Generation Date: {info['summary'].get('generation_date', 'Unknown')}")
        print(f"Data Type: {info['summary'].get('data_type', 'Unknown')}")
    print("=" * 80 + "\n")


def show_profiles(args):
    """Show available personality profiles."""
    from ai_trading_bot.core.personality_profiles import PERSONALITY_PROFILES

    print("\n" + "=" * 80)
    print("AVAILABLE PERSONALITY PROFILES")
    print("=" * 80 + "\n")

    for name, profile in PERSONALITY_PROFILES.items():
        print(f"{name}")
        print(f"  Description: {profile.description}")
        print(f"  Risk Tolerance: {profile.risk_tolerance}")
        print(f"  Trading Style: {profile.trading_style}")
        print(f"  Preferred Strategies: {', '.join(profile.preferred_strategies)}")
        print()

    print("=" * 80 + "\n")


def show_config(args):
    """Show current configuration."""
    import yaml

    print("\n" + "=" * 80)
    print("CURRENT CONFIGURATION")
    print("=" * 80 + "\n")

    print(yaml.dump(config.raw, default_flow_style=False))
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Bot - Multifaceted automated trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run paper trading with balanced growth personality
  python main.py trade --mode paper --personality balanced_growth

  # Run backtesting on SPY
  python main.py backtest --symbol SPY --start-date 2020-01-01 --end-date 2023-12-31

  # Show available personality profiles
  python main.py profiles

  # Show current configuration
  python main.py config
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run')

    # Trade command
    trade_parser = subparsers.add_parser('trade', help='Run the trading bot')
    trade_parser.add_argument(
        '--mode',
        choices=['paper', 'live'],
        default='paper',
        help='Trading mode (default: paper)'
    )
    trade_parser.add_argument(
        '--capital',
        type=float,
        default=100000.0,
        help='Initial capital (default: 100000)'
    )
    trade_parser.add_argument(
        '--personality',
        choices=list_profiles(),
        default='balanced_growth',
        help='Trading personality profile (default: balanced_growth)'
    )

    # Backtest command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtesting')
    backtest_parser.add_argument(
        '--symbol',
        default='SPY',
        help='Symbol to backtest (default: SPY)'
    )
    backtest_parser.add_argument(
        '--start-date',
        default='2020-01-01',
        help='Start date (default: 2020-01-01)'
    )
    backtest_parser.add_argument(
        '--end-date',
        default='2023-12-31',
        help='End date (default: 2023-12-31)'
    )
    backtest_parser.add_argument(
        '--output',
        help='Output file for results (optional)'
    )
    backtest_parser.add_argument(
        '--static',
        action='store_true',
        help='Use static pre-downloaded data instead of fetching from API'
    )
    backtest_parser.add_argument(
        '--strategy',
        default='momentum',
        help='Strategy to use for backtest (default: momentum)'
    )
    backtest_parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all strategies instead of running just one'
    )

    # Static data list command
    subparsers.add_parser('list-static', help='List available symbols in static data')

    # Profiles command
    subparsers.add_parser('profiles', help='Show available personality profiles')

    # Config command
    subparsers.add_parser('config', help='Show current configuration')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Route to appropriate handler
    if args.command == 'trade':
        run_trading_bot(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'list-static':
        list_static_symbols(args)
    elif args.command == 'profiles':
        show_profiles(args)
    elif args.command == 'config':
        show_config(args)


if __name__ == '__main__':
    main()
