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
sys.path.insert(0, str(Path(__file__).parent))

from core.trading_bot import TradingBot
from core.personality_profiles import list_profiles
from backtesting import BacktestEngine, StrategyEvaluator
from data import MarketDataCollector
from utils.logger import get_logger
from config import config

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
    """Run backtesting with live data fetch."""
    logger.info("Starting backtest mode with LIVE DATA FETCH")

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
    header = f"{'Strategy':<20} {'Return %':>10} {'Sharpe':>10} {'Win Rate %':>12} {'Max DD %':>10}"
    print(header)
    print("-" * len(header))
    for _, row in results.iterrows():
        print(f"{row['strategy_name']:<20} {row['total_return']:>10.2f} {row['sharpe_ratio']:>10.2f} {row['win_rate']:>12.2f} {row['max_drawdown']:>10.2f}")
    print("=" * 80)

    # Save results
    if args.output:
        results.to_csv(args.output, index=False)
        logger.info(f"Results saved to {args.output}")


def show_profiles(args):
    """Show available personality profiles."""
    from core.personality_profiles import PERSONALITY_PROFILES

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


def show_data_status(args):
    """Show data source status."""
    market_data = MarketDataCollector()
    status = market_data.get_data_source_status()

    print("\n" + "=" * 80)
    print("DATA SOURCE STATUS")
    print("=" * 80 + "\n")

    print(f"Alpaca:")
    print(f"  SDK Available: {status['alpaca']['available']}")
    print(f"  Client Initialized: {status['alpaca']['initialized']}")
    print(f"  Is Primary: {status['alpaca']['is_primary']}")

    print(f"\nyfinance:")
    print(f"  Available: {status['yfinance']['available']}")
    print(f"  Is Backup: {status['yfinance']['is_backup']}")

    print(f"\nFallback Order: {' -> '.join(status['fallback_order'])}")
    print("=" * 80 + "\n")


def run_auto_trader():
    """Start the auto-trading scheduler (default behavior when no args)."""
    from auto_trader import run_scheduler
    logger.info("=" * 60)
    logger.info("AI TRADING BOT - AUTO MODE")
    logger.info("=" * 60)
    logger.info("Starting auto-trader scheduler...")
    logger.info("  - Morning: Set daily goals at 8:30 AM")
    logger.info("  - Trading: Active during market hours")
    logger.info("  - Evening: Generate reports at 4:05 PM")
    logger.info("=" * 60)
    run_scheduler()


def main():
    parser = argparse.ArgumentParser(
        description="AI Trading Bot - Multifaceted automated trading system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run auto-trader (DEFAULT - just run without arguments)
  python main.py

  # Run paper trading with balanced growth personality
  python main.py trade --mode paper --personality balanced_growth

  # Run backtesting on SPY
  python main.py backtest --symbol SPY --start-date 2020-01-01 --end-date 2023-12-31

  # Show available personality profiles
  python main.py profiles

  # Show current configuration
  python main.py config

  # Show data source status
  python main.py status
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
        '--strategy',
        default='momentum',
        help='Strategy to use for backtest (default: momentum)'
    )
    backtest_parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare all strategies instead of running just one'
    )

    # Profiles command
    subparsers.add_parser('profiles', help='Show available personality profiles')

    # Config command
    subparsers.add_parser('config', help='Show current configuration')

    # Status command
    subparsers.add_parser('status', help='Show data source status')

    args = parser.parse_args()

    # DEFAULT: If no command given, run the auto-trader
    if not args.command:
        run_auto_trader()
        return

    # Route to appropriate handler
    if args.command == 'trade':
        run_trading_bot(args)
    elif args.command == 'backtest':
        run_backtest(args)
    elif args.command == 'profiles':
        show_profiles(args)
    elif args.command == 'config':
        show_config(args)
    elif args.command == 'status':
        show_data_status(args)


if __name__ == '__main__':
    main()
