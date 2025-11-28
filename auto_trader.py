#!/usr/bin/env python3
"""
Auto Paper Trading Scheduler

Automatically starts paper trading every weekday at 8:30 AM ET (1 hour before market open).
Uses AI Adaptive personality and trades stocks currently held in positions.
Runs as a background process (hidden, no console window).

Usage:
    python auto_trader.py              # Run the scheduler (keeps running)
    python auto_trader.py --now        # Start paper trading immediately
    python auto_trader.py --status     # Check current status
    python auto_trader.py --background # Start scheduler as hidden background process
    python auto_trader.py --stop       # Stop the background process
"""

import sys
import os
import time
import argparse
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import schedule
from utils.logger import get_logger
from utils.database import Database
from utils.daily_summary import (
    get_daily_summary_generator,
    get_self_reflection_analyzer,
    set_daily_goals
)
from config import config

logger = get_logger(__name__)

# PID file for tracking background process
PID_FILE = Path(__file__).parent / "auto_trader.pid"

# Configuration
AUTO_START_TIME = "08:30"  # 1 hour before market open (9:30 AM ET)
AUTO_END_TIME = "16:05"  # 5 minutes after market close for daily report
PERSONALITY = "ai_adaptive"  # AI Adaptive trading personality
MODE = "paper"


def get_current_holdings() -> list:
    """Get list of symbols currently held in positions from Alpaca account."""
    symbols = []

    # Try to get positions from Alpaca (live account data)
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca.api_key', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca.secret_key', '')

        # Remove ${} wrapper if present (from config file)
        if api_key.startswith('${'):
            api_key = os.getenv(api_key[2:-1], '')
        if secret_key.startswith('${'):
            secret_key = os.getenv(secret_key[2:-1], '')

        if api_key and secret_key:
            # Use paper=True since we're paper trading
            client = TradingClient(api_key, secret_key, paper=True)

            # Get all positions from Alpaca
            positions = client.get_all_positions()

            for position in positions:
                symbols.append(position.symbol)
                logger.debug(f"  {position.symbol}: {position.qty} shares @ ${float(position.avg_entry_price):.2f} "
                           f"(P/L: ${float(position.unrealized_pl):.2f})")

            if symbols:
                logger.info(f"Alpaca holdings ({len(symbols)}): {', '.join(symbols)}")
            else:
                logger.info("No positions found in Alpaca account")

    except ImportError:
        logger.warning("Alpaca SDK not available - cannot fetch live positions")
    except Exception as e:
        logger.warning(f"Error fetching Alpaca positions: {e}")

    # If no Alpaca positions, fall back to defaults
    if not symbols:
        logger.info("Using default watchlist")
        symbols = config.get('data.universe.initial_stocks', ['SPY', 'QQQ', 'IWM'])

    return symbols


def get_account_info() -> dict:
    """Get account information from Alpaca."""
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca.api_key', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca.secret_key', '')

        # Remove ${} wrapper if present
        if api_key.startswith('${'):
            api_key = os.getenv(api_key[2:-1], '')
        if secret_key.startswith('${'):
            secret_key = os.getenv(secret_key[2:-1], '')

        if api_key and secret_key:
            client = TradingClient(api_key, secret_key, paper=True)
            account = client.get_account()

            return {
                'buying_power': float(account.buying_power),
                'cash': float(account.cash),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'daytrade_count': account.daytrade_count,
                'pattern_day_trader': account.pattern_day_trader
            }
    except Exception as e:
        logger.warning(f"Error fetching account info: {e}")

    return {}


def generate_morning_goals():
    """Set daily goals at the start of trading day."""
    now = datetime.now()

    # Skip weekends
    if now.weekday() >= 5:
        return

    logger.info("=" * 60)
    logger.info("SETTING DAILY GOALS")
    logger.info("=" * 60)

    try:
        goals = set_daily_goals()
        logger.info(f"Set {len(goals)} goals for today:")
        for goal in goals:
            logger.info(f"  - {goal.description}")
    except Exception as e:
        logger.error(f"Error setting daily goals: {e}")


def generate_end_of_day_report():
    """Generate daily summary and self-reflection reports at end of trading day."""
    now = datetime.now()

    # Skip weekends
    if now.weekday() >= 5:
        return

    logger.info("=" * 60)
    logger.info("GENERATING END-OF-DAY REPORTS")
    logger.info("=" * 60)

    try:
        # Generate daily summary
        summary_gen = get_daily_summary_generator()
        summary = summary_gen.generate_summary()

        logger.info("\nDAILY PERFORMANCE SUMMARY")
        logger.info("-" * 40)
        logger.info(f"Portfolio Value: ${summary.ending_portfolio_value:,.2f}")
        logger.info(f"Daily P&L: ${summary.daily_pnl:+,.2f} ({summary.daily_return_pct:+.2f}%)")
        logger.info(f"Trades: {summary.total_trades} ({summary.winning_trades}W / {summary.losing_trades}L)")
        logger.info(f"Win Rate: {summary.win_rate:.1f}%")
        logger.info(f"Goals Achieved: {summary.goals_achieved}/{summary.goals_total}")

        if summary.best_performing_stock:
            logger.info(f"Best Stock: {summary.best_performing_stock}")
        if summary.worst_performing_stock:
            logger.info(f"Worst Stock: {summary.worst_performing_stock}")

        # Generate self-reflection
        analyzer = get_self_reflection_analyzer()
        analysis = analyzer.analyze_performance(days=30)
        analyzer.save_reflection_report()

        logger.info("\nSELF-REFLECTION INSIGHTS")
        logger.info("-" * 40)
        logger.info(analysis.get('overall_assessment', 'No assessment available'))

        if analysis.get('areas_to_improve'):
            logger.info(f"Areas to improve: {', '.join(analysis['areas_to_improve'][:3])}")

        logger.info("\nReports saved to logs/daily_summaries/ and logs/self_reflection/")

    except Exception as e:
        logger.error(f"Error generating end-of-day report: {e}")
        import traceback
        traceback.print_exc()


def start_paper_trading():
    """Start the paper trading bot."""
    now = datetime.now()

    # Skip weekends
    if now.weekday() >= 5:
        logger.info(f"Skipping - today is {now.strftime('%A')} (weekend)")
        return

    # Set daily goals at start of trading
    generate_morning_goals()

    logger.info("=" * 60)
    logger.info("AUTO PAPER TRADING - STARTING")
    logger.info("=" * 60)
    logger.info(f"Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Mode: {MODE}")
    logger.info(f"Personality: {PERSONALITY}")

    try:
        from core.trading_bot import TradingBot
        from core.personality_profiles import get_profile

        # Get current holdings for watchlist
        holdings = get_current_holdings()
        logger.info(f"Watchlist: {', '.join(holdings)}")

        # Create and configure the bot
        bot = TradingBot(mode=MODE, personality=PERSONALITY)

        # Override watchlist with current holdings
        bot._holdings_watchlist = holdings

        # Store original _get_watchlist and override
        original_get_watchlist = bot._get_watchlist
        def custom_watchlist():
            # Return holdings first, then add any new positions
            symbols = list(holdings)
            current_positions = list(bot.portfolio.position_tracker.get_all_positions().keys())
            for pos in current_positions:
                if pos not in symbols:
                    symbols.append(pos)
            return symbols

        bot._get_watchlist = custom_watchlist

        logger.info("Starting trading bot...")
        bot.start()

    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user")
    except Exception as e:
        logger.error(f"Error starting paper trading: {e}")
        import traceback
        traceback.print_exc()


def check_status():
    """Check current status of auto trader."""
    print("\n" + "=" * 60)
    print("  AUTO PAPER TRADING STATUS")
    print("=" * 60)

    now = datetime.now()
    print(f"\n  Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Day: {now.strftime('%A')}")
    print(f"  Scheduled Start: {AUTO_START_TIME} (daily on weekdays)")
    print(f"  Trading Frequency: Every 1 minute during market hours")
    print(f"  Mode: {MODE}")
    print(f"  Personality: {PERSONALITY}")

    # Check if weekend
    if now.weekday() >= 5:
        print(f"\n  Market Status: WEEKEND - Trading will resume Monday")
    else:
        # Calculate next start time
        scheduled_time = datetime.strptime(AUTO_START_TIME, "%H:%M").time()
        market_open = now.replace(hour=9, minute=30, second=0)
        market_close = now.replace(hour=16, minute=0, second=0)

        if now < market_open:
            print(f"\n  Market Status: Pre-market (opens at 9:30 AM)")
        elif now > market_close:
            print(f"\n  Market Status: After-hours (closed)")
        else:
            print(f"\n  Market Status: OPEN (closes at 4:00 PM)")

    # Show account info
    print("\n  Alpaca Account:")
    account = get_account_info()
    if account:
        print(f"    Portfolio Value: ${account.get('portfolio_value', 0):,.2f}")
        print(f"    Cash: ${account.get('cash', 0):,.2f}")
        print(f"    Buying Power: ${account.get('buying_power', 0):,.2f}")
        print(f"    Long Market Value: ${account.get('long_market_value', 0):,.2f}")
        print(f"    Day Trades: {account.get('daytrade_count', 0)}")
    else:
        print("    (unable to fetch account info)")

    # Show current holdings with details
    print("\n  Current Holdings (from Alpaca):")
    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv('ALPACA_API_KEY') or config.get('api_keys.alpaca.api_key', '')
        secret_key = os.getenv('ALPACA_SECRET_KEY') or config.get('api_keys.alpaca.secret_key', '')

        if api_key.startswith('${'):
            api_key = os.getenv(api_key[2:-1], '')
        if secret_key.startswith('${'):
            secret_key = os.getenv(secret_key[2:-1], '')

        if api_key and secret_key:
            client = TradingClient(api_key, secret_key, paper=True)
            positions = client.get_all_positions()

            if positions:
                total_pl = 0
                print(f"    {'Symbol':<8} {'Qty':>8} {'Avg Price':>12} {'Current':>12} {'P/L':>12} {'P/L %':>8}")
                print("    " + "-" * 64)
                for pos in positions:
                    qty = float(pos.qty)
                    avg_price = float(pos.avg_entry_price)
                    current_price = float(pos.current_price)
                    pl = float(pos.unrealized_pl)
                    pl_pct = float(pos.unrealized_plpc) * 100
                    total_pl += pl
                    print(f"    {pos.symbol:<8} {qty:>8.2f} ${avg_price:>10.2f} ${current_price:>10.2f} ${pl:>10.2f} {pl_pct:>7.2f}%")
                print("    " + "-" * 64)
                print(f"    {'TOTAL':<8} {len(positions):>8} positions{' '*24} ${total_pl:>10.2f}")
            else:
                print("    (no positions)")
        else:
            print("    (Alpaca API keys not configured)")
    except Exception as e:
        print(f"    (error: {e})")

    print("\n" + "=" * 60)


def run_scheduler():
    """Run the auto-start scheduler."""
    logger.info("=" * 60)
    logger.info("AUTO PAPER TRADING SCHEDULER")
    logger.info("=" * 60)
    logger.info(f"Scheduled start time: {AUTO_START_TIME} (weekdays)")
    logger.info(f"Scheduled report time: {AUTO_END_TIME} (weekdays)")
    logger.info(f"Mode: {MODE}")
    logger.info(f"Personality: {PERSONALITY}")
    logger.info("Press Ctrl+C to stop the scheduler")
    logger.info("=" * 60)

    # Schedule daily start (morning - sets goals and starts trading)
    schedule.every().day.at(AUTO_START_TIME).do(start_paper_trading)

    # Schedule end-of-day report (after market close)
    schedule.every().day.at(AUTO_END_TIME).do(generate_end_of_day_report)

    # Also check if we should start now (if within trading window)
    now = datetime.now()
    scheduled_time = datetime.strptime(AUTO_START_TIME, "%H:%M").time()
    market_end = now.replace(hour=16, minute=0, second=0)

    if now.weekday() < 5:  # Weekday
        if now.time() >= scheduled_time and now < market_end:
            logger.info("Within trading window - starting immediately")
            start_paper_trading()

    # Save PID for background tracking
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))

    # Run scheduler loop
    try:
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    except KeyboardInterrupt:
        logger.info("Scheduler stopped by user")
    finally:
        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()


def start_background():
    """Start the auto trader as a hidden background process (no console window)."""
    if is_running():
        print("  Auto trader is already running in background!")
        return False

    script_path = Path(__file__).resolve()
    python_exe = sys.executable

    try:
        if sys.platform == 'win32':
            # Windows: Use pythonw.exe for no console, or CREATE_NO_WINDOW flag
            pythonw = python_exe.replace('python.exe', 'pythonw.exe')
            if os.path.exists(pythonw):
                # Use pythonw (no console window)
                subprocess.Popen(
                    [pythonw, str(script_path)],
                    cwd=str(script_path.parent),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            else:
                # Fallback: use CREATE_NO_WINDOW flag
                DETACHED_PROCESS = 0x00000008
                CREATE_NO_WINDOW = 0x08000000
                subprocess.Popen(
                    [python_exe, str(script_path)],
                    cwd=str(script_path.parent),
                    creationflags=DETACHED_PROCESS | CREATE_NO_WINDOW,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
        else:
            # Linux/Mac: Use nohup-style background
            subprocess.Popen(
                [python_exe, str(script_path)],
                cwd=str(script_path.parent),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True
            )

        print("\n  Auto Paper Trading started in background!")
        print(f"  - Scheduled: {AUTO_START_TIME} daily (weekdays)")
        print(f"  - Personality: {PERSONALITY}")
        print(f"  - Mode: {MODE}")
        print("\n  Use 'python auto_trader.py --status' to check status")
        print("  Use 'python auto_trader.py --stop' to stop")
        return True

    except Exception as e:
        print(f"  Error starting background process: {e}")
        return False


def stop_background():
    """Stop the background auto trader process."""
    if not is_running():
        print("  Auto trader is not running.")
        return False

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        if sys.platform == 'win32':
            # Windows: use taskkill
            subprocess.run(['taskkill', '/F', '/PID', str(pid)],
                         capture_output=True)
        else:
            # Linux/Mac: use kill
            os.kill(pid, 9)

        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

        print(f"  Auto trader stopped (PID: {pid})")
        return True

    except Exception as e:
        print(f"  Error stopping process: {e}")
        # Try to clean up PID file anyway
        if PID_FILE.exists():
            PID_FILE.unlink()
        return False


def is_running() -> bool:
    """Check if the auto trader is running in background."""
    if not PID_FILE.exists():
        return False

    try:
        with open(PID_FILE, 'r') as f:
            pid = int(f.read().strip())

        if sys.platform == 'win32':
            # Windows: check if process exists
            result = subprocess.run(
                ['tasklist', '/FI', f'PID eq {pid}'],
                capture_output=True, text=True
            )
            return str(pid) in result.stdout
        else:
            # Linux/Mac: check if process exists
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(description="Auto Paper Trading Scheduler")
    parser.add_argument("--now", action="store_true", help="Start paper trading immediately")
    parser.add_argument("--status", action="store_true", help="Check current status")
    parser.add_argument("--background", action="store_true", help="Start as hidden background process")
    parser.add_argument("--stop", action="store_true", help="Stop the background process")
    parser.add_argument("--report", action="store_true", help="Generate daily report now")
    parser.add_argument("--goals", action="store_true", help="Set daily goals now")

    args = parser.parse_args()

    if args.status:
        check_status()
        if is_running():
            print("  Background Process: RUNNING")
        else:
            print("  Background Process: NOT RUNNING")
    elif args.stop:
        stop_background()
    elif args.background:
        start_background()
    elif args.now:
        start_paper_trading()
    elif args.report:
        generate_end_of_day_report()
    elif args.goals:
        generate_morning_goals()
    else:
        run_scheduler()


if __name__ == "__main__":
    main()
