#!/usr/bin/env python3
"""
AI Trading Bot - Interactive Dashboard

Simple numbered menu interface for the trading bot.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def clear_screen():
    """Clear the terminal screen."""
    import os
    os.system('cls' if os.name == 'nt' else 'clear')


def print_header():
    """Print the dashboard header."""
    print("\n" + "=" * 60)
    print("           AI TRADING BOT - DASHBOARD")
    print("=" * 60)


def print_main_menu():
    """Print main menu options."""
    print("\n  MAIN MENU")
    print("  ---------")
    print("  1. Check Data Source Status")
    print("  2. Run Backtest")
    print("  3. Start Trading Bot")
    print("  4. View Personality Profiles")
    print("  5. View Configuration")
    print("  6. Quick Data Test")
    print("  7. View Trade Log")
    print("  0. Exit")
    print()


def check_data_status():
    """Check and display data source status."""
    print("\n  Checking data sources...")

    try:
        from data import MarketDataCollector
        market_data = MarketDataCollector()
        status = market_data.get_data_source_status()

        print("\n" + "-" * 40)
        print("  DATA SOURCE STATUS")
        print("-" * 40)
        print(f"  Alpaca SDK Installed: {status['alpaca']['available']}")
        print(f"  Alpaca Connected: {status['alpaca']['initialized']}")
        print(f"  yfinance Available: {status['yfinance']['available']}")
        print(f"\n  Fallback Order: {' -> '.join(status['fallback_order'])}")
        print("-" * 40)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def run_backtest_menu():
    """Backtest submenu."""
    while True:
        print("\n" + "-" * 40)
        print("  BACKTEST MENU")
        print("-" * 40)
        print("  1. Backtest SPY (default)")
        print("  2. Backtest QQQ")
        print("  3. Backtest AAPL")
        print("  4. Backtest Custom Symbol")
        print("  0. Back to Main Menu")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            run_backtest("SPY")
        elif choice == "2":
            run_backtest("QQQ")
        elif choice == "3":
            run_backtest("AAPL")
        elif choice == "4":
            symbol = input("  Enter symbol: ").strip().upper()
            if symbol:
                run_backtest(symbol)
        else:
            print("  Invalid choice. Try again.")


def run_backtest(symbol: str):
    """Run backtest for a symbol."""
    print(f"\n  Running backtest for {symbol}...")
    print("  Fetching data from Alpaca/yfinance...")

    try:
        from data import MarketDataCollector
        from backtesting import StrategyEvaluator

        market_data = MarketDataCollector()
        evaluator = StrategyEvaluator()

        # Get data
        df = market_data.get_historical_data(
            symbol,
            start_date="2023-01-01",
            end_date="2024-12-31"
        )

        if df.empty:
            print(f"\n  No data available for {symbol}")
            input("\n  Press Enter to continue...")
            return

        print(f"  Retrieved {len(df)} bars of data")
        print("  Evaluating strategies...")

        # Run evaluation
        results = evaluator.evaluate_all_strategies(df)

        # Print results
        print("\n" + "=" * 60)
        print(f"  BACKTEST RESULTS - {symbol}")
        print("=" * 60)

        if not results.empty:
            for _, row in results.iterrows():
                print(f"\n  {row['strategy_name']}")
                print(f"    Return: {row['total_return']:.2f}%")
                print(f"    Sharpe: {row['sharpe_ratio']:.2f}")
                print(f"    Win Rate: {row['win_rate']:.1f}%")
                print(f"    Max Drawdown: {row['max_drawdown']:.2f}%")

        print("=" * 60)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def start_trading_menu():
    """Trading bot submenu."""
    print("\n" + "-" * 40)
    print("  START TRADING BOT")
    print("-" * 40)
    print("  1. Paper Trading (Safe)")
    print("  2. Live Trading (Real Money)")
    print("  0. Back to Main Menu")
    print()

    choice = input("  Enter choice: ").strip()

    if choice == "1":
        start_trading("paper")
    elif choice == "2":
        confirm = input("  Are you sure? This uses REAL money! (yes/no): ").strip().lower()
        if confirm == "yes":
            start_trading("live")
        else:
            print("  Cancelled.")
            input("\n  Press Enter to continue...")
    elif choice == "0":
        return
    else:
        print("  Invalid choice.")
        input("\n  Press Enter to continue...")


def start_trading(mode: str):
    """Start the trading bot."""
    print(f"\n  Starting trading bot in {mode.upper()} mode...")
    print("  Press Ctrl+C to stop\n")

    try:
        from core.trading_bot import TradingBot

        bot = TradingBot(
            initial_capital=100000.0,
            personality="balanced_growth",
            mode=mode
        )

        bot.start()

    except KeyboardInterrupt:
        print("\n  Trading bot stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_profiles():
    """View personality profiles."""
    try:
        from core.personality_profiles import PERSONALITY_PROFILES

        print("\n" + "=" * 60)
        print("  PERSONALITY PROFILES")
        print("=" * 60)

        for i, (name, profile) in enumerate(PERSONALITY_PROFILES.items(), 1):
            print(f"\n  {i}. {name}")
            print(f"     {profile.description}")
            print(f"     Risk: {profile.risk_tolerance}")
            print(f"     Style: {profile.trading_style}")

        print("=" * 60)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_config():
    """View current configuration."""
    try:
        from config import config
        import yaml

        print("\n" + "=" * 60)
        print("  CONFIGURATION")
        print("=" * 60)

        # Show key settings
        print(f"\n  Trading Mode: {config.get('trading.mode', 'paper')}")
        print(f"  Initial Capital: ${config.get('trading.initial_capital', 100000):,.2f}")
        print(f"  Max Position Size: {config.get('trading.max_position_size', 0.1)*100:.0f}%")
        print(f"  Data Source: {config.get('data.sources.market_data.primary', 'alpaca')}")
        print(f"  Backup Source: {config.get('data.sources.market_data.backup', 'yfinance')}")

        print("=" * 60)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def quick_data_test():
    """Quick test to fetch data."""
    print("\n  Testing data fetch for SPY...")

    try:
        from data import MarketDataCollector

        market_data = MarketDataCollector()

        # Get quote
        print("\n  Fetching real-time quote...")
        quote = market_data.get_real_time_quote("SPY")

        if quote:
            print(f"\n  SPY Quote:")
            print(f"    Price: ${quote.get('price', 'N/A')}")
            print(f"    Bid: ${quote.get('bid', 'N/A')}")
            print(f"    Ask: ${quote.get('ask', 'N/A')}")
            print(f"    Source: {quote.get('source', 'N/A')}")
        else:
            print("  Could not get quote")

        # Get historical
        print("\n  Fetching historical data (last 5 days)...")
        df = market_data.get_historical_data("SPY")

        if not df.empty:
            print(f"\n  Last 5 days:")
            print(df[['open', 'high', 'low', 'close', 'volume']].tail().to_string())
        else:
            print("  Could not get historical data")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def view_trade_log():
    """View trade log with reasoning."""
    while True:
        print("\n" + "-" * 60)
        print("  TRADE LOG VIEWER")
        print("-" * 60)
        print("  1. View Recent Trades")
        print("  2. View Trades by Symbol")
        print("  3. View Trades by Strategy")
        print("  4. View Trades by Mode (live/paper/backtest)")
        print("  5. View Trade Summary")
        print("  6. View Trade Details")
        print("  0. Back to Main Menu")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            view_recent_trades()
        elif choice == "2":
            symbol = input("  Enter symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                view_trades_by_filter(symbol=symbol)
        elif choice == "3":
            view_trades_by_strategy()
        elif choice == "4":
            view_trades_by_mode()
        elif choice == "5":
            view_trade_summary()
        elif choice == "6":
            trade_id = input("  Enter trade ID: ").strip()
            if trade_id:
                view_trade_details(trade_id)
        else:
            print("  Invalid choice.")


def view_recent_trades(n: int = 20):
    """View recent trades."""
    try:
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            print("\n  No trades logged yet.")
            input("\n  Press Enter to continue...")
            return

        df = df.head(n)

        print("\n" + "=" * 110)
        print("  RECENT TRADES")
        print("=" * 110)
        print(f"  {'Trade ID':<18} {'Mode':<8} {'Time':<10} {'Symbol':<8} {'Action':<6} {'Qty':<8} {'Price':<10} {'P&L':<10} {'Reason':<20}")
        print("-" * 110)

        for _, row in df.iterrows():
            pnl_str = f"${row['realized_pnl']:.2f}" if row['realized_pnl'] else "-"
            reason = row['primary_signal'][:19] if row['primary_signal'] else "N/A"
            time_str = row['timestamp'][11:19] if len(row['timestamp']) > 19 else row['timestamp'][:8]
            mode = row.get('mode', 'backtest')[:7] if row.get('mode') else "backtest"

            print(f"  {row['trade_id']:<18} {mode:<8} {time_str:<10} {row['symbol']:<8} "
                  f"{row['action']:<6} {row['quantity']:<8.2f} ${row['price']:<9.2f} "
                  f"{pnl_str:<10} {reason:<20}")

        print("=" * 100)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_trades_by_filter(symbol: str = None, strategy: str = None):
    """View trades filtered by symbol or strategy."""
    try:
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades(symbol=symbol, strategy=strategy)

        if df.empty:
            filter_desc = symbol if symbol else strategy
            print(f"\n  No trades found for {filter_desc}.")
            input("\n  Press Enter to continue...")
            return

        filter_desc = symbol if symbol else strategy
        print("\n" + "=" * 100)
        print(f"  TRADES FOR: {filter_desc}")
        print("=" * 100)
        print(f"  {'Trade ID':<18} {'Time':<12} {'Action':<6} {'Qty':<8} {'Price':<10} {'P&L':<10} {'Reason':<30}")
        print("-" * 100)

        for _, row in df.iterrows():
            pnl_str = f"${row['realized_pnl']:.2f}" if row['realized_pnl'] else "-"
            reason = row['primary_signal'][:29] if row['primary_signal'] else "N/A"
            time_str = row['timestamp'][11:19] if len(row['timestamp']) > 19 else row['timestamp'][:8]

            print(f"  {row['trade_id']:<18} {time_str:<12} "
                  f"{row['action']:<6} {row['quantity']:<8.2f} ${row['price']:<9.2f} "
                  f"{pnl_str:<10} {reason:<30}")

        print("=" * 100)
        print(f"  Total trades: {len(df)}")

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_trades_by_strategy():
    """View trades grouped by strategy."""
    try:
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            print("\n  No trades logged yet.")
            input("\n  Press Enter to continue...")
            return

        strategies = df['strategy_name'].unique()

        print("\n" + "=" * 60)
        print("  TRADES BY STRATEGY")
        print("=" * 60)

        for i, strategy in enumerate(strategies, 1):
            print(f"  {i}. {strategy}")

        print()
        choice = input("  Select strategy (number): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(strategies):
                view_trades_by_filter(strategy=strategies[idx])
                return
        except ValueError:
            pass

        print("  Invalid selection.")

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_trades_by_mode():
    """View trades filtered by mode (live/paper/backtest)."""
    try:
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            print("\n  No trades logged yet.")
            input("\n  Press Enter to continue...")
            return

        # Get unique modes
        modes = df['mode'].unique() if 'mode' in df.columns else ['backtest']

        print("\n" + "=" * 60)
        print("  TRADES BY MODE")
        print("=" * 60)

        for i, mode in enumerate(modes, 1):
            count = len(df[df['mode'] == mode]) if 'mode' in df.columns else len(df)
            print(f"  {i}. {mode.upper()} ({count} trades)")

        print()
        choice = input("  Select mode (number): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(modes):
                mode = modes[idx]
                filtered_df = df[df['mode'] == mode] if 'mode' in df.columns else df

                print("\n" + "=" * 110)
                print(f"  {mode.upper()} TRADES")
                print("=" * 110)
                print(f"  {'Trade ID':<18} {'Time':<10} {'Symbol':<8} {'Strategy':<15} {'Action':<6} {'Qty':<8} {'Price':<10} {'P&L':<10}")
                print("-" * 110)

                for _, row in filtered_df.head(30).iterrows():
                    pnl_str = f"${row['realized_pnl']:.2f}" if row['realized_pnl'] else "-"
                    time_str = row['timestamp'][11:19] if len(row['timestamp']) > 19 else row['timestamp'][:8]
                    strategy = row['strategy_name'][:14] if row['strategy_name'] else "N/A"

                    print(f"  {row['trade_id']:<18} {time_str:<10} {row['symbol']:<8} "
                          f"{strategy:<15} {row['action']:<6} {row['quantity']:<8.2f} "
                          f"${row['price']:<9.2f} {pnl_str:<10}")

                print("=" * 110)
                print(f"  Total {mode} trades: {len(filtered_df)}")

                input("\n  Press Enter to continue...")
                return
        except ValueError:
            pass

        print("  Invalid selection.")

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_trade_summary():
    """View trade summary statistics."""
    try:
        from utils.trade_logger import get_trade_logger

        trade_logger = get_trade_logger()
        summary = trade_logger.get_trade_summary()

        print("\n" + "=" * 60)
        print("  TRADE SUMMARY")
        print("=" * 60)
        print(f"  Total Trades: {summary.get('total_trades', 0)}")
        print(f"  Buy Trades: {summary.get('buy_trades', 0)}")
        print(f"  Sell Trades: {summary.get('sell_trades', 0)}")
        print(f"  Completed Round-trips: {summary.get('completed_trades', 0)}")
        print()
        print(f"  Total Realized P&L: ${summary.get('total_realized_pnl', 0):.2f}")
        print(f"  Average P&L per Trade: ${summary.get('avg_realized_pnl', 0):.2f}")
        print(f"  Win Rate: {summary.get('win_rate', 0):.1f}%")
        print()
        print(f"  Strategies Used: {', '.join(summary.get('strategies_used', []))}")
        print(f"  Symbols Traded: {', '.join(summary.get('symbols_traded', []))}")
        print(f"  Most Used Strategy: {summary.get('most_used_strategy', 'N/A')}")

        # Show breakdown by mode
        print()
        print("  BREAKDOWN BY MODE")
        print("  -----------------")
        for mode in ['backtest', 'paper', 'live']:
            mode_summary = trade_logger.get_trade_summary(mode=mode)
            if mode_summary.get('total_trades', 0) > 0:
                print(f"  {mode.upper():8} | Trades: {mode_summary['total_trades']:4} | "
                      f"P&L: ${mode_summary.get('total_realized_pnl', 0):>10.2f} | "
                      f"Win Rate: {mode_summary.get('win_rate', 0):>5.1f}%")

        print("=" * 60)

    except Exception as e:
        print(f"\n  Error: {e}")

    input("\n  Press Enter to continue...")


def view_trade_details(trade_id: str):
    """View detailed information about a specific trade."""
    try:
        from utils.trade_logger import get_trade_logger
        import json

        trade_logger = get_trade_logger()
        df = trade_logger.get_trades()

        if df.empty:
            print("\n  No trades found.")
            input("\n  Press Enter to continue...")
            return

        # Find the trade
        trade = df[df['trade_id'] == trade_id]

        if trade.empty:
            print(f"\n  Trade {trade_id} not found.")
            input("\n  Press Enter to continue...")
            return

        row = trade.iloc[0]

        print("\n" + "=" * 70)
        print(f"  TRADE DETAILS: {trade_id}")
        print("=" * 70)

        print(f"\n  BASIC INFO")
        print(f"  -----------")
        print(f"  Timestamp:    {row['timestamp']}")
        print(f"  Symbol:       {row['symbol']}")
        print(f"  Action:       {row['action']}")
        print(f"  Side:         {row['side']}")
        print(f"  Quantity:     {row['quantity']:.4f}")
        print(f"  Price:        ${row['price']:.2f}")
        print(f"  Total Value:  ${row['total_value']:.2f}")
        print(f"  Mode:         {row['mode']}")

        print(f"\n  STRATEGY")
        print(f"  --------")
        print(f"  Name:         {row['strategy_name']}")

        print(f"\n  DECISION REASONING")
        print(f"  ------------------")
        print(f"  Primary Signal:   {row['primary_signal']}")
        print(f"  Signal Value:     {row['signal_value']}")
        print(f"  Threshold:        {row['threshold']}")
        print(f"  Direction:        {row['direction']}")
        print(f"  Trend:            {row['trend_direction']}")
        print(f"  Volatility:       {row['volatility_level']}")
        print(f"  Volume:           {row['volume_condition']}")

        if row['explanation']:
            print(f"\n  EXPLANATION")
            print(f"  -----------")
            # Word wrap the explanation
            explanation = row['explanation']
            while len(explanation) > 60:
                print(f"  {explanation[:60]}")
                explanation = explanation[60:]
            if explanation:
                print(f"  {explanation}")

        if row['realized_pnl']:
            print(f"\n  PERFORMANCE")
            print(f"  -----------")
            print(f"  Realized P&L:     ${row['realized_pnl']:.2f}")
            print(f"  Realized P&L %:   {row['realized_pnl_pct']:.2f}%")
            print(f"  Holding Period:   {row['holding_period_days']:.1f} days")
            print(f"  Entry Trade ID:   {row['entry_trade_id']}")

        if row['supporting_indicators']:
            print(f"\n  SUPPORTING INDICATORS")
            print(f"  ---------------------")
            try:
                indicators = json.loads(row['supporting_indicators'])
                for key, value in indicators.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")
            except:
                print(f"  {row['supporting_indicators']}")

        print("=" * 70)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def main():
    """Main dashboard loop."""
    while True:
        clear_screen()
        print_header()
        print_main_menu()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            print("\n  Goodbye!")
            break
        elif choice == "1":
            check_data_status()
        elif choice == "2":
            run_backtest_menu()
        elif choice == "3":
            start_trading_menu()
        elif choice == "4":
            view_profiles()
        elif choice == "5":
            view_config()
        elif choice == "6":
            quick_data_test()
        elif choice == "7":
            view_trade_log()
        else:
            print("  Invalid choice. Try again.")
            input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
