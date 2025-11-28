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
    print("  4. Live Market Monitor & Learning")
    print("  5. View Personality Profiles")
    print("  6. View Configuration")
    print("  7. Quick Data Test")
    print("  8. View Trade Log")
    print("  9. Bot Watchdog Status")
    print("  10. Auto Paper Trading (AI Adaptive)")
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

        print("\n" + "=" * 120)
        print("  RECENT TRADES")
        print("=" * 120)
        print(f"  {'Trade ID':<22} {'Mode':<9} {'Date':<12} {'Symbol':<7} {'Action':<6} {'Qty':>10} {'Price':>10} {'P&L':>12} {'Reason':<20}")
        print("-" * 120)

        for _, row in df.iterrows():
            # Format P&L with color indication
            if row['realized_pnl']:
                pnl_val = row['realized_pnl']
                pnl_str = f"${pnl_val:>+10,.2f}"
            else:
                pnl_str = "-".center(12)

            reason = row['primary_signal'][:18] if row['primary_signal'] else "N/A"

            # Extract date from timestamp (format: 2023-01-31T00:00:00-05:00)
            timestamp = str(row['timestamp'])
            if 'T' in timestamp:
                date_str = timestamp[:10]  # Get YYYY-MM-DD part
            else:
                date_str = timestamp[:10]

            mode = row.get('mode', 'backtest')[:8] if row.get('mode') else "backtest"

            print(f"  {row['trade_id']:<22} {mode:<9} {date_str:<12} {row['symbol']:<7} "
                  f"{row['action']:<6} {row['quantity']:>10.2f} ${row['price']:>9.2f} "
                  f"{pnl_str:>12} {reason:<20}")

        print("=" * 120)

        # Show summary
        sells = df[df['realized_pnl'].notna()]
        if not sells.empty:
            total_pnl = sells['realized_pnl'].sum()
            wins = (sells['realized_pnl'] > 0).sum()
            losses = (sells['realized_pnl'] < 0).sum()
            print(f"\n  Summary: {len(sells)} closed trades | Total P&L: ${total_pnl:,.2f} | Wins: {wins} | Losses: {losses}")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

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
        print("\n" + "=" * 110)
        print(f"  TRADES FOR: {filter_desc}")
        print("=" * 110)
        print(f"  {'Trade ID':<22} {'Date':<12} {'Action':<6} {'Qty':>10} {'Price':>10} {'P&L':>12} {'Reason':<25}")
        print("-" * 110)

        for _, row in df.iterrows():
            if row['realized_pnl']:
                pnl_str = f"${row['realized_pnl']:>+10,.2f}"
            else:
                pnl_str = "-".center(12)

            reason = row['primary_signal'][:24] if row['primary_signal'] else "N/A"

            # Extract date from timestamp
            timestamp = str(row['timestamp'])
            date_str = timestamp[:10] if 'T' in timestamp else timestamp[:10]

            print(f"  {row['trade_id']:<22} {date_str:<12} "
                  f"{row['action']:<6} {row['quantity']:>10.2f} ${row['price']:>9.2f} "
                  f"{pnl_str:>12} {reason:<25}")

        print("=" * 110)

        # Summary
        sells = df[df['realized_pnl'].notna()]
        if not sells.empty:
            total_pnl = sells['realized_pnl'].sum()
            print(f"\n  Total trades: {len(df)} | Closed: {len(sells)} | Total P&L: ${total_pnl:,.2f}")
        else:
            print(f"\n  Total trades: {len(df)}")

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

        # Always show all three modes
        all_modes = ['live', 'paper', 'backtest']

        print("\n" + "=" * 60)
        print("  TRADES BY MODE")
        print("=" * 60)

        for i, mode in enumerate(all_modes, 1):
            if 'mode' in df.columns:
                count = len(df[df['mode'] == mode])
            else:
                # Legacy: assume all trades are backtest if no mode column
                count = len(df) if mode == 'backtest' else 0
            print(f"  {i}. {mode.upper()} ({count} trades)")

        print()
        choice = input("  Select mode (number): ").strip()

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_modes):
                mode = all_modes[idx]
                if 'mode' in df.columns:
                    filtered_df = df[df['mode'] == mode]
                else:
                    # Legacy: assume all trades are backtest if no mode column
                    filtered_df = df if mode == 'backtest' else df.head(0)

                print("\n" + "=" * 120)
                print(f"  {mode.upper()} TRADES")
                print("=" * 120)
                print(f"  {'Trade ID':<22} {'Date':<12} {'Symbol':<7} {'Strategy':<18} {'Action':<6} {'Qty':>10} {'Price':>10} {'P&L':>12}")
                print("-" * 120)

                for _, row in filtered_df.head(30).iterrows():
                    if row['realized_pnl']:
                        pnl_str = f"${row['realized_pnl']:>+10,.2f}"
                    else:
                        pnl_str = "-".center(12)

                    # Extract date from timestamp
                    timestamp = str(row['timestamp'])
                    date_str = timestamp[:10] if 'T' in timestamp else timestamp[:10]

                    strategy = row['strategy_name'][:17] if row['strategy_name'] else "N/A"

                    print(f"  {row['trade_id']:<22} {date_str:<12} {row['symbol']:<7} "
                          f"{strategy:<18} {row['action']:<6} {row['quantity']:>10.2f} "
                          f"${row['price']:>9.2f} {pnl_str:>12}")

                print("=" * 120)

                # Summary
                sells = filtered_df[filtered_df['realized_pnl'].notna()]
                if not sells.empty:
                    total_pnl = sells['realized_pnl'].sum()
                    wins = (sells['realized_pnl'] > 0).sum()
                    losses = (sells['realized_pnl'] < 0).sum()
                    print(f"\n  Total {mode} trades: {len(filtered_df)} | P&L: ${total_pnl:,.2f} | Wins: {wins} | Losses: {losses}")
                else:
                    print(f"\n  Total {mode} trades: {len(filtered_df)}")

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


def market_monitor_menu():
    """Live market monitor and learning submenu."""
    while True:
        print("\n" + "=" * 60)
        print("  LIVE MARKET MONITOR & LEARNING")
        print("=" * 60)
        print("  1. Start Live Monitor (continuous)")
        print("  2. Run Single Analysis (all symbols)")
        print("  3. Analyze Specific Symbol")
        print("  4. View Prediction History")
        print("  5. View Learning Statistics")
        print("  6. View Signal Performance")
        print("  7. View Learned Weights")
        print("  0. Back to Main Menu")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            start_live_monitor()
        elif choice == "2":
            run_single_analysis()
        elif choice == "3":
            symbol = input("  Enter symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                run_single_analysis(symbol)
        elif choice == "4":
            view_prediction_history()
        elif choice == "5":
            view_learning_stats()
        elif choice == "6":
            view_signal_performance()
        elif choice == "7":
            view_learned_weights()
        else:
            print("  Invalid choice.")


def start_live_monitor():
    """Start continuous live market monitoring."""
    print("\n  Starting Live Market Monitor...")
    print("  Press Ctrl+C to stop\n")

    try:
        from core.market_monitor import MarketMonitor

        # Get symbols from user or use defaults
        print("  Default symbols: SPY, QQQ, AAPL, MSFT, TSLA")
        custom = input("  Enter custom symbols (comma-separated) or press Enter for defaults: ").strip()

        if custom:
            symbols = [s.strip().upper() for s in custom.split(",")]
        else:
            symbols = None

        monitor = MarketMonitor(symbols=symbols)
        monitor.start()

        print("\n" + "=" * 60)
        print("  LIVE MONITOR RUNNING")
        print("=" * 60)
        print("  The bot is now monitoring markets and making predictions.")
        print("  It will learn from outcomes and adjust signal weights.")
        print()
        print("  Press Ctrl+C to stop monitoring...")
        print("=" * 60)

        # Keep running until interrupted
        import time
        while monitor.is_running:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n  Stopping monitor...")
        monitor.stop()
        print("  Monitor stopped.")
    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def run_single_analysis(symbol: str = None):
    """Run a single analysis pass."""
    try:
        from core.market_monitor import MarketMonitor
        from data import MarketDataCollector

        market_data = MarketDataCollector()

        if symbol:
            symbols = [symbol]
        else:
            symbols = ['SPY', 'QQQ', 'AAPL', 'MSFT', 'TSLA']

        monitor = MarketMonitor(symbols=symbols)

        print("\n" + "=" * 70)
        print("  MARKET ANALYSIS")
        print("=" * 70)

        for sym in symbols:
            print(f"\n  Analyzing {sym}...")

            # Get data
            df = market_data.get_historical_data(sym)
            if df.empty:
                print(f"    Could not get data for {sym}")
                continue

            # Make prediction
            prediction = monitor._analyze_and_predict(sym, df)

            if prediction:
                print(f"\n  {sym} PREDICTION:")
                print(f"  ----------------")
                print(f"    Direction: {prediction.predicted_direction.upper()}")
                print(f"    Confidence: {prediction.confidence:.1f}%")
                print(f"    Current Price: ${prediction.entry_price:.2f}")
                print(f"    Target Price: ${prediction.target_price:.2f}")
                print(f"    Timeframe: {prediction.timeframe}")
                print()
                print(f"    Signals:")
                for signal, value in prediction.signals.items():
                    if isinstance(value, float):
                        print(f"      {signal}: {value:.4f}")
                    else:
                        print(f"      {signal}: {value}")
            else:
                print(f"    No prediction generated for {sym}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def view_prediction_history():
    """View historical predictions."""
    try:
        from core.market_monitor import PredictionTracker

        tracker = PredictionTracker()

        print("\n" + "=" * 90)
        print("  PREDICTION HISTORY")
        print("=" * 90)

        # Get predictions from database
        from utils.database import Database
        db = Database()

        with db.get_connection() as conn:
            import pandas as pd
            df = pd.read_sql_query(
                """SELECT * FROM ai_predictions
                   ORDER BY timestamp DESC
                   LIMIT 50""",
                conn
            )

        if df.empty:
            print("\n  No predictions recorded yet.")
            print("  Start the Live Monitor to begin making predictions.")
            input("\n  Press Enter to continue...")
            return

        print(f"\n  {'Time':<12} {'Symbol':<8} {'Dir':<8} {'Conf':<8} {'Entry':<10} {'Target':<10} {'Result':<10}")
        print("-" * 90)

        for _, row in df.iterrows():
            time_str = row['timestamp'][11:19] if len(str(row['timestamp'])) > 19 else str(row['timestamp'])[:8]
            outcome = "Correct" if row['was_correct'] == 1 else ("Wrong" if row['was_correct'] == 0 else "Pending")
            direction = row['predicted_direction'] if 'predicted_direction' in row else row.get('direction', '?')

            print(f"  {time_str:<12} {row['symbol']:<8} {direction:<8} "
                  f"{row['confidence']:>5.1f}%   ${row['entry_price']:<9.2f} "
                  f"${row['target_price']:<9.2f} {outcome:<10}")

        print("=" * 90)

        # Summary stats
        resolved = df[df['was_correct'].notna()]
        if not resolved.empty:
            correct = (resolved['was_correct'] == 1).sum()
            total = len(resolved)
            accuracy = correct / total * 100
            print(f"\n  Accuracy: {correct}/{total} ({accuracy:.1f}%)")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def view_learning_stats():
    """View learning and accuracy statistics."""
    try:
        from utils.database import Database
        import pandas as pd

        db = Database()

        print("\n" + "=" * 70)
        print("  LEARNING STATISTICS")
        print("=" * 70)

        with db.get_connection() as conn:
            # Overall stats
            df = pd.read_sql_query(
                """SELECT
                      COUNT(*) as total_predictions,
                      SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct,
                      SUM(CASE WHEN was_correct = 0 THEN 1 ELSE 0 END) as incorrect,
                      SUM(CASE WHEN was_correct IS NULL THEN 1 ELSE 0 END) as pending
                   FROM ai_predictions""",
                conn
            )

            if df.iloc[0]['total_predictions'] == 0:
                print("\n  No predictions recorded yet.")
                print("  Start the Live Monitor to begin learning.")
                input("\n  Press Enter to continue...")
                return

            row = df.iloc[0]
            total = row['total_predictions']
            correct = row['correct']
            incorrect = row['incorrect']
            pending = row['pending']
            resolved = correct + incorrect

            print(f"\n  OVERALL")
            print(f"  -------")
            print(f"  Total Predictions: {total}")
            print(f"  Resolved: {resolved}")
            print(f"  Pending: {pending}")
            if resolved > 0:
                accuracy = correct / resolved * 100
                print(f"  Accuracy: {accuracy:.1f}%")

            # By symbol
            df_symbol = pd.read_sql_query(
                """SELECT symbol,
                      COUNT(*) as total,
                      SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
                   FROM ai_predictions
                   WHERE was_correct IS NOT NULL
                   GROUP BY symbol
                   ORDER BY total DESC""",
                conn
            )

            if not df_symbol.empty:
                print(f"\n  BY SYMBOL")
                print(f"  ---------")
                for _, r in df_symbol.iterrows():
                    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
                    print(f"  {r['symbol']:<8} {r['correct']}/{r['total']} ({acc:.1f}%)")

            # By direction
            df_dir = pd.read_sql_query(
                """SELECT predicted_direction as direction,
                      COUNT(*) as total,
                      SUM(CASE WHEN was_correct = 1 THEN 1 ELSE 0 END) as correct
                   FROM ai_predictions
                   WHERE was_correct IS NOT NULL
                   GROUP BY predicted_direction""",
                conn
            )

            if not df_dir.empty:
                print(f"\n  BY DIRECTION")
                print(f"  ------------")
                for _, r in df_dir.iterrows():
                    acc = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
                    print(f"  {r['direction'].upper():<8} {r['correct']}/{r['total']} ({acc:.1f}%)")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def view_signal_performance():
    """View performance of individual signals."""
    try:
        from utils.database import Database
        import pandas as pd
        import json

        db = Database()

        print("\n" + "=" * 70)
        print("  SIGNAL PERFORMANCE")
        print("=" * 70)

        with db.get_connection() as conn:
            # Get all resolved predictions with their signals
            df = pd.read_sql_query(
                """SELECT signals, was_correct
                   FROM ai_predictions
                   WHERE was_correct IS NOT NULL""",
                conn
            )

        if df.empty:
            print("\n  No resolved predictions yet.")
            print("  Wait for predictions to resolve to see signal performance.")
            input("\n  Press Enter to continue...")
            return

        # Analyze signal performance
        signal_stats = {}

        for _, row in df.iterrows():
            try:
                signals = json.loads(row['signals'])
                was_correct = row['was_correct'] == 1

                for signal, value in signals.items():
                    if signal not in signal_stats:
                        signal_stats[signal] = {'correct': 0, 'total': 0, 'values_correct': [], 'values_incorrect': []}

                    signal_stats[signal]['total'] += 1
                    if was_correct:
                        signal_stats[signal]['correct'] += 1
                        if isinstance(value, (int, float)):
                            signal_stats[signal]['values_correct'].append(value)
                    else:
                        if isinstance(value, (int, float)):
                            signal_stats[signal]['values_incorrect'].append(value)
            except:
                continue

        print(f"\n  {'Signal':<25} {'Accuracy':<12} {'Correct':<10} {'Total':<8}")
        print("-" * 70)

        for signal, stats in sorted(signal_stats.items(), key=lambda x: x[1]['correct']/max(x[1]['total'],1), reverse=True):
            accuracy = stats['correct'] / stats['total'] * 100 if stats['total'] > 0 else 0
            print(f"  {signal:<25} {accuracy:>5.1f}%      {stats['correct']:<10} {stats['total']:<8}")

        print("\n" + "=" * 70)

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def view_learned_weights():
    """View the current learned signal weights."""
    try:
        from core.market_monitor import MarketMonitor

        monitor = MarketMonitor()

        print("\n" + "=" * 60)
        print("  LEARNED SIGNAL WEIGHTS")
        print("=" * 60)
        print("\n  These weights are adjusted based on prediction accuracy.")
        print("  Higher weight = more influence on predictions.")
        print()
        print(f"  {'Signal':<30} {'Weight':<10}")
        print("-" * 45)

        for signal, weight in sorted(monitor.signal_weights.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(weight * 10)
            print(f"  {signal:<30} {weight:.3f}  {bar}")

        print("\n" + "=" * 60)
        print("\n  Weights start at 1.0 and adjust based on accuracy.")
        print("  Signals that lead to correct predictions gain weight.")

    except Exception as e:
        print(f"\n  Error: {e}")
        import traceback
        traceback.print_exc()

    input("\n  Press Enter to continue...")


def auto_paper_trading_menu():
    """Auto Paper Trading menu - manages background paper trading with AI Adaptive personality."""
    from pathlib import Path

    # Import auto_trader functions
    script_path = Path(__file__).parent / "auto_trader.py"

    while True:
        clear_screen()
        print("\n" + "=" * 60)
        print("  AUTO PAPER TRADING (AI Adaptive)")
        print("=" * 60)

        # Check if running
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent))
            from auto_trader import is_running, start_background, stop_background, check_status, get_current_holdings, AUTO_START_TIME, PERSONALITY, MODE

            running = is_running()

            print(f"\n  Status: {'RUNNING' if running else 'NOT RUNNING'}")
            print(f"  Scheduled Start: {AUTO_START_TIME} daily (weekdays)")
            print(f"  Trading Frequency: Every 1 minute during market hours")
            print(f"  Personality: {PERSONALITY}")
            print(f"  Mode: {MODE}")

            # Show current holdings
            print("\n  Current Holdings (Watchlist):")
            try:
                holdings = get_current_holdings()
                if holdings:
                    for symbol in holdings:
                        print(f"    - {symbol}")
                else:
                    print("    (none - will use defaults)")
            except Exception:
                print("    (unable to retrieve)")

        except ImportError as e:
            print(f"\n  Error: Could not import auto_trader module: {e}")
            running = False

        print("\n  Options:")
        if running:
            print("  1. Stop Background Trading")
            print("  2. Check Detailed Status")
        else:
            print("  1. Start Background Trading (Hidden)")
            print("  2. Check Detailed Status")
        print("  3. Start Now (Foreground - visible)")
        print("  0. Back to Main Menu")

        print()
        choice = input("  Enter choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            if running:
                print("\n  Stopping background trading...")
                stop_background()
            else:
                print("\n  Starting background trading...")
                start_background()
            input("\n  Press Enter to continue...")
        elif choice == "2":
            print()
            check_status()
            input("\n  Press Enter to continue...")
        elif choice == "3":
            print("\n  Starting paper trading in foreground...")
            print("  (Press Ctrl+C to stop)")
            input("  Press Enter to start...")
            try:
                from auto_trader import start_paper_trading
                start_paper_trading()
            except KeyboardInterrupt:
                print("\n  Stopped by user.")
            input("\n  Press Enter to continue...")
        else:
            print("  Invalid choice.")
            input("\n  Press Enter to continue...")


def view_watchdog_status():
    """View bot watchdog status and controls."""
    from pathlib import Path
    import json
    from datetime import datetime

    while True:
        print("\n" + "=" * 60)
        print("  BOT WATCHDOG STATUS")
        print("=" * 60)

        # Check status file
        status_file = Path(__file__).parent / "logs" / "bot_status.json"

        if status_file.exists():
            try:
                with open(status_file) as f:
                    status = json.load(f)

                print(f"\n  Status: {status.get('status', 'Unknown').upper()}")
                print(f"  Message: {status.get('message', 'N/A')}")
                print(f"  Last Update: {status.get('timestamp', 'N/A')}")
                print(f"  Bot PID: {status.get('pid', 'N/A')}")
                print(f"  Restart Count: {status.get('restart_count', 0)}")
                print(f"  Market Hours: {'Yes' if status.get('is_market_hours') else 'No'}")
                print(f"  Core Hours: {'Yes' if status.get('is_core_hours') else 'No'}")

            except Exception as e:
                print(f"\n  Error reading status: {e}")
        else:
            print("\n  Watchdog not running or no status file found.")
            print("  Run 'start_watchdog.bat' to start the watchdog.")

        # Check if market is open
        now = datetime.now()
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute

        print("\n  CURRENT TIME INFO")
        print("  -----------------")
        print(f"  Current Time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Day: {['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'][weekday]}")

        if weekday < 5:  # Weekday
            if 8 <= hour < 18:
                if 9*60+30 <= hour*60+minute <= 16*60:
                    print("  Market Status: OPEN (Core Hours)")
                else:
                    print("  Market Status: Extended Hours")
            else:
                print("  Market Status: CLOSED")
        else:
            print("  Market Status: WEEKEND - CLOSED")

        print("\n" + "-" * 60)
        print("  OPTIONS")
        print("-" * 60)
        print("  1. Refresh Status")
        print("  2. Start Watchdog (opens new window)")
        print("  3. Setup Auto-Start on Windows Login")
        print("  0. Back to Main Menu")
        print()

        choice = input("  Enter choice: ").strip()

        if choice == "0":
            break
        elif choice == "1":
            continue  # Refresh
        elif choice == "2":
            import subprocess
            bat_file = Path(__file__).parent / "start_watchdog.bat"
            if bat_file.exists():
                subprocess.Popen(['cmd', '/c', 'start', str(bat_file)], shell=True)
                print("\n  Watchdog starting in new window...")
                input("  Press Enter to continue...")
            else:
                print("\n  start_watchdog.bat not found!")
                input("  Press Enter to continue...")
        elif choice == "3":
            import subprocess
            bat_file = Path(__file__).parent / "setup_startup.bat"
            if bat_file.exists():
                subprocess.Popen(['cmd', '/c', 'start', str(bat_file)], shell=True)
                print("\n  Setup instructions opening in new window...")
                input("  Press Enter to continue...")
            else:
                print("\n  setup_startup.bat not found!")
                input("  Press Enter to continue...")


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
            market_monitor_menu()
        elif choice == "5":
            view_profiles()
        elif choice == "6":
            view_config()
        elif choice == "7":
            quick_data_test()
        elif choice == "8":
            view_trade_log()
        elif choice == "9":
            view_watchdog_status()
        elif choice == "10":
            auto_paper_trading_menu()
        else:
            print("  Invalid choice. Try again.")
            input("\n  Press Enter to continue...")


if __name__ == "__main__":
    main()
