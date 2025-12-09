"""
Backtest Workflow Analysis Script
Runs a backtest and traces all significant steps
"""
import asyncio
import logging
import sys
from datetime import datetime, timedelta
from historical_trainer import HistoricalTrainer

# Set up detailed logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)s | %(name)s:%(funcName)s:%(lineno)d | %(message)s',
    stream=sys.stdout
)

# Reduce noise from some modules
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)

async def run_backtest():
    print('=' * 80)
    print('STARTING BACKTEST WORKFLOW ANALYSIS')
    print('=' * 80)

    # Define symbols to test
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'NFLX', 'SPY']

    trainer = HistoricalTrainer(symbols=symbols)

    # Use a 2-week period to get enough data
    end_date = datetime(2024, 11, 15)
    start_date = datetime(2024, 11, 1)

    print(f'\nDate Range: {start_date.date()} to {end_date.date()}')
    print(f'Initial Capital: ${trainer.initial_capital:,.2f}')
    print(f'Min Confidence: {trainer.min_confidence:.1%}')
    print(f'Symbols to test: {symbols}')
    print()

    results = trainer.train_on_historical(
        start_date=start_date.strftime('%Y-%m-%d'),
        end_date=end_date.strftime('%Y-%m-%d'),
        verbose=True
    )

    print('\n' + '=' * 80)
    print('BACKTEST COMPLETE - FINAL RESULTS')
    print('=' * 80)

    if results:
        print(f"\nPrediction Stats:")
        print(f"  Total Predictions: {results.get('total_predictions', 0)}")
        print(f"  Correct: {results.get('correct_predictions', 0)}")
        print(f"  Accuracy: {results.get('accuracy', 0):.1%}")

        trading = results.get('trading', {})
        print(f"\nTrading Stats:")
        print(f"  Total Trades: {trading.get('total_trades', 0)}")
        print(f"  Winning Trades: {trading.get('winning_trades', 0)}")
        print(f"  Losing Trades: {trading.get('losing_trades', 0)}")
        print(f"  Win Rate: {trading.get('win_rate', 0):.1%}")
        print(f"  Total P&L: ${trading.get('total_pnl', 0):,.2f}")
        print(f"  Final Equity: ${trading.get('final_equity', 0):,.2f}")
        print(f"  Return: {trading.get('total_return', 0):.2%}")

    return results

if __name__ == '__main__':
    asyncio.run(run_backtest())
