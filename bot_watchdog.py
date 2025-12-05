"""
Bot Watchdog - Ensures the trading bot is always running during market hours.

This script:
1. Runs on Windows startup
2. Monitors if the bot is running
3. Restarts the bot if it crashes during market hours
4. Sends notifications if there are issues
"""

import os
import sys
import time
import subprocess
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Setup logging
LOG_FILE = PROJECT_ROOT / "logs" / "watchdog.log"
LOG_FILE.parent.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BotWatchdog:
    """Watchdog to keep the trading bot running."""

    def __init__(self):
        self.bot_process = None
        self.restart_count = 0
        self.max_restarts_per_hour = 5
        self.restart_times = []
        self.status_file = PROJECT_ROOT / "logs" / "bot_status.json"

        # Load settings from config
        try:
            from config import config
            self.check_interval = config.get('watchdog.check_interval_seconds', 30)
            self.market_open_hour = config.get('market_hours.open_hour', 9)
            self.market_open_minute = config.get('market_hours.open_minute', 30)
            self.market_close_hour = config.get('market_hours.close_hour', 16)
            self.market_close_minute = config.get('market_hours.close_minute', 0)
            self.extended_start_hour = config.get('market_hours.extended_start_hour', 8)
            self.extended_end_hour = config.get('market_hours.extended_end_hour', 18)
        except ImportError:
            # Fallback defaults if config not available
            self.check_interval = 30
            self.market_open_hour = 9
            self.market_open_minute = 30
            self.market_close_hour = 16
            self.market_close_minute = 0
            self.extended_start_hour = 8
            self.extended_end_hour = 18

    def is_market_hours(self) -> bool:
        """Check if we're in market hours (including extended)."""
        now = datetime.now()

        # Skip weekends
        if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return False

        current_time = now.hour * 60 + now.minute
        extended_start = self.extended_start_hour * 60
        extended_end = self.extended_end_hour * 60

        return extended_start <= current_time <= extended_end

    def is_core_market_hours(self) -> bool:
        """Check if we're in core market hours (9:30 AM - 4:00 PM)."""
        now = datetime.now()

        if now.weekday() >= 5:
            return False

        current_time = now.hour * 60 + now.minute
        market_open = self.market_open_hour * 60 + self.market_open_minute
        market_close = self.market_close_hour * 60 + self.market_close_minute

        return market_open <= current_time <= market_close

    def can_restart(self) -> bool:
        """Check if we haven't exceeded restart limits."""
        now = datetime.now()

        # Remove restart times older than 1 hour
        self.restart_times = [t for t in self.restart_times
                             if now - t < timedelta(hours=1)]

        return len(self.restart_times) < self.max_restarts_per_hour

    def is_bot_running(self) -> bool:
        """Check if the bot process is still running."""
        if self.bot_process is None:
            return False

        # Check if process is still alive
        poll = self.bot_process.poll()
        return poll is None  # None means still running

    def start_bot(self, mode: str = "paper") -> bool:
        """Start the trading bot."""
        try:
            logger.info(f"Starting trading bot in {mode} mode...")

            # Build command to run the bot
            python_exe = sys.executable
            bot_script = PROJECT_ROOT / "run_bot.py"

            # Create run_bot.py if it doesn't exist
            if not bot_script.exists():
                self._create_run_bot_script()

            # Start the bot process
            self.bot_process = subprocess.Popen(
                [python_exe, str(bot_script), "--mode", mode],
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )

            # Record restart
            self.restart_times.append(datetime.now())
            self.restart_count += 1

            # Update status
            self._update_status("running", f"Bot started (restart #{self.restart_count})")

            logger.info(f"Bot started with PID: {self.bot_process.pid}")
            return True

        except Exception as e:
            logger.error(f"Failed to start bot: {e}")
            self._update_status("error", str(e))
            return False

    def stop_bot(self):
        """Stop the trading bot gracefully."""
        if self.bot_process:
            logger.info("Stopping trading bot...")
            try:
                self.bot_process.terminate()
                self.bot_process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("Bot didn't stop gracefully, forcing kill...")
                self.bot_process.kill()

            self.bot_process = None
            self._update_status("stopped", "Bot stopped by watchdog")

    def _create_run_bot_script(self):
        """Create the run_bot.py script."""
        script_content = '''"""Run the trading bot."""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.trading_bot import TradingBot

def main():
    parser = argparse.ArgumentParser(description='Run the AI Trading Bot')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode (paper or live)')
    parser.add_argument('--capital', type=float, default=100000.0,
                       help='Initial capital')
    parser.add_argument('--personality', default='balanced_growth',
                       help='Trading personality profile')

    args = parser.parse_args()

    bot = TradingBot(
        initial_capital=args.capital,
        personality=args.personality,
        mode=args.mode
    )

    bot.start()

if __name__ == "__main__":
    main()
'''
        with open(PROJECT_ROOT / "run_bot.py", 'w') as f:
            f.write(script_content)

        logger.info("Created run_bot.py script")

    def _update_status(self, status: str, message: str):
        """Update the status file."""
        status_data = {
            "status": status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "pid": self.bot_process.pid if self.bot_process else None,
            "restart_count": self.restart_count,
            "is_market_hours": self.is_market_hours(),
            "is_core_hours": self.is_core_market_hours()
        }

        with open(self.status_file, 'w') as f:
            json.dump(status_data, f, indent=2)

    def run(self, mode: str = "paper"):
        """Main watchdog loop."""
        logger.info("=" * 60)
        logger.info("BOT WATCHDOG STARTED")
        logger.info("=" * 60)
        logger.info(f"Mode: {mode}")
        logger.info(f"Check interval: {self.check_interval} seconds")
        logger.info(f"Max restarts per hour: {self.max_restarts_per_hour}")
        logger.info("=" * 60)

        self._update_status("watchdog_started", "Watchdog monitoring active")

        try:
            while True:
                try:
                    self._check_and_maintain()
                except Exception as e:
                    logger.error(f"Error in watchdog loop: {e}")

                time.sleep(self.check_interval)

        except KeyboardInterrupt:
            logger.info("Watchdog interrupted by user")
            self.stop_bot()
            self._update_status("stopped", "Watchdog stopped by user")

    def _check_and_maintain(self):
        """Check bot status and restart if needed."""
        now = datetime.now()

        # During market hours, ensure bot is running
        if self.is_market_hours():
            if not self.is_bot_running():
                if self.can_restart():
                    logger.warning("Bot is not running during market hours! Restarting...")
                    self.start_bot()
                else:
                    logger.error("Too many restarts in the past hour. Manual intervention needed!")
                    self._update_status("error", "Too many restarts - manual intervention needed")
            else:
                # Bot is running fine
                if now.minute % 5 == 0 and now.second < self.check_interval:
                    logger.info(f"Bot running normally (PID: {self.bot_process.pid})")
                    self._update_status("running", "Bot running normally")
        else:
            # Outside market hours
            if self.is_bot_running():
                # Keep running for learning/analysis but log it
                if now.minute == 0 and now.second < self.check_interval:
                    logger.info("Bot running outside market hours (for learning/analysis)")
                    self._update_status("running_off_hours", "Running for analysis")
            else:
                # Not running outside hours is OK, but log status
                self._update_status("idle", "Outside market hours - bot idle")


def create_startup_task():
    """Create Windows scheduled task to run watchdog on startup."""
    import subprocess

    python_exe = sys.executable
    script_path = Path(__file__).absolute()

    # Create the scheduled task XML
    task_xml = f'''<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Description>AI Trading Bot Watchdog - Ensures bot runs during market hours</Description>
  </RegistrationInfo>
  <Triggers>
    <LogonTrigger>
      <Enabled>true</Enabled>
    </LogonTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <AllowHardTerminate>true</AllowHardTerminate>
    <StartWhenAvailable>true</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>false</RunOnlyIfNetworkAvailable>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>false</WakeToRun>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author">
    <Exec>
      <Command>"{python_exe}"</Command>
      <Arguments>"{script_path}"</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>'''

    # Save task XML
    task_file = PROJECT_ROOT / "watchdog_task.xml"
    with open(task_file, 'w', encoding='utf-16') as f:
        f.write(task_xml)

    print("To enable automatic startup, run this command as Administrator:")
    print(f'  schtasks /create /tn "AI_Trading_Bot_Watchdog" /xml "{task_file}"')
    print()
    print("To remove the scheduled task:")
    print('  schtasks /delete /tn "AI_Trading_Bot_Watchdog" /f')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Trading Bot Watchdog')
    parser.add_argument('--mode', choices=['paper', 'live'], default='paper',
                       help='Trading mode')
    parser.add_argument('--setup-startup', action='store_true',
                       help='Setup Windows startup task')
    parser.add_argument('--status', action='store_true',
                       help='Show current bot status')

    args = parser.parse_args()

    if args.setup_startup:
        create_startup_task()
        return

    if args.status:
        status_file = PROJECT_ROOT / "logs" / "bot_status.json"
        if status_file.exists():
            with open(status_file) as f:
                status = json.load(f)
            print("\nBot Status:")
            print("-" * 40)
            for key, value in status.items():
                print(f"  {key}: {value}")
        else:
            print("No status file found. Bot may not have been started yet.")
        return

    # Run the watchdog
    watchdog = BotWatchdog()
    watchdog.run(mode=args.mode)


if __name__ == "__main__":
    main()
