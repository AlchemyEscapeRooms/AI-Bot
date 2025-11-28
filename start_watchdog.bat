@echo off
echo.
echo ========================================
echo   AI Trading Bot Watchdog
echo ========================================
echo.
echo This will start the watchdog that keeps
echo the bot running during market hours.
echo.
echo Press Ctrl+C to stop.
echo.
cd /d "%~dp0"
python bot_watchdog.py --mode paper
pause
