@echo off
echo.
echo ========================================
echo   Setup Bot Watchdog Auto-Start
echo ========================================
echo.
echo This will show you how to set up the bot
echo to start automatically when Windows starts.
echo.
cd /d "%~dp0"
python bot_watchdog.py --setup-startup
echo.
pause
