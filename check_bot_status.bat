@echo off
echo.
echo ========================================
echo   AI Trading Bot Status
echo ========================================
echo.
cd /d "%~dp0"
python bot_watchdog.py --status
echo.
pause
