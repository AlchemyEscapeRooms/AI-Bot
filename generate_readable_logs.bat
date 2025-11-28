@echo off
echo.
echo ========================================
echo   Generating Readable Trade Logs...
echo ========================================
echo.

cd /d "%~dp0"
python -c "from utils.readable_trade_log import generate_readable_logs; generate_readable_logs()"

echo.
echo ========================================
echo   Done! Files created in trade_logs/
echo ========================================
echo.
echo   - trades_readable.txt (detailed report)
echo   - trades_simple.csv (spreadsheet)
echo.
pause
