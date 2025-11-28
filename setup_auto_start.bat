@echo off
echo.
echo ========================================
echo   AI Trading Bot - Auto-Start Setup
echo ========================================
echo.
echo This will configure Windows to automatically start
echo the trading bot when your computer starts.
echo.
echo Choose an option:
echo   1. Start bot at system startup (recommended)
echo   2. Start bot at 8:00 AM daily
echo   3. Start bot NOW in background
echo   4. Remove auto-start
echo   5. Check status
echo   6. Exit
echo.

set /p choice="Enter choice (1-6): "

if "%choice%"=="1" goto startup
if "%choice%"=="2" goto scheduled
if "%choice%"=="3" goto startnow
if "%choice%"=="4" goto remove
if "%choice%"=="5" goto status
if "%choice%"=="6" goto end

:startup
echo.
echo Setting up auto-start at system startup...
echo.

:: Get the pythonw.exe path (no console window)
for /f "tokens=*" %%i in ('where pythonw 2^>nul') do set PYTHONW=%%i
if "%PYTHONW%"=="" (
    for /f "tokens=*" %%i in ('where python') do set PYTHONW=%%i
    set PYTHONW=%PYTHONW:python.exe=pythonw.exe%
)

:: Create the scheduled task to run at logon
schtasks /create /tn "AI_Trading_Bot" /tr "\"%PYTHONW%\" \"%~dp0start_trading_bot.pyw\"" /sc onlogon /rl highest /f

if %errorlevel%==0 (
    echo.
    echo SUCCESS! Trading bot will start automatically when you log in.
    echo.
    echo The bot will:
    echo   - Set daily goals at 8:30 AM
    echo   - Trade during market hours (9:30 AM - 4:00 PM)
    echo   - Generate daily reports at 4:05 PM
    echo.
) else (
    echo.
    echo ERROR: Could not create scheduled task.
    echo Try running this script as Administrator.
    echo.
)
goto end

:scheduled
echo.
echo Setting up auto-start at 8:00 AM daily...
echo.

for /f "tokens=*" %%i in ('where pythonw 2^>nul') do set PYTHONW=%%i
if "%PYTHONW%"=="" (
    for /f "tokens=*" %%i in ('where python') do set PYTHONW=%%i
    set PYTHONW=%PYTHONW:python.exe=pythonw.exe%
)

:: Create the scheduled task to run at 8:00 AM weekdays
schtasks /create /tn "AI_Trading_Bot" /tr "\"%PYTHONW%\" \"%~dp0start_trading_bot.pyw\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 08:00 /rl highest /f

if %errorlevel%==0 (
    echo.
    echo SUCCESS! Trading bot will start at 8:00 AM on weekdays.
    echo.
) else (
    echo.
    echo ERROR: Could not create scheduled task.
    echo Try running this script as Administrator.
    echo.
)
goto end

:startnow
echo.
echo Starting trading bot in background...
echo.
cd /d "%~dp0"
python auto_trader.py --background
goto end

:remove
echo.
echo Removing auto-start...
echo.
schtasks /delete /tn "AI_Trading_Bot" /f
if %errorlevel%==0 (
    echo Auto-start removed successfully.
) else (
    echo No auto-start task found or error removing it.
)
goto end

:status
echo.
echo Checking status...
echo.
schtasks /query /tn "AI_Trading_Bot" 2>nul
if %errorlevel%==0 (
    echo.
    echo Auto-start is CONFIGURED.
) else (
    echo Auto-start is NOT configured.
)
echo.
cd /d "%~dp0"
python auto_trader.py --status
goto end

:end
echo.
pause
