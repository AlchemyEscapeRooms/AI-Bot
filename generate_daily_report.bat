@echo off
echo.
echo ========================================
echo   Generating Daily Trading Report...
echo ========================================
echo.

cd /d "%~dp0"

echo Setting daily goals...
python -c "from utils.daily_summary import set_daily_goals; goals = set_daily_goals(); print(f'Set {len(goals)} goals for today')"

echo.
echo Generating daily summary...
python -c "from utils.daily_summary import print_daily_summary; print_daily_summary()"

echo.
echo Generating self-reflection report...
python -c "from utils.daily_summary import generate_self_reflection; print(generate_self_reflection())"

echo.
echo ========================================
echo   Done! Reports saved to logs/
echo ========================================
echo.
echo   - logs/daily_summaries/daily_summary_YYYY-MM-DD.txt
echo   - logs/self_reflection/reflection_YYYY-MM-DD.txt
echo.
pause
