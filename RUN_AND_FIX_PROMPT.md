# AI Bot - Run, Test, and Fix Until Working

You are tasked with getting this AI Trading Bot fully operational. Your job is to systematically start the application, identify ALL errors, fix them, and repeat until everything works perfectly.

## Your Mission

**DO NOT STOP** until the application runs without errors and all features work.

---

## Step 1: Environment Setup

First, check the environment:

```bash
# Check Python version
python --version

# Check if required packages are installed
pip list | grep -E "fastapi|uvicorn|pandas|numpy|alpaca|yfinance"

# If missing, install requirements
pip install -r requirements.txt
```

Check for required environment variables:
```bash
# These should be set for Alpaca API
echo $APCA_API_KEY_ID
echo $APCA_API_SECRET_KEY
```

If not set, check if there's a `.env` file or config that needs them.

---

## Step 2: Start the Web API (Frontend)

Start the main web dashboard:

```bash
python web_api.py
```

**Watch for errors.** Common issues:
- Import errors (missing modules)
- Database errors (missing tables/columns)
- Config errors (missing config values)
- API connection errors

**If it starts successfully**, open http://localhost:8000 in a browser (or use curl/fetch to test endpoints).

---

## Step 3: Test ALL API Endpoints

Test each endpoint and fix any that fail:

```bash
# Health/Status
curl http://localhost:8000/api/status
curl http://localhost:8000/api/health

# Portfolio & Account
curl http://localhost:8000/api/portfolio
curl http://localhost:8000/api/account
curl http://localhost:8000/api/positions

# Trading Signals & Predictions
curl http://localhost:8000/api/signals
curl http://localhost:8000/api/predictions
curl http://localhost:8000/api/predicted-profit

# Market Data
curl http://localhost:8000/api/trending
curl http://localhost:8000/api/market-status

# AI Brain / Learning
curl http://localhost:8000/api/brain
curl http://localhost:8000/api/brain/details
curl http://localhost:8000/api/learning-stats

# Trades
curl http://localhost:8000/api/trades
curl http://localhost:8000/api/trades?mode=paper

# Performance
curl http://localhost:8000/api/performance
curl http://localhost:8000/api/daily-summary
```

**For each endpoint that returns an error:**
1. Read the error message carefully
2. Find the relevant code in `web_api.py` or the module it calls
3. Fix the issue (missing column, wrong variable name, missing import, etc.)
4. Save the file
5. Restart the server
6. Test again

---

## Step 4: Test the Dashboard HTML

Verify the dashboard loads correctly:

```bash
curl http://localhost:8000/ -o /dev/null -w "%{http_code}"
```

Should return 200. If there's a static file issue, check:
- `alchemy_dashboard.html` exists
- `static/` folder exists with required files
- FastAPI static file mounting is correct

---

## Step 5: Test Background Services

Try starting the background service:

```bash
python background_service.py
```

Watch for errors and fix them.

Try the bot watchdog:

```bash
python bot_watchdog.py
```

Try the main trading bot:

```bash
python unified_bot.py
```

Or the dashboard:

```bash
python dashboard.py
```

---

## Step 6: Database Verification

Check database schema matches code expectations:

```python
import sqlite3

# Check ai_predictions table
conn = sqlite3.connect('data/trading_bot.db')  # or wherever db is
cursor = conn.cursor()
cursor.execute("PRAGMA table_info(ai_predictions)")
print("ai_predictions columns:", cursor.fetchall())

# Check for other important tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("All tables:", cursor.fetchall())
conn.close()
```

**Common database issues:**
- Column name mismatch (`signals` vs `signals_used`)
- Missing tables
- Wrong database path

---

## Step 7: Fix Checklist

As you find and fix errors, track them:

### Errors Found and Fixed:
- [ ] Error 1: _________________ | File: _________ | Line: ___ | Fix: _________________
- [ ] Error 2: _________________ | File: _________ | Line: ___ | Fix: _________________
- [ ] Error 3: _________________ | File: _________ | Line: ___ | Fix: _________________
(add more as needed)

---

## Step 8: Final Verification

Once all errors are fixed, do a complete test:

1. **Start the web API:**
   ```bash
   python web_api.py
   ```
   Confirm: "Uvicorn running on http://0.0.0.0:8000"

2. **Test critical endpoints (all should return valid JSON, no errors):**
   ```bash
   curl -s http://localhost:8000/api/status | python -m json.tool
   curl -s http://localhost:8000/api/portfolio | python -m json.tool
   curl -s http://localhost:8000/api/signals | python -m json.tool
   curl -s http://localhost:8000/api/brain | python -m json.tool
   ```

3. **Check server logs** - no ERROR messages should appear

4. **Test the HTML dashboard** - load http://localhost:8000 and verify:
   - Page loads without JavaScript errors
   - Portfolio data displays
   - Charts render (if applicable)
   - No console errors in browser dev tools

---

## Step 9: Report Success

When everything works, provide a summary:

```
## AI Bot Status: OPERATIONAL

### Components Tested:
- [x] Web API (web_api.py) - Running on port 8000
- [x] All API endpoints responding
- [x] Dashboard HTML loading
- [x] Database connections working
- [x] No errors in logs

### Fixes Applied:
1. [List each fix you made]
2. ...

### Endpoints Verified:
- /api/status - OK
- /api/portfolio - OK
- /api/signals - OK
- [etc.]

### Ready for Trading: YES
```

---

## Common Errors and Fixes

### "no such column: X"
- Database schema mismatch
- Fix: Either ALTER TABLE to add column, or change code to use correct column name

### "ModuleNotFoundError: No module named 'X'"
- Missing package
- Fix: `pip install X`

### "KeyError: 'X'"
- Config or dict missing expected key
- Fix: Add default value or check if key exists first

### "Connection refused" / "Cannot connect to Alpaca"
- API keys not set or invalid
- Fix: Set APCA_API_KEY_ID and APCA_API_SECRET_KEY environment variables

### "Address already in use"
- Port 8000 already taken
- Fix: Kill existing process or use different port

### "FileNotFoundError: database/X.db"
- Database file doesn't exist
- Fix: Create directory or let code create database on first run

---

## Important Files to Check

Priority order for debugging:

1. `web_api.py` - Main API server
2. `core/market_monitor.py` - Signal processing, database queries
3. `core/order_executor.py` - Trade execution
4. `core/trading_bot.py` - Main bot logic
5. `utils/database.py` - Database operations
6. `config/config.yaml` - Configuration
7. `session_manager.py` - Process management

---

## DO NOT STOP UNTIL:

1. `python web_api.py` starts without errors
2. ALL `/api/*` endpoints return valid responses (no 500 errors)
3. The dashboard HTML loads and displays data
4. Server logs show no ERROR messages
5. You can confirm: "The AI Bot is fully operational"

**Fix every error you encounter. Do not skip any. Do not give up.**
