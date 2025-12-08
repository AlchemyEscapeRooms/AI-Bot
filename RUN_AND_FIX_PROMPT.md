# AI Trading Bot - Run, Test & Fix Until Working

You are testing the AI Trading Bot. Your job is to start the application, find ALL errors, fix them, and repeat until fully operational.

## **DO NOT STOP** until everything works with zero errors.

---

## Project Structure

**Main Entry Point:** `api_server.py` (runs on port 8000)
**Frontend:** `alchemy_dashboard.html` (served at http://localhost:8000)
**Database:** SQLite (`data/trading_bot.db` and `data/predictions.db`)

**Key Files:**
- `api_server.py` - FastAPI server (PRIMARY)
- `alchemy_dashboard.html` - Web dashboard
- `background_service.py` - Trading service
- `core/market_monitor.py` - Signal processing & predictions
- `core/order_executor.py` - Trade execution via Alpaca
- `core/trading_bot.py` - Main bot logic
- `learning_trader.py` - Prediction database & learning
- `config/config.yaml` - Configuration

---

## Step 1: Check Environment

```bash
# Verify Python
python --version

# Check required packages
pip install -r requirements.txt

# Verify Alpaca API keys are set
echo %APCA_API_KEY_ID%
echo %APCA_API_SECRET_KEY%
```

If Alpaca keys aren't set, check `.env` file or set them:
```bash
set APCA_API_KEY_ID=your_key_here
set APCA_API_SECRET_KEY=your_secret_here
```

---

## Step 2: Start the API Server

```bash
python api_server.py
```

Or with uvicorn:
```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

**Expected output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**If it fails:** Read the error, fix it, try again. Common issues:
- Import errors → install missing package
- Database errors → check schema
- Config errors → check config/config.yaml

---

## Step 3: Test ALL API Endpoints

Test each endpoint. **Fix any that return errors:**

### Core Status
```bash
curl http://localhost:8000/api/service/status
curl http://localhost:8000/api/service/state
curl http://localhost:8000/api/today
```

### Portfolio & Account
```bash
curl http://localhost:8000/api/portfolio
curl http://localhost:8000/api/positions
curl http://localhost:8000/api/account
```

### Trading Signals
```bash
curl http://localhost:8000/api/signals
curl http://localhost:8000/api/predictions/pending
curl http://localhost:8000/api/predictions/recent
```

### AI Brain & Learning
```bash
curl http://localhost:8000/api/learning/stats
curl http://localhost:8000/api/learning/weights
curl http://localhost:8000/api/brain
curl http://localhost:8000/api/brain/details
```

### Trades & Performance
```bash
curl http://localhost:8000/api/trades
curl http://localhost:8000/api/performance
curl http://localhost:8000/api/pnl
```

### Stock Management
```bash
curl http://localhost:8000/api/stocks
curl http://localhost:8000/api/stocks/excluded
```

### Bot Control
```bash
curl http://localhost:8000/api/bot/status
curl http://localhost:8000/api/settings
```

### Dashboard
```bash
curl http://localhost:8000/ -o /dev/null -w "%{http_code}\n"
```
Should return `200`.

---

## Step 4: For Each Error

When you see an error like:
```
ERROR: no such column: signals_used
```

1. **Find the query** - Search for the column name in the codebase:
   ```bash
   findstr /s /n "signals_used" *.py
   ```

2. **Check the database schema:**
   ```python
   python -c "import sqlite3; conn=sqlite3.connect('data/trading_bot.db'); print([c[1] for c in conn.execute('PRAGMA table_info(ai_predictions)').fetchall()])"
   ```

3. **Fix the mismatch** - Either:
   - Change the query to use the correct column name
   - Or add the missing column: `ALTER TABLE ai_predictions ADD COLUMN signals_used TEXT`

4. **Restart and test again**

---

## Step 5: Test the Dashboard UI

1. Open http://localhost:8000 in browser
2. Check browser console (F12 → Console) for JavaScript errors
3. Verify:
   - [ ] Page loads without errors
   - [ ] Portfolio value displays
   - [ ] P&L chart renders
   - [ ] Signals table shows data (or "No signals" message)
   - [ ] Bot status shows correctly
   - [ ] All buttons work (Start/Stop/Settings)

---

## Step 6: Test Background Services

```bash
# Test the terminal dashboard
python dashboard.py

# Test background service
python background_service.py

# Test the watchdog
python bot_watchdog.py
```

Fix any errors in each.

---

## Step 7: Database Schema Reference

The `ai_predictions` table should have these columns:
- id, prediction_id, timestamp, symbol
- predicted_direction, confidence, predicted_change_pct
- timeframe, target_price, entry_price, exit_price
- actual_change_pct, was_correct, resolved, resolved_at
- **signals** (NOT signals_used), reasoning

If queries fail, verify column names match.

---

## Step 8: Common Fixes

### "no such column: X"
```python
# Add missing column
import sqlite3
conn = sqlite3.connect('data/trading_bot.db')
conn.execute('ALTER TABLE table_name ADD COLUMN column_name TEXT')
conn.commit()
```

### "ModuleNotFoundError"
```bash
pip install module_name
```

### "Connection refused" (Alpaca)
- Check API keys are set correctly
- Verify internet connection
- Check if using paper vs live endpoint

### "Address already in use"
```bash
# Find and kill process on port 8000
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

### Config errors
- Check `config/config.yaml` for syntax errors
- Verify all required keys exist

---

## Success Criteria

**DO NOT STOP** until ALL of these pass:

1. [ ] `python api_server.py` starts with no errors
2. [ ] ALL `/api/*` endpoints return valid JSON (no 500 errors)
3. [ ] Dashboard loads at http://localhost:8000
4. [ ] No ERROR messages in server logs
5. [ ] Browser console shows no JavaScript errors
6. [ ] Bot can switch between Learning/Paper/Live modes

---

## Final Report

When everything works, provide:

```
## AI Trading Bot Status: ✅ OPERATIONAL

### Server
- api_server.py running on port 8000
- All endpoints responding

### Endpoints Tested (X/X passed)
- /api/service/status ✅
- /api/portfolio ✅
- /api/signals ✅
- /api/brain ✅
- [list all]

### Dashboard
- HTML loads ✅
- No JS errors ✅
- All widgets render ✅

### Fixes Applied
1. [List any fixes made]

### Ready for Trading: YES
```

---

## REMEMBER

- Fix EVERY error you find
- Do not skip any endpoint
- Do not give up until it works
- Test after every fix
- **This system handles real money - be thorough**
