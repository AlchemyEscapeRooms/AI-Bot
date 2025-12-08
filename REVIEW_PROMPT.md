# AI Code Review Prompt: AI Trading Bot Comprehensive Review

You are tasked with performing a meticulous, line-by-line code review of recent enhancements to an AI Trading Bot system. This review must be thorough, critical, and identify ALL issues before production deployment. **This system handles real money - miss nothing.**

---

## Project Context

This is a Python-based AI trading bot that:
- Connects to Alpaca API for live/paper trading
- Uses multiple technical analysis signals (RSI, MACD, Bollinger Bands, Volume, Momentum, etc.)
- Has adaptive learning that adjusts signal weights based on prediction accuracy
- Learns per-stock signal preferences (e.g., MACD works well for AAPL but not TSLA)
- Includes web dashboards for monitoring (alchemy_dashboard.html, static/dashboard.html)
- Has walk-forward incremental learning in backtesting

---

## Scope of Work to Review (Recent Commits)

### Commit 1: Alchemy Dashboard with Real Alpaca Portfolio Integration
**Files changed (19 files, +7002 lines):**
- `alchemy_dashboard.html` - New web dashboard (2795 lines)
- `web_api.py` - FastAPI backend (1536 lines)
- `static/dashboard.html` - Additional dashboard (1231 lines)
- `core/order_executor.py` - Alpaca connection, get_positions(), get_portfolio_history()
- `core/trading_bot.py` - Trading logic enhancements (+281 lines)
- `session_manager.py` - Bot lifecycle management (+207 lines)
- `api_server.py` - API enhancements (+547 lines)
- `background_service.py` - Background process changes
- `backtesting/strategies.py` - Strategy enhancements (+164 lines)
- `ml_models/feature_engineering.py` - Feature calculation changes (+128 lines)
- `data/market_data.py` - Data fetching changes
- `config/config.yaml` - Configuration updates
- `bot_watchdog.py`, `dashboard.py`, `unified_bot.py`, `historical_trainer.py`, `utils/daily_summary.py`, `portfolio/risk_manager.py`

### Commit 2: Hybrid Per-Stock Signal Weight Learning
**Files changed:**
- `core/market_monitor.py` (+232 lines) - Per-stock weight learning
- `dashboard.py` (+54 lines) - Dashboard updates

**Key features:**
- `stock_signal_weights` table for per-stock weights
- `get_signal_performance_by_stock()` - Per-stock accuracy tracking
- `get_stock_signal_weights()` / `update_stock_signal_weight()` - Weight management
- `get_weights_for_symbol()` - Hybrid weight calculation (blends global + per-stock)
- `_learn_per_stock_weights()` - Updates weights based on performance

### Commit 3: AI Brain Status Dashboard
**Files changed:**
- `dashboard.py` (+92 lines) - Visual signal weight display

### Commit 4: Walk-Forward Incremental Learning
**Files changed:**
- `backtesting/backtest_engine.py` (+39 lines) - Incremental learning in backtests

---

## Critical Review Checklist

### 1. SECURITY (Highest Priority)

#### API & Authentication
- [ ] Are ALL API endpoints authenticated/authorized?
- [ ] Are Alpaca API keys loaded from environment variables (not hardcoded)?
- [ ] Search for: `api_key`, `secret`, `password`, `token` in code
- [ ] Is CORS configuration restrictive enough?
- [ ] Are there rate limits on API endpoints?
- [ ] Can unauthenticated users trigger trades?

#### Input Validation
- [ ] Is ALL user input sanitized before database queries?
- [ ] Are SQL queries parameterized (no string concatenation)?
- [ ] Check for SQL injection in: `web_api.py`, `api_server.py`, `core/market_monitor.py`
- [ ] Are file paths validated (no path traversal)?

#### Frontend Security
- [ ] XSS vulnerabilities in `alchemy_dashboard.html`?
- [ ] Are dynamic HTML insertions sanitized?
- [ ] CSRF protection on forms?
- [ ] Sensitive data exposed in JavaScript?

#### Data Exposure
- [ ] Can API responses leak portfolio data to unauthorized users?
- [ ] Are error messages too verbose (exposing internal details)?
- [ ] Is logging exposing sensitive information?

### 2. FINANCIAL SAFETY (Critical)

#### Trade Execution
- [ ] Can the bot accidentally place LIVE trades when in PAPER mode?
- [ ] Is there a maximum position size limit that CANNOT be bypassed?
- [ ] Are there circuit breakers for unusual market conditions?
- [ ] Is there protection against rapid consecutive trades?
- [ ] What happens if API calls fail mid-trade?
- [ ] Are partial fills handled correctly?
- [ ] Are rejected orders handled gracefully?

#### Position Management
- [ ] Can positions exceed account buying power?
- [ ] Is margin calculated correctly?
- [ ] Are stop losses enforced?
- [ ] What happens if position data is stale?

#### Order Executor Review (`core/order_executor.py`)
- [ ] `get_positions()` returns accurate data structure?
- [ ] `get_portfolio_history()` handles date ranges correctly?
- [ ] Network failure handling (retries, timeouts)?
- [ ] Rate limit handling for Alpaca API?
- [ ] Paper vs Live mode clearly separated?

### 3. DATABASE INTEGRITY

#### Schema Consistency
- [ ] **CRITICAL**: Check if `signals_used` column exists in `ai_predictions` table
  - File `core/market_monitor.py:85-103` defines schema with `signals` column
  - But some code may reference `signals_used` - causing errors
- [ ] Are all table schemas consistent across files?
- [ ] Are migrations needed for schema changes?

#### Data Integrity
- [ ] Can weight updates result in NaN or infinite values?
- [ ] Division by zero protection in weight calculations?
- [ ] What happens when sample_size is 0?
- [ ] Are database transactions properly committed/rolled back?
- [ ] Thread safety for concurrent database writes?

#### Query Review
- [ ] All queries use parameterized values?
- [ ] No raw string formatting in SQL?
- [ ] Proper error handling for query failures?

### 4. ALGORITHM CORRECTNESS

#### Signal Weight Learning (`core/market_monitor.py`)
- [ ] Is the hybrid weight formula mathematically correct?
  - `hybrid = global_weight * (1 - blend_factor) + stock_weight * blend_factor`
- [ ] Does `blend_factor` calculation make sense?
- [ ] Are weights bounded (0.5 to 2.0)?
- [ ] Can weights drift to extreme values over time?
- [ ] Is the learning rate appropriate?

#### Per-Stock Learning
- [ ] Does `get_signal_performance_by_stock()` calculate correctly?
- [ ] Are there enough samples before trusting per-stock weights?
- [ ] How does it handle new stocks with no history?

#### Walk-Forward Backtesting (`backtesting/backtest_engine.py`)
- [ ] **CRITICAL**: Is there look-ahead bias (using future data)?
- [ ] Is training/test split done correctly?
- [ ] Are walk-forward windows properly sized?
- [ ] Is data leakage prevented?

#### Strategy Implementations (`backtesting/strategies.py`)
- [ ] Are all 9 strategies mathematically correct?
- [ ] RSI calculation correct?
- [ ] MACD calculation correct?
- [ ] Bollinger Bands calculation correct?
- [ ] Volume analysis correct?
- [ ] Momentum calculation correct?

### 5. ERROR HANDLING & RELIABILITY

#### Exception Handling
- [ ] Are exceptions caught and handled (not swallowed silently)?
- [ ] Do error handlers log sufficient information?
- [ ] Are there try/catch blocks around all external API calls?
- [ ] What happens when Alpaca API is down?

#### Process Management (`session_manager.py`, `background_service.py`, `bot_watchdog.py`)
- [ ] Clean startup/shutdown procedures?
- [ ] Zombie process prevention?
- [ ] Crash recovery - what state is preserved?
- [ ] Resource cleanup (file handles, connections, sockets)?
- [ ] Memory leaks in long-running processes?

#### Network Resilience
- [ ] Retry logic for failed API calls?
- [ ] Exponential backoff implemented?
- [ ] Timeout handling?
- [ ] Connection pooling?

### 6. DATA HANDLING

#### Feature Engineering (`ml_models/feature_engineering.py`)
- [ ] Are technical indicators calculated correctly?
- [ ] NaN handling for missing data?
- [ ] Off-by-one errors in time series?
- [ ] Data alignment across different timeframes?

#### Market Data (`data/market_data.py`)
- [ ] Error handling for API failures?
- [ ] Cache invalidation correct?
- [ ] Timezone handling correct?
- [ ] Weekend/holiday handling?

### 7. CONFIGURATION (`config/config.yaml`)

- [ ] No API keys or secrets in config file?
- [ ] Safe default values for all parameters?
- [ ] Config validation at startup?
- [ ] What happens with invalid config values?

### 8. CODE QUALITY

#### General
- [ ] Infinite loops possible?
- [ ] Deadlocks possible with threading?
- [ ] Memory accumulation over time?
- [ ] Proper logging for debugging?

#### Type Safety
- [ ] Type hints used consistently?
- [ ] Type mismatches possible?

---

## Files to Review (Priority Order)

**Priority 1 - Security & Financial Risk:**
1. `web_api.py` (1536 lines) - Main API, highest risk
2. `core/order_executor.py` - Trade execution
3. `core/trading_bot.py` - Core logic
4. `api_server.py` - Secondary API

**Priority 2 - Data Integrity:**
5. `core/market_monitor.py` - Signal learning
6. `session_manager.py` - Process lifecycle
7. `utils/database.py` - Database operations

**Priority 3 - Algorithm Correctness:**
8. `backtesting/backtest_engine.py` - Walk-forward learning
9. `backtesting/strategies.py` - All strategies
10. `ml_models/feature_engineering.py` - Feature calculations

**Priority 4 - Frontend & Config:**
11. `alchemy_dashboard.html` - Frontend security
12. `static/dashboard.html` - Dashboard
13. `config/config.yaml` - Configuration
14. `dashboard.py` - Terminal dashboard

**Priority 5 - Supporting Files:**
15. `background_service.py`
16. `bot_watchdog.py`
17. `data/market_data.py`
18. `unified_bot.py`

---

## Known Issues to Verify

1. **Database Schema Mismatch**: Query references `signals_used` column but table has `signals` column
   - Error: `no such column: signals_used`
   - Location: Check all files querying `ai_predictions` table

2. **Multiple Database Schemas**: Different files may define different schemas
   - `core/market_monitor.py` defines one schema
   - `learning_trader.py` may define another
   - Verify consistency

---

## Deliverables Required

### 1. Critical Issues (MUST FIX before any trading)
List all security vulnerabilities, financial safety issues, and data integrity risks.

### 2. High Priority Issues
Logic bugs, error handling gaps, algorithm errors.

### 3. Medium Priority Issues
Code quality, best practice violations, performance concerns.

### 4. Low Priority Issues
Style improvements, documentation, minor optimizations.

### 5. Test Recommendations
- Unit tests needed
- Integration tests needed
- Edge cases to test
- Stress test scenarios

### 6. Database Schema Fix
Provide corrected schema and migration script if needed.

---

## Review Instructions

1. **Read each file completely** - don't skim
2. **Trace data flow** from user input to database to trade execution
3. **Test edge cases mentally** - what if value is 0, None, negative, very large?
4. **Check every SQL query** for injection vulnerabilities
5. **Verify every API endpoint** has proper authentication
6. **Confirm paper/live mode separation** is bulletproof
7. **Document EVERY issue found** with file:line reference

**This system handles real money. Be paranoid. Miss nothing.**
