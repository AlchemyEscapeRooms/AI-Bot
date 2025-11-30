"""
Trading Bot API Server
=======================

FastAPI backend that:
- Serves the dashboard
- Provides REST endpoints for bot data
- WebSocket for real-time updates
- Connects to the background trading service

Run with: uvicorn api_server:app --reload --port 8000

Author: Claude AI
Date: November 29, 2025
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import threading

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our trading bot components
from core.background_service import (
    BackgroundTradingService,
    ServiceConfig,
    ServiceState,
    TradingMode
)
from core.learning_trader import PredictionDatabase, StockLearningProfile
from core.historical_trainer import HistoricalTrainer
from utils.logger import get_logger
from config import config

logger = get_logger(__name__)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class StockCommand(BaseModel):
    symbol: str

class TradingModeCommand(BaseModel):
    mode: str  # 'learning', 'paper', 'live'

class BacktestRequest(BaseModel):
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    initial_capital: float = 100000
    mode: str = "test"  # 'learn' or 'test'

class UserCommand(BaseModel):
    command: str

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="Alchemy Trading Bot API",
    description="API for the learning trading bot",
    version="1.0.0"
)

# CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
trading_service: Optional[BackgroundTradingService] = None
websocket_clients: List[WebSocket] = []
data_dir = Path("data")

# ============================================================================
# SERVICE MANAGEMENT
# ============================================================================

def get_service() -> BackgroundTradingService:
    """Get or create the trading service."""
    global trading_service
    
    if trading_service is None:
        # Default configuration
        service_config = ServiceConfig(
            symbols=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "AMD", "SPY", "QQQ"],
            trading_mode=TradingMode.LEARNING_ONLY,
            prediction_interval_minutes=60,
            verification_interval_minutes=5,
        )
        
        trading_service = BackgroundTradingService(
            config=service_config,
            api_key=config.get('alpaca.api_key'),
            api_secret=config.get('alpaca.api_secret'),
            data_dir=str(data_dir)
        )
        
        # Set up callbacks for WebSocket broadcasts
        trading_service.on_prediction = lambda p: asyncio.create_task(broadcast_event('prediction', {
            'symbol': p.symbol,
            'horizon': p.horizon.value,
            'direction': p.predicted_direction.value,
            'confidence': p.confidence
        }))
        
        trading_service.on_trade_signal = lambda s: asyncio.create_task(broadcast_event('trade_signal', {
            'symbol': s.symbol,
            'action': s.action,
            'confidence': s.confidence
        }))
    
    return trading_service

async def broadcast_event(event_type: str, data: Dict):
    """Broadcast event to all connected WebSocket clients."""
    message = json.dumps({
        'type': event_type,
        'data': data,
        'timestamp': datetime.now().isoformat()
    })
    
    disconnected = []
    for client in websocket_clients:
        try:
            await client.send_text(message)
        except:
            disconnected.append(client)
    
    # Clean up disconnected clients
    for client in disconnected:
        websocket_clients.remove(client)

# ============================================================================
# STARTUP / SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    logger.info("API Server starting up...")
    data_dir.mkdir(parents=True, exist_ok=True)

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    global trading_service
    if trading_service and trading_service.state != ServiceState.STOPPED:
        trading_service.stop()
    logger.info("API Server shut down")

# ============================================================================
# STATIC FILES & DASHBOARD
# ============================================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard."""
    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    else:
        # Return embedded dashboard if file doesn't exist
        return HTMLResponse(content=get_embedded_dashboard(), status_code=200)

# ============================================================================
# SERVICE CONTROL ENDPOINTS
# ============================================================================

@app.post("/api/service/start")
async def start_service():
    """Start the background trading service."""
    service = get_service()
    
    if service.state == ServiceState.RUNNING:
        return {"status": "already_running", "message": "Service is already running"}
    
    service.start()
    return {"status": "started", "message": "Trading service started"}

@app.post("/api/service/stop")
async def stop_service():
    """Stop the background trading service."""
    service = get_service()
    
    if service.state == ServiceState.STOPPED:
        return {"status": "already_stopped", "message": "Service is already stopped"}
    
    service.stop()
    return {"status": "stopped", "message": "Trading service stopped"}

@app.post("/api/service/pause")
async def pause_service():
    """Pause the trading service."""
    service = get_service()
    service.pause()
    return {"status": "paused", "message": "Trading service paused"}

@app.post("/api/service/resume")
async def resume_service():
    """Resume the trading service."""
    service = get_service()
    service.resume()
    return {"status": "resumed", "message": "Trading service resumed"}

@app.get("/api/service/status")
async def get_service_status():
    """Get current service status."""
    service = get_service()
    return service.get_status()

# ============================================================================
# PORTFOLIO & ACCOUNT ENDPOINTS
# ============================================================================

@app.get("/api/portfolio")
async def get_portfolio():
    """Get current portfolio positions."""
    service = get_service()
    
    if service.trade_executor is None:
        return {"positions": [], "error": "Service not initialized"}
    
    positions = service.trade_executor.get_positions()
    account = service.trade_executor.get_account()
    
    # Format positions for frontend
    formatted_positions = []
    for symbol, data in positions.items():
        formatted_positions.append({
            "symbol": symbol,
            "quantity": data['quantity'],
            "avgCost": data['avg_cost'],
            "currentPrice": data['current_price'],
            "marketValue": data['market_value'],
            "unrealizedPL": data['unrealized_pl'],
            "unrealizedPLPct": data['unrealized_pl_pct'] * 100
        })
    
    return {
        "positions": formatted_positions,
        "account": {
            "equity": account.get('equity', 0),
            "cash": account.get('cash', 0),
            "buyingPower": account.get('buying_power', 0)
        }
    }

@app.get("/api/account")
async def get_account():
    """Get account information."""
    service = get_service()
    
    if service.trade_executor is None:
        return {"error": "Service not initialized"}
    
    return service.trade_executor.get_account()

# ============================================================================
# PREDICTIONS & LEARNING ENDPOINTS
# ============================================================================

@app.get("/api/predictions/active")
async def get_active_predictions():
    """Get currently active (unverified) predictions."""
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    # Get predictions that haven't been verified yet
    pending = db.get_pending_predictions(before_time=datetime.now() + timedelta(days=1))
    
    # Format for frontend
    formatted = []
    for pred in pending[:50]:  # Limit to 50 most recent
        formatted.append({
            "id": pred['id'],
            "symbol": pred['symbol'],
            "horizon": pred['horizon'],
            "direction": pred['predicted_direction'],
            "predictedChange": pred['predicted_change_pct'],
            "confidence": pred['confidence'],
            "predictionTime": pred['prediction_time'],
            "targetTime": pred['target_time']
        })
    
    return {"predictions": formatted}

@app.get("/api/predictions/stats")
async def get_prediction_stats(days: int = 7):
    """Get prediction statistics."""
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    stats = db.get_prediction_stats(days=days)
    
    return {
        "period_days": days,
        "by_horizon": stats
    }

@app.get("/api/learning/stocks")
async def get_stock_learning_profiles():
    """Get learning profiles for all stocks."""
    service = get_service()
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    profiles = []
    for symbol in service.config.symbols:
        profile = db.get_stock_profile(symbol)
        
        if profile:
            ready = (
                profile.total_predictions >= 100 and
                profile.overall_accuracy >= 0.55
            )
            
            profiles.append({
                "symbol": symbol,
                "totalPredictions": profile.total_predictions,
                "accuracy": profile.overall_accuracy,
                "accuracy1h": profile.accuracy_1h,
                "accuracyEod": profile.accuracy_eod,
                "accuracyNextDay": profile.accuracy_next_day,
                "ready": ready,
                "status": "ready" if ready else ("learning" if profile.total_predictions > 0 else "new"),
                "excluded": symbol in service.config.excluded_symbols
            })
        else:
            profiles.append({
                "symbol": symbol,
                "totalPredictions": 0,
                "accuracy": 0,
                "accuracy1h": 0.5,
                "accuracyEod": 0.5,
                "accuracyNextDay": 0.5,
                "ready": False,
                "status": "new",
                "excluded": symbol in service.config.excluded_symbols
            })
    
    return {"stocks": profiles}

# ============================================================================
# STOCK MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/stocks/add")
async def add_stock(command: StockCommand, background_tasks: BackgroundTasks):
    """Add a new stock to monitoring."""
    service = get_service()
    symbol = command.symbol.upper()
    
    if symbol in service.config.symbols:
        return {"success": False, "message": f"{symbol} is already being monitored"}
    
    # Add and train in background
    result = service.add_symbol(symbol, train_first=True)
    
    return {
        "success": result.get('success', False),
        "symbol": symbol,
        "predictions": result.get('predictions', 0),
        "accuracy": result.get('accuracy', 0),
        "readyToTrade": result.get('ready_to_trade', False),
        "message": f"Added {symbol} to watchlist"
    }

@app.post("/api/stocks/remove")
async def remove_stock(command: StockCommand):
    """Remove a stock from monitoring."""
    service = get_service()
    symbol = command.symbol.upper()
    
    if symbol not in service.config.symbols:
        return {"success": False, "message": f"{symbol} is not being monitored"}
    
    service.remove_symbol(symbol)
    return {"success": True, "message": f"Removed {symbol} from watchlist"}

@app.post("/api/stocks/exclude")
async def exclude_stock(command: StockCommand):
    """Exclude a stock from trading today."""
    service = get_service()
    symbol = command.symbol.upper()
    
    service.exclude_symbol(symbol)
    return {"success": True, "message": f"Excluded {symbol} from trading today"}

@app.post("/api/stocks/include")
async def include_stock(command: StockCommand):
    """Re-include a previously excluded stock."""
    service = get_service()
    symbol = command.symbol.upper()
    
    service.include_symbol(symbol)
    return {"success": True, "message": f"Re-included {symbol} for trading"}

# ============================================================================
# TRADING MODE ENDPOINTS
# ============================================================================

@app.post("/api/trading/mode")
async def set_trading_mode(command: TradingModeCommand):
    """Set the trading mode."""
    service = get_service()
    
    mode_map = {
        'learning': TradingMode.LEARNING_ONLY,
        'paper': TradingMode.PAPER_TRADING,
        'live': TradingMode.LIVE_TRADING
    }
    
    if command.mode not in mode_map:
        raise HTTPException(status_code=400, detail=f"Invalid mode: {command.mode}")
    
    if command.mode == 'live':
        return {
            "success": False,
            "requiresConfirmation": True,
            "message": "Live trading requires confirmation. Use /api/trading/confirm-live"
        }
    
    service.set_trading_mode(mode_map[command.mode])
    return {"success": True, "mode": command.mode}

@app.post("/api/trading/confirm-live")
async def confirm_live_trading():
    """Confirm and enable live trading."""
    service = get_service()
    service.set_trading_mode(TradingMode.LIVE_TRADING)
    return {"success": True, "mode": "live", "warning": "LIVE TRADING ENABLED - Real money will be used"}

# ============================================================================
# BACKTESTING ENDPOINTS
# ============================================================================

@app.post("/api/backtest/run")
async def run_backtest(request: BacktestRequest):
    """Run a backtest on historical data."""
    
    try:
        trainer = HistoricalTrainer(
            symbols=[request.symbol.upper()],
            db_path=str(data_dir / "backtest_temp.db") if request.mode == "test" else str(data_dir / "predictions.db")
        )
        
        results = trainer.train_on_historical(
            start_date=request.start_date,
            end_date=request.end_date,
            prediction_interval=1,
            verbose=False
        )
        
        profile = trainer.profiles.get(request.symbol.upper())
        
        if profile:
            return {
                "success": True,
                "symbol": request.symbol.upper(),
                "results": {
                    "totalPredictions": profile.total_predictions,
                    "accuracy": profile.overall_accuracy,
                    "accuracy1h": profile.accuracy_1h,
                    "accuracyEod": profile.accuracy_eod,
                    "accuracyNextDay": profile.accuracy_next_day,
                    "topFeatures": sorted(
                        profile.feature_weights.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                }
            }
        else:
            return {"success": False, "error": "No results generated"}
            
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return {"success": False, "error": str(e)}

# ============================================================================
# MORNING BRIEFING ENDPOINT
# ============================================================================

@app.get("/api/briefing")
async def get_morning_briefing():
    """Get the morning briefing data."""
    service = get_service()
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    # Account info
    account = {}
    if service.trade_executor:
        account = service.trade_executor.get_account()
    
    # Yesterday's stats
    stats = db.get_prediction_stats(days=1)
    total_preds = sum(s.get('total', 0) for s in stats.values())
    correct_preds = sum(s.get('correct', 0) for s in stats.values())
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    
    # Get best performers (stocks with highest accuracy)
    best_performers = []
    for symbol in service.config.symbols:
        profile = db.get_stock_profile(symbol)
        if profile and profile.total_predictions > 50:
            best_performers.append({
                "symbol": symbol,
                "accuracy": profile.overall_accuracy,
                "predictions": profile.total_predictions
            })
    
    best_performers.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Excluded stocks
    excluded = service.config.excluded_symbols
    
    return {
        "date": datetime.now().strftime("%A, %B %d, %Y"),
        "account": {
            "equity": account.get('equity', 0),
            "cash": account.get('cash', 0),
            "buyingPower": account.get('buying_power', 0)
        },
        "yesterdayPerformance": {
            "predictions": total_preds,
            "correct": correct_preds,
            "accuracy": accuracy,
            "byHorizon": stats
        },
        "bestPerformers": best_performers[:5],
        "monitoringCount": len(service.config.symbols),
        "excludedToday": excluded,
        "tradingMode": service.config.trading_mode.value,
        "serviceState": service.state.value if service.state else "unknown"
    }

# ============================================================================
# TRADE SIGNALS ENDPOINT
# ============================================================================

@app.get("/api/signals")
async def get_trade_signals():
    """Get current trade signals/suggestions."""
    service = get_service()
    db = PredictionDatabase(str(data_dir / "predictions.db"))
    
    signals = []
    
    # Get recent high-confidence predictions
    pending = db.get_pending_predictions()
    
    for pred in pending:
        if pred['confidence'] >= 0.65:
            profile = db.get_stock_profile(pred['symbol'])
            
            # Only include if stock is ready
            if profile and profile.total_predictions >= 100 and profile.overall_accuracy >= 0.55:
                signals.append({
                    "symbol": pred['symbol'],
                    "action": "buy" if pred['predicted_direction'] == 'up' else "sell",
                    "confidence": pred['confidence'],
                    "horizon": pred['horizon'],
                    "reasoning": f"{pred['horizon']} prediction with {pred['confidence']:.0%} confidence",
                    "stockAccuracy": profile.overall_accuracy
                })
    
    # Sort by confidence
    signals.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "signals": signals[:10],  # Top 10
        "tradingMode": service.config.trading_mode.value
    }

# ============================================================================
# WEBSOCKET FOR REAL-TIME UPDATES
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    websocket_clients.append(websocket)
    
    logger.info(f"WebSocket client connected. Total clients: {len(websocket_clients)}")
    
    try:
        # Send initial status
        service = get_service()
        await websocket.send_json({
            "type": "connected",
            "data": service.get_status()
        })
        
        # Keep connection alive and listen for messages
        while True:
            try:
                # Wait for messages from client
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30)
                
                # Handle client commands
                message = json.loads(data)
                
                if message.get('type') == 'ping':
                    await websocket.send_json({"type": "pong"})
                elif message.get('type') == 'get_status':
                    await websocket.send_json({
                        "type": "status",
                        "data": service.get_status()
                    })
                    
            except asyncio.TimeoutError:
                # Send heartbeat
                await websocket.send_json({"type": "heartbeat"})
                
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    finally:
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

# ============================================================================
# COMMAND ENDPOINT (for natural language commands)
# ============================================================================

@app.post("/api/command")
async def process_command(command: UserCommand):
    """Process a natural language command."""
    service = get_service()
    cmd = command.command.strip().upper()
    
    # Parse command
    if cmd in ['STATUS', 'STATS']:
        return {"response": "status", "data": service.get_status()}
    
    elif cmd == 'START':
        if service.state != ServiceState.RUNNING:
            service.start()
        return {"response": "Service started"}
    
    elif cmd == 'STOP':
        if service.state != ServiceState.STOPPED:
            service.stop()
        return {"response": "Service stopped"}
    
    elif cmd == 'PAUSE':
        service.pause()
        return {"response": "Service paused"}
    
    elif cmd == 'RESUME':
        service.resume()
        return {"response": "Service resumed"}
    
    elif 'SKIP' in cmd or 'EXCLUDE' in cmd:
        # Extract symbols
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd)
        exclude_words = {'SKIP', 'EXCLUDE', 'AND', 'OR', 'THE', 'TODAY'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        for symbol in symbols:
            service.exclude_symbol(symbol)
        
        return {"response": f"Excluded: {', '.join(symbols)}"}
    
    elif 'ADD' in cmd or 'WATCH' in cmd:
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', cmd)
        exclude_words = {'ADD', 'WATCH', 'AND', 'OR', 'THE', 'TO'}
        symbols = [s for s in symbols if s not in exclude_words]
        
        results = []
        for symbol in symbols:
            result = service.add_symbol(symbol, train_first=True)
            results.append(f"{symbol}: {'Added' if result.get('success') else 'Failed'}")
        
        return {"response": '\n'.join(results)}
    
    else:
        return {"response": f"Unknown command: {command.command}"}

# ============================================================================
# EMBEDDED DASHBOARD (fallback if static file doesn't exist)
# ============================================================================

def get_embedded_dashboard():
    """Return the embedded dashboard HTML."""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>Alchemy Trading Bot</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        body { font-family: system-ui; background: #0d1117; color: #f0f6fc; padding: 20px; }
        .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; margin: 10px 0; }
        .title { font-size: 24px; font-weight: bold; margin-bottom: 20px; }
        .btn { padding: 12px 24px; border-radius: 8px; border: none; cursor: pointer; margin: 5px; }
        .btn-green { background: #238636; color: white; }
        .btn-red { background: #da3633; color: white; }
        .status { padding: 8px 16px; border-radius: 8px; display: inline-block; }
        .status-running { background: rgba(63,185,80,0.2); color: #3fb950; }
        .status-stopped { background: rgba(248,81,73,0.2); color: #f85149; }
        pre { background: #0d1117; padding: 15px; border-radius: 8px; overflow: auto; }
    </style>
</head>
<body>
    <div class="title">⚗️ Alchemy Trading Bot</div>
    
    <div class="card">
        <h3>Service Status</h3>
        <div id="status" class="status status-stopped">Loading...</div>
        <div style="margin-top: 15px;">
            <button class="btn btn-green" onclick="startService()">Start</button>
            <button class="btn btn-red" onclick="stopService()">Stop</button>
            <button class="btn" style="background:#30363d;color:white" onclick="refreshStatus()">Refresh</button>
        </div>
    </div>
    
    <div class="card">
        <h3>Status Details</h3>
        <pre id="details">Loading...</pre>
    </div>
    
    <script>
        async function refreshStatus() {
            try {
                const res = await fetch('/api/service/status');
                const data = await res.json();
                
                const statusEl = document.getElementById('status');
                statusEl.textContent = data.state || 'unknown';
                statusEl.className = 'status ' + (data.state === 'running' ? 'status-running' : 'status-stopped');
                
                document.getElementById('details').textContent = JSON.stringify(data, null, 2);
            } catch (e) {
                document.getElementById('status').textContent = 'Error: ' + e.message;
            }
        }
        
        async function startService() {
            await fetch('/api/service/start', {method: 'POST'});
            refreshStatus();
        }
        
        async function stopService() {
            await fetch('/api/service/stop', {method: 'POST'});
            refreshStatus();
        }
        
        refreshStatus();
        setInterval(refreshStatus, 5000);
    </script>
</body>
</html>
"""

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
