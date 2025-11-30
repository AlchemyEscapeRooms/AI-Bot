"""
Alchemy Trading Bot - Startup Script
=====================================

This script starts everything you need:
1. Background trading service
2. API server
3. Opens dashboard in browser

Usage:
    python run_bot.py

Author: Claude AI
Date: November 29, 2025
"""

import os
import sys
import webbrowser
import time
import threading
from pathlib import Path

def main():
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     ⚗️  ALCHEMY TRADING BOT                               ║
    ║                                                           ║
    ║     Starting up...                                        ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Check dependencies
    print("Checking dependencies...")
    
    required = ['fastapi', 'uvicorn', 'alpaca-py', 'pandas', 'numpy']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        sys.exit(1)
    
    print("✓ All dependencies installed")
    
    # Check for API keys
    from config import config
    
    api_key = config.get('alpaca.api_key')
    api_secret = config.get('alpaca.api_secret')
    
    if not api_key or not api_secret:
        print("\n❌ Alpaca API keys not configured!")
        print("\nPlease set your API keys in config.py or environment variables:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_API_SECRET=your_secret")
        sys.exit(1)
    
    print("✓ API keys configured")
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("✓ Data directory ready")
    
    # Start the API server
    print("\n🚀 Starting API server on http://localhost:8000")
    print("   Dashboard will open in your browser...\n")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:8000')
    
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run uvicorn
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )


if __name__ == "__main__":
    main()
