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
    ===========================================================

         ALCHEMY TRADING BOT

         Starting up...

    ===========================================================
    """)
    
    # Check dependencies
    print("Checking dependencies...")

    # Map package names to their import names
    required = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'alpaca-py': 'alpaca',  # alpaca-py imports as 'alpaca'
        'pandas': 'pandas',
        'numpy': 'numpy'
    }
    missing = []

    for pkg_name, import_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(pkg_name)
    
    if missing:
        print(f"\n[X] Missing packages: {', '.join(missing)}")
        print(f"\nInstall with: pip install {' '.join(missing)}")
        sys.exit(1)

    print("[OK] All dependencies installed")

    # Load .env file first
    from dotenv import load_dotenv
    load_dotenv()

    # Check for API keys (check environment variables directly)
    api_key = os.environ.get('ALPACA_API_KEY')
    api_secret = os.environ.get('ALPACA_SECRET_KEY')

    if not api_key or not api_secret:
        print("\n[X] Alpaca API keys not configured!")
        print("\nPlease set your API keys in .env file:")
        print("  ALPACA_API_KEY=your_key")
        print("  ALPACA_SECRET_KEY=your_secret")
        sys.exit(1)

    print("[OK] API keys configured")

    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    print("[OK] Data directory ready")

    # Start the API server
    print("\n>>> Starting API server on http://localhost:8000")
    print("    Dashboard will open in your browser...\n")
    
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
