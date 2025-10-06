#!/usr/bin/env python3
"""
Simple script to run the Image Retrieval API server
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import uvicorn
from route_search import app

def run_server(host="0.0.0.0", port=8000, reload=True):
    """Run the FastAPI server"""
    print(f"🚀 Starting Image Retrieval API Server...")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Reload: {reload}")
    print(f"   API Docs: http://{host}:{port}/docs")
    print("=" * 50)
    
    uvicorn.run(
        "route_search:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Image Retrieval API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    run_server(
      host=args.host,
      port=args.port,
      reload=not args.no_reload
    )