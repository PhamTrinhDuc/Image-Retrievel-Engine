#!/usr/bin/env python3
"""
Run Streamlit app for Image Retrieval System
Usage: python run_app.py
"""

import subprocess
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    """Run the Streamlit application"""
    
    # Set environment variables
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Command to run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "frontend/app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    print("Starting Image Retrieval Streamlit App...")
    print("App will be available at: http://localhost:8501")
    print("Make sure API server is running at: http://localhost:8000")
    print("-" * 50)
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()