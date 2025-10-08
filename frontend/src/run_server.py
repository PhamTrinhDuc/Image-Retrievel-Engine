#!/usr/bin/env python3
"""
Run Streamlit app for Image Retrieval System
Usage: python run_app.py
"""

import subprocess
import sys
import os
from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))


def main():
    """Run the Streamlit application"""
    
    # Set environment variables
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    
    # Command to run streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "src/app.py",
        "--server.port", f"{os.getenv('FRONTEND_PORT', 8501)}",
        "--server.address", f"{os.getenv('FRONTEND_HOST', 'localhost')}",
        "--server.headless", "true"
    ]
    
    print("Starting Image Retrieval Streamlit App...")
    print(f"App will be available at: http://{os.getenv('FRONTEND_HOST', 'localhost')}:{os.getenv('FRONTEND_PORT', 8501)}")
    print(f"Make sure API server is running at: http://{os.getenv('BACKEND_HOST', 'localhost')}:{os.getenv('BACKEND_PORT', 8000)}")

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nApp stopped by user")
    except Exception as e:
        print(f"Error running app: {e}")

if __name__ == "__main__":
    main()