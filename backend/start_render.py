#!/usr/bin/env python3
"""
Startup script for Render deployment
This script initializes the database and starts the FastAPI application
"""
import os
import sys
import subprocess
from pathlib import Path

def init_database():
    """Initialize database tables"""
    try:
        print("Initializing database...")
        from create_db import main as create_db_main
        create_db_main()
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        print("Continuing with startup...")

def main():
    """Main startup function"""
    print("🚀 Starting NOESIS Backend on Render...")
    
    # Initialize database
    init_database()
    
    # Get port from environment variable (Render sets this)
    port = os.environ.get("PORT", "8000")
    
    # Start the FastAPI application
    print(f"Starting uvicorn server on port {port}...")
    subprocess.run([
        "uvicorn", 
        "main:app", 
        "--host", "0.0.0.0", 
        "--port", port
    ])

if __name__ == "__main__":
    main()
