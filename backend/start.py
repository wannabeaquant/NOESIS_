#!/usr/bin/env python3
"""
NOESIS Backend Server Startup Script
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def main():
    """Start the NOESIS backend server"""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    
    print("🚀 Starting NOESIS Backend Server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print("📚 API Docs: http://localhost:8000/docs")
    print("🔍 Interactive API: http://localhost:8000/redoc")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 