#!/usr/bin/env python3
"""
NOESIS Setup Script
Quick setup for the NOESIS OSINT Civil Unrest Detection System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Run a shell command"""
    print(f"🔄 Running: {command}")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            cwd=cwd, 
            check=check,
            capture_output=True,
            text=True
        )
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        return False

def setup_backend():
    """Set up the backend environment"""
    print("\n🐍 Setting up Python backend...")
    
    backend_dir = Path("backend")
    if not backend_dir.exists():
        print("❌ Backend directory not found!")
        return False
    
    # Create virtual environment
    venv_path = backend_dir / "venv"
    if not venv_path.exists():
        print("📦 Creating virtual environment...")
        if not run_command("python -m venv venv", cwd=backend_dir):
            return False
    
    # Determine activation script
    if os.name == 'nt':  # Windows
        activate_script = venv_path / "Scripts" / "activate.bat"
        pip_path = venv_path / "Scripts" / "pip.exe"
    else:  # Unix/Linux/Mac
        activate_script = venv_path / "bin" / "activate"
        pip_path = venv_path / "bin" / "pip"
    
    # Install dependencies
    print("📦 Installing Python dependencies...")
    if not run_command(f"{pip_path} install -r requirements.txt", cwd=backend_dir):
        return False
    
    # Download spaCy model
    print("🧠 Downloading spaCy model...")
    if not run_command(f"{pip_path} install spacy", cwd=backend_dir):
        return False
    if not run_command(f"{pip_path} -m spacy download en_core_web_sm", cwd=backend_dir):
        print("⚠️  Warning: Could not download spaCy model. You may need to install it manually.")
    
    # Create .env file if it doesn't exist
    env_file = backend_dir / ".env"
    env_example = backend_dir / "env.example"
    if not env_file.exists() and env_example.exists():
        print("📝 Creating .env file from template...")
        shutil.copy(env_example, env_file)
        print("✅ Created .env file. Please edit it with your API keys.")
    
    # Initialize database
    print("🗄️  Initializing database...")
    if not run_command(f"{pip_path} -m python create_db.py", cwd=backend_dir):
        print("⚠️  Warning: Could not initialize database. You may need to run it manually.")
    
    return True

def setup_frontend():
    """Set up the frontend environment"""
    print("\n⚛️  Setting up React frontend...")
    
    frontend_dir = Path("frontend")
    if not frontend_dir.exists():
        print("❌ Frontend directory not found!")
        return False
    
    # Install Node.js dependencies
    print("📦 Installing Node.js dependencies...")
    if not run_command("npm install", cwd=frontend_dir):
        return False
    
    return True

def main():
    """Main setup function"""
    print("🚀 NOESIS Setup Script")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("❌ Python 3.8+ is required!")
        return
    
    print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro} detected")
    
    # Check Node.js
    if not run_command("node --version", check=False):
        print("❌ Node.js is required! Please install Node.js 16+")
        return
    
    # Setup backend
    if not setup_backend():
        print("❌ Backend setup failed!")
        return
    
    # Setup frontend
    if not setup_frontend():
        print("❌ Frontend setup failed!")
        return
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Edit backend/.env with your API keys (optional)")
    print("2. Start the backend: cd backend && python start.py")
    print("3. Start the frontend: cd frontend && npm run dev")
    print("4. Open http://localhost:5173 in your browser")
    print("\n📚 For more information, see README.md")

if __name__ == "__main__":
    main() 