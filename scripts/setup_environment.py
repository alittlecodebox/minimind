#!/usr/bin/env python3
"""
Environment Setup Script for MiniMind
Automatically detects platform and installs appropriate dependencies
"""

import os
import sys
import platform
import subprocess
import shutil


def detect_platform():
    """Detect the current platform"""
    system = platform.system().lower()
    machine = platform.machine().lower()
    
    if system == "darwin":
        if machine == "arm64":
            return "mac_arm64"
        else:
            return "mac_x86"
    elif system == "linux":
        return "linux"
    elif system == "windows":
        return "windows"
    else:
        return "unknown"


def check_uv_installed():
    """Check if uv is installed"""
    return shutil.which("uv") is not None


def install_uv():
    """Install uv package manager"""
    print("Installing uv package manager...")
    try:
        if platform.system().lower() == "windows":
            subprocess.run(["powershell", "-c", "irm https://astral.sh/uv/install.ps1 | iex"], check=True)
        else:
            subprocess.run(["curl", "-LsSf", "https://astral.sh/uv/install.sh"], check=True)
        print("✅ uv installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("❌ Failed to install uv")
        return False


def setup_environment():
    """Setup the environment based on platform"""
    platform_type = detect_platform()
    print(f"Detected platform: {platform_type}")
    
    # Check if uv is installed
    if not check_uv_installed():
        print("uv not found. Installing...")
        if not install_uv():
            print("❌ Failed to install uv. Please install manually from https://astral.sh/uv/")
            return False
    
    # Create virtual environment and install dependencies
    print("Creating virtual environment and installing dependencies...")
    
    try:
        # Create virtual environment
        subprocess.run(["uv", "venv"], check=True)
        print("✅ Virtual environment created")
        
        # Use platform-specific pyproject.toml
        if platform_type.startswith("mac"):
            print("🍎 Using Mac-specific configuration...")
            # Copy Mac-specific pyproject.toml
            import shutil
            shutil.copy("pyproject_mac.toml", "pyproject.toml")
            print("✅ Mac configuration applied")
        
        # Install dependencies
        subprocess.run(["uv", "pip", "install", "-e", "."], check=True)
        print("✅ Dependencies installed")
        
        # Platform-specific setup
        if platform_type.startswith("mac"):
            print("\n🍎 Mac-specific setup:")
            print("✅ PyTorch with MPS support installed")
            print("✅ CPU-optimized PyTorch for Apple Silicon")
            
        elif platform_type == "linux":
            print("\n🐧 Linux-specific setup:")
            print("✅ PyTorch with CUDA 12.9 support installed")
            print("✅ GPU acceleration available")
            
        elif platform_type == "windows":
            print("\n🪟 Windows-specific setup:")
            print("✅ PyTorch with CUDA 12.9 support installed")
            print("✅ GPU acceleration available")
        
        print("\n🎉 Environment setup complete!")
        print("\nNext steps:")
        print("1. Activate the virtual environment:")
        if platform.system().lower() == "windows":
            print("   .venv\\Scripts\\activate")
        else:
            print("   source .venv/bin/activate")
        
        print("2. Run the setup checker:")
        if platform_type.startswith("mac"):
            print("   python scripts/check_setup_mac.py")
        else:
            print("   python scripts/check_setup.py")
        
        print("3. Start training:")
        if platform_type.startswith("mac"):
            print("   python trainer/train_pretrain_mac.py")
        else:
            print("   python trainer/train_pretrain_optimized.py")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Setup failed: {e}")
        return False


def main():
    """Main setup function"""
    print("MiniMind Environment Setup")
    print("=" * 50)
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 10):
        print(f"❌ Python 3.10+ required, found {python_version.major}.{python_version.minor}")
        return False
    else:
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Setup environment
    return setup_environment()


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
