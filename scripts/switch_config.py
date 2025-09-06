#!/usr/bin/env python3
"""
Configuration Switcher for MiniMind
Switch between different platform configurations
"""

import os
import sys
import platform
import shutil
import argparse


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


def switch_config(platform_type=None):
    """Switch to platform-specific configuration"""
    if platform_type is None:
        platform_type = detect_platform()
    
    print(f"Switching to {platform_type} configuration...")
    
    # Backup current pyproject.toml
    if os.path.exists("pyproject.toml"):
        shutil.copy("pyproject.toml", "pyproject.toml.backup")
        print("✅ Backed up current pyproject.toml")
    
    # Copy platform-specific configuration
    if platform_type.startswith("mac"):
        if os.path.exists("pyproject_mac.toml"):
            shutil.copy("pyproject_mac.toml", "pyproject.toml")
            print("✅ Switched to Mac configuration (CPU PyTorch with MPS)")
        else:
            print("❌ pyproject_mac.toml not found")
            return False
    else:
        # Default to Linux/Windows configuration
        if os.path.exists("pyproject_linux.toml"):
            shutil.copy("pyproject_linux.toml", "pyproject.toml")
            print("✅ Switched to Linux/Windows configuration (CUDA PyTorch)")
        else:
            print("ℹ️  Using default configuration (CUDA PyTorch)")
    
    return True


def restore_config():
    """Restore backed up configuration"""
    if os.path.exists("pyproject.toml.backup"):
        shutil.copy("pyproject.toml.backup", "pyproject.toml")
        print("✅ Restored backed up configuration")
        return True
    else:
        print("❌ No backup configuration found")
        return False


def show_current_config():
    """Show current configuration"""
    if os.path.exists("pyproject.toml"):
        with open("pyproject.toml", "r") as f:
            content = f.read()
        
        if "pytorch-cu129" in content:
            print("Current configuration: Linux/Windows (CUDA PyTorch)")
        elif "pytorch" in content and "cu129" not in content:
            print("Current configuration: Mac (CPU PyTorch with MPS)")
        else:
            print("Current configuration: Unknown")
    else:
        print("No pyproject.toml found")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Switch MiniMind configuration")
    parser.add_argument("--platform", choices=["mac", "linux", "windows"], 
                       help="Target platform (auto-detect if not specified)")
    parser.add_argument("--restore", action="store_true", 
                       help="Restore backed up configuration")
    parser.add_argument("--show", action="store_true", 
                       help="Show current configuration")
    
    args = parser.parse_args()
    
    if args.show:
        show_current_config()
        return
    
    if args.restore:
        restore_config()
        return
    
    # Switch configuration
    platform_type = args.platform
    if platform_type:
        # Map platform to full type
        if platform_type == "mac":
            platform_type = "mac_arm64"  # Default to ARM64
        elif platform_type == "linux":
            platform_type = "linux"
        elif platform_type == "windows":
            platform_type = "windows"
    
    success = switch_config(platform_type)
    
    if success:
        print("\nNext steps:")
        print("1. Reinstall dependencies:")
        print("   uv pip install -e .")
        print("2. Check setup:")
        if platform_type and platform_type.startswith("mac"):
            print("   python scripts/check_setup_mac.py")
        else:
            print("   python scripts/check_setup.py")
    else:
        print("❌ Configuration switch failed")


if __name__ == "__main__":
    main()
