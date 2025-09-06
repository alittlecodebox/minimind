#!/usr/bin/env python3
"""
Memory Monitoring Script for Mac (Apple Silicon)
Monitors unified memory usage during MiniMind training
"""

import time
import psutil
import argparse
from datetime import datetime
import subprocess
import sys


def get_memory_info():
    """Get memory information for Mac"""
    # Get system memory info
    memory = psutil.virtual_memory()
    
    # Get memory pressure (macOS specific)
    try:
        result = subprocess.run(['memory_pressure'], capture_output=True, text=True, timeout=5)
        pressure_info = result.stdout
    except:
        pressure_info = "N/A"
    
    return {
        'total_gb': memory.total / 1024**3,
        'available_gb': memory.available / 1024**3,
        'used_gb': memory.used / 1024**3,
        'percent': memory.percent,
        'pressure': pressure_info
    }


def get_gpu_info():
    """Get GPU information for Mac (if available)"""
    try:
        # Try to get GPU info using system_profiler
        result = subprocess.run([
            'system_profiler', 'SPDisplaysDataType', '-json'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            gpus = data.get('SPDisplaysDataType', [])
            return gpus
    except:
        pass
    
    return []


def monitor_memory(interval=5, log_file=None):
    """Monitor memory usage on Mac"""
    
    print(f"Starting memory monitoring on Mac (interval: {interval}s)")
    print("=" * 80)
    print(f"{'Time':<20} {'Used (GB)':<12} {'Available (GB)':<15} {'Usage (%)':<12} {'Pressure':<15}")
    print("=" * 80)
    
    if log_file:
        log_f = open(log_file, 'w')
        log_f.write("timestamp,used_gb,available_gb,usage_percent,pressure\n")
    
    # Get initial GPU info
    gpus = get_gpu_info()
    if gpus:
        print(f"Detected GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            name = gpu.get('_name', 'Unknown')
            print(f"  GPU {i}: {name}")
        print()
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get memory info
            mem_info = get_memory_info()
            
            # Extract pressure level
            pressure = "Normal"
            if "WARNING" in mem_info['pressure']:
                pressure = "Warning"
            elif "CRITICAL" in mem_info['pressure']:
                pressure = "Critical"
            
            # Print status
            print(f"{timestamp:<20} {mem_info['used_gb']:<12.1f} {mem_info['available_gb']:<15.1f} {mem_info['percent']:<12.1f} {pressure:<15}")
            
            # Log to file if specified
            if log_file:
                log_f.write(f"{timestamp},{mem_info['used_gb']:.1f},{mem_info['available_gb']:.1f},{mem_info['percent']:.1f},{pressure}\n")
                log_f.flush()
            
            # Warning if memory usage is high
            if mem_info['percent'] > 80:
                print(f"⚠️  WARNING: High memory usage: {mem_info['percent']:.1f}%")
            if pressure == "Warning":
                print(f"⚠️  WARNING: Memory pressure detected")
            if pressure == "Critical":
                print(f"🚨 CRITICAL: High memory pressure!")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        if log_file:
            log_f.close()


def check_system_info():
    """Check system information"""
    print("Mac System Information")
    print("=" * 50)
    
    # CPU info
    print(f"CPU: {psutil.cpu_count()} cores")
    print(f"CPU Usage: {psutil.cpu_percent()}%")
    
    # Memory info
    mem_info = get_memory_info()
    print(f"Total Memory: {mem_info['total_gb']:.1f} GB")
    print(f"Available Memory: {mem_info['available_gb']:.1f} GB")
    print(f"Memory Usage: {mem_info['percent']:.1f}%")
    
    # GPU info
    gpus = get_gpu_info()
    if gpus:
        print(f"GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            name = gpu.get('_name', 'Unknown')
            print(f"  GPU {i}: {name}")
    else:
        print("No GPUs detected")
    
    # PyTorch info
    try:
        import torch
        print(f"PyTorch Version: {torch.__version__}")
        print(f"MPS Available: {torch.backends.mps.is_available()}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed")
    
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor memory usage on Mac")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", type=str, help="Log to CSV file")
    parser.add_argument("--check-system", action="store_true", help="Check system information and exit")
    
    args = parser.parse_args()
    
    if args.check_system:
        check_system_info()
    else:
        monitor_memory(args.interval, args.log_file)
