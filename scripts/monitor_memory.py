#!/usr/bin/env python3
"""
GPU Memory Monitoring Script for MiniMind Training
Run this in a separate terminal to monitor memory usage during training
"""

import time
import psutil
import GPUtil
import argparse
from datetime import datetime


def monitor_memory(interval=5, log_file=None):
    """Monitor GPU and CPU memory usage"""
    
    print(f"Starting memory monitoring (interval: {interval}s)")
    print("=" * 60)
    print(f"{'Time':<20} {'GPU Mem (GB)':<12} {'GPU Util (%)':<12} {'CPU Mem (%)':<12}")
    print("=" * 60)
    
    if log_file:
        log_f = open(log_file, 'w')
        log_f.write("timestamp,gpu_memory_gb,gpu_utilization,cpu_memory_percent\n")
    
    try:
        while True:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Get GPU info
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    gpu_mem_gb = gpu.memoryUsed / 1024  # Convert MB to GB
                    gpu_util = gpu.load * 100
                else:
                    gpu_mem_gb = 0
                    gpu_util = 0
            except:
                gpu_mem_gb = 0
                gpu_util = 0
            
            # Get CPU memory
            cpu_mem = psutil.virtual_memory().percent
            
            # Print status
            print(f"{timestamp:<20} {gpu_mem_gb:<12.1f} {gpu_util:<12.1f} {cpu_mem:<12.1f}")
            
            # Log to file if specified
            if log_file:
                log_f.write(f"{timestamp},{gpu_mem_gb:.1f},{gpu_util:.1f},{cpu_mem:.1f}\n")
                log_f.flush()
            
            # Warning if memory usage is high
            if gpu_mem_gb > 70:
                print(f"⚠️  WARNING: High GPU memory usage: {gpu_mem_gb:.1f} GB")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped")
        if log_file:
            log_f.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor GPU and CPU memory usage")
    parser.add_argument("--interval", type=int, default=5, help="Monitoring interval in seconds")
    parser.add_argument("--log-file", type=str, help="Log to CSV file")
    
    args = parser.parse_args()
    
    monitor_memory(args.interval, args.log_file)
