#!/usr/bin/env python3
"""
Mac Setup Checker for MiniMind Training
Checks Apple Silicon compatibility and memory requirements
"""

import os
import sys
import torch
import json
import subprocess
import platform

# Add project root to path
sys.path.append(os.path.abspath('.'))

def check_apple_silicon():
    """Check if running on Apple Silicon"""
    print("🍎 Apple Silicon Check")
    print("=" * 50)
    
    # Check architecture
    arch = platform.machine()
    print(f"Architecture: {arch}")
    
    if arch == "arm64":
        print("✅ Running on Apple Silicon (ARM64)")
        return True
    else:
        print("⚠️  Not running on Apple Silicon")
        return False

def check_pytorch_mps():
    """Check PyTorch MPS support"""
    print("\n🔥 PyTorch MPS Check")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    
    # Check MPS availability
    mps_available = torch.backends.mps.is_available()
    print(f"MPS available: {mps_available}")
    
    if mps_available:
        print("✅ MPS (Metal Performance Shaders) is available")
        
        # Test MPS device
        try:
            device = torch.device("mps")
            test_tensor = torch.randn(100, 100).to(device)
            result = torch.mm(test_tensor, test_tensor)
            print("✅ MPS device test passed")
            return True
        except Exception as e:
            print(f"❌ MPS device test failed: {e}")
            return False
    else:
        print("❌ MPS not available")
        return False

def check_memory_requirements():
    """Check memory requirements for different model sizes"""
    print("\n💾 Memory Requirements Check")
    print("=" * 50)
    
    # Get system memory
    import psutil
    total_memory_gb = psutil.virtual_memory().total / 1024**3
    available_memory_gb = psutil.virtual_memory().available / 1024**3
    
    print(f"Total system memory: {total_memory_gb:.1f} GB")
    print(f"Available memory: {available_memory_gb:.1f} GB")
    
    # Memory requirements for different model sizes
    model_configs = [
        {"name": "Small (26M)", "hidden_size": 512, "layers": 8, "memory_gb": 4},
        {"name": "Medium (104M)", "hidden_size": 768, "layers": 16, "memory_gb": 12},
        {"name": "Large (1B)", "hidden_size": 768, "layers": 24, "memory_gb": 24},
    ]
    
    print("\nModel memory requirements:")
    for config in model_configs:
        status = "✅" if available_memory_gb > config["memory_gb"] else "❌"
        print(f"  {config['name']}: {config['memory_gb']} GB {status}")
    
    return available_memory_gb

def check_model_config():
    """Check model configuration and calculate parameters"""
    print("\n🤖 Model Configuration Check")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        
        # Test different configurations suitable for Mac
        configs = [
            {"name": "Mac Small (26M)", "hidden_size": 512, "num_hidden_layers": 8},
            {"name": "Mac Medium (104M)", "hidden_size": 768, "num_hidden_layers": 16},
            {"name": "Mac Large (1B)", "hidden_size": 768, "num_hidden_layers": 24},
        ]
        
        for config in configs:
            lm_config = MiniMindConfig(
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_hidden_layers"],
                num_attention_heads=config["hidden_size"] // 64,
                num_key_value_heads=config["hidden_size"] // 192,
            )
            
            model = MiniMindForCausalLM(lm_config)
            total_params = sum(p.numel() for p in model.parameters())
            
            print(f"{config['name']}:")
            print(f"  Hidden size: {config['hidden_size']}")
            print(f"  Layers: {config['num_hidden_layers']}")
            print(f"  Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
            print()
            
    except ImportError as e:
        print(f"❌ Error importing model: {e}")
        print("Make sure you're in the correct directory and model files exist")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_tokenizer():
    """Check tokenizer setup"""
    print("\n🔤 Tokenizer Check")
    print("=" * 50)
    
    try:
        from transformers import AutoTokenizer
        
        model_path = os.path.abspath('./model/')
        print(f"Looking for tokenizer in: {model_path}")
        
        if os.path.exists(model_path):
            print("✅ Model directory exists")
            
            # Check for required files
            required_files = ['tokenizer.json', 'tokenizer_config.json']
            for file in required_files:
                file_path = os.path.join(model_path, file)
                if os.path.exists(file_path):
                    print(f"✅ {file} found")
                else:
                    print(f"❌ {file} missing")
            
            # Try to load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
                print(f"✅ Tokenizer loaded successfully")
                print(f"  Vocab size: {tokenizer.vocab_size}")
                print(f"  Pad token: {tokenizer.pad_token}")
            except Exception as e:
                print(f"❌ Error loading tokenizer: {e}")
        else:
            print(f"❌ Model directory not found: {model_path}")
            
    except ImportError as e:
        print(f"❌ Error importing transformers: {e}")

def check_dataset():
    """Check dataset availability"""
    print("\n📊 Dataset Check")
    print("=" * 50)
    
    dataset_path = "./dataset/pretrain_hq.jsonl"
    print(f"Looking for dataset: {dataset_path}")
    
    if os.path.exists(dataset_path):
        print("✅ Dataset file exists")
        
        # Check file size
        file_size = os.path.getsize(dataset_path) / 1024**3
        print(f"  File size: {file_size:.2f} GB")
        
        # Count lines (first 1000 lines)
        try:
            with open(dataset_path, 'r', encoding='utf-8') as f:
                line_count = sum(1 for _ in f)
            print(f"  Total lines: {line_count:,}")
        except Exception as e:
            print(f"❌ Error reading dataset: {e}")
    else:
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please download the dataset from the provided links")

def estimate_memory_usage_mac():
    """Estimate memory usage for Mac training"""
    print("\n💾 Mac Memory Usage Estimation")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        
        # Mac-friendly configurations
        configs = [
            {"name": "Mac Small", "hidden_size": 512, "layers": 8, "batch_size": 16},
            {"name": "Mac Medium", "hidden_size": 768, "layers": 16, "batch_size": 8},
            {"name": "Mac Large", "hidden_size": 768, "layers": 24, "batch_size": 4},
        ]
        
        for config in configs:
            lm_config = MiniMindConfig(
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["layers"],
                num_attention_heads=config["hidden_size"] // 64,
                num_key_value_heads=config["hidden_size"] // 192,
            )
            
            model = MiniMindForCausalLM(lm_config)
            total_params = sum(p.numel() for p in model.parameters())
            
            def calc_memory(batch_size, seq_len):
                # Model parameters (float16)
                model_memory = total_params * 2 / 1024**3
                
                # Activations
                attention_activations = batch_size * seq_len * config["hidden_size"] * config["layers"] * 4
                mlp_activations = batch_size * seq_len * config["hidden_size"] * 8 * config["layers"]
                total_activations = (attention_activations + mlp_activations) * 2 / 1024**3
                
                # Optimizer states (AdamW)
                optimizer_memory = total_params * 8 / 1024**3
                
                # Gradients
                grad_memory = total_params * 2 / 1024**3
                
                total_memory = model_memory + total_activations + optimizer_memory + grad_memory
                
                return {
                    'model': model_memory,
                    'activations': total_activations,
                    'optimizer': optimizer_memory,
                    'gradients': grad_memory,
                    'total': total_memory
                }
            
            batch_size = config["batch_size"]
            seq_len = 512
            
            mem = calc_memory(batch_size, seq_len)
            status = "✅ OK" if mem['total'] < 16 else "⚠️  High" if mem['total'] < 32 else "❌ Too High"
            
            print(f"{config['name']} ({total_params/1e6:.0f}M params, batch={batch_size}): {mem['total']:.1f} GB {status}")
            print(f"  Model: {mem['model']:.1f} GB, Activations: {mem['activations']:.1f} GB")
            print(f"  Optimizer: {mem['optimizer']:.1f} GB, Gradients: {mem['gradients']:.1f} GB")
            print()
            
    except Exception as e:
        print(f"❌ Error calculating memory: {e}")

def check_recommendations():
    """Provide Mac-specific recommendations"""
    print("\n💡 Mac Training Recommendations")
    print("=" * 50)
    
    print("1. Use MPS backend for GPU acceleration:")
    print("   python trainer/train_pretrain_mac.py --device mps")
    print()
    
    print("2. Start with smaller model for testing:")
    print("   python trainer/train_pretrain_mac.py --hidden_size 512 --num_hidden_layers 8")
    print()
    
    print("3. Use smaller batch sizes:")
    print("   python trainer/train_pretrain_mac.py --batch_size 4 --accumulation_steps 16")
    print()
    
    print("4. Monitor memory usage:")
    print("   python scripts/monitor_memory_mac.py --interval 2")
    print()
    
    print("5. Use float16 for memory efficiency:")
    print("   python trainer/train_pretrain_mac.py --dtype float16")
    print()

if __name__ == "__main__":
    print("MiniMind Mac Training Environment Checker")
    print("=" * 60)
    print()
    
    is_apple_silicon = check_apple_silicon()
    mps_available = check_pytorch_mps()
    available_memory = check_memory_requirements()
    check_model_config()
    check_tokenizer()
    check_dataset()
    estimate_memory_usage_mac()
    check_recommendations()
    
    print("✅ Mac setup check complete!")
    print()
    
    if is_apple_silicon and mps_available:
        print("🎉 Your Mac is ready for MiniMind training!")
        print("Run: python trainer/train_pretrain_mac.py")
    else:
        print("⚠️  Some issues detected. Please check the recommendations above.")
