#!/usr/bin/env python3
"""
Setup checker for MiniMind training environment
Run this to verify your environment and model configuration
"""

import os
import sys
import torch
import json

# Add project root to path
sys.path.append(os.path.abspath('.'))

def check_environment():
    """Check basic environment setup"""
    print("🔍 Environment Check")
    print("=" * 50)
    
    # Python version
    print(f"Python version: {sys.version}")
    
    # PyTorch version and CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    
    print()

def check_model_config():
    """Check model configuration and calculate parameters"""
    print("🤖 Model Configuration Check")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        
        # Test different configurations
        configs = [
            {"name": "Default (26M)", "hidden_size": 512, "num_hidden_layers": 8},
            {"name": "Medium (104M)", "hidden_size": 768, "num_hidden_layers": 16},
            {"name": "Large (1B)", "hidden_size": 768, "num_hidden_layers": 24},
        ]
        
        for config in configs:
            lm_config = MiniMindConfig(
                hidden_size=config["hidden_size"],
                num_hidden_layers=config["num_hidden_layers"],
                num_attention_heads=config["hidden_size"] // 64,  # Standard ratio
                num_key_value_heads=config["hidden_size"] // 192,  # GQA ratio
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
    print("🔤 Tokenizer Check")
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
    print("📊 Dataset Check")
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

def estimate_memory_usage():
    """Estimate memory usage for different configurations"""
    print("💾 Memory Usage Estimation")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        
        # 1B model configuration
        config = MiniMindConfig(
            hidden_size=768,
            num_hidden_layers=24,
            num_attention_heads=12,
            num_key_value_heads=4
        )
        
        model = MiniMindForCausalLM(config)
        total_params = sum(p.numel() for p in model.parameters())
        
        def calc_memory(batch_size, seq_len):
            # Model parameters (bfloat16)
            model_memory = total_params * 2 / 1024**3
            
            # Activations
            attention_activations = batch_size * seq_len * 768 * 24 * 4
            mlp_activations = batch_size * seq_len * 768 * 8 * 24
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
        
        batch_sizes = [32, 16, 8, 4, 2]
        seq_len = 512
        
        print(f"1B Model ({total_params/1e6:.0f}M parameters) - Sequence length: {seq_len}")
        print()
        
        for batch_size in batch_sizes:
            mem = calc_memory(batch_size, seq_len)
            status = "✅ OK" if mem['total'] < 80 else "❌ OOM"
            print(f"Batch size {batch_size:2d}: {mem['total']:5.1f} GB {status}")
            print(f"  Model: {mem['model']:.1f} GB, Activations: {mem['activations']:.1f} GB")
            print(f"  Optimizer: {mem['optimizer']:.1f} GB, Gradients: {mem['gradients']:.1f} GB")
            print()
            
    except Exception as e:
        print(f"❌ Error calculating memory: {e}")

if __name__ == "__main__":
    print("MiniMind Training Environment Checker")
    print("=" * 60)
    print()
    
    check_environment()
    check_model_config()
    check_tokenizer()
    check_dataset()
    estimate_memory_usage()
    
    print("✅ Setup check complete!")
    print()
    print("Next steps:")
    print("1. If all checks pass, try the optimized training script")
    print("2. Use batch_size=4 or smaller to avoid OOM")
    print("3. Monitor memory usage during training")
