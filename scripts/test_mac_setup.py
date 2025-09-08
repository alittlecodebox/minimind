#!/usr/bin/env python3
"""
Quick test script for Mac setup
Tests basic functionality without full training
"""

import os
import sys
import torch
import time

# Add project root to path
sys.path.append(os.path.abspath('.'))

def test_mps_availability():
    """Test MPS availability and basic functionality"""
    print("🍎 Testing MPS (Metal Performance Shaders)")
    print("=" * 50)
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    if not torch.backends.mps.is_available():
        print("❌ MPS not available. Please check PyTorch installation.")
        return False
    
    try:
        # Test basic MPS operations
        device = torch.device("mps")
        print(f"✅ MPS device created: {device}")
        
        # Test tensor operations
        x = torch.randn(100, 100).to(device)
        y = torch.randn(100, 100).to(device)
        z = torch.mm(x, y)
        
        print("✅ Basic tensor operations work on MPS")
        
        # Test memory management
        torch.mps.empty_cache()
        print("✅ MPS memory cache clearing works")
        
        return True
        
    except Exception as e:
        print(f"❌ MPS test failed: {e}")
        return False

def test_model_loading():
    """Test model loading and basic forward pass"""
    print("\n🤖 Testing Model Loading")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        from transformers import AutoTokenizer
        
        # Create small test model
        config = MiniMindConfig(
            hidden_size=512,
            num_hidden_layers=4,  # Small for testing
            num_attention_heads=8,
            num_key_value_heads=2
        )
        
        model = MiniMindForCausalLM(config)
        print(f"✅ Model created: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
        
        # Test tokenizer loading
        model_path = os.path.abspath('./model/')
        if os.path.exists(model_path):
            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            print("✅ Tokenizer loaded")
        else:
            print("❌ Model directory not found")
            return False
        
        # Test forward pass on MPS
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            model = model.to(device)
            
            # Create test input
            test_text = "Hello, world!"
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            input_ids = inputs.input_ids.to(device)
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                outputs = model(input_ids)
                end_time = time.time()
            
            print(f"✅ Forward pass successful on MPS")
            print(f"   Input shape: {input_ids.shape}")
            print(f"   Output shape: {outputs.logits.shape}")
            print(f"   Time: {(end_time - start_time)*1000:.1f}ms")
            
            # Test memory usage
            torch.mps.empty_cache()
            print("✅ Memory cleanup successful")
            
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_training_step():
    """Test a single training step"""
    print("\n🏋️ Testing Training Step")
    print("=" * 50)
    
    try:
        from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
        from transformers import AutoTokenizer
        import torch.nn as nn
        
        # Create small test model
        config = MiniMindConfig(
            hidden_size=256,  # Very small for testing
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2
        )
        
        model = MiniMindForCausalLM(config)
        device = torch.device("mps")
        model = model.to(device)
        
        # Create test data
        batch_size = 2
        seq_len = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        labels = torch.randint(0, 1000, (batch_size, seq_len)).to(device)
        
        # Test forward pass
        outputs = model(input_ids)
        logits = outputs.logits
        
        # Test loss calculation
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        print(f"✅ Forward pass: {logits.shape}")
        print(f"✅ Loss calculation: {loss.item():.4f}")
        
        # Test backward pass (without scaler for MPS)
        loss.backward()
        print("✅ Backward pass successful")
        
        # Test optimizer step
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad()
        print("✅ Optimizer step successful")
        
        # Cleanup
        torch.mps.empty_cache()
        print("✅ Memory cleanup successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Training step test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("MiniMind Mac Setup Test")
    print("=" * 60)
    
    tests = [
        ("MPS Availability", test_mps_availability),
        ("Model Loading", test_model_loading),
        ("Training Step", test_training_step),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n📊 Test Results Summary")
    print("=" * 50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Your Mac setup is ready for training.")
        print("\nNext steps:")
        print("1. Run full setup check: python scripts/check_setup_mac.py")
        print("2. Start training: python trainer/train_pretrain_mac.py")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        print("Make sure you have:")
        print("- PyTorch with MPS support")
        print("- Model files in ./model/ directory")
        print("- Proper virtual environment activated")

if __name__ == "__main__":
    main()
