# MiniMind Mac Setup Guide

This guide helps you set up MiniMind training on Mac with Apple Silicon.

## 🍎 Quick Start

### Option 1: Automatic Setup (Recommended)
```bash
# Run the automatic setup script
python scripts/setup_environment.py
```

### Option 2: Manual Setup
```bash
# 1. Create virtual environment
uv venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Install dependencies (Mac-optimized)
uv pip install -e .

# 4. Check setup
python scripts/check_setup_mac.py

# 5. Start training
python trainer/train_pretrain_mac.py
```

## 🔧 Platform-Specific Configuration

MiniMind uses separate configuration files for different platforms:

- **`pyproject.toml`**: Default (Linux/Windows with CUDA)
- **`pyproject_mac.toml`**: Mac-specific (CPU PyTorch with MPS)
- **`pyproject_linux.toml`**: Linux/Windows (CUDA PyTorch)

### Automatic Configuration
The setup script automatically detects your platform and uses the correct configuration.

### Manual Configuration Switch
```bash
# Switch to Mac configuration
python scripts/switch_config.py --platform mac

# Switch to Linux/Windows configuration  
python scripts/switch_config.py --platform linux

# Show current configuration
python scripts/switch_config.py --show

# Restore backed up configuration
python scripts/switch_config.py --restore
```

## 📊 Mac-Specific Features

### Memory Management
- **Unified Memory**: No GPU/CPU memory transfer overhead
- **MPS Acceleration**: Uses Apple's Metal Performance Shaders
- **Automatic Memory Clearing**: `torch.mps.empty_cache()`

### Recommended Model Sizes
| Model | Parameters | Memory | Mac M1 | Mac M1 Pro | Mac M2 Pro |
|-------|------------|--------|--------|------------|------------|
| Small | 26M | 4GB | ✅ Fast | ✅ Very Fast | ✅ Very Fast |
| Medium | 104M | 12GB | ✅ Good | ✅ Fast | ✅ Very Fast |
| Large | 1B | 24GB | ⚠️ Slow | ✅ Good | ✅ Fast |

## 🚀 Training Commands

### Small Model (Recommended for testing)
```bash
python trainer/train_pretrain_mac.py \
    --hidden_size 512 \
    --num_hidden_layers 8 \
    --batch_size 8 \
    --accumulation_steps 16
```

### Medium Model (Good balance)
```bash
python trainer/train_pretrain_mac.py \
    --hidden_size 768 \
    --num_hidden_layers 16 \
    --batch_size 4 \
    --accumulation_steps 32
```

### Large Model (If you have enough memory)
```bash
python trainer/train_pretrain_mac.py \
    --hidden_size 768 \
    --num_hidden_layers 24 \
    --batch_size 2 \
    --accumulation_steps 64
```

## 📈 Memory Monitoring

Monitor memory usage during training:
```bash
# In another terminal
python scripts/monitor_memory_mac.py --interval 2 --log-file mac_memory_log.csv
```

## 🔍 Troubleshooting

### Common Issues

1. **MPS not available**
   ```bash
   # Check PyTorch version
   python -c "import torch; print(torch.__version__)"
   
   # Check MPS availability
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **Memory issues**
   - Reduce batch size: `--batch_size 2`
   - Increase accumulation steps: `--accumulation_steps 64`
   - Use smaller model: `--hidden_size 512 --num_hidden_layers 8`

3. **Slow training**
   - Ensure MPS is enabled: `torch.backends.mps.is_available()`
   - Use float16: `--dtype float16`
   - Reduce sequence length: `--max_seq_len 256`

## 📝 Notes

- **MPS Limitations**: Some operations may fall back to CPU
- **Memory Efficiency**: Mac uses unified memory, so GPU and CPU share the same pool
- **Power Efficiency**: Mac training is more power-efficient than discrete GPUs
- **Silent Operation**: No fan noise during training

## 🆚 Mac vs Linux Comparison

| Feature | Mac (MPS) | Linux (CUDA) |
|---------|-----------|--------------|
| Memory | Unified (efficient) | Separate GPU/CPU |
| Power | Low consumption | High consumption |
| Noise | Silent | Fan noise |
| Setup | Simple | Complex |
| Performance | Good | Excellent |
| Cost | High upfront | Lower upfront |

Happy training on your Mac! 🎉
