import os
import sys
__package__ = "trainer"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext
from transformers import AutoTokenizer
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from dataset.lm_dataset import PretrainDataset

warnings.filterwarnings('ignore')


def Logger(content):
    print(content)


def get_lr(current_step, total_steps, lr):
    return lr / 10 + 0.5 * lr * (1 + math.cos(math.pi * current_step / total_steps))


def clear_memory_cache():
    """Clear memory cache for Apple Silicon"""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def force_garbage_collection():
    """Force Python garbage collection to free memory"""
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def log_memory_usage(step, prefix=""):
    """Log current memory usage"""
    if torch.backends.mps.is_available():
        # MPS doesn't have detailed memory tracking like CUDA
        Logger(f"{prefix}Step {step}: MPS Memory (unified with system)")
    elif torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        Logger(f"{prefix}Step {step}: Allocated={allocated:.1f}GB, Reserved={reserved:.1f}GB")


def get_device_info():
    """Get device information and set optimal device"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        Logger("✅ Using Apple Silicon MPS (Metal Performance Shaders)")
        return device
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        Logger("✅ Using CUDA GPU")
        return device
    else:
        device = torch.device("cpu")
        Logger("⚠️  Using CPU (slow)")
        return device


def train_epoch(epoch, wandb):
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    start_time = time.time()
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        # Log memory at start of step
        if step % 10 == 0:
            log_memory_usage(step, "START ")
        
        X = X.to(args.device, non_blocking=True)
        Y = Y.to(args.device, non_blocking=True)
        loss_mask = loss_mask.to(args.device, non_blocking=True)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args.learning_rate)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            res = model(X)
            loss = loss_fct(
                res.logits.view(-1, res.logits.size(-1)),
                Y.view(-1)
            ).view(Y.size())
            loss = (loss * loss_mask).sum() / loss_mask.sum()
            loss += res.aux_loss
            loss = loss / args.accumulation_steps

        scaler.scale(loss).backward()
        
        # Clear intermediate tensors immediately after backward pass
        del res, loss
        
        # Clear input tensors to prevent memory leak
        del X, Y, loss_mask
        
        # Clear cache more frequently to prevent oscillation
        if step % args.clear_cache_freq == 0:
            clear_memory_cache()
        
        # Force garbage collection every 5 steps to prevent memory leak
        if step % 5 == 0:
            force_garbage_collection()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)
            
            # Clear cache after every optimizer step
            clear_memory_cache()
            
            # Log memory after optimizer step
            if step % 10 == 0:
                log_memory_usage(step, "AFTER_OPT ")

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.12f} epoch_Time:{}min:'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))

            if (wandb is not None):
                wandb.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr'],
                    "epoch_Time": spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60
                })

        if (step + 1) % args.save_interval == 0:
            model.eval()
            moe_path = '_moe' if lm_config.use_moe else ''
            ckp = f'{args.save_dir}/pretrain_{lm_config.hidden_size}{moe_path}.pth'

            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()

            state_dict = {k: v.half() for k, v in state_dict.items()}  # 半精度保存
            torch.save(state_dict, ckp)
            model.train()
            
            # Clear cache after saving
            clear_memory_cache()


def init_model(lm_config):
    # Try multiple tokenizer loading methods
    model_path = os.path.abspath('../model/')
    
    try:
        # Method 1: Direct path with local_files_only
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        Logger(f"✅ Tokenizer loaded from: {model_path}")
    except Exception as e1:
        Logger(f"❌ Method 1 failed: {e1}")
        try:
            # Method 2: Try without local_files_only
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            Logger(f"✅ Tokenizer loaded from: {model_path} (without local_files_only)")
        except Exception as e2:
            Logger(f"❌ Method 2 failed: {e2}")
            try:
                # Method 3: Try relative path
                tokenizer = AutoTokenizer.from_pretrained('./model/', local_files_only=True)
                Logger(f"✅ Tokenizer loaded from: ./model/")
            except Exception as e3:
                Logger(f"❌ All methods failed. Please check model directory exists.")
                raise e3
    
    model = MiniMindForCausalLM(lm_config).to(args.device)
    
    # Enable gradient checkpointing to save memory (optional)
    # if hasattr(model, 'gradient_checkpointing_enable'):
    #     model.gradient_checkpointing_enable()
    #     Logger("✅ Gradient checkpointing enabled")
    
    Logger(f'LLM可训练总参数量：{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.3f} 百万')
    return model, tokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MiniMind Pretraining - Mac Optimized")
    parser.add_argument("--out_dir", type=str, default="../out")
    parser.add_argument("--epochs", type=int, default=1)
    
    # MAC OPTIMIZED SETTINGS
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller for Mac
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--device", type=str, default="auto")  # Auto-detect best device
    parser.add_argument("--dtype", type=str, default="float16")  # Use float16 for Mac
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain-Mac")
    parser.add_argument("--num_workers", type=int, default=0)  # Mac optimization
    parser.add_argument("--ddp", action="store_true")
    parser.add_argument("--accumulation_steps", type=int, default=16)  # Adjusted for Mac
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_iters", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--clear_cache_freq", type=int, default=1, help="Clear memory cache every N steps")
    
    # MAC-FRIENDLY MODEL CONFIGURATION
    parser.add_argument('--hidden_size', default=512, type=int)  # Smaller for Mac
    parser.add_argument('--num_hidden_layers', default=8, type=int)  # Smaller for Mac
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument("--data_path", type=str, default="../dataset/pretrain_hq.jsonl")
    
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        args.device = get_device_info()
    else:
        args.device = torch.device(args.device)

    lm_config = MiniMindConfig(
        hidden_size=args.hidden_size, 
        num_hidden_layers=args.num_hidden_layers, 
        num_attention_heads=args.hidden_size // 64,  # Standard ratio
        num_key_value_heads=args.hidden_size // 192,  # GQA ratio
        use_moe=args.use_moe
    )
    
    args.save_dir = os.path.join(args.out_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_per_iter = args.batch_size * args.max_seq_len

    args.wandb_run_name = f"MiniMind-Pretrain-Mac-Epoch-{args.epochs}-BatchSize-{args.batch_size}-LearningRate-{args.learning_rate}"

    # Set up context manager based on device
    if args.device.type == "mps":
        # MPS doesn't support autocast yet, use regular context
        ctx = nullcontext()
        Logger("ℹ️  MPS autocast not available, using regular context")
    elif args.device.type == "cuda":
        ctx = torch.cuda.amp.autocast()
    else:
        ctx = nullcontext()

    # Set up distributed training (disabled for Mac)
    ddp = False  # Mac typically doesn't use DDP
    Logger("ℹ️  Distributed training disabled for Mac")

    base_seed = 1337
    torch.manual_seed(base_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(base_seed)

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)
    else:
        wandb = None

    model, tokenizer = init_model(lm_config)
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=False,  # Disabled for Mac
        drop_last=True,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=False
    )

    # Set up scaler based on device
    if args.device.type == "mps":
        # MPS doesn't support GradScaler yet
        scaler = None
        Logger("ℹ️  MPS GradScaler not available, using regular backward")
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, eps=1e-8, weight_decay=0.01)

    iter_per_epoch = len(train_loader)
    
    # Clear initial cache
    clear_memory_cache()
    
    Logger(f"Starting training with {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")
    Logger(f"Device: {args.device}")
    Logger(f"Batch size: {args.batch_size}, Accumulation steps: {args.accumulation_steps}")
    Logger(f"Effective batch size: {args.batch_size * args.accumulation_steps}")
    
    for epoch in range(args.epochs):
        train_epoch(epoch, wandb)
        # Clear cache after each epoch
        clear_memory_cache()
