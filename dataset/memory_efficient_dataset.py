import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class MemoryEfficientPretrainDataset(Dataset):
    """Memory-efficient dataset that doesn't load all data into memory at once"""
    
    def __init__(self, data_path, tokenizer, max_length=512, cache_size=10000):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        
        # Count total lines without loading data
        self.total_lines = self._count_lines()
        
        # Cache for recently accessed items
        self.cache = {}
        self.cache_order = []
        
    def _count_lines(self):
        """Count total lines in the file"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _get_line(self, index):
        """Get a specific line from the file"""
        if index in self.cache:
            return self.cache[index]
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    data = json.loads(line.strip())
                    
                    # Add to cache
                    if len(self.cache) >= self.cache_size:
                        # Remove oldest item
                        oldest = self.cache_order.pop(0)
                        del self.cache[oldest]
                    
                    self.cache[index] = data
                    self.cache_order.append(index)
                    return data
        
        raise IndexError(f"Index {index} out of range")

    def __len__(self):
        return self.total_lines

    def __getitem__(self, index):
        sample = self._get_line(index)

        # 构建输入文本
        encoding = self.tokenizer(
            str(sample['text']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = encoding.input_ids.squeeze()
        loss_mask = (input_ids != self.tokenizer.pad_token_id)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)
        
        # Clean up intermediate tensors
        del encoding, input_ids
        
        return X, Y, loss_mask


class MemoryEfficientSFTDataset(Dataset):
    """Memory-efficient SFT dataset"""
    
    def __init__(self, jsonl_path, tokenizer, max_length=1024, cache_size=5000):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.cache_size = cache_size
        
        # Count total lines
        self.total_lines = self._count_lines()
        
        # Cache for recently accessed items
        self.cache = {}
        self.cache_order = []
        
        self.bos_id = tokenizer('<|im_start|>assistant', add_special_tokens=False).input_ids
        self.eos_id = tokenizer('<|im_end|>', add_special_tokens=False).input_ids

    def _count_lines(self):
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    
    def _get_line(self, index):
        if index in self.cache:
            return self.cache[index]
            
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i == index:
                    data = json.loads(line.strip())
                    
                    # Add to cache
                    if len(self.cache) >= self.cache_size:
                        oldest = self.cache_order.pop(0)
                        del self.cache[oldest]
                    
                    self.cache[index] = data
                    self.cache_order.append(index)
                    return data
        
        raise IndexError(f"Index {index} out of range")

    def __len__(self):
        return self.total_lines

    def _create_chat_prompt(self, conversations):
        """构建符合ChatML格式的对话"""
        messages = []
        for i, turn in enumerate(conversations):
            role = 'user' if i % 2 == 0 else 'assistant'
            messages.append({"role": role, "content": turn['content']})
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end:end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(start + 1, min(end + len(self.eos_id) + 1, self.max_length)):
                    loss_mask[j] = 1
                i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index):
        sample = self._get_line(index)
        
        # 构建对话提示
        prompt = self._create_chat_prompt(sample['conversations'])
        input_ids = self.tokenizer(prompt).input_ids[:self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (self.max_length - len(input_ids))

        # 生成动态损失掩码
        loss_mask = self._generate_loss_mask(input_ids)

        # 构建训练数据
        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        return X, Y, loss_mask
