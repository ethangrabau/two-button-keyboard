"""
Data processor for training the pattern-constrained transformer model.
Handles:
- Chat data loading and preprocessing
- Token and pattern generation
- Training batch creation with masks
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path
from keyboard_mapping import KeyboardMapper
from transformers import PreTrainedTokenizerFast
import numpy as np

class PatternDataset(Dataset):
    """Dataset for pattern-constrained transformer training"""
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        max_context_length: int = 128,
        max_pattern_length: int = 32,
    ):
        self.tokenizer = tokenizer
        self.keyboard_mapper = KeyboardMapper()
        self.max_context_length = max_context_length
        self.max_pattern_length = max_pattern_length
        
        # Pattern vocab: L=0, R=1, Space=2, Padding=3
        self.pattern_vocab = {'L': 0, 'R': 1, ' ': 2, '[PAD]': 3}
        
        # Initialize empty data lists
        self.context_ids: List[List[int]] = []
        self.pattern_ids: List[List[int]] = []
        self.target_ids: List[List[int]] = []
        
    def preprocess_message(self, message: str) -> Tuple[List[int], List[str]]:
        """Convert message to token IDs and patterns."""
        # Tokenize message
        tokens = self.tokenizer.tokenize(message)
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Generate patterns for each token
        patterns = []
        for token in tokens:
            # Convert token back to text to get pattern
            token_text = self.tokenizer.convert_tokens_to_string([token])
            pattern = self.keyboard_mapper.token_to_pattern(token_text)
            if pattern:  # Only add if pattern is valid
                patterns.append(pattern)
            else:
                # For tokens without valid patterns (e.g. punctuation),
                # use a single L or R based on keyboard side
                patterns.append('L')
                
        return token_ids, patterns
        
    def add_conversation(self, messages: List[str]) -> None:
        """Process a conversation and add to dataset."""
        for i in range(1, len(messages)):  # Start from 1 to use previous as context
            context = messages[i-1]
            target = messages[i]
            
            # Process context and target
            context_ids, _ = self.preprocess_message(context)
            target_ids, target_patterns = self.preprocess_message(target)
            
            # Only add if we have valid patterns
            if target_patterns:
                self.context_ids.append(context_ids)
                self.target_ids.append(target_ids)
                
                # Convert patterns to ids
                pattern_ids = [self.pattern_vocab[p] for p in target_patterns[0]]
                self.pattern_ids.append(pattern_ids)
                
    def load_chat_data(self, data_path: str) -> None:
        """Load chat data from json file."""
        data_path = Path(data_path)
        if not data_path.exists():
            raise FileNotFoundError(f"Chat data not found at {data_path}")
            
        with open(data_path) as f:
            chat_data = json.load(f)
            
        # Process each conversation
        for conversation in chat_data:
            messages = conversation.get('messages', [])
            if len(messages) > 1:  # Need at least 2 messages for context
                self.add_conversation(messages)
                
    def __len__(self) -> int:
        return len(self.target_ids)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example."""
        # Get raw ids
        context = self.context_ids[idx]
        pattern = self.pattern_ids[idx]
        target = self.target_ids[idx]
        
        # Truncate or pad context
        if len(context) > self.max_context_length:
            context = context[-self.max_context_length:]
        else:
            pad_length = self.max_context_length - len(context)
            context = context + [self.tokenizer.pad_token_id] * pad_length
            
        # Create context mask (1 for real tokens, 0 for padding)
        context_mask = [1] * min(len(self.context_ids[idx]), self.max_context_length) + \
                      [0] * max(0, self.max_context_length - len(self.context_ids[idx]))
                      
        # Truncate or pad pattern
        if len(pattern) > self.max_pattern_length:
            pattern = pattern[:self.max_pattern_length]
        else:
            pad_length = self.max_pattern_length - len(pattern)
            pattern = pattern + [self.pattern_vocab['[PAD]']] * pad_length
            
        # Create pattern mask
        pattern_mask = [1] * min(len(self.pattern_ids[idx]), self.max_pattern_length) + \
                      [0] * max(0, self.max_pattern_length - len(self.pattern_ids[idx]))
                      
        # For decoder input, shift target right and add BOS token
        decoder_input = [self.tokenizer.bos_token_id] + target[:-1]
        if len(decoder_input) > self.max_pattern_length:
            decoder_input = decoder_input[:self.max_pattern_length]
        else:
            pad_length = self.max_pattern_length - len(decoder_input)
            decoder_input = decoder_input + [self.tokenizer.pad_token_id] * pad_length
            
        # Create target with padding
        if len(target) > self.max_pattern_length:
            target = target[:self.max_pattern_length]
        else:
            pad_length = self.max_pattern_length - len(target)
            target = target + [self.tokenizer.pad_token_id] * pad_length
            
        return {
            'input_ids': torch.tensor(context),
            'attention_mask': torch.tensor(context_mask),
            'pattern_ids': torch.tensor(pattern),
            'pattern_attention_mask': torch.tensor(pattern_mask),
            'decoder_input_ids': torch.tensor(decoder_input),
            'labels': torch.tensor(target)
        }

class DataCollator:
    """Collate data samples into batches."""
    def __init__(self, pad_token_id: int, pattern_pad_id: int = 3):
        self.pad_token_id = pad_token_id
        self.pattern_pad_id = pattern_pad_id
        
    def __call__(self, samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of samples."""
        batch = {
            key: torch.stack([sample[key] for sample in samples])
            for key in samples[0].keys()
        }
        
        return batch

def create_pattern_mask(tokenizer: PreTrainedTokenizerFast, keyboard_mapper: KeyboardMapper) -> torch.Tensor:
    """Create pattern constraint mask for the full vocabulary."""
    vocab_size = tokenizer.vocab_size
    pattern_mask = torch.zeros(vocab_size, dtype=torch.bool)
    
    for token_id in range(vocab_size):
        token = tokenizer.convert_ids_to_tokens(token_id)
        # Convert token to text to check pattern
        token_text = tokenizer.convert_tokens_to_string([token])
        pattern = keyboard_mapper.token_to_pattern(token_text)
        # Token is valid if it has a pattern
        pattern_mask[token_id] = bool(pattern)
        
    return pattern_mask

def get_dataloaders(
    tokenizer: PreTrainedTokenizerFast,
    data_path: str,
    batch_size: int = 32,
    max_context_length: int = 128,
    max_pattern_length: int = 32,
    val_split: float = 0.1,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """Create training and validation dataloaders."""
    
    # Create dataset
    dataset = PatternDataset(
        tokenizer=tokenizer,
        max_context_length=max_context_length,
        max_pattern_length=max_pattern_length
    )
    
    # Load data
    dataset.load_chat_data(data_path)
    
    # Split into train/val
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    # Use random split with fixed seed
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size],
        generator=generator
    )
    
    # Create data collator
    collator = DataCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    # Test data processing
    from transformers import GPT2TokenizerFast
    
    print("Testing data processor...")
    
    # Initialize tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create test data
    test_data = [{
        "messages": [
            "Hello, how are you?",
            "I'm doing well, thanks!",
            "That's great to hear!"
        ]
    }]
    
    # Save test data
    test_path = Path("test_chat_data.json")
    with open(test_path, 'w') as f:
        json.dump(test_data, f)
    
    try:
        # Create dataloaders
        train_loader, val_loader = get_dataloaders(
            tokenizer=tokenizer,
            data_path=str(test_path),
            batch_size=2
        )
        
        # Test batch iteration
        batch = next(iter(train_loader))
        print("\nBatch structure:")
        for key, value in batch.items():
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            
        print("\nDecoding first sequence:")
        decoded = tokenizer.decode(batch['input_ids'][0])
        print(f"Input text: {decoded}")
        
        # Test pattern mask creation
        pattern_mask = create_pattern_mask(tokenizer, KeyboardMapper())
        print(f"\nPattern mask shape: {pattern_mask.shape}")
        print(f"Valid tokens: {pattern_mask.sum().item()}/{len(pattern_mask)}")
        
        print("\nAll tests passed! 🎉")
        
    finally:
        # Cleanup test file
        test_path.unlink()
