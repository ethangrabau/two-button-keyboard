"""
Data loader for training direct pattern transformer model.
Handles both raw Google Messages JSON and preprocessed CSV formats.
"""

import json
import pandas as pd
import torch
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from keyboard_mapping import KeyboardMapper
import logging
from datetime import datetime
from tqdm import tqdm
import os

# Set tokenizer parallelism explicitly
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

def pad_sequence(sequence: List[int], max_length: int, pad_token_id: int) -> List[int]:
    """Pad a sequence to max_length"""
    if len(sequence) > max_length - 1:  # Leave room for EOS token
        return sequence[:max_length-1] + [pad_token_id]
    return sequence + [pad_token_id] * (max_length - len(sequence))

class MessageDataset:
    """Dataset for loading and preprocessing chat messages"""
    
    def __init__(
        self,
        tokenizer,
        keyboard_mapper: Optional[KeyboardMapper] = None,
        max_context_length: int = 128,
        max_pattern_length: int = 32,
        min_message_length: int = 4,
        word_freq_file: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.keyboard_mapper = keyboard_mapper or KeyboardMapper()
        self.max_context_length = max_context_length
        self.max_pattern_length = max_pattern_length
        self.min_message_length = min_message_length
        
        # Load word frequencies if provided
        self.word_frequencies = {}
        if word_freq_file:
            with open(word_freq_file) as f:
                self.word_frequencies = json.load(f)
                
        # Initialize data lists
        self.context_messages: List[str] = []
        self.target_messages: List[str] = []
        self.patterns: List[str] = []
        
    def load_csv_data(self, csv_file: str):
        """Load messages from preprocessed CSV file"""
        try:
            logger.info(f"Loading CSV data from {csv_file}")
            df = pd.read_csv(csv_file)
            logger.info(f"Loaded {len(df)} rows from CSV")
            
            # Process each message
            for _, row in df.iterrows():
                try:
                    message = str(row['Message'])
                    pattern = str(row['RL_Translation'])
                    
                    if len(message) >= self.min_message_length:
                        self.context_messages.append(message)
                        self.target_messages.append(message)
                        self.patterns.append(pattern)
                except Exception as e:
                    logger.error(f"Error processing row: {e}")
                    continue
                    
            logger.info(f"Successfully processed {len(self.context_messages)} messages")
            
        except Exception as e:
            logger.error(f"Error loading CSV file {csv_file}: {e}")
            raise
        
    def __len__(self) -> int:
        return len(self.patterns)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single training example"""
        # Get raw text and pattern
        context = self.context_messages[idx]
        target = self.target_messages[idx]
        pattern = self.patterns[idx]
        
        # Tokenize text and add EOS token
        context_ids = self.tokenizer.encode(
            context,
            add_special_tokens=False,  # We'll add EOS manually
            max_length=self.max_context_length - 1,  # Leave room for EOS
            truncation=True
        ) + [self.tokenizer.eos_token_id]  # Add EOS token
        
        target_ids = self.tokenizer.encode(
            target,
            add_special_tokens=False,  # We'll add EOS manually
            max_length=self.max_pattern_length - 1,  # Leave room for EOS
            truncation=True
        ) + [self.tokenizer.eos_token_id]  # Add EOS token
        
        # Pad sequences
        context_ids = pad_sequence(context_ids, self.max_context_length, self.tokenizer.pad_token_id)
        target_ids = pad_sequence(target_ids, self.max_pattern_length, self.tokenizer.pad_token_id)
        
        # Convert pattern to IDs (L=0, R=1, Space=2, Padding=3)
        pattern_ids = []
        for char in pattern:
            if char == 'L':
                pattern_ids.append(0)
            elif char == 'R':
                pattern_ids.append(1)
            elif char == ' ':
                pattern_ids.append(2)
            else:
                continue  # Skip invalid characters
                
        # Add an end token (2 = space) and pad pattern
        pattern_ids.append(2)
        pattern_ids = pad_sequence(pattern_ids, self.max_pattern_length, 3)  # 3 is padding token
            
        # Create attention masks (1 for real tokens including EOS, 0 for padding)
        context_mask = [1 if id != self.tokenizer.pad_token_id else 0 for id in context_ids]
        pattern_mask = [1 if id != 3 else 0 for id in pattern_ids]
        
        # For decoder input, shift target right and add BOS token
        decoder_input_ids = [self.tokenizer.bos_token_id] + target_ids[:-1]
            
        # Create tensor dictionary
        return {
            'input_ids': torch.tensor(context_ids, dtype=torch.long),
            'attention_mask': torch.tensor(context_mask, dtype=torch.long),
            'pattern_ids': torch.tensor(pattern_ids, dtype=torch.long),
            'pattern_attention_mask': torch.tensor(pattern_mask, dtype=torch.long),
            'decoder_input_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(target_ids, dtype=torch.long)
        }
        
def create_datasets(
    tokenizer,
    data_files: List[str],
    val_split: float = 0.1,
    word_freq_file: Optional[str] = None,
    **kwargs
) -> Tuple[MessageDataset, MessageDataset]:
    """Create training and validation datasets"""
    # Create full dataset
    dataset = MessageDataset(
        tokenizer=tokenizer,
        word_freq_file=word_freq_file,
        **kwargs
    )
    
    # Load all data files
    for file in data_files:
        if file.endswith('.csv'):
            dataset.load_csv_data(file)
        else:
            logger.warning(f"Unsupported file format: {file}")
            
    # Split into train/val
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset = MessageDataset(
        tokenizer=tokenizer,
        word_freq_file=word_freq_file,
        **kwargs
    )
    val_dataset = MessageDataset(
        tokenizer=tokenizer,
        word_freq_file=word_freq_file,
        **kwargs
    )
    
    # Copy data
    train_dataset.context_messages = dataset.context_messages[:train_size]
    train_dataset.target_messages = dataset.target_messages[:train_size]
    train_dataset.patterns = dataset.patterns[:train_size]
    
    val_dataset.context_messages = dataset.context_messages[train_size:]
    val_dataset.target_messages = dataset.target_messages[train_size:]
    val_dataset.patterns = dataset.patterns[train_size:]
    
    return train_dataset, val_dataset