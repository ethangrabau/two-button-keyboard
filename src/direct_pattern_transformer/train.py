"""
Training script for the pattern-constrained transformer model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Tokenizer
from tqdm import tqdm
import wandb
import numpy as np
from typing import List, Tuple, Optional
import os

from model import SimplePatternTransformer
from model_common import PatternMatchConfig
from keyboard_mapping import KeyboardMapper

# Set environment variable for MPS fallback
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class TextPatternDataset(Dataset):
    def __init__(self, texts: List[str], max_length: int = 32):
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.keyboard = KeyboardMapper()
        self.max_length = max_length
        
        # Process texts and patterns
        self.examples = []
        for text in texts:
            # Get pattern for text
            pattern = self.keyboard.token_to_pattern(text)
            if not pattern:
                continue
                
            # Tokenize text
            tokens = self.tokenizer.encode(text)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                pattern = pattern[:max_length]
                
            # Convert pattern to ids (L=0, R=1, Space=2, Pad=3)
            pattern_ids = []
            for p in pattern:
                if p == 'L':
                    pattern_ids.append(0)
                elif p == 'R':
                    pattern_ids.append(1)
                elif p == ' ':
                    pattern_ids.append(2)
                    
            # Pad sequences
            tokens = tokens + [self.tokenizer.pad_token_id] * (max_length - len(tokens))
            pattern_ids = pattern_ids + [3] * (max_length - len(pattern_ids))
            
            self.examples.append((tokens, pattern_ids))
            
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        tokens, pattern = self.examples[idx]
        return {
            'input_ids': torch.tensor(tokens),
            'pattern_ids': torch.tensor(pattern)
        }

def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None,
    epochs: int = 10,
    lr: float = 1e-4,
    device: str = 'cuda',
    log_wandb: bool = False
):
    """Train the model"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    if log_wandb:
        wandb.init(project='pattern-transformer')
        
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        progress = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch in progress:
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            pattern_ids = batch['pattern_ids'].to(device)
            
            # Forward pass
            loss = model(input_ids, pattern_ids=pattern_ids)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update progress
            total_loss += loss.item()
            num_batches += 1
            progress.set_postfix({'loss': total_loss / num_batches})
            
            if log_wandb:
                wandb.log({
                    'train_loss': loss.item(),
                    'epoch': epoch,
                    'learning_rate': scheduler.get_last_lr()[0]
                })
                
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    pattern_ids = batch['pattern_ids'].to(device)
                    loss = model(input_ids, pattern_ids=pattern_ids)
                    val_loss += loss.item()
                    num_val_batches += 1
                    
            val_loss /= num_val_batches
            print(f'Validation loss: {val_loss:.4f}')
            
            if log_wandb:
                wandb.log({
                    'val_loss': val_loss,
                    'epoch': epoch
                })
                
            model.train()
            
        scheduler.step()
        
    if log_wandb:
        wandb.finish()
        
def main():
    # Load sample texts (replace with your dataset)
    sample_texts = [
        "Hello world!",
        "This is a test.",
        "Pattern matching works!",
        "Left and right keys.",
        "Typing with two buttons.",
        "The quick brown fox.",
        "Jumps over lazy dog!",
        "How are you today?",
        "I am doing great!",
        "Let's try this out."
    ]
    
    # Create dataset
    dataset = TextPatternDataset(sample_texts)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize model
    config = PatternMatchConfig()
    model = SimplePatternTransformer(config)
    
    # Determine device
    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print(f'Using device: {device}')
    
    model = model.to(device)
    
    # Train model
    train(
        model=model,
        train_loader=train_loader,
        epochs=10,
        lr=1e-4,
        device=device,
        log_wandb=False  # Set to True to log metrics
    )
    
    # Save model
    torch.save(model.state_dict(), 'pattern_transformer.pt')
    print('Model saved to pattern_transformer.pt')
    
if __name__ == '__main__':
    main()
