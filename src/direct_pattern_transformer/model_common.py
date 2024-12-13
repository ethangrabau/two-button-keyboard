"""
Common components for pattern-constrained model
- TokenEmbedding: Combined token and position embeddings
- PatternMask: Enforces L/R pattern constraints
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
from dataclasses import dataclass
from keyboard_mapping import KeyboardMapper

@dataclass
class PatternMatchConfig:
    max_pattern_length: int = 32
    max_context_length: int = 128
    hidden_size: int = 256  # Reduced from 512
    vocab_size: int = 50257  # GPT-2 vocab size
    num_layers: int = 2
    num_attention_heads: int = 8
    dropout: float = 0.3  # Increased dropout
    embedding_scale: float = 0.1
    weight_decay: float = 0.1  # Added L2 regularization
    pattern_loss_weight: float = 0.5  # Weight for pattern matching loss

class PatternMask:
    """Creates masks to enforce pattern constraints"""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.keyboard_mapper = KeyboardMapper()  # Use actual keyboard mapping
        self.token_patterns = {}
        self._build_token_patterns()

    def _build_token_patterns(self):
        """Build mapping of tokens to their L/R patterns"""
        for token_id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.decode([token_id])
            if token.strip():  # Skip empty tokens
                pattern = self.keyboard_mapper.token_to_pattern(token)
                if pattern:  # Only store valid patterns
                    self.token_patterns[token_id] = pattern
    
    def _get_token_pattern(self, token: str) -> str:
        """Use keyboard mapper for pattern generation"""
        return self.keyboard_mapper.token_to_pattern(token)

    def create_pattern_masks(self, pattern_ids: torch.Tensor) -> torch.Tensor:
        """Create vocabulary masks efficiently"""
        batch_size, seq_length = pattern_ids.size()
        vocab_size = self.tokenizer.vocab_size
        device = pattern_ids.device
        
        # Initialize masks with special tokens allowed
        masks = torch.zeros(batch_size, seq_length, vocab_size, device=device)
        special_tokens = {
            self.tokenizer.bos_token_id,
            self.tokenizer.eos_token_id,
            self.tokenizer.pad_token_id
        }
        masks[:, :, list(special_tokens)] = 1
        
        # Convert pattern IDs to strings
        pattern_map = {0: 'L', 1: 'R', 2: ' ', 3: ''}
        patterns = []
        
        for b in range(batch_size):
            pattern = []
            for pos in range(seq_length):
                pid = pattern_ids[b, pos].item()
                if pid in pattern_map:
                    char = pattern_map[pid]
                    if char:  # Skip empty/padding
                        pattern.append(char)
            patterns.append(''.join(pattern))
        
        # Match tokens efficiently
        for token_id, token_pattern in self.token_patterns.items():
            if token_pattern:  # Skip invalid patterns
                for b, pattern in enumerate(patterns):
                    if token_pattern in pattern:
                        # Allow this token at positions where it could start
                        for pos in range(min(len(pattern), seq_length)):
                            if pattern[pos:].startswith(token_pattern):
                                masks[b, pos, token_id] = 1
                    
        return masks

class TokenEmbedding(nn.Module):
    """Improved token embedding with better initialization and scaling"""
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        max_length: int,
        dropout: float = 0.1,
        scale: Optional[float] = None
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = scale or hidden_size ** -0.5
        
        # Initialize embeddings with small values
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get sequence length
        seq_length = x.size(1)
        
        # Create position ids
        positions = torch.arange(seq_length, dtype=torch.long, device=x.device)
        positions = positions.unsqueeze(0).expand_as(x)
        
        # Get embeddings
        token_embeddings = self.token_embedding(x) * self.scale
        position_embeddings = self.position_embedding(positions) * self.scale
        
        # Combine and normalize
        embeddings = token_embeddings + position_embeddings
        embeddings = self.norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings