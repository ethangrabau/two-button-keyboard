"""
Pattern Constraint Model Prototype
- Enforces L/R pattern matching during generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from dataclasses import dataclass

@dataclass
class PatternMatchConfig:
    max_pattern_length: int = 32
    max_context_length: int = 128
    hidden_size: int = 256
    vocab_size: int = 32000
    num_layers: int = 3
    num_attention_heads: int = 8
    
class TokenEmbedding(nn.Module):
    """Handles token embeddings and position encoding"""
    def __init__(self, vocab_size: int, hidden_size: int, max_length: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_length = x.size(1)
        positions = torch.arange(seq_length, device=x.device)
        positions = positions.unsqueeze(0).expand(x.size(0), -1)
        
        embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding(positions)
        
        return embeddings + position_embeddings

class PatternEncoder(nn.Module):
    """Encodes L/R patterns into hidden states"""
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        self.config = config
        
        # L=0, R=1, Space=2, Padding=3
        self.embedding = TokenEmbedding(
            vocab_size=4,
            hidden_size=config.hidden_size,
            max_length=config.max_pattern_length
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(config.hidden_size)
        )
        
    def forward(self, pattern_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Convert input to long for embedding
        pattern_ids = pattern_ids.long()
        
        # Get embeddings
        hidden_states = self.embedding(pattern_ids)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            
        encoded = self.encoder(hidden_states, src_key_padding_mask=attention_mask)
        return encoded

class ContextEncoder(nn.Module):
    """Encodes previous message context"""
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        self.config = config
        
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_length=config.max_context_length
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=2,
            norm=nn.LayerNorm(config.hidden_size)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # Convert input to long for embedding
        input_ids = input_ids.long()
        
        # Get embeddings
        hidden_states = self.embedding(input_ids)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            
        encoded = self.encoder(hidden_states, src_key_padding_mask=attention_mask)
        return encoded

class PatternConstrainedDecoder(nn.Module):
    """Generates tokens while enforcing pattern constraints"""
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        self.config = config
        
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_length=config.max_context_length
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.hidden_size * 4,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_size)
        )
        
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        
    def forward(
        self,
        decoder_input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pattern_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None
    ):
        # Convert input to long for embedding
        decoder_input_ids = decoder_input_ids.long()
        
        # Get embeddings
        hidden_states = self.embedding(decoder_input_ids)
        
        # Create causal mask for decoder
        seq_len = decoder_input_ids.size(1)
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float('-inf'), device=decoder_input_ids.device),
            diagonal=1
        )
        
        # Process attention masks
        if attention_mask is not None:
            attention_mask = attention_mask.float()
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
            attention_mask = attention_mask.masked_fill(attention_mask == 1, float(0.0))
            
        if encoder_attention_mask is not None:
            encoder_attention_mask = encoder_attention_mask.float()
            encoder_attention_mask = encoder_attention_mask.masked_fill(
                encoder_attention_mask == 0, float('-inf')
            )
            encoder_attention_mask = encoder_attention_mask.masked_fill(
                encoder_attention_mask == 1, float(0.0)
            )
        
        # Decode
        hidden_states = self.decoder(
            tgt=hidden_states,
            memory=encoder_hidden_states,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=attention_mask,
            memory_key_padding_mask=encoder_attention_mask
        )
        
        logits = self.output(hidden_states)
        
        if pattern_mask is not None:
            # Apply pattern constraints
            pattern_mask = pattern_mask.unsqueeze(1).expand(-1, seq_len, -1)
            logits = logits.masked_fill(~pattern_mask, float('-inf'))
            
        return logits

class PatternConstrainedModel(nn.Module):
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        self.config = config
        
        self.pattern_encoder = PatternEncoder(config)
        self.context_encoder = ContextEncoder(config)
        self.decoder = PatternConstrainedDecoder(config)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        pattern_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        pattern_mask: Optional[torch.Tensor] = None,
        pattern_attention_mask: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None
    ):
        # Encode context
        context_encoded = self.context_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Encode pattern
        pattern_encoded = self.pattern_encoder(
            pattern_ids=pattern_ids,
            attention_mask=pattern_attention_mask
        )
        
        # Combine encodings
        encoder_hidden_states = torch.cat([context_encoded, pattern_encoded], dim=1)
        encoder_attention_mask = None
        if attention_mask is not None and pattern_attention_mask is not None:
            encoder_attention_mask = torch.cat([attention_mask, pattern_attention_mask], dim=1)
            
        # Generate with pattern constraints
        logits = self.decoder(
            decoder_input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_hidden_states,
            pattern_mask=pattern_mask,
            attention_mask=decoder_attention_mask,
            encoder_attention_mask=encoder_attention_mask
        )
        
        return logits