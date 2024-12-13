"""
Pattern Constraint Model Prototype
- Enforces L/R pattern matching during generation
- Added numerical stability improvements
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from model_common import PatternMatchConfig, TokenEmbedding, PatternMask

class SimpleEncoder(nn.Module):
    """Simplified encoder with LSTM and improved stability"""
    def __init__(self, config: PatternMatchConfig, is_pattern_encoder: bool = False):
        super().__init__()
        
        # Use different vocab sizes for pattern and text encoders
        vocab_size = 4 if is_pattern_encoder else config.vocab_size
        max_length = config.max_pattern_length if is_pattern_encoder else config.max_context_length
        
        # Embedding with improved initialization
        self.embedding = TokenEmbedding(
            vocab_size=vocab_size,
            hidden_size=config.hidden_size,
            max_length=max_length,
            dropout=config.dropout
        )
        
        # LSTM with gradient clipping
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            dropout=0,
            batch_first=True
        )
        
        # Output projection with scaled initialization
        self.output = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.hidden_size)
        )
        
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # Initialize parameters with scaled normal distribution
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Get embeddings
        x = self.embedding(input_ids)
        
        # Apply mask if provided
        if attention_mask is not None:
            mask = attention_mask.bool()
            x = x * mask.unsqueeze(-1)
        
        # LSTM with gradient clipping
        x, _ = self.lstm(x)
        torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)
        
        # Project output
        x = self.output(x)
        
        return self.norm(x)

class SimpleDecoder(nn.Module):
    """Simplified decoder with improved numerical stability"""
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        
        # Embedding
        self.embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            max_length=config.max_pattern_length,
            dropout=config.dropout
        )
        
        # LSTM for decoding
        self.lstm = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=1,
            dropout=0,
            batch_first=True
        )
        
        # Attention components with improved scaling
        self.query_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.key_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.attention_scale = (config.hidden_size ** -0.5) * 0.5  # Reduced scale factor
        
        # Output with layer norm
        self.combine_layer = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.output = nn.Linear(config.hidden_size, config.vocab_size)
        self.dropout = nn.Dropout(config.dropout)
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # Initialize parameters
        self._init_weights()
        
    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.normal_(param, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(param)
                
    def attention(self, query: torch.Tensor, keys: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Project and normalize query and keys
        query = self.norm1(self.query_proj(query))  # [B, H]
        keys = self.norm1(self.key_proj(keys))      # [B, T, H]
        
        # Calculate attention scores with improved numerical stability
        scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2))  # [B, 1, T]
        scores = scores * self.attention_scale
        
        # Apply mask and attention
        if mask is not None:
            mask = mask.unsqueeze(1).bool()
            scores = scores.masked_fill(~mask, -1e4)  # Use smaller negative value
        
        # Compute attention with improved numerical stability
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - scores_max)
        
        if mask is not None:
            exp_scores = exp_scores.masked_fill(~mask, 0)
            
        weights = exp_scores / (exp_scores.sum(dim=-1, keepdim=True) + 1e-6)
        context = torch.bmm(weights, keys)
        
        return context.squeeze(1), weights.squeeze(1)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pattern_masks: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Get embeddings
        x = self.embedding(input_ids)
        
        # LSTM decoding with gradient clipping
        lstm_out, _ = self.lstm(x)
        torch.nn.utils.clip_grad_norm_(self.lstm.parameters(), max_norm=1.0)
        
        # Process each position
        outputs = []
        for pos in range(lstm_out.size(1)):
            query = lstm_out[:, pos]  # [B, H]
            
            # Get context through attention
            context, _ = self.attention(
                query,
                encoder_hidden_states,
                mask=attention_mask
            )
            
            # Combine LSTM state and context with normalization
            hidden = self.norm2(self.combine_layer(
                torch.cat([query, context], dim=-1)
            ))
            
            # Get logits with dropout
            logits = self.output(self.dropout(hidden))
            
            # Apply pattern mask if provided
            if pattern_masks is not None:
                mask = pattern_masks[:, pos].bool()
                logits = logits.masked_fill(~mask, -1e4)  # Use smaller negative value
                
            outputs.append(logits)
            
        # Stack all outputs
        return torch.stack(outputs, dim=1)

class PatternMatchModel(nn.Module):
    def __init__(self, config: PatternMatchConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.encoder = SimpleEncoder(config, is_pattern_encoder=False)
        self.pattern_encoder = SimpleEncoder(config, is_pattern_encoder=True)
        self.decoder = SimpleDecoder(config)
        
        # Pattern masking
        self.pattern_mask = None
        
    def set_tokenizer(self, tokenizer):
        """Initialize pattern masking with tokenizer"""
        self.pattern_mask = PatternMask(tokenizer)
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pattern_ids: torch.Tensor,
        pattern_attention_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # Encode input and pattern with gradient clipping
        encoder_states = self.encoder(input_ids, attention_mask)
        pattern_states = self.pattern_encoder(pattern_ids, pattern_attention_mask)
        
        # Combine states
        combined_states = torch.cat([encoder_states, pattern_states], dim=1)
        combined_mask = torch.cat([attention_mask, pattern_attention_mask], dim=1)
        
        # Create pattern masks with validation
        pattern_masks = None
        if self.pattern_mask is not None:
            pattern_masks = self.pattern_mask.create_pattern_masks(pattern_ids)
            # Validate pattern masks
            if torch.any(torch.isnan(pattern_masks)):
                pattern_masks = None
        
        # Decode
        logits = self.decoder(
            decoder_input_ids,
            combined_states,
            pattern_masks=pattern_masks,
            attention_mask=combined_mask
        )
        
        if labels is not None:
            # Main prediction loss with label smoothing
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=-100,
                label_smoothing=0.1  # Add label smoothing
            )
            prediction_loss = loss_fct(
                logits.view(-1, self.config.vocab_size),
                labels.view(-1)
            )
            
            # Pattern matching loss with improved stability
            if pattern_masks is not None and not torch.isnan(prediction_loss):
                pattern_pred = logits.argmax(dim=-1)
                pattern_match = torch.zeros_like(pattern_pred, dtype=torch.float)
                
                # Check predictions against pattern
                valid_mask = (labels != -100)
                for b in range(pattern_pred.size(0)):
                    for pos in range(pattern_pred.size(1)):
                        if valid_mask[b, pos]:
                            pred_id = pattern_pred[b, pos].item()
                            if pattern_masks[b, pos, pred_id]:
                                pattern_match[b, pos] = 1.0
                
                # Calculate pattern loss with stability checks
                valid_sum = valid_mask.float().sum()
                if valid_sum > 0:
                    pattern_loss = 1.0 - (pattern_match * valid_mask.float()).sum() / valid_sum
                    pattern_loss = torch.clamp(pattern_loss, 0.0, 1.0)  # Ensure loss is bounded
                else:
                    pattern_loss = torch.tensor(0.0, device=prediction_loss.device)
                
                # Combine losses with stability check
                loss = prediction_loss + self.config.pattern_loss_weight * pattern_loss
            else:
                loss = prediction_loss
                
            return loss, logits
            
        return logits