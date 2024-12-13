"""
Training configuration for pattern transformer model
Optimized for MPS performance
"""
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Model architecture
    hidden_size: int = 256
    num_layers: int = 1
    num_attention_heads: int = 4
    dropout: float = 0.2
    embedding_scale: float = 0.1
    
    # Sequence lengths
    max_context_length: int = 48
    max_pattern_length: int = 24
    
    # Training parameters
    batch_size: int = 8
    learning_rate: float = 5e-5
    min_learning_rate: float = 1e-6
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 8
    
    # Training loop
    num_epochs: int = 10
    early_stopping_patience: int = 3
    val_check_interval: int = 50
    log_predictions_every: int = 10
    
    # Mixed precision training
    use_mixed_precision: bool = True
    
    # Data processing
    num_workers: int = 1
    val_split: float = 0.1
    shuffle_buffer_size: int = 5000
    
    # Device
    device: str = "mps"
    
    # Test mode
    test_mode: bool = False
    
    def __post_init__(self):
        import torch
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            self.device = "cpu"
            
        if self.device == "cpu":
            self.batch_size = 4
            self.gradient_accumulation_steps = 16
            
        # If test mode is enabled, use minimal configuration
        if self.test_mode:
            print("Running in test mode with minimal configuration")
            self.num_epochs = 1
            self.batch_size = 4
            self.max_context_length = 32
            self.max_pattern_length = 16
            self.val_check_interval = 10
            self.log_predictions_every = 5