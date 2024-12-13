"""
Test run script for model stability verification
"""
import torch
from train_simple import main
from training_config import TrainingConfig

if __name__ == "__main__":
    # Clear MPS cache if available
    if torch.backends.mps.is_available():
        print("Clearing MPS cache...")
        torch.mps.empty_cache()
    
    # Create test configuration
    config = TrainingConfig(test_mode=True)
    
    # Print PyTorch version and device info
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"Using device: {config.device}")
    
    # Run test training
    main(config)