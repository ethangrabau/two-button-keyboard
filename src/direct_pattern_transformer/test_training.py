import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from data_loader import create_datasets
from model import PatternMatchConfig, PatternMatchModel
from trainer import PatternMatchTrainer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_test_training():
    # Get data path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "Matthew_Halle_Training_with_RL_and_Spaces.csv"

    logger.info(f"Data path: {data_path}")

    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Configure model with more conservative settings
    logger.info("Creating model configuration...")
    config = PatternMatchConfig(
        max_pattern_length=32,
        max_context_length=64,  # Reduced from 128
        hidden_size=256,
        vocab_size=len(tokenizer),
        num_layers=1,  # Reduced from 2
        num_attention_heads=8,
        dropout=0.3,  # Increased from 0.1
        embedding_scale=0.1,
        weight_decay=0.1,
        pattern_loss_weight=0.5
    )

    # Create datasets with shorter sequences
    logger.info("Loading datasets...")
    train_dataset, val_dataset = create_datasets(
        tokenizer=tokenizer,
        data_files=[str(data_path)],
        val_split=0.1,
        max_context_length=config.max_context_length,
        max_pattern_length=config.max_pattern_length
    )

    # Create dataloaders with larger batch size
    logger.info("Creating dataloaders...")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,  # Increased from 8
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # Initialize model
    logger.info("Initializing model...")
    model = PatternMatchModel(config)
    model.set_tokenizer(tokenizer)  # Important: Set tokenizer for pattern masking

    # Calculate training steps for warmup
    num_epochs = 10
    total_steps = len(train_dataloader) * num_epochs

    # Set up trainer with more conservative learning
    logger.info("Setting up trainer...")
    trainer = PatternMatchTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        tokenizer=tokenizer,
        save_dir=str(project_root / "checkpoints" / "test_run"),
        learning_rate=1e-5,  # Reduced from 5e-5
        weight_decay=config.weight_decay,
        warmup_steps=total_steps // 3,  # Increased warmup
        max_grad_norm=0.5,
        gradient_accumulation_steps=4,
        num_epochs=num_epochs,
        device="mps" if torch.backends.mps.is_available() else "cpu",
        log_predictions_every=25
    )

    # Run training
    logger.info("Starting training loop...")
    try:
        trainer.train()
        logger.info("Training completed successfully!")
        
        # Print final metrics
        logger.info("\nFinal Metrics:")
        logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")
        logger.info(f"Final training loss: {trainer.train_losses[-1]:.4f}")
        logger.info(f"Final validation loss: {trainer.val_losses[-1]:.4f}")

        # Print some sample predictions from history
        logger.info("\nSample Predictions from Training:")
        for pred in trainer.sample_predictions[-3:]:
            logger.info(f"\nPattern: {pred['pattern']}")
            logger.info(f"Predicted: {pred['predicted']}")
            logger.info(f"Target: {pred['target']}")
            logger.info(f"Confidence: {pred['confidence']:.4f}")

    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    run_test_training()