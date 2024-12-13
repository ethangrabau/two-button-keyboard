"""
Simplified training script with improved stability and MPS optimization
"""
import logging
from pathlib import Path
import torch
import torch.backends.mps
from torch.amp import autocast
from torch.utils.data import DataLoader
from transformers import GPT2TokenizerFast
from tqdm import tqdm

from data_loader import create_datasets
from model import PatternMatchConfig, PatternMatchModel
from training_config import TrainingConfig  # Added this import

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

def train_epoch(
    model, 
    train_loader, 
    optimizer, 
    scheduler, 
    config: TrainingConfig,
    epoch: int
):
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    optimizer.zero_grad()
    
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
    
    for step, batch in progress_bar:
        # Move batch to device
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        # Forward pass
        with autocast(device_type=config.device):
            loss, logits = model(**batch)
            loss = loss / config.gradient_accumulation_steps
        
        # Backward pass
        loss.backward()
        
        # Update weights if needed
        if (step + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Calculate accuracy
        predictions = logits.argmax(dim=-1)
        labels = batch['labels']
        mask = labels != -100
        accuracy = (predictions[mask] == labels[mask]).float().mean()
        
        # Update metrics
        total_loss += loss.item() * config.gradient_accumulation_steps
        total_accuracy += accuracy.item()
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'epoch': epoch,
            'loss': f'{total_loss / num_batches:.3f}',
            'acc': f'{total_accuracy / num_batches:.3f}',
            'lr': f'{scheduler.get_last_lr()[0]:.2e}'
        })
        
    return total_loss / num_batches, total_accuracy / num_batches

@torch.no_grad()
def evaluate(model, val_loader, config: TrainingConfig):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    for batch in val_loader:
        batch = {k: v.to(config.device) for k, v in batch.items()}
        
        with autocast(device_type=config.device):
            loss, logits = model(**batch)
        
        predictions = logits.argmax(dim=-1)
        labels = batch['labels']
        mask = labels != -100
        accuracy = (predictions[mask] == labels[mask]).float().mean()
        
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
    return total_loss / num_batches, total_accuracy / num_batches

def main():
    # Clear MPS cache if available
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Load config
    config = TrainingConfig()
    
    # Set up paths
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent
    data_path = project_root / "data" / "Matthew_Halle_Training_with_RL_and_Spaces.csv"
    save_dir = project_root / "checkpoints" / "test_run"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"MPS available: {torch.backends.mps.is_available()}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Training configuration:\n{config}")
    logger.info(f"Data path: {data_path}")
    logger.info(f"Save directory: {save_dir}")
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets
    logger.info("Loading datasets...")
    train_dataset, val_dataset = create_datasets(
        tokenizer=tokenizer,
        data_files=[str(data_path)],
        val_split=config.val_split,
        max_context_length=config.max_context_length,
        max_pattern_length=config.max_pattern_length
    )
    
    # Create dataloaders with adjusted settings for MPS
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,  # Reduced for MPS
        pin_memory=False  # Disabled for MPS
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,  # Reduced for MPS
        pin_memory=False  # Disabled for MPS
    )
    
    # Initialize model
    logger.info("Initializing model...")
    model_config = PatternMatchConfig(
        max_pattern_length=config.max_pattern_length,
        max_context_length=config.max_context_length,
        hidden_size=config.hidden_size,
        vocab_size=len(tokenizer),
        num_layers=config.num_layers,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        embedding_scale=config.embedding_scale
    )
    
    model = PatternMatchModel(model_config)
    model.set_tokenizer(tokenizer)
    model = model.to(config.device)
    
    # Initialize optimizer with larger eps for stability
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        eps=1e-7
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    num_warmup_steps = int(num_training_steps * config.warmup_ratio)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config.learning_rate,
        total_steps=num_training_steps,
        pct_start=config.warmup_ratio,
        anneal_strategy='linear',
        final_div_factor=config.learning_rate / config.min_learning_rate
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(config.num_epochs):
        # Training phase
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            epoch=epoch
        )
        
        # Validation phase
        val_loss, val_acc = evaluate(model, val_loader, config)
        
        # Log metrics
        logger.info(
            f"\nEpoch {epoch + 1}/{config.num_epochs}:\n"
            f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}\n"
            f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}"
        )
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'config': config
            }
            torch.save(checkpoint, save_dir / 'best_model.pt')
            logger.info(f"Saved new best model with validation loss: {val_loss:.4f}")
        else:
            epochs_without_improvement += 1
            
        # Early stopping
        if epochs_without_improvement >= config.early_stopping_patience:
            logger.info(f"\nStopping early after {epoch + 1} epochs")
            break
            
    logger.info("Training completed!")
    
if __name__ == "__main__":
    main()