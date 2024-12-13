"""
Training infrastructure for pattern-constrained model
"""
import json
from pathlib import Path
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from typing import Dict, Optional, List, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class PatternMatchTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        tokenizer,
        save_dir: str,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_grad_norm: float = 1.0,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 10,
        device: Optional[str] = None,
        log_predictions_every: int = 25  # Log predictions every N steps
    ):
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.log_predictions_every = log_predictions_every
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Calculate total steps
        self.total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        
        # Learning rate scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_steps
        )
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_val_loss = float('inf')
        self.sample_predictions = []  # Store periodic predictions
        
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'sample_predictions': self.sample_predictions
        }
        
        if is_best:
            path = self.save_dir / 'best_model.pt'
        else:
            path = self.save_dir / f'checkpoint_epoch_{epoch}.pt'
            
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
        
    def _log_predictions(self, batch, outputs):
        # Get predictions
        predictions = outputs.logits.argmax(dim=-1)
        
        # Convert L/R pattern IDs back to string
        pattern_map = {0: 'L', 1: 'R', 2: ' ', 3: ''}
        pattern = ''.join(pattern_map[id.item()] for id in batch['pattern_ids'][0])  # First batch item
        
        # Decode predictions and targets
        predicted = self.tokenizer.decode(predictions[0])
        target = self.tokenizer.decode(batch['labels'][0])
        
        # Calculate confidence
        probs = torch.softmax(outputs.logits, dim=-1)
        confidence = probs.max(dim=-1).values[0].mean().item()
        
        # Store prediction
        prediction = {
            'pattern': pattern.strip(),
            'predicted': predicted.strip(),
            'target': target.strip(),
            'confidence': confidence
        }
        self.sample_predictions.append(prediction)
        
        # Log
        logger.info("\nSample Prediction:")
        logger.info(f"Pattern: {pattern.strip()}")
        logger.info(f"Predicted: {predicted.strip()}")
        logger.info(f"Target: {target.strip()}")
        logger.info(f"Confidence: {confidence:.4f}")
        
    def move_batch_to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}
        
    def train_epoch(self, epoch: int):
        self.model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}/{self.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            loss.backward()
            
            # Update metrics
            total_loss += loss.item() * self.gradient_accumulation_steps
            
            # Calculate accuracy
            if 'labels' in batch:
                predictions = outputs.logits.argmax(-1)
                labels = batch['labels']
                mask = labels != -100
                correct_predictions += (predictions[mask] == labels[mask]).sum().item()
                total_predictions += mask.sum().item()
            
            # Update weights if needed
            if (step + 1) % self.gradient_accumulation_steps == 0:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Log predictions periodically
            if step % self.log_predictions_every == 0:
                self._log_predictions(batch, outputs)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / (step + 1),
                'acc': correct_predictions / max(1, total_predictions)
            })
        
        return {
            'loss': total_loss / len(self.train_dataloader),
            'accuracy': correct_predictions / max(1, total_predictions)
        }

    @torch.no_grad()
    def validate_checkpoint(self):
        """Test if saved model can be loaded and run"""
        state_dict = self.model.state_dict()
        
        # Create new model instance
        test_model = type(self.model)(self.model.config)
        test_model.load_state_dict(state_dict)
        test_model.to(self.device)
        
        # Try forward pass
        batch = next(iter(self.val_dataloader))
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        try:
            test_model(**batch)
            return True
        except Exception as e:
            logger.error(f"Checkpoint validation failed: {e}")
            return False

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        
        progress_bar = tqdm(self.val_dataloader, desc="Validating")
        
        for batch in progress_bar:
            # Move batch to device
            batch = self.move_batch_to_device(batch)
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Calculate accuracy (excluding padding tokens)
            mask = batch['labels'] != -100
            predictions = outputs.logits.argmax(-1)
            correct = ((predictions[mask] == batch['labels'][mask])).sum().item()
            total_correct += correct
            total_tokens += mask.sum().item()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': total_loss / len(self.val_dataloader),
                'acc': total_correct / max(1, total_tokens)
            })
            
        return total_loss / len(self.val_dataloader), total_correct / max(1, total_tokens)

    def train(self):
        logger.info(f"Starting training on {self.device}")
        logger.info(f"Model has {sum(p.numel() for p in self.model.parameters())} parameters")
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_result = self.train_epoch(epoch)
            self.train_losses.append(train_result['loss'])
            self.train_accuracies.append(train_result['accuracy'])
            
            # Validation phase
            val_loss, val_accuracy = self.validate()
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_accuracy)
            
            # Log metrics
            logger.info(
                f"\nEpoch {epoch+1}/{self.num_epochs}:\n"
                f"Train Loss: {train_result['loss']:.4f}, Train Accuracy: {train_result['accuracy']:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )
            
            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                logger.info("New best validation loss!")
                
            self.save_checkpoint(epoch, is_best)
            self.validate_checkpoint()