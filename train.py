"""
Training Script for Healthcare AI - Chest X-ray Pneumonia Detection
Handles model training with proper logging, checkpointing, and monitoring.
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import wandb
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Import our modules
from data_loader import DataLoaderManager
from model import ModelManager, MedicalImageClassifier
from evaluate import MetricsCalculator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, restore_best_weights: bool = True):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                logger.info("Restored best weights")
            return True
        
        return False


class Trainer:
    """Main trainer class for the healthcare AI model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration."""
        self.config = self._load_config(config_path)
        
        # Extract configuration sections
        self.training_config = self.config.get('training', {})
        self.data_config = self.config.get('data', {})
        self.model_config = self.config.get('model', {})
        self.hardware_config = self.config.get('hardware', {})
        self.logging_config = self.config.get('logging', {})
        
        # Set up device
        self.device = self._setup_device()
        
        # Initialize components
        self.data_manager = DataLoaderManager(config_path)
        self.model_manager = ModelManager(config_path)
        self.metrics_calculator = MetricsCalculator(
            self.data_config.get('class_names', ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia'])
        )
        
        # Training state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.dataloaders = None
        self.writer = None
        self.early_stopping = None
        
        # Create output directories
        self._create_directories()
        
        logger.info(f"Trainer initialized with device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.error(f"Config file {config_path} not found")
            sys.exit(1)
    
    def _setup_device(self) -> torch.device:
        """Set up compute device."""
        device_config = self.hardware_config.get('device', 'auto')
        
        if device_config == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device_config == 'cuda':
            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, falling back to CPU")
                device = torch.device('cpu')
            else:
                device = torch.device('cuda')
        else:
            device = torch.device(device_config)
        
        if device.type == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        return device
    
    def _create_directories(self):
        """Create necessary directories."""
        directories = [
            'models',
            'logs',
            self.logging_config.get('log_dir', 'logs'),
            'checkpoints'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """Set up model, optimizer, criterion, and scheduler."""
        # Create model
        self.model = self.model_manager.create_model()
        self.model.to(self.device)
        
        # Count parameters
        self.model_manager.count_parameters(self.model)
        
        # Set up optimizer
        optimizer_name = self.training_config.get('optimizer', 'adam').lower()
        learning_rate = self.training_config.get('learning_rate', 0.001)
        
        if optimizer_name == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        elif optimizer_name == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=0.9,
                weight_decay=1e-4
            )
        elif optimizer_name == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
        
        # Set up loss function
        loss_function = self.training_config.get('loss_function', 'categorical_crossentropy')
        if loss_function == 'categorical_crossentropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_function == 'focal_loss':
            # You might want to implement focal loss for imbalanced datasets
            self.criterion = nn.CrossEntropyLoss()  # Fallback for now
        else:
            self.criterion = nn.CrossEntropyLoss()
        
        # Set up learning rate scheduler
        lr_config = self.training_config.get('lr_scheduler', {})
        if lr_config.get('enabled', True):
            scheduler_type = lr_config.get('type', 'reduce_on_plateau')
            
            if scheduler_type == 'reduce_on_plateau':
                self.scheduler = ReduceLROnPlateau(
                    self.optimizer,
                    mode='min',
                    factor=lr_config.get('factor', 0.5),
                    patience=lr_config.get('patience', 5),
                    min_lr=lr_config.get('min_lr', 1e-6)
                )
            elif scheduler_type == 'step':
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=lr_config.get('step_size', 10),
                    gamma=lr_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.training_config.get('epochs', 50)
                )
        
        # Set up early stopping
        early_stopping_config = self.training_config.get('early_stopping', {})
        if early_stopping_config.get('enabled', True):
            self.early_stopping = EarlyStopping(
                patience=early_stopping_config.get('patience', 10),
                min_delta=0.001,
                restore_best_weights=True
            )
        
        # Set up mixed precision if enabled
        self.use_mixed_precision = self.hardware_config.get('mixed_precision', True) and self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            logger.info("Mixed precision training enabled")
        
        logger.info(f"Model setup completed:")
        logger.info(f"  Optimizer: {optimizer_name}")
        logger.info(f"  Learning rate: {learning_rate}")
        logger.info(f"  Loss function: {loss_function}")
        logger.info(f"  Scheduler: {lr_config.get('type', 'reduce_on_plateau') if lr_config.get('enabled') else 'None'}")
    
    def setup_data(self, data_path: Optional[str] = None):
        """Set up data loaders."""
        if data_path and data_path.endswith('.csv'):
            self.dataloaders = self.data_manager.load_dataset_from_csv(data_path)
        else:
            self.dataloaders = self.data_manager.load_dataset_from_folders()
        
        if not self.dataloaders:
            logger.error("No data loaded. Please check your data directory structure.")
            sys.exit(1)
        
        # Validate dataset
        if not self.data_manager.validate_dataset(self.dataloaders):
            logger.error("Dataset validation failed")
            sys.exit(1)
        
        # Handle class imbalance
        if 'train' in self.dataloaders:
            class_weights = self.data_manager.get_class_weights(self.dataloaders['train'])
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
            logger.info("Applied class weights for imbalanced dataset")
    
    def setup_logging(self, experiment_name: Optional[str] = None):
        """Set up logging and monitoring."""
        # TensorBoard
        if self.logging_config.get('tensorboard', True):
            log_dir = Path(self.logging_config.get('log_dir', 'logs')) / 'tensorboard'
            if experiment_name:
                log_dir = log_dir / experiment_name
            self.writer = SummaryWriter(log_dir)
            logger.info(f"TensorBoard logging to: {log_dir}")
        
        # Weights & Biases (optional)
        if self.logging_config.get('wandb', False):
            try:
                wandb.init(
                    project="healthcare-ai-pneumonia",
                    name=experiment_name,
                    config=self.config
                )
                logger.info("Weights & Biases logging enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize Weights & Biases: {e}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        train_loader = self.dataloaders['train']
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (images, labels) in enumerate(progress_bar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision:
                with torch.cuda.amp.autocast():
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
            
            running_loss += loss.item()
            
            # Collect predictions for metrics
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Avg Loss': f"{running_loss/(batch_idx+1):.4f}"
            })
        
        # Calculate epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        
        # Calculate additional metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        val_loader = self.dataloaders.get('val', self.dataloaders.get('test'))
        
        if val_loader is None:
            logger.warning("No validation data available")
            return {}
        
        running_loss = 0.0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                
                # Get predictions and probabilities
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Calculate metrics
        epoch_loss = running_loss / len(val_loader)
        epoch_accuracy = accuracy_score(all_labels, all_predictions)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted', zero_division=0
        )
        
        # Calculate AUC (for multi-class, use ovr strategy)
        try:
            auc = roc_auc_score(all_labels, all_probabilities, multi_class='ovr', average='weighted')
        except ValueError:
            auc = 0.0
        
        metrics = {
            'loss': epoch_loss,
            'accuracy': epoch_accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        return metrics
    
    def save_checkpoint(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_dir = Path('checkpoints')
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch+1}.pth"
        
        self.model_manager.save_model(
            model=self.model,
            filepath=str(checkpoint_path),
            optimizer=self.optimizer,
            epoch=epoch,
            loss=val_metrics.get('loss', train_metrics.get('loss')),
            metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
        )
        
        # Save best model
        if not hasattr(self, 'best_val_loss') or val_metrics.get('loss', float('inf')) < self.best_val_loss:
            self.best_val_loss = val_metrics.get('loss', float('inf'))
            best_model_path = checkpoint_dir / "best_model.pth"
            self.model_manager.save_model(
                model=self.model,
                filepath=str(best_model_path),
                optimizer=self.optimizer,
                epoch=epoch,
                loss=val_metrics.get('loss', train_metrics.get('loss')),
                metrics={**train_metrics, **{f"val_{k}": v for k, v in val_metrics.items()}}
            )
            logger.info(f"New best model saved with validation loss: {self.best_val_loss:.4f}")
    
    def log_metrics(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics to various logging systems."""
        # TensorBoard
        if self.writer:
            for metric, value in train_metrics.items():
                self.writer.add_scalar(f'Train/{metric}', value, epoch)
            for metric, value in val_metrics.items():
                self.writer.add_scalar(f'Validation/{metric}', value, epoch)
            
            # Log learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar('Learning_Rate', current_lr, epoch)
        
        # Weights & Biases
        if wandb.run:
            log_dict = {f"train_{k}": v for k, v in train_metrics.items()}
            log_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
            log_dict['learning_rate'] = self.optimizer.param_groups[0]['lr']
            log_dict['epoch'] = epoch
            wandb.log(log_dict)
        
        # Console logging
        logger.info(f"Epoch {epoch+1}/{self.training_config.get('epochs', 50)}:")
        logger.info(f"  Train - Loss: {train_metrics.get('loss', 0):.4f}, "
                   f"Acc: {train_metrics.get('accuracy', 0):.4f}, "
                   f"F1: {train_metrics.get('f1', 0):.4f}")
        if val_metrics:
            logger.info(f"  Val   - Loss: {val_metrics.get('loss', 0):.4f}, "
                       f"Acc: {val_metrics.get('accuracy', 0):.4f}, "
                       f"F1: {val_metrics.get('f1', 0):.4f}, "
                       f"AUC: {val_metrics.get('auc', 0):.4f}")
    
    def train(self, experiment_name: Optional[str] = None, data_path: Optional[str] = None):
        """Main training loop."""
        logger.info("Starting training...")
        
        # Setup
        self.setup_model()
        self.setup_data(data_path)
        self.setup_logging(experiment_name)
        
        # Training parameters
        num_epochs = self.training_config.get('epochs', 50)
        self.best_val_loss = float('inf')
        
        start_time = time.time()
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_metrics = self.train_epoch(epoch)
            
            # Validation phase
            val_metrics = self.validate_epoch(epoch)
            
            # Learning rate scheduling
            if self.scheduler:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics.get('loss', train_metrics['loss']))
                else:
                    self.scheduler.step()
            
            # Logging
            self.log_metrics(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            self.save_checkpoint(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if self.early_stopping and val_metrics:
                if self.early_stopping(val_metrics.get('loss', float('inf')), self.model):
                    logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    break
            
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s")
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        
        # Close logging
        if self.writer:
            self.writer.close()
        if wandb.run:
            wandb.finish()
        
        return self.model


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Train Healthcare AI Model")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default=None,
                       help='Path to data directory or CSV file')
    parser.add_argument('--experiment', type=str, default=None,
                       help='Experiment name for logging')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = Trainer(args.config)
    
    # Resume from checkpoint if specified
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # Implementation for resuming from checkpoint would go here
        # This would involve loading the model state, optimizer state, etc.
    
    # Start training
    try:
        model = trainer.train(
            experiment_name=args.experiment,
            data_path=args.data
        )
        logger.info("Training completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()