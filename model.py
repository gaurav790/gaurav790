"""
Model Module for Healthcare AI - Chest X-ray Pneumonia Detection
Defines the neural network architecture for medical image classification.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, Any, Optional, Tuple
import yaml
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageClassifier(nn.Module):
    """
    Medical image classifier based on ResNet architecture.
    Designed for chest X-ray pneumonia detection with 3 classes:
    Normal, Bacterial Pneumonia, Viral Pneumonia
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        architecture: str = "resnet50",
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5,
        input_channels: int = 1
    ):
        """
        Initialize the medical image classifier.
        
        Args:
            num_classes: Number of output classes
            architecture: Backbone architecture (resnet18, resnet34, resnet50, etc.)
            pretrained: Whether to use pretrained weights
            freeze_backbone: Whether to freeze backbone parameters
            dropout_rate: Dropout rate for regularization
            input_channels: Number of input channels (1 for grayscale, 3 for RGB)
        """
        super(MedicalImageClassifier, self).__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        self.dropout_rate = dropout_rate
        self.input_channels = input_channels
        
        # Load backbone model
        self.backbone = self._create_backbone(architecture, pretrained)
        
        # Modify first layer for grayscale input if needed
        if input_channels != 3:
            self._modify_input_layer()
        
        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()
        
        # Get backbone output features
        backbone_features = self._get_backbone_features()
        
        # Create classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(backbone_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate / 4),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_classifier()
        
        logger.info(f"MedicalImageClassifier initialized:")
        logger.info(f"  Architecture: {architecture}")
        logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Input channels: {input_channels}")
        logger.info(f"  Pretrained: {pretrained}")
        logger.info(f"  Frozen backbone: {freeze_backbone}")
    
    def _create_backbone(self, architecture: str, pretrained: bool) -> nn.Module:
        """Create the backbone model."""
        if architecture == "resnet18":
            backbone = models.resnet18(pretrained=pretrained)
        elif architecture == "resnet34":
            backbone = models.resnet34(pretrained=pretrained)
        elif architecture == "resnet50":
            backbone = models.resnet50(pretrained=pretrained)
        elif architecture == "resnet101":
            backbone = models.resnet101(pretrained=pretrained)
        elif architecture == "resnet152":
            backbone = models.resnet152(pretrained=pretrained)
        elif architecture == "densenet121":
            backbone = models.densenet121(pretrained=pretrained)
        elif architecture == "densenet169":
            backbone = models.densenet169(pretrained=pretrained)
        elif architecture == "efficientnet_b0":
            backbone = models.efficientnet_b0(pretrained=pretrained)
        elif architecture == "efficientnet_b1":
            backbone = models.efficientnet_b1(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        
        # Remove the final classification layer
        if hasattr(backbone, 'fc'):
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif hasattr(backbone, 'classifier'):
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        
        return backbone
    
    def _modify_input_layer(self):
        """Modify the first layer to accept different number of input channels."""
        if self.architecture.startswith('resnet'):
            # Get the original first conv layer
            original_conv = self.backbone[0][0]  # backbone -> Sequential -> first conv
            
            # Create new conv layer with same parameters but different input channels
            new_conv = nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )
            
            # Initialize weights
            if self.input_channels == 1 and original_conv.in_channels == 3:
                # Average the RGB weights for grayscale
                with torch.no_grad():
                    new_conv.weight = nn.Parameter(
                        original_conv.weight.mean(dim=1, keepdim=True)
                    )
            else:
                # Initialize randomly
                nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            
            # Replace the first conv layer
            self.backbone[0][0] = new_conv
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone parameters frozen")
    
    def _get_backbone_features(self) -> int:
        """Get the number of features from the backbone."""
        if self.architecture.startswith('resnet'):
            if self.architecture in ['resnet18', 'resnet34']:
                return 512
            else:  # resnet50, resnet101, resnet152
                return 2048
        elif self.architecture.startswith('densenet'):
            if self.architecture == 'densenet121':
                return 1024
            elif self.architecture == 'densenet169':
                return 1664
        elif self.architecture.startswith('efficientnet'):
            # This is approximate, might need adjustment
            return 1280
        else:
            # Default fallback
            return 2048
    
    def _initialize_classifier(self):
        """Initialize classifier weights."""
        for module in self.classifier.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using backbone
        features = self.backbone(x)
        
        # Classify using the classifier head
        logits = self.classifier(features)
        
        return logits
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Probability tensor of shape (batch_size, num_classes)
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with confidence scores.
        
        Args:
            x: Input tensor
            threshold: Confidence threshold
            
        Returns:
            Tuple of (predictions, confidence_scores)
        """
        probabilities = self.predict_proba(x)
        confidence_scores, predictions = torch.max(probabilities, dim=1)
        
        # Apply threshold
        low_confidence = confidence_scores < threshold
        predictions[low_confidence] = -1  # Mark as uncertain
        
        return predictions, confidence_scores
    
    def get_feature_maps(self, x: torch.Tensor, layer_name: str = 'backbone') -> torch.Tensor:
        """
        Extract feature maps from a specific layer.
        
        Args:
            x: Input tensor
            layer_name: Name of the layer to extract features from
            
        Returns:
            Feature maps tensor
        """
        if layer_name == 'backbone':
            return self.backbone(x)
        else:
            # For more specific layer extraction, you might need to register hooks
            raise NotImplementedError(f"Feature extraction for layer '{layer_name}' not implemented")


class ModelManager:
    """Manages model creation, loading, and saving."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.model_config = self.config.get('model', {})
        self.data_config = self.config.get('data', {})
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def create_model(self) -> MedicalImageClassifier:
        """Create model based on configuration."""
        model = MedicalImageClassifier(
            num_classes=self.data_config.get('num_classes', 3),
            architecture=self.model_config.get('architecture', 'resnet50'),
            pretrained=self.model_config.get('pretrained', True),
            freeze_backbone=self.model_config.get('freeze_backbone', False),
            dropout_rate=self.model_config.get('dropout_rate', 0.5),
            input_channels=self.data_config.get('channels', 1)
        )
        
        return model
    
    def save_model(
        self,
        model: MedicalImageClassifier,
        filepath: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: Optional[int] = None,
        loss: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Save model checkpoint.
        
        Args:
            model: Model to save
            filepath: Path to save the model
            optimizer: Optimizer state to save
            epoch: Current epoch
            loss: Current loss
            metrics: Current metrics
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'model_config': {
                'num_classes': model.num_classes,
                'architecture': model.architecture,
                'dropout_rate': model.dropout_rate,
                'input_channels': model.input_channels
            },
            'config': self.config
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
        if epoch is not None:
            checkpoint['epoch'] = epoch
        if loss is not None:
            checkpoint['loss'] = loss
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        # Create directory if it doesn't exist
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(
        self,
        filepath: str,
        device: str = 'cpu'
    ) -> Tuple[MedicalImageClassifier, Dict[str, Any]]:
        """
        Load model from checkpoint.
        
        Args:
            filepath: Path to the model checkpoint
            device: Device to load the model on
            
        Returns:
            Tuple of (model, checkpoint_info)
        """
        checkpoint = torch.load(filepath, map_location=device)
        
        # Create model with saved configuration
        model_config = checkpoint.get('model_config', {})
        model = MedicalImageClassifier(
            num_classes=model_config.get('num_classes', 3),
            architecture=model_config.get('architecture', 'resnet50'),
            pretrained=False,  # Don't load pretrained weights when loading checkpoint
            freeze_backbone=False,
            dropout_rate=model_config.get('dropout_rate', 0.5),
            input_channels=model_config.get('input_channels', 1)
        )
        
        # Load model weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        
        # Extract checkpoint info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch'),
            'loss': checkpoint.get('loss'),
            'metrics': checkpoint.get('metrics', {}),
            'config': checkpoint.get('config', {})
        }
        
        logger.info(f"Model loaded from {filepath}")
        if checkpoint_info['epoch'] is not None:
            logger.info(f"  Epoch: {checkpoint_info['epoch']}")
        if checkpoint_info['loss'] is not None:
            logger.info(f"  Loss: {checkpoint_info['loss']:.4f}")
        
        return model, checkpoint_info
    
    def count_parameters(self, model: MedicalImageClassifier) -> Dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        param_info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': total_params - trainable_params
        }
        
        logger.info(f"Model parameters:")
        logger.info(f"  Total: {total_params:,}")
        logger.info(f"  Trainable: {trainable_params:,}")
        logger.info(f"  Non-trainable: {total_params - trainable_params:,}")
        
        return param_info


def main():
    """Test the model."""
    # Initialize model manager
    model_manager = ModelManager()
    
    # Create model
    model = model_manager.create_model()
    
    # Count parameters
    model_manager.count_parameters(model)
    
    # Test forward pass
    batch_size = 4
    channels = 1
    height, width = 512, 512
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        probabilities = model.predict_proba(dummy_input)
        predictions, confidence = model.predict(dummy_input)
    
    logger.info(f"Input shape: {dummy_input.shape}")
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Probabilities shape: {probabilities.shape}")
    logger.info(f"Predictions: {predictions}")
    logger.info(f"Confidence scores: {confidence}")
    
    # Test save/load
    model_path = "test_model.pth"
    model_manager.save_model(model, model_path)
    
    loaded_model, checkpoint_info = model_manager.load_model(model_path)
    logger.info(f"Model successfully saved and loaded")
    
    # Clean up
    Path(model_path).unlink(missing_ok=True)
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    main()