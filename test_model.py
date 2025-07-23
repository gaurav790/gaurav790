"""
Unit Tests for Healthcare AI - Chest X-ray Pneumonia Detection
Comprehensive test suite for all model components.
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
import cv2

# Import modules to test
from model import MedicalImageClassifier, ModelManager
from data_loader import MedicalImageDataset, DataLoaderManager
from evaluate import MetricsCalculator
from inference import MedicalImageInference


class TestMedicalImageClassifier:
    """Test cases for the MedicalImageClassifier model."""
    
    @pytest.fixture
    def model(self):
        """Create a test model instance."""
        return MedicalImageClassifier(
            num_classes=3,
            architecture="resnet18",  # Use smaller model for testing
            pretrained=False,
            freeze_backbone=False,
            dropout_rate=0.5,
            input_channels=1
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.num_classes == 3
        assert model.architecture == "resnet18"
        assert model.input_channels == 1
        assert model.dropout_rate == 0.5
    
    def test_forward_pass(self, model):
        """Test forward pass with dummy input."""
        batch_size = 2
        channels = 1
        height, width = 224, 224
        
        dummy_input = torch.randn(batch_size, channels, height, width)
        output = model(dummy_input)
        
        assert output.shape == (batch_size, 3)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_predict_proba(self, model):
        """Test probability prediction."""
        dummy_input = torch.randn(1, 1, 224, 224)
        probabilities = model.predict_proba(dummy_input)
        
        assert probabilities.shape == (1, 3)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(1), atol=1e-6)
        assert (probabilities >= 0).all()
        assert (probabilities <= 1).all()
    
    def test_predict_with_threshold(self, model):
        """Test prediction with confidence threshold."""
        dummy_input = torch.randn(1, 1, 224, 224)
        predictions, confidence = model.predict(dummy_input, threshold=0.5)
        
        assert predictions.shape == (1,)
        assert confidence.shape == (1,)
        assert 0 <= confidence.item() <= 1
    
    def test_different_architectures(self):
        """Test different model architectures."""
        architectures = ["resnet18", "resnet34", "resnet50"]
        
        for arch in architectures:
            model = MedicalImageClassifier(
                num_classes=3,
                architecture=arch,
                pretrained=False,
                input_channels=1
            )
            
            dummy_input = torch.randn(1, 1, 224, 224)
            output = model(dummy_input)
            
            assert output.shape == (1, 3)


class TestModelManager:
    """Test cases for the ModelManager class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        config = {
            'model': {
                'architecture': 'resnet18',
                'pretrained': False,
                'freeze_backbone': False,
                'dropout_rate': 0.3
            },
            'data': {
                'num_classes': 3,
                'channels': 1,
                'class_names': ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia']
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    def test_model_creation(self, temp_config):
        """Test model creation from config."""
        manager = ModelManager(temp_config)
        model = manager.create_model()
        
        assert isinstance(model, MedicalImageClassifier)
        assert model.num_classes == 3
        assert model.architecture == "resnet18"
    
    def test_save_and_load_model(self, temp_config):
        """Test model saving and loading."""
        manager = ModelManager(temp_config)
        model = manager.create_model()
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as f:
            model_path = f.name
        
        manager.save_model(model, model_path, epoch=10, loss=0.5)
        
        # Load model
        loaded_model, checkpoint_info = manager.load_model(model_path)
        
        assert isinstance(loaded_model, MedicalImageClassifier)
        assert checkpoint_info['epoch'] == 10
        assert checkpoint_info['loss'] == 0.5
        
        # Test that models produce same output
        dummy_input = torch.randn(1, 1, 224, 224)
        original_output = model(dummy_input)
        loaded_output = loaded_model(dummy_input)
        
        # Models should produce similar outputs (allowing for small numerical differences)
        assert torch.allclose(original_output, loaded_output, atol=1e-5)
        
        # Cleanup
        Path(model_path).unlink()
    
    def test_parameter_counting(self, temp_config):
        """Test parameter counting functionality."""
        manager = ModelManager(temp_config)
        model = manager.create_model()
        
        param_info = manager.count_parameters(model)
        
        assert 'total_parameters' in param_info
        assert 'trainable_parameters' in param_info
        assert 'non_trainable_parameters' in param_info
        assert param_info['total_parameters'] > 0
        assert param_info['trainable_parameters'] > 0


class TestMedicalImageDataset:
    """Test cases for the MedicalImageDataset class."""
    
    @pytest.fixture
    def temp_images(self):
        """Create temporary test images."""
        temp_dir = tempfile.mkdtemp()
        image_paths = []
        labels = []
        
        # Create dummy images
        for i, label in enumerate(['Normal', 'Pneumonia', 'Normal']):
            image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
            image_path = Path(temp_dir) / f"image_{i}.png"
            cv2.imwrite(str(image_path), image)
            
            image_paths.append(str(image_path))
            labels.append(label)
        
        yield image_paths, labels
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_dataset_initialization(self, temp_images):
        """Test dataset initialization."""
        image_paths, labels = temp_images
        
        dataset = MedicalImageDataset(
            image_paths=image_paths,
            labels=labels,
            image_size=(224, 224)
        )
        
        assert len(dataset) == 3
        assert len(dataset.class_names) == 2  # 'Normal' and 'Pneumonia'
    
    def test_dataset_getitem(self, temp_images):
        """Test dataset item retrieval."""
        image_paths, labels = temp_images
        
        dataset = MedicalImageDataset(
            image_paths=image_paths,
            labels=labels,
            image_size=(224, 224)
        )
        
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, (int, np.integer))
        assert image.shape == (1, 224, 224)  # Grayscale image
        assert 0 <= label < len(dataset.label_encoder.classes_)
    
    def test_error_handling(self):
        """Test error handling for invalid images."""
        # Create dataset with invalid image path
        dataset = MedicalImageDataset(
            image_paths=["nonexistent.png"],
            labels=["Normal"],
            image_size=(224, 224)
        )
        
        # Should return blank image and label 0 for invalid path
        image, label = dataset[0]
        
        assert isinstance(image, torch.Tensor)
        assert image.shape == (1, 224, 224)
        assert label == 0


class TestDataLoaderManager:
    """Test cases for the DataLoaderManager class."""
    
    @pytest.fixture
    def temp_config(self):
        """Create temporary config file."""
        config = {
            'data': {
                'data_dir': 'test_data',
                'image_size': [224, 224],
                'num_classes': 3,
                'class_names': ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia'],
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'augmentation': {
                    'enabled': True,
                    'rotation_range': 15,
                    'horizontal_flip': True
                }
            },
            'training': {
                'batch_size': 4
            },
            'hardware': {
                'num_workers': 0  # Use 0 for testing to avoid multiprocessing issues
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f)
            return f.name
    
    def test_transform_creation(self, temp_config):
        """Test augmentation transform creation."""
        manager = DataLoaderManager(temp_config)
        
        train_transform = manager.create_transforms(is_training=True)
        val_transform = manager.create_transforms(is_training=False)
        
        assert train_transform is not None
        assert val_transform is not None
        
        # Test transform application
        dummy_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        train_result = train_transform(image=dummy_image)
        val_result = val_transform(image=dummy_image)
        
        assert 'image' in train_result
        assert 'image' in val_result
        assert isinstance(train_result['image'], torch.Tensor)
        assert isinstance(val_result['image'], torch.Tensor)


class TestMetricsCalculator:
    """Test cases for the MetricsCalculator class."""
    
    @pytest.fixture
    def calculator(self):
        """Create MetricsCalculator instance."""
        class_names = ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia']
        return MetricsCalculator(class_names)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample prediction data."""
        np.random.seed(42)
        y_true = np.random.randint(0, 3, 100)
        y_pred = np.random.randint(0, 3, 100)
        y_proba = np.random.rand(100, 3)
        # Normalize probabilities
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)
        
        return y_true, y_pred, y_proba
    
    def test_basic_metrics_calculation(self, calculator, sample_data):
        """Test basic metrics calculation."""
        y_true, y_pred, _ = sample_data
        
        metrics = calculator.calculate_basic_metrics(y_true, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision_macro' in metrics
        assert 'recall_macro' in metrics
        assert 'f1_macro' in metrics
        assert 'precision_weighted' in metrics
        assert 'recall_weighted' in metrics
        assert 'f1_weighted' in metrics
        
        # Check per-class metrics
        for class_name in calculator.class_names:
            assert f'precision_{class_name}' in metrics
            assert f'recall_{class_name}' in metrics
            assert f'f1_{class_name}' in metrics
            assert f'support_{class_name}' in metrics
        
        # Check value ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision_macro'] <= 1
        assert 0 <= metrics['recall_macro'] <= 1
        assert 0 <= metrics['f1_macro'] <= 1
    
    def test_auc_metrics_calculation(self, calculator, sample_data):
        """Test AUC metrics calculation."""
        y_true, _, y_proba = sample_data
        
        metrics = calculator.calculate_auc_metrics(y_true, y_proba)
        
        assert 'auc_macro' in metrics
        assert 'auc_weighted' in metrics
        
        # Check per-class AUC
        for class_name in calculator.class_names:
            assert f'auc_{class_name}' in metrics
        
        # Check value ranges
        assert 0 <= metrics['auc_macro'] <= 1
        assert 0 <= metrics['auc_weighted'] <= 1
    
    def test_classification_report_generation(self, calculator, sample_data):
        """Test classification report generation."""
        y_true, y_pred, _ = sample_data
        
        report = calculator.generate_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert len(report) > 0
        
        # Check that class names appear in report
        for class_name in calculator.class_names:
            assert class_name in report


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    @pytest.fixture
    def temp_setup(self):
        """Set up temporary directory structure for testing."""
        temp_dir = tempfile.mkdtemp()
        
        # Create config
        config = {
            'model': {
                'architecture': 'resnet18',
                'pretrained': False,
                'freeze_backbone': False,
                'dropout_rate': 0.3
            },
            'data': {
                'data_dir': str(Path(temp_dir) / 'data'),
                'image_size': [224, 224],
                'num_classes': 2,
                'channels': 1,
                'class_names': ['Normal', 'Pneumonia'],
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1
            },
            'training': {
                'batch_size': 2,
                'epochs': 1,
                'learning_rate': 0.001,
                'optimizer': 'adam'
            },
            'hardware': {
                'num_workers': 0
            },
            'inference': {
                'confidence_threshold': 0.7
            }
        }
        
        config_path = Path(temp_dir) / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Create sample data structure
        data_dir = Path(temp_dir) / 'data' / 'images'
        for class_name in ['Normal', 'Pneumonia']:
            class_dir = data_dir / class_name
            class_dir.mkdir(parents=True, exist_ok=True)
            
            # Create dummy images
            for i in range(5):
                image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
                image_path = class_dir / f"image_{i}.png"
                cv2.imwrite(str(image_path), image)
        
        yield temp_dir, str(config_path)
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    def test_end_to_end_pipeline(self, temp_setup):
        """Test the complete pipeline from data loading to inference."""
        temp_dir, config_path = temp_setup
        
        # Test model creation and training setup
        model_manager = ModelManager(config_path)
        model = model_manager.create_model()
        
        assert isinstance(model, MedicalImageClassifier)
        
        # Test data loading
        data_manager = DataLoaderManager(config_path)
        dataloaders = data_manager.load_dataset_from_folders()
        
        assert 'train' in dataloaders or 'val' in dataloaders or 'test' in dataloaders
        
        # Test model saving
        model_path = Path(temp_dir) / 'test_model.pth'
        model_manager.save_model(model, str(model_path))
        
        assert model_path.exists()
        
        # Test model loading and inference (if we have data)
        if dataloaders:
            loaded_model, _ = model_manager.load_model(str(model_path))
            assert isinstance(loaded_model, MedicalImageClassifier)


def test_model_device_handling():
    """Test model behavior on different devices."""
    model = MedicalImageClassifier(
        num_classes=3,
        architecture="resnet18",
        pretrained=False,
        input_channels=1
    )
    
    # Test CPU
    model = model.to('cpu')
    dummy_input = torch.randn(1, 1, 224, 224)
    output = model(dummy_input)
    assert output.device.type == 'cpu'
    
    # Test CUDA if available
    if torch.cuda.is_available():
        model = model.to('cuda')
        dummy_input = dummy_input.to('cuda')
        output = model(dummy_input)
        assert output.device.type == 'cuda'


def test_batch_size_handling():
    """Test model with different batch sizes."""
    model = MedicalImageClassifier(
        num_classes=3,
        architecture="resnet18",
        pretrained=False,
        input_channels=1
    )
    
    batch_sizes = [1, 2, 4, 8]
    
    for batch_size in batch_sizes:
        dummy_input = torch.randn(batch_size, 1, 224, 224)
        output = model(dummy_input)
        
        assert output.shape == (batch_size, 3)
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])