"""
Data Loader Module for Healthcare AI - Chest X-ray Pneumonia Detection
Handles loading, preprocessing, and augmentation of medical images.
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict, Any
from pathlib import Path
import cv2
from PIL import Image
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pydicom
import SimpleITK as sitk

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageDataset(Dataset):
    """Custom dataset for medical images with augmentation support."""
    
    def __init__(
        self,
        image_paths: List[str],
        labels: List[str],
        transform: Optional[A.Compose] = None,
        image_size: Tuple[int, int] = (512, 512),
        is_dicom: bool = False
    ):
        """
        Initialize the dataset.
        
        Args:
            image_paths: List of paths to image files
            labels: List of corresponding labels
            transform: Albumentations transform pipeline
            image_size: Target image size (height, width)
            is_dicom: Whether images are in DICOM format
        """
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.image_size = image_size
        self.is_dicom = is_dicom
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.encoded_labels = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Dataset initialized with {len(image_paths)} images")
        logger.info(f"Classes: {self.label_encoder.classes_}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single item from the dataset."""
        try:
            # Load image
            image_path = self.image_paths[idx]
            image = self._load_image(image_path)
            
            # Apply transforms
            if self.transform:
                transformed = self.transform(image=image)
                image = transformed['image']
            else:
                # Default preprocessing
                image = cv2.resize(image, self.image_size)
                image = torch.from_numpy(image).float()
                if len(image.shape) == 2:
                    image = image.unsqueeze(0)  # Add channel dimension
                image = image / 255.0  # Normalize to [0, 1]
            
            label = self.encoded_labels[idx]
            
            return image, label
            
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image and label 0 as fallback
            blank_image = torch.zeros((1, *self.image_size), dtype=torch.float32)
            return blank_image, 0
    
    def _load_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess a single image."""
        if self.is_dicom:
            return self._load_dicom_image(image_path)
        else:
            return self._load_standard_image(image_path)
    
    def _load_dicom_image(self, image_path: str) -> np.ndarray:
        """Load DICOM image using pydicom and SimpleITK."""
        try:
            # Try pydicom first
            dicom_data = pydicom.dcmread(image_path)
            image = dicom_data.pixel_array
            
            # Handle different photometric interpretations
            if hasattr(dicom_data, 'PhotometricInterpretation'):
                if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                    image = np.max(image) - image
            
            # Normalize to 8-bit
            image = self._normalize_to_uint8(image)
            
        except Exception as e:
            logger.warning(f"pydicom failed for {image_path}, trying SimpleITK: {str(e)}")
            try:
                # Fallback to SimpleITK
                sitk_image = sitk.ReadImage(image_path)
                image = sitk.GetArrayFromImage(sitk_image)
                image = self._normalize_to_uint8(image)
            except Exception as e2:
                logger.error(f"Both pydicom and SimpleITK failed for {image_path}: {str(e2)}")
                raise e2
        
        return image
    
    def _load_standard_image(self, image_path: str) -> np.ndarray:
        """Load standard image formats (PNG, JPEG, etc.)."""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Try PIL as fallback
            pil_image = Image.open(image_path).convert('L')
            image = np.array(pil_image)
        
        return image
    
    def _normalize_to_uint8(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to uint8 range [0, 255]."""
        if image.dtype == np.uint8:
            return image
        
        # Handle different bit depths
        image = image.astype(np.float64)
        image_min, image_max = np.min(image), np.max(image)
        
        if image_max > image_min:
            image = (image - image_min) / (image_max - image_min) * 255.0
        else:
            image = np.zeros_like(image)
        
        return image.astype(np.uint8)


class DataLoaderManager:
    """Manages data loading, preprocessing, and augmentation for the healthcare AI project."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration."""
        self.config = self._load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.training_config = self.config.get('training', {})
        
        # Set up paths
        self.data_dir = Path(self.data_config.get('data_dir', 'data'))
        self.image_size = tuple(self.data_config.get('image_size', [512, 512]))
        self.num_classes = self.data_config.get('num_classes', 3)
        self.class_names = self.data_config.get('class_names', ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia'])
        
        logger.info(f"DataLoaderManager initialized with config: {config_path}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def create_transforms(self, is_training: bool = True) -> A.Compose:
        """Create augmentation transforms."""
        if is_training and self.data_config.get('augmentation', {}).get('enabled', True):
            # Training transforms with augmentation
            aug_config = self.data_config.get('augmentation', {})
            transforms_list = [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Rotate(limit=aug_config.get('rotation_range', 15), p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=aug_config.get('width_shift_range', 0.1),
                    scale_limit=aug_config.get('zoom_range', 0.1),
                    rotate_limit=0,
                    p=0.5
                ),
                A.HorizontalFlip(p=0.5 if aug_config.get('horizontal_flip', True) else 0),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.1),
                A.Normalize(mean=0.5, std=0.5),  # Normalize to [-1, 1]
                ToTensorV2()
            ]
        else:
            # Validation/test transforms without augmentation
            transforms_list = [
                A.Resize(height=self.image_size[0], width=self.image_size[1]),
                A.Normalize(mean=0.5, std=0.5),
                ToTensorV2()
            ]
        
        return A.Compose(transforms_list)
    
    def load_dataset_from_folders(self, split_data: bool = True) -> Dict[str, DataLoader]:
        """
        Load dataset from folder structure: data/train/class1, data/train/class2, etc.
        """
        dataloaders = {}
        
        if split_data:
            # Load all data and split
            all_image_paths = []
            all_labels = []
            
            for class_name in self.class_names:
                class_dir = self.data_dir / "images" / class_name
                if class_dir.exists():
                    image_files = list(class_dir.glob("*"))
                    image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']]
                    
                    all_image_paths.extend([str(f) for f in image_files])
                    all_labels.extend([class_name] * len(image_files))
            
            # Split the data
            train_paths, temp_paths, train_labels, temp_labels = train_test_split(
                all_image_paths, all_labels,
                test_size=(1 - self.data_config.get('train_split', 0.7)),
                stratify=all_labels,
                random_state=42
            )
            
            val_size = self.data_config.get('val_split', 0.15) / (self.data_config.get('val_split', 0.15) + self.data_config.get('test_split', 0.15))
            val_paths, test_paths, val_labels, test_labels = train_test_split(
                temp_paths, temp_labels,
                test_size=(1 - val_size),
                stratify=temp_labels,
                random_state=42
            )
            
            # Create datasets
            splits = {
                'train': (train_paths, train_labels, True),
                'val': (val_paths, val_labels, False),
                'test': (test_paths, test_labels, False)
            }
            
        else:
            # Load from separate train/val/test folders
            splits = {}
            for split in ['train', 'val', 'test']:
                split_dir = self.data_dir / split
                if split_dir.exists():
                    paths, labels = [], []
                    for class_name in self.class_names:
                        class_dir = split_dir / class_name
                        if class_dir.exists():
                            image_files = list(class_dir.glob("*"))
                            image_files = [f for f in image_files if f.suffix.lower() in ['.png', '.jpg', '.jpeg', '.dcm', '.dicom']]
                            
                            paths.extend([str(f) for f in image_files])
                            labels.extend([class_name] * len(image_files))
                    
                    splits[split] = (paths, labels, split == 'train')
        
        # Create DataLoaders
        for split_name, (paths, labels, is_training) in splits.items():
            if len(paths) > 0:
                dataset = MedicalImageDataset(
                    image_paths=paths,
                    labels=labels,
                    transform=self.create_transforms(is_training),
                    image_size=self.image_size,
                    is_dicom=any(Path(p).suffix.lower() in ['.dcm', '.dicom'] for p in paths[:10])
                )
                
                dataloaders[split_name] = DataLoader(
                    dataset,
                    batch_size=self.training_config.get('batch_size', 32),
                    shuffle=is_training,
                    num_workers=self.config.get('hardware', {}).get('num_workers', 4),
                    pin_memory=torch.cuda.is_available(),
                    drop_last=is_training
                )
                
                logger.info(f"{split_name} dataset: {len(dataset)} images")
        
        return dataloaders
    
    def load_dataset_from_csv(self, csv_path: str) -> Dict[str, DataLoader]:
        """Load dataset from CSV file with image paths and labels."""
        df = pd.read_csv(csv_path)
        
        # Assume CSV has columns: 'image_path', 'label'
        image_paths = df['image_path'].tolist()
        labels = df['label'].tolist()
        
        # Split the data
        train_paths, temp_paths, train_labels, temp_labels = train_test_split(
            image_paths, labels,
            test_size=(1 - self.data_config.get('train_split', 0.7)),
            stratify=labels,
            random_state=42
        )
        
        val_size = self.data_config.get('val_split', 0.15) / (self.data_config.get('val_split', 0.15) + self.data_config.get('test_split', 0.15))
        val_paths, test_paths, val_labels, test_labels = train_test_split(
            temp_paths, temp_labels,
            test_size=(1 - val_size),
            stratify=temp_labels,
            random_state=42
        )
        
        # Create datasets and dataloaders
        dataloaders = {}
        splits = {
            'train': (train_paths, train_labels, True),
            'val': (val_paths, val_labels, False),
            'test': (test_paths, test_labels, False)
        }
        
        for split_name, (paths, labels, is_training) in splits.items():
            dataset = MedicalImageDataset(
                image_paths=paths,
                labels=labels,
                transform=self.create_transforms(is_training),
                image_size=self.image_size,
                is_dicom=any(Path(p).suffix.lower() in ['.dcm', '.dicom'] for p in paths[:10])
            )
            
            dataloaders[split_name] = DataLoader(
                dataset,
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=is_training,
                num_workers=self.config.get('hardware', {}).get('num_workers', 4),
                pin_memory=torch.cuda.is_available(),
                drop_last=is_training
            )
            
            logger.info(f"{split_name} dataset: {len(dataset)} images")
        
        return dataloaders
    
    def validate_dataset(self, dataloaders: Dict[str, DataLoader]) -> bool:
        """Validate the loaded dataset."""
        try:
            for split_name, dataloader in dataloaders.items():
                logger.info(f"Validating {split_name} dataset...")
                
                # Check if we can load a batch
                batch = next(iter(dataloader))
                images, labels = batch
                
                logger.info(f"{split_name} - Batch shape: {images.shape}, Labels shape: {labels.shape}")
                logger.info(f"{split_name} - Image range: [{images.min():.3f}, {images.max():.3f}]")
                logger.info(f"{split_name} - Unique labels: {torch.unique(labels).tolist()}")
                
            logger.info("Dataset validation completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Dataset validation failed: {str(e)}")
            return False
    
    def get_class_weights(self, dataloader: DataLoader) -> torch.Tensor:
        """Calculate class weights for handling imbalanced datasets."""
        all_labels = []
        for _, labels in dataloader:
            all_labels.extend(labels.tolist())
        
        class_counts = np.bincount(all_labels)
        total_samples = len(all_labels)
        
        # Calculate inverse frequency weights
        class_weights = total_samples / (len(class_counts) * class_counts)
        
        logger.info(f"Class counts: {class_counts}")
        logger.info(f"Class weights: {class_weights}")
        
        return torch.FloatTensor(class_weights)


def main():
    """Test the data loader."""
    # Initialize data loader manager
    data_manager = DataLoaderManager()
    
    # Create sample data structure for testing
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    for split in ['train', 'val', 'test']:
        split_dir = data_dir / split
        split_dir.mkdir(exist_ok=True)
        for class_name in ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia']:
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
    
    logger.info("Sample data structure created. Please add your images to the appropriate folders.")
    
    # Try to load dataloaders (will be empty without actual images)
    try:
        dataloaders = data_manager.load_dataset_from_folders(split_data=False)
        if dataloaders:
            data_manager.validate_dataset(dataloaders)
        else:
            logger.info("No images found. Please add images to the data folders.")
    except Exception as e:
        logger.error(f"Error testing data loader: {str(e)}")


if __name__ == "__main__":
    main()