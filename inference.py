"""
Inference Script for Healthcare AI - Chest X-ray Pneumonia Detection
Performs inference on new images and provides predictions with confidence scores.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd

# Import our modules
from model import ModelManager, MedicalImageClassifier
from data_loader import DataLoaderManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MedicalImageInference:
    """Handles inference for medical image classification."""
    
    def __init__(
        self,
        model_path: str,
        config_path: str = "config.yaml",
        device: Optional[str] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            config_path: Path to configuration file
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.config = self._load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.inference_config = self.config.get('inference', {})
        
        # Set up device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model_manager = ModelManager(config_path)
        self.model, self.checkpoint_info = self.model_manager.load_model(model_path, str(self.device))
        self.model.eval()
        
        # Set up data preprocessing
        self.data_manager = DataLoaderManager(config_path)
        self.transform = self.data_manager.create_transforms(is_training=False)
        
        # Class information
        self.class_names = self.data_config.get('class_names', ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia'])
        self.num_classes = len(self.class_names)
        self.confidence_threshold = self.inference_config.get('confidence_threshold', 0.7)
        
        logger.info(f"Model loaded successfully:")
        logger.info(f"  Classes: {self.class_names}")
        logger.info(f"  Confidence threshold: {self.confidence_threshold}")
        if self.checkpoint_info.get('epoch'):
            logger.info(f"  Trained for {self.checkpoint_info['epoch']} epochs")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        Preprocess a single image for inference.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        if str(image_path).lower().endswith(('.dcm', '.dicom')):
            # Handle DICOM files
            try:
                import pydicom
                dicom_data = pydicom.dcmread(image_path)
                image = dicom_data.pixel_array
                
                # Handle photometric interpretation
                if hasattr(dicom_data, 'PhotometricInterpretation'):
                    if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                        image = np.max(image) - image
                
                # Normalize to 8-bit
                image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                
            except ImportError:
                logger.error("pydicom not available for DICOM files")
                raise
            except Exception as e:
                logger.error(f"Error loading DICOM file {image_path}: {e}")
                raise
        else:
            # Handle standard image formats
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                # Try PIL as fallback
                pil_image = Image.open(image_path).convert('L')
                image = np.array(pil_image)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
        else:
            # Fallback preprocessing
            image = cv2.resize(image, (512, 512))
            image_tensor = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        
        return image_tensor.unsqueeze(0)  # Add batch dimension
    
    def predict_single(
        self,
        image_path: str,
        return_probabilities: bool = True
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        """
        Make prediction on a single image.
        
        Args:
            image_path: Path to the image file
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary containing prediction results
        """
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image_path)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                
                confidence = confidence.item()
                predicted_class = predicted_class.item()
                probabilities_np = probabilities.cpu().numpy()[0]
            
            # Prepare results
            results = {
                'image_path': str(image_path),
                'predicted_class': self.class_names[predicted_class],
                'predicted_class_index': predicted_class,
                'confidence': confidence,
                'is_confident': confidence >= self.confidence_threshold,
                'status': 'success'
            }
            
            if return_probabilities:
                results['probabilities'] = probabilities_np
                results['class_probabilities'] = {
                    class_name: prob for class_name, prob in zip(self.class_names, probabilities_np)
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error predicting image {image_path}: {e}")
            return {
                'image_path': str(image_path),
                'error': str(e),
                'status': 'error'
            }
    
    def predict_batch(
        self,
        image_paths: List[str],
        batch_size: int = 32,
        return_probabilities: bool = True
    ) -> List[Dict]:
        """
        Make predictions on a batch of images.
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction results
        """
        results = []
        
        # Process in batches
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
            batch_paths = image_paths[i:i + batch_size]
            batch_tensors = []
            valid_indices = []
            
            # Preprocess batch
            for j, image_path in enumerate(batch_paths):
                try:
                    image_tensor = self._preprocess_image(image_path)
                    batch_tensors.append(image_tensor)
                    valid_indices.append(j)
                except Exception as e:
                    logger.error(f"Error preprocessing {image_path}: {e}")
                    results.append({
                        'image_path': str(image_path),
                        'error': str(e),
                        'status': 'error'
                    })
            
            if not batch_tensors:
                continue
            
            # Stack tensors and move to device
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Make predictions
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidences, predicted_classes = torch.max(probabilities, 1)
                
                probabilities_np = probabilities.cpu().numpy()
                confidences_np = confidences.cpu().numpy()
                predicted_classes_np = predicted_classes.cpu().numpy()
            
            # Process results
            for j, (prob, conf, pred_class) in enumerate(zip(probabilities_np, confidences_np, predicted_classes_np)):
                original_index = valid_indices[j]
                image_path = batch_paths[original_index]
                
                result = {
                    'image_path': str(image_path),
                    'predicted_class': self.class_names[pred_class],
                    'predicted_class_index': int(pred_class),
                    'confidence': float(conf),
                    'is_confident': conf >= self.confidence_threshold,
                    'status': 'success'
                }
                
                if return_probabilities:
                    result['probabilities'] = prob
                    result['class_probabilities'] = {
                        class_name: float(prob_val) for class_name, prob_val in zip(self.class_names, prob)
                    }
                
                results.append(result)
        
        return results
    
    def predict_directory(
        self,
        directory_path: str,
        output_path: Optional[str] = None,
        batch_size: int = 32,
        extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.dcm', '.dicom')
    ) -> pd.DataFrame:
        """
        Make predictions on all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            output_path: Path to save results CSV (optional)
            batch_size: Batch size for processing
            extensions: Valid image file extensions
            
        Returns:
            DataFrame containing results
        """
        directory_path = Path(directory_path)
        
        # Find all image files
        image_paths = []
        for ext in extensions:
            image_paths.extend(directory_path.glob(f"**/*{ext}"))
            image_paths.extend(directory_path.glob(f"**/*{ext.upper()}"))
        
        logger.info(f"Found {len(image_paths)} images in {directory_path}")
        
        if not image_paths:
            logger.warning("No images found in the specified directory")
            return pd.DataFrame()
        
        # Make predictions
        results = self.predict_batch([str(p) for p in image_paths], batch_size)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results if requested
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        
        return df
    
    def visualize_prediction(
        self,
        image_path: str,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """
        Visualize prediction results for a single image.
        
        Args:
            image_path: Path to the image file
            save_path: Path to save the visualization (optional)
            figsize: Figure size for the plot
        """
        # Make prediction
        result = self.predict_single(image_path, return_probabilities=True)
        
        if result['status'] == 'error':
            logger.error(f"Cannot visualize due to prediction error: {result['error']}")
            return
        
        # Load and display image
        if str(image_path).lower().endswith(('.dcm', '.dicom')):
            try:
                import pydicom
                dicom_data = pydicom.dcmread(image_path)
                image = dicom_data.pixel_array
                if hasattr(dicom_data, 'PhotometricInterpretation'):
                    if dicom_data.PhotometricInterpretation == 'MONOCHROME1':
                        image = np.max(image) - image
            except:
                image = np.zeros((512, 512))
        else:
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                pil_image = Image.open(image_path).convert('L')
                image = np.array(pil_image)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Display image
        ax1.imshow(image, cmap='gray')
        ax1.set_title(f"Input Image\n{Path(image_path).name}")
        ax1.axis('off')
        
        # Display prediction probabilities
        probabilities = result['probabilities']
        colors = ['green' if i == result['predicted_class_index'] else 'lightblue' 
                 for i in range(len(self.class_names))]
        
        bars = ax2.bar(self.class_names, probabilities, color=colors)
        ax2.set_title(f"Prediction: {result['predicted_class']}\n"
                     f"Confidence: {result['confidence']:.3f}")
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Add confidence threshold line
        ax2.axhline(y=self.confidence_threshold, color='red', linestyle='--', 
                   label=f'Confidence Threshold ({self.confidence_threshold})')
        ax2.legend()
        
        # Rotate x-axis labels if needed
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Add probability values on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, results: List[Dict]) -> Dict[str, Union[int, float]]:
        """
        Generate a summary report from prediction results.
        
        Args:
            results: List of prediction results
            
        Returns:
            Summary statistics
        """
        successful_results = [r for r in results if r['status'] == 'success']
        
        if not successful_results:
            return {'error': 'No successful predictions to analyze'}
        
        # Count predictions by class
        class_counts = {}
        confident_counts = {}
        total_confident = 0
        
        for result in successful_results:
            predicted_class = result['predicted_class']
            is_confident = result['is_confident']
            
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            
            if is_confident:
                confident_counts[predicted_class] = confident_counts.get(predicted_class, 0) + 1
                total_confident += 1
        
        # Calculate statistics
        total_predictions = len(successful_results)
        confidence_rate = total_confident / total_predictions if total_predictions > 0 else 0
        
        # Average confidence by class
        avg_confidence_by_class = {}
        for class_name in self.class_names:
            class_results = [r for r in successful_results if r['predicted_class'] == class_name]
            if class_results:
                avg_confidence_by_class[class_name] = np.mean([r['confidence'] for r in class_results])
        
        report = {
            'total_images': len(results),
            'successful_predictions': total_predictions,
            'failed_predictions': len(results) - total_predictions,
            'confidence_rate': confidence_rate,
            'class_distribution': class_counts,
            'confident_predictions_by_class': confident_counts,
            'average_confidence_by_class': avg_confidence_by_class,
            'overall_average_confidence': np.mean([r['confidence'] for r in successful_results])
        }
        
        return report


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Healthcare AI Inference")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save results (CSV for batch, image for single)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for processing multiple images')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cpu, cuda, or auto)')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualization for single image prediction')
    parser.add_argument('--report', action='store_true',
                       help='Generate summary report')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    try:
        inference_engine = MedicalImageInference(
            model_path=args.model,
            config_path=args.config,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize inference engine: {e}")
        sys.exit(1)
    
    input_path = Path(args.input)
    
    # Single image prediction
    if input_path.is_file():
        logger.info(f"Processing single image: {input_path}")
        
        result = inference_engine.predict_single(str(input_path))
        
        if result['status'] == 'success':
            logger.info(f"Prediction: {result['predicted_class']}")
            logger.info(f"Confidence: {result['confidence']:.3f}")
            logger.info(f"Confident: {result['is_confident']}")
            
            if 'class_probabilities' in result:
                logger.info("Class probabilities:")
                for class_name, prob in result['class_probabilities'].items():
                    logger.info(f"  {class_name}: {prob:.3f}")
        else:
            logger.error(f"Prediction failed: {result['error']}")
        
        # Visualization
        if args.visualize:
            inference_engine.visualize_prediction(str(input_path), args.output)
    
    # Directory processing
    elif input_path.is_dir():
        logger.info(f"Processing directory: {input_path}")
        
        results_df = inference_engine.predict_directory(
            str(input_path),
            output_path=args.output,
            batch_size=args.batch_size
        )
        
        if not results_df.empty:
            logger.info(f"Processed {len(results_df)} images")
            
            # Show summary
            successful = results_df[results_df['status'] == 'success']
            if len(successful) > 0:
                logger.info("Class distribution:")
                class_dist = successful['predicted_class'].value_counts()
                for class_name, count in class_dist.items():
                    logger.info(f"  {class_name}: {count}")
                
                avg_confidence = successful['confidence'].mean()
                logger.info(f"Average confidence: {avg_confidence:.3f}")
            
            # Generate detailed report
            if args.report:
                results_list = results_df.to_dict('records')
                report = inference_engine.generate_report(results_list)
                
                logger.info("\n=== DETAILED REPORT ===")
                for key, value in report.items():
                    if isinstance(value, dict):
                        logger.info(f"{key}:")
                        for sub_key, sub_value in value.items():
                            logger.info(f"  {sub_key}: {sub_value}")
                    else:
                        logger.info(f"{key}: {value}")
        else:
            logger.warning("No results to display")
    
    else:
        logger.error(f"Input path does not exist: {input_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()