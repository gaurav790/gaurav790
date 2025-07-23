"""
Evaluation Script for Healthcare AI - Chest X-ray Pneumonia Detection
Provides comprehensive evaluation metrics and visualizations.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, roc_curve, auc, roc_auc_score, 
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Import our modules
from model import ModelManager, MedicalImageClassifier
from data_loader import DataLoaderManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculates and visualizes comprehensive evaluation metrics."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
        
    def calculate_basic_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate basic classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Dictionary of metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class and averaged metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        }
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]
        
        return metrics
    
    def calculate_auc_metrics(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate AUC-related metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary of AUC metrics
        """
        metrics = {}
        
        try:
            # Multi-class AUC (one-vs-rest)
            auc_macro = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            auc_weighted = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            
            metrics['auc_macro'] = auc_macro
            metrics['auc_weighted'] = auc_weighted
            
            # Per-class AUC
            y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
            
            for i, class_name in enumerate(self.class_names):
                if self.num_classes == 2:
                    # Binary classification
                    class_auc = roc_auc_score(y_true, y_proba[:, 1])
                else:
                    # Multi-class: one-vs-rest for each class
                    class_auc = roc_auc_score(y_true_binarized[:, i], y_proba[:, i])
                
                metrics[f'auc_{class_name}'] = class_auc
                
        except ValueError as e:
            logger.warning(f"Could not calculate AUC metrics: {e}")
            metrics['auc_macro'] = 0.0
            metrics['auc_weighted'] = 0.0
        
        return metrics
    
    def plot_confusion_matrix(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        Plot confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Counts)')
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')
        
        # Plot percentages
        sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues',
                   xticklabels=self.class_names, yticklabels=self.class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (Percentages)')
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig
    
    def plot_roc_curves(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot ROC curves for each class.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize labels for multi-class ROC
        y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot ROC curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if self.num_classes == 2 and i == 0:
                # Skip first class in binary classification
                continue
                
            if self.num_classes == 2:
                # Binary classification
                fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
                roc_auc = auc(fpr, tpr)
                label = f'{class_name} (AUC = {roc_auc:.3f})'
            else:
                # Multi-class: one-vs-rest
                fpr, tpr, _ = roc_curve(y_true_binarized[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                label = f'{class_name} (AUC = {roc_auc:.3f})'
            
            ax.plot(fpr, tpr, color=color, linewidth=2, label=label)
        
        # Plot diagonal line
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curves')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        return fig
    
    def plot_precision_recall_curves(
        self, 
        y_true: np.ndarray, 
        y_proba: np.ndarray,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """
        Plot Precision-Recall curves for each class.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Binarize labels for multi-class PR curves
        y_true_binarized = label_binarize(y_true, classes=range(self.num_classes))
        
        # Plot PR curve for each class
        colors = plt.cm.Set1(np.linspace(0, 1, self.num_classes))
        
        for i, (class_name, color) in enumerate(zip(self.class_names, colors)):
            if self.num_classes == 2 and i == 0:
                continue
                
            if self.num_classes == 2:
                # Binary classification
                precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
                avg_precision = average_precision_score(y_true, y_proba[:, 1])
                label = f'{class_name} (AP = {avg_precision:.3f})'
            else:
                # Multi-class: one-vs-rest
                precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], y_proba[:, i])
                avg_precision = average_precision_score(y_true_binarized[:, i], y_proba[:, i])
                label = f'{class_name} (AP = {avg_precision:.3f})'
            
            ax.plot(recall, precision, color=color, linewidth=2, label=label)
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall Curves')
        ax.legend(loc="lower left")
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Precision-Recall curves saved to {save_path}")
        
        return fig
    
    def plot_class_distribution(
        self, 
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> plt.Figure:
        """
        Plot class distribution comparison.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels (optional)
            save_path: Path to save the plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 2 if y_pred is not None else 1, figsize=figsize)
        if y_pred is None:
            axes = [axes]
        
        # True distribution
        true_counts = np.bincount(y_true, minlength=self.num_classes)
        true_percentages = true_counts / len(y_true) * 100
        
        bars1 = axes[0].bar(self.class_names, true_counts, color='skyblue', alpha=0.7)
        axes[0].set_title('True Class Distribution')
        axes[0].set_ylabel('Count')
        
        # Add percentage labels
        for bar, pct in zip(bars1, true_percentages):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{pct:.1f}%', ha='center', va='bottom')
        
        # Predicted distribution (if provided)
        if y_pred is not None:
            pred_counts = np.bincount(y_pred, minlength=self.num_classes)
            pred_percentages = pred_counts / len(y_pred) * 100
            
            bars2 = axes[1].bar(self.class_names, pred_counts, color='lightcoral', alpha=0.7)
            axes[1].set_title('Predicted Class Distribution')
            axes[1].set_ylabel('Count')
            
            # Add percentage labels
            for bar, pct in zip(bars2, pred_percentages):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{pct:.1f}%', ha='center', va='bottom')
        
        # Rotate x-axis labels if needed
        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Class distribution plot saved to {save_path}")
        
        return fig
    
    def generate_classification_report(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        save_path: Optional[str] = None
    ) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the report
            
        Returns:
            Classification report as string
        """
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Classification report saved to {save_path}")
        
        return report


class ModelEvaluator:
    """Comprehensive model evaluation pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator with configuration."""
        self.config = self._load_config(config_path)
        self.data_config = self.config.get('data', {})
        self.evaluation_config = self.config.get('evaluation', {})
        
        # Set up device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.data_manager = DataLoaderManager(config_path)
        self.model_manager = ModelManager(config_path)
        
        # Class information
        self.class_names = self.data_config.get('class_names', ['Normal', 'Bacterial_Pneumonia', 'Viral_Pneumonia'])
        self.metrics_calculator = MetricsCalculator(self.class_names)
        
        logger.info(f"ModelEvaluator initialized with device: {self.device}")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    
    def evaluate_model(
        self, 
        model_path: str,
        data_split: str = 'test',
        output_dir: str = 'evaluation_results'
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            model_path: Path to trained model checkpoint
            data_split: Data split to evaluate on ('test', 'val')
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary containing all evaluation results
        """
        logger.info(f"Starting model evaluation on {data_split} set")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Load model
        model, checkpoint_info = self.model_manager.load_model(model_path, str(self.device))
        model.eval()
        
        # Load data
        dataloaders = self.data_manager.load_dataset_from_folders()
        
        if data_split not in dataloaders:
            logger.error(f"Data split '{data_split}' not found. Available: {list(dataloaders.keys())}")
            return {}
        
        dataloader = dataloaders[data_split]
        logger.info(f"Evaluating on {len(dataloader.dataset)} images")
        
        # Collect predictions
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = model(images)
                probabilities = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        y_true = np.array(all_labels)
        y_pred = np.array(all_predictions)
        y_proba = np.array(all_probabilities)
        
        # Calculate metrics
        logger.info("Calculating metrics...")
        basic_metrics = self.metrics_calculator.calculate_basic_metrics(y_true, y_pred)
        auc_metrics = self.metrics_calculator.calculate_auc_metrics(y_true, y_proba)
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **auc_metrics}
        
        # Generate classification report
        classification_report = self.metrics_calculator.generate_classification_report(
            y_true, y_pred, save_path=output_path / "classification_report.txt"
        )
        
        # Create visualizations
        logger.info("Creating visualizations...")
        
        # Confusion matrix
        self.metrics_calculator.plot_confusion_matrix(
            y_true, y_pred, save_path=output_path / "confusion_matrix.png"
        )
        plt.close()
        
        # ROC curves
        self.metrics_calculator.plot_roc_curves(
            y_true, y_proba, save_path=output_path / "roc_curves.png"
        )
        plt.close()
        
        # Precision-Recall curves
        self.metrics_calculator.plot_precision_recall_curves(
            y_true, y_proba, save_path=output_path / "precision_recall_curves.png"
        )
        plt.close()
        
        # Class distribution
        self.metrics_calculator.plot_class_distribution(
            y_true, y_pred, save_path=output_path / "class_distribution.png"
        )
        plt.close()
        
        # Save detailed results
        results = {
            'model_path': model_path,
            'data_split': data_split,
            'num_samples': len(y_true),
            'checkpoint_info': checkpoint_info,
            'metrics': all_metrics,
            'classification_report': classification_report,
            'predictions': {
                'y_true': y_true.tolist(),
                'y_pred': y_pred.tolist(),
                'y_proba': y_proba.tolist()
            }
        }
        
        # Save results to JSON
        import json
        with open(output_path / "evaluation_results.json", 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            json_results = results.copy()
            json_results['predictions']['y_proba'] = y_proba.tolist()
            json.dump(json_results, f, indent=2)
        
        # Save metrics to CSV
        metrics_df = pd.DataFrame([all_metrics])
        metrics_df.to_csv(output_path / "metrics.csv", index=False)
        
        # Print summary
        logger.info("\n=== EVALUATION SUMMARY ===")
        logger.info(f"Accuracy: {all_metrics['accuracy']:.4f}")
        logger.info(f"F1 Score (Macro): {all_metrics['f1_macro']:.4f}")
        logger.info(f"F1 Score (Weighted): {all_metrics['f1_weighted']:.4f}")
        logger.info(f"AUC (Macro): {all_metrics.get('auc_macro', 'N/A')}")
        logger.info(f"AUC (Weighted): {all_metrics.get('auc_weighted', 'N/A')}")
        
        logger.info("\nPer-class metrics:")
        for class_name in self.class_names:
            precision = all_metrics.get(f'precision_{class_name}', 0)
            recall = all_metrics.get(f'recall_{class_name}', 0)
            f1 = all_metrics.get(f'f1_{class_name}', 0)
            support = all_metrics.get(f'support_{class_name}', 0)
            logger.info(f"  {class_name}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}, Support={support}")
        
        logger.info(f"\nResults saved to: {output_path}")
        
        return results


def main():
    """Main function with CLI interface."""
    parser = argparse.ArgumentParser(description="Evaluate Healthcare AI Model")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Data split to evaluate on')
    parser.add_argument('--output', type=str, default='evaluation_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    try:
        evaluator = ModelEvaluator(args.config)
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        sys.exit(1)
    
    # Run evaluation
    try:
        results = evaluator.evaluate_model(
            model_path=args.model,
            data_split=args.split,
            output_dir=args.output
        )
        
        if results:
            logger.info("Evaluation completed successfully!")
        else:
            logger.error("Evaluation failed")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Evaluation failed with error: {str(e)}")
        raise


if __name__ == "__main__":
    main()