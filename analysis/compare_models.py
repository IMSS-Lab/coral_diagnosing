"""
Model comparison module for coral bleaching prediction.

This module compares the performance of different models for the Duke Coral Health project.
It includes functions for:
- Training and evaluating multiple models
- Comparing model performance with statistical tests
- Visualizing learning curves and performance metrics
- Analyzing model robustness and generalization
- Generating comparison reports and visualizations
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve, 
    average_precision_score
)

# Import models
from models.cnn_lstm_attention import CoralNet as CNNLSTMModel
from models.vit import DualTransformerModel 
from models.tcn import TCNCoralModel
from models.xgboost_model import XGBoostCoralModel
from models.ensemble import EnsembleModel


class ModelEvaluator:
    """
    Evaluates and compares multiple models for coral bleaching prediction.
    
    Handles model training, evaluation, and comparison.
    """
    
    def __init__(
        self,
        models_config: Dict[str, Dict[str, Any]],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        results_dir: str = './results/model_comparison'
    ):
        """
        Initialize model evaluator.
        
        Args:
            models_config: Dictionary of model configurations
            device: Device to use for computation
            results_dir: Directory to save results
        """
        self.models_config = models_config
        self.device = device
        self.results_dir = results_dir
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Initialize model instances
        self.models = {}
        self.optimizers = {}
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Initialize models
        for model_name, config in models_config.items():
            print(f"Initializing {model_name} model...")
            if model_name == 'cnn_lstm':
                self.models[model_name] = CNNLSTMModel(**config['model_params']).to(device)
                self.optimizers[model_name] = torch.optim.AdamW(
                    self.models[model_name].parameters(), 
                    **config.get('optimizer_params', {'lr': 0.001})
                )
            elif model_name == 'transformer':
                self.models[model_name] = DualTransformerModel(**config['model_params']).to(device)
                self.optimizers[model_name] = torch.optim.AdamW(
                    self.models[model_name].parameters(), 
                    **config.get('optimizer_params', {'lr': 0.001})
                )
            elif model_name == 'tcn':
                self.models[model_name] = TCNCoralModel(**config['model_params']).to(device)
                self.optimizers[model_name] = torch.optim.AdamW(
                    self.models[model_name].parameters(), 
                    **config.get('optimizer_params', {'lr': 0.001})
                )
            elif model_name == 'xgboost':
                self.models[model_name] = XGBoostCoralModel(**config['model_params'])
                # XGBoost doesn't use PyTorch optimizers
            elif model_name == 'ensemble':
                # Ensemble model will be initialized after training individual models
                pass
            else:
                raise ValueError(f"Unsupported model: {model_name}")
        
        # Metrics history
        self.history = {model_name: {'train_loss': [], 'val_loss': [], 'val_metrics': {}} for model_name in models_config}
    
    def train_model(
        self, 
        model_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10, 
        wavelet_features: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Dictionary of training history
        """
        print(f"Training {model_name} model...")
        
        # Check if model exists
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized!")
        
        # Handle XGBoost separately
        if model_name == 'xgboost':
            return self._train_xgboost(model_name, train_loader, val_loader, wavelet_features)
        
        model = self.models[model_name]
        optimizer = self.optimizers[model_name]
        history = {'train_loss': [], 'val_loss': [], 'val_metrics': {}}
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
                # Get data
                if wavelet_features:
                    images, timeseries, wavelet, labels = batch
                    images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    if model_name == 'cnn_lstm':
                        outputs, _ = model(images, timeseries, wavelet)
                    else:  # transformer and tcn don't use wavelet features
                        outputs, _ = model(images, timeseries)
                else:
                    images, timeseries, labels = batch
                    images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs, _ = model(images, timeseries)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Calculate average training loss
            train_loss /= len(train_loader)
            history['train_loss'].append(train_loss)
            
            # Validation phase
            val_loss, val_metrics = self.evaluate_model(model_name, val_loader, wavelet_features)
            history['val_loss'].append(val_loss)
            
            # Store validation metrics
            if epoch == 0:
                history['val_metrics'] = {metric: [value] for metric, value in val_metrics.items()}
            else:
                for metric, value in val_metrics.items():
                    history['val_metrics'][metric].append(value)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_metrics['accuracy']:.4f}")
        
        # Update history
        self.history[model_name] = history
        
        # Save model
        self._save_model(model_name)
        
        # Save history
        history_path = os.path.join(self.results_dir, f"{model_name}_history.csv")
        self._save_history(history, history_path)
        
        return history
    
    def _train_xgboost(
        self, 
        model_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader,
        wavelet_features: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train XGBoost model.
        
        Args:
            model_name: Model name
            train_loader: Training data loader
            val_loader: Validation data loader
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Dictionary of training history
        """
        print(f"Training {model_name} model (XGBoost)...")
        
        # Extract data from loaders
        train_images, train_timeseries, train_labels = self._extract_data_from_loader(train_loader, wavelet_features)
        val_images, val_timeseries, val_labels = self._extract_data_from_loader(val_loader, wavelet_features)
        
        # Train XGBoost model
        xgb_model = self.models[model_name]
        
        # Train model
        xgb_model.train(
            images=train_images,
            time_series=train_timeseries,
            labels=train_labels,
            use_cv=True
        )
        
        # Evaluate on validation set
        val_pred_proba, val_pred = xgb_model.predict(val_images, val_timeseries)
        
        # Calculate metrics
        val_metrics = {
            'accuracy': accuracy_score(val_labels, val_pred),
            'precision': precision_score(val_labels, val_pred),
            'recall': recall_score(val_labels, val_pred),
            'f1': f1_score(val_labels, val_pred),
            'auc': roc_auc_score(val_labels, val_pred_proba)
        }
        
        # Create history
        history = {
            'train_loss': [0.0],  # XGBoost doesn't provide loss values directly
            'val_loss': [0.0],    # XGBoost doesn't provide loss values directly
            'val_metrics': {metric: [value] for metric, value in val_metrics.items()}
        }
        
        # Update history
        self.history[model_name] = history
        
        # Save model
        model_path = os.path.join(self.results_dir, f"{model_name}_model.json")
        xgb_model.save_model(model_path)
        
        # Save history
        history_path = os.path.join(self.results_dir, f"{model_name}_history.csv")
        self._save_history(history, history_path)
        
        print(f"XGBoost training complete. Validation Accuracy: {val_metrics['accuracy']:.4f}")
        
        return history
    
    def _extract_data_from_loader(self, loader: DataLoader, wavelet_features: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data from data loader.
        
        Args:
            loader: Data loader
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Tuple of (images, timeseries, labels)
        """
        images, timeseries, labels = [], [], []
        
        for batch in loader:
            if wavelet_features:
                img, ts, _, lbl = batch
            else:
                img, ts, lbl = batch
            
            # Convert to numpy and append
            images.append(img.cpu().numpy())
            timeseries.append(ts.cpu().numpy())
            labels.append(lbl.cpu().numpy())
        
        # Concatenate
        images = np.concatenate(images)
        timeseries = np.concatenate(timeseries)
        labels = np.concatenate(labels)
        
        # Transpose images from [N, C, H, W] to [N, H, W, C]
        images = np.transpose(images, (0, 2, 3, 1))
        
        return images, timeseries, labels
    
    def evaluate_model(
        self, 
        model_name: str, 
        val_loader: DataLoader, 
        wavelet_features: bool = True
    ) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on validation data.
        
        Args:
            model_name: Name of the model to evaluate
            val_loader: Validation data loader
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Tuple of (validation loss, validation metrics)
        """
        # Check if model exists
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized!")
        
        # Handle XGBoost separately
        if model_name == 'xgboost':
            val_images, val_timeseries, val_labels = self._extract_data_from_loader(val_loader, wavelet_features)
            val_pred_proba, val_pred = self.models[model_name].predict(val_images, val_timeseries)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(val_labels, val_pred),
                'precision': precision_score(val_labels, val_pred),
                'recall': recall_score(val_labels, val_pred),
                'f1': f1_score(val_labels, val_pred),
                'auc': roc_auc_score(val_labels, val_pred_proba)
            }
            
            return 0.0, metrics  # XGBoost doesn't provide loss values directly
        
        model = self.models[model_name]
        model.eval()
        
        val_loss = 0.0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Get data
                if wavelet_features:
                    images, timeseries, wavelet, labels = batch
                    images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    if model_name == 'cnn_lstm':
                        outputs, _ = model(images, timeseries, wavelet)
                    else:  # transformer and tcn don't use wavelet features
                        outputs, _ = model(images, timeseries)
                else:
                    images, timeseries, labels = batch
                    images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs, _ = model(images, timeseries)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                # Store predictions and labels
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Concatenate predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
            'precision': precision_score(all_labels, (all_preds > 0.5).astype(int)),
            'recall': recall_score(all_labels, (all_preds > 0.5).astype(int)),
            'f1': f1_score(all_labels, (all_preds > 0.5).astype(int)),
            'auc': roc_auc_score(all_labels, all_preds)
        }
        
        return val_loss, metrics
    
    def test_model(
        self, 
        model_name: str, 
        test_loader: DataLoader, 
        wavelet_features: bool = True
    ) -> Dict[str, float]:
        """
        Test model on test data.
        
        Args:
            model_name: Name of the model to test
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Dictionary of test metrics
        """
        print(f"Testing {model_name} model...")
        
        # Check if model exists
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not initialized!")
        
        # Handle XGBoost separately
        if model_name == 'xgboost':
            test_images, test_timeseries, test_labels = self._extract_data_from_loader(test_loader, wavelet_features)
            test_pred_proba, test_pred = self.models[model_name].predict(test_images, test_timeseries)
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(test_labels, test_pred),
                'precision': precision_score(test_labels, test_pred),
                'recall': recall_score(test_labels, test_pred),
                'f1': f1_score(test_labels, test_pred),
                'auc': roc_auc_score(test_labels, test_pred_proba)
            }
            
            return metrics
        
        model = self.models[model_name]
        model.eval()
        
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Testing {model_name}"):
                # Get data
                if wavelet_features:
                    images, timeseries, wavelet, labels = batch
                    images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    if model_name == 'cnn_lstm':
                        outputs, _ = model(images, timeseries, wavelet)
                    else:  # transformer and tcn don't use wavelet features
                        outputs, _ = model(images, timeseries)
                else:
                    images, timeseries, labels = batch
                    images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs, _ = model(images, timeseries)
                
                # Store predictions and labels
                all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate predictions and labels
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, (all_preds > 0.5).astype(int)),
            'precision': precision_score(all_labels, (all_preds > 0.5).astype(int)),
            'recall': recall_score(all_labels, (all_preds > 0.5).astype(int)),
            'f1': f1_score(all_labels, (all_preds > 0.5).astype(int)),
            'auc': roc_auc_score(all_labels, all_preds)
        }
        
        # Save confusion matrix
        cm = confusion_matrix(all_labels, (all_preds > 0.5).astype(int))
        cm_df = pd.DataFrame(cm, index=['Actual Negative', 'Actual Positive'], columns=['Predicted Negative', 'Predicted Positive'])
        cm_df.to_csv(os.path.join(self.results_dir, f"{model_name}_confusion_matrix.csv"))
        
        # Save predictions
        pred_df = pd.DataFrame({
            'true_label': all_labels,
            'prediction': all_preds
        })
        pred_df.to_csv(os.path.join(self.results_dir, f"{model_name}_predictions.csv"), index=False)
        
        return metrics
    
    def train_all_models(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10, 
        wavelet_features: bool = True
    ) -> Dict[str, Dict[str, List[float]]]:
        """
        Train all models.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Dictionary of training histories for all models
        """
        histories = {}
        
        for model_name in self.models:
            if model_name != 'ensemble':  # Skip ensemble model for now
                histories[model_name] = self.train_model(model_name, train_loader, val_loader, num_epochs, wavelet_features)
        
        # Initialize and train ensemble model if it's in the config
        if 'ensemble' in self.models_config:
            print("Training ensemble model...")
            # TODO: Implement ensemble model training
            pass
        
        return histories
    
    def test_all_models(
        self, 
        test_loader: DataLoader, 
        wavelet_features: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Test all models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
            
        Returns:
            Dictionary of test metrics for all models
        """
        results = {}
        
        for model_name in self.models:
            results[model_name] = self.test_model(model_name, test_loader, wavelet_features)
        
        # Save results
        results_df = pd.DataFrame(results).T
        results_df.to_csv(os.path.join(self.results_dir, "model_comparison_results.csv"))
        
        return results
    
    def _save_model(self, model_name: str) -> None:
        """
        Save model to disk.
        
        Args:
            model_name: Name of the model to save
        """
        if model_name == 'xgboost':
            model_path = os.path.join(self.results_dir, f"{model_name}_model.json")
            self.models[model_name].save_model(model_path)
        else:
            model_path = os.path.join(self.results_dir, f"{model_name}_model.pt")
            torch.save(self.models[model_name].state_dict(), model_path)
    
    def _save_history(self, history: Dict[str, Any], history_path: str) -> None:
        """
        Save training history to disk.
        
        Args:
            history: Training history
            history_path: Path to save history
        """
        # Convert history to dataframe
        history_data = {
            'epoch': list(range(1, len(history['train_loss']) + 1)),
            'train_loss': history['train_loss'],
            'val_loss': history['val_loss']
        }
        
        # Add validation metrics
        for metric, values in history['val_metrics'].items():
            history_data[f'val_{metric}'] = values
        
        # Create dataframe and save
        history_df = pd.DataFrame(history_data)
        history_df.to_csv(history_path, index=False)
    
    def plot_learning_curves(self, save_path: Optional[str] = None) -> None:
        """
        Plot learning curves for all models.
        
        Args:
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(12, 8))
        
        for model_name, history in self.history.items():
            if len(history['train_loss']) > 0:
                plt.plot(history['train_loss'], label=f"{model_name} - Train Loss")
                plt.plot(history['val_loss'], label=f"{model_name} - Val Loss")
        
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_metrics_comparison(self, metrics: List[str] = ['accuracy', 'auc'], save_path: Optional[str] = None) -> None:
        """
        Plot metrics comparison for all models.
        
        Args:
            metrics: List of metrics to plot
            save_path: Path to save plot (optional)
        """
        # Create figure with multiple subplots
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 6))
        
        # Handle case with only one metric
        if n_metrics == 1:
            axes = [axes]
        
        # Get models and their final validation metrics
        model_names = []
        model_values = {metric: [] for metric in metrics}
        
        for model_name, history in self.history.items():
            if 'val_metrics' in history and all(metric in history['val_metrics'] for metric in metrics):
                model_names.append(model_name)
                for metric in metrics:
                    model_values[metric].append(history['val_metrics'][metric][-1])
        
        # Plot each metric
        for i, metric in enumerate(metrics):
            axes[i].bar(model_names, model_values[metric])
            axes[i].set_title(f'Validation {metric.capitalize()}')
            axes[i].set_ylim([0, 1])
            for j, v in enumerate(model_values[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, test_loader: DataLoader, wavelet_features: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot ROC curves for all models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        for model_name in self.models:
            # Get predictions
            if model_name == 'xgboost':
                test_images, test_timeseries, test_labels = self._extract_data_from_loader(test_loader, wavelet_features)
                test_pred_proba, _ = self.models[model_name].predict(test_images, test_timeseries)
            else:
                model = self.models[model_name]
                model.eval()
                
                all_labels = []
                all_preds = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        # Get data
                        if wavelet_features:
                            images, timeseries, wavelet, labels = batch
                            images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            if model_name == 'cnn_lstm':
                                outputs, _ = model(images, timeseries, wavelet)
                            else:  # transformer and tcn don't use wavelet features
                                outputs, _ = model(images, timeseries)
                        else:
                            images, timeseries, labels = batch
                            images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            outputs, _ = model(images, timeseries)
                        
                        # Store predictions and labels
                        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                
                # Concatenate predictions and labels
                test_pred_proba = np.concatenate(all_preds)
                test_labels = np.concatenate(all_labels)
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(test_labels, test_pred_proba)
            roc_auc = roc_auc_score(test_labels, test_pred_proba)
            
            # Plot ROC curve
            plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], 'k--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc='lower right')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_pr_curves(self, test_loader: DataLoader, wavelet_features: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot precision-recall curves for all models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
            save_path: Path to save plot (optional)
        """
        plt.figure(figsize=(10, 8))
        
        for model_name in self.models:
            # Get predictions
            if model_name == 'xgboost':
                test_images, test_timeseries, test_labels = self._extract_data_from_loader(test_loader, wavelet_features)
                test_pred_proba, _ = self.models[model_name].predict(test_images, test_timeseries)
            else:
                model = self.models[model_name]
                model.eval()
                
                all_labels = []
                all_preds = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        # Get data
                        if wavelet_features:
                            images, timeseries, wavelet, labels = batch
                            images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            if model_name == 'cnn_lstm':
                                outputs, _ = model(images, timeseries, wavelet)
                            else:  # transformer and tcn don't use wavelet features
                                outputs, _ = model(images, timeseries)
                        else:
                            images, timeseries, labels = batch
                            images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            outputs, _ = model(images, timeseries)
                        
                        # Store predictions and labels
                        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
                        all_labels.append(labels.cpu().numpy())
                
                # Concatenate predictions and labels
                test_pred_proba = np.concatenate(all_preds)
                test_labels = np.concatenate(all_labels)
            
            # Calculate precision-recall curve
            precision, recall, _ = precision_recall_curve(test_labels, test_pred_proba)
            ap = average_precision_score(test_labels, test_pred_proba)
            
            # Plot precision-recall curve
            plt.plot(recall, precision, label=f'{model_name} (AP = {ap:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc='lower left')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrices(self, test_loader: DataLoader, wavelet_features: bool = True, save_path: Optional[str] = None) -> None:
        """
        Plot confusion matrices for all models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
            save_path: Path to save plot (optional)
        """
        # Determine number of models and create figure
        n_models = len(self.models)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
        
        # Handle single model case
        if n_models == 1:
            axes = np.array([axes])
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, (model_name, model) in enumerate(self.models.items()):
            # Get predictions
            if model_name == 'xgboost':
                test_images, test_timeseries, test_labels = self._extract_data_from_loader(test_loader, wavelet_features)
                _, test_pred = self.models[model_name].predict(test_images, test_timeseries)
            else:
                model.eval()
                
                all_labels = []
                all_preds = []
                
                with torch.no_grad():
                    for batch in test_loader:
                        # Get data
                        if wavelet_features:
                            images, timeseries, wavelet, labels = batch
                            images, timeseries, wavelet, labels = images.to(self.device), timeseries.to(self.device), wavelet.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            if model_name == 'cnn_lstm':
                                outputs, _ = model(images, timeseries, wavelet)
                            else:  # transformer and tcn don't use wavelet features
                                outputs, _ = model(images, timeseries)
                        else:
                            images, timeseries, labels = batch
                            images, timeseries, labels = images.to(self.device), timeseries.to(self.device), labels.to(self.device)
                            
                            # Forward pass
                            outputs, _ = model(images, timeseries)
                        
                        # Store predictions and labels
                        all_preds.append((torch.sigmoid(outputs) > 0.5).cpu().numpy().astype(int))
                        all_labels.append(labels.cpu().numpy())
                
                # Concatenate predictions and labels
                test_pred = np.concatenate(all_preds)
                test_labels = np.concatenate(all_labels)
            
            # Calculate confusion matrix
            cm = confusion_matrix(test_labels, test_pred)
            
            # Plot confusion matrix
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
            axes[i].set_title(f'{model_name} Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
            axes[i].set_xticklabels(['Negative', 'Positive'])
            axes[i].set_yticklabels(['Negative', 'Positive'])
        
        # Hide empty subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def generate_comparison_report(self, test_loader: DataLoader, wavelet_features: bool = True) -> None:
        """
        Generate comprehensive comparison report for all models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
        """
        print("Generating model comparison report...")
        
        # Test all models
        test_results = self.test_all_models(test_loader, wavelet_features)
        
        # Plot learning curves
        self.plot_learning_curves(save_path=os.path.join(self.results_dir, 'learning_curves.png'))
        
        # Plot metrics comparison
        self.plot_metrics_comparison(save_path=os.path.join(self.results_dir, 'metrics_comparison.png'))
        
        # Plot ROC curves
        self.plot_roc_curves(test_loader, wavelet_features, save_path=os.path.join(self.results_dir, 'roc_curves.png'))
        
        # Plot PR curves
        self.plot_pr_curves(test_loader, wavelet_features, save_path=os.path.join(self.results_dir, 'pr_curves.png'))
        
        # Plot confusion matrices
        self.plot_confusion_matrices(test_loader, wavelet_features, save_path=os.path.join(self.results_dir, 'confusion_matrices.png'))
        
        # Create change over time CSV
        change_over_time = {}
        
        for model_name, history in self.history.items():
            if len(history['train_loss']) > 0:
                change_over_time[f'{model_name}_train_loss'] = history['train_loss']
                change_over_time[f'{model_name}_val_loss'] = history['val_loss']
                
                for metric, values in history['val_metrics'].items():
                    change_over_time[f'{model_name}_val_{metric}'] = values
        
        # Create dataframe and save
        change_df = pd.DataFrame(change_over_time)
        change_df.to_csv(os.path.join(self.results_dir, 'change_over_time.csv'), index=False)
        
        # Create feature importance CSV
        self._generate_feature_importance_analysis(test_loader, wavelet_features)
        
        print("Model comparison report generated successfully!")
    
    def _generate_feature_importance_analysis(self, test_loader: DataLoader, wavelet_features: bool = True) -> None:
        """
        Generate feature importance analysis for applicable models.
        
        Args:
            test_loader: Test data loader
            wavelet_features: Whether data includes wavelet features
        """
        # Only XGBoost model currently supports built-in feature importance
        if 'xgboost' in self.models:
            print("Analyzing feature importance for XGBoost model...")
            
            # Get feature importance
            feature_importance = self.models['xgboost'].feature_importances
            
            if feature_importance:
                # Create dataframe
                importance_df = pd.DataFrame({
                    'feature': list(feature_importance.keys()),
                    'importance': list(feature_importance.values())
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('importance', ascending=False)
                
                # Save to CSV
                importance_df.to_csv(os.path.join(self.results_dir, 'feature_importance.csv'), index=False)
                
                # Plot top 20 features
                plt.figure(figsize=(12, 8))
                sns.barplot(data=importance_df.head(20), x='importance', y='feature')
                plt.title('Top 20 Feature Importance (XGBoost)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.show()


def compare_models(
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    models_config: Dict[str, Dict[str, Any]],
    num_epochs: int = 10,
    wavelet_features: bool = True,
    results_dir: str = './results/model_comparison'
) -> None:
    """
    Compare multiple models for coral bleaching prediction.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        models_config: Dictionary of model configurations
        num_epochs: Number of training epochs
        wavelet_features: Whether data includes wavelet features
        results_dir: Directory to save results
    """
    # Initialize model evaluator
    evaluator = ModelEvaluator(models_config, results_dir=results_dir)
    
    # Train all models
    evaluator.train_all_models(train_loader, val_loader, num_epochs, wavelet_features)
    
    # Generate comparison report
    evaluator.generate_comparison_report(test_loader, wavelet_features)


if __name__ == "__main__":
    import sys
    sys.path.append('..')  # Add parent directory to path
    import preprocessing
    import feature_engineering
    
    # Load data
    data_dir = "../data/processed"
    train_data, val_data, test_data, _ = preprocessing.load_processed_data(data_dir)
    
    # Create data loaders
    train_loader, val_loader, test_loader = preprocessing.create_data_loaders(
        train_data, val_data, test_data
    )
    
    # Define model configurations
    models_config = {
        'cnn_lstm': {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'num_features': train_data['timeseries'].shape[2],
                'wavelet_dim': 128  # This should match the actual wavelet dimension
            },
            'optimizer_params': {
                'lr': 0.001,
                'weight_decay': 1e-4
            }
        },
        'transformer': {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'time_features': train_data['timeseries'].shape[2],
                'embed_dim': 256,
                'vision_depth': 6,
                'temporal_depth': 4,
                'num_heads': 8
            },
            'optimizer_params': {
                'lr': 0.0005,
                'weight_decay': 1e-5
            }
        },
        'tcn': {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'num_features': train_data['timeseries'].shape[2],
                'tcn_hidden_dims': [64, 128, 256]
            },
            'optimizer_params': {
                'lr': 0.001,
                'weight_decay': 1e-4
            }
        },
        'xgboost': {
            'model_params': {}
        }
    }
    
    # Compare models
    compare_models(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        models_config=models_config,
        num_epochs=10,
        wavelet_features=True
    )