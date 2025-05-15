"""
Ensemble model combining multiple approaches for coral bleaching prediction.
Integrates CNN-LSTM, Transformer, TCN, and XGBoost models.

This model incorporates:
- Weighted ensemble of multiple base models
- Model confidence estimation
- Uncertainty quantification
- Feature importance analysis
- Early warning signal detection
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import polars as plrs
from typing import Tuple, Dict, List, Optional, Union, Any
import pytorch_lightning as pl
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt

# Import base models
from cnn_lstm_attention import CoralNet as CNNLSTMModel
from models.vit import DualTransformerModel
from models.tcn import TCNCoralModel
from xgboost_model import XGBoostCoralModel, FeatureExtractor


class CoralDataset(Dataset):
    """Dataset class for coral bleaching prediction."""
    
    def __init__(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        wavelet_features: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
        transform = None
    ):
        """
        Initialize coral dataset.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            wavelet_features: Optional wavelet features of shape [num_samples, feature_dim]
            labels: Optional labels of shape [num_samples]
            transform: Transformations to apply to images
        """
        # Convert to PyTorch tensors
        # Ensure images are in the correct format (N, C, H, W)
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.time_series = torch.FloatTensor(time_series)
        
        if wavelet_features is not None:
            self.wavelet_features = torch.FloatTensor(wavelet_features)
        else:
            self.wavelet_features = None
        
        if labels is not None:
            self.labels = torch.FloatTensor(labels)
        else:
            self.labels = None
        
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, time_series, [wavelet_features], [label])
        """
        image = self.images[idx]
        time_series = self.time_series[idx]
        
        if self.transform:
            image = self.transform(image)
        
        if self.wavelet_features is not None:
            wavelet = self.wavelet_features[idx]
            if self.labels is not None:
                label = self.labels[idx]
                return image, time_series, wavelet, label
            else:
                return image, time_series, wavelet
        else:
            if self.labels is not None:
                label = self.labels[idx]
                return image, time_series, label
            else:
                return image, time_series


class WaveletExtractor:
    """
    Extracts wavelet features from time series data.
    
    Used to preprocess data for models that require wavelet features.
    """
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize wavelet extractor.
        
        Args:
            wavelet: Wavelet type
            level: Decomposition level
        """
        self.wavelet = wavelet
        self.level = level
    
    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features from time series data.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Wavelet features of shape [num_samples, feature_dim]
        """
        # Import pywt here to avoid circular import
        import pywt
        
        num_samples, time_steps, num_features = time_series.shape
        
        # Container for wavelet features
        all_features = []
        
        for i in range(num_samples):
            sample_features = []
            
            for f in range(num_features):
                # Get time series for this feature
                ts = time_series[i, :, f]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(ts, self.wavelet, level=self.level)
                
                # Extract statistical features from each coefficient level
                for coeff in coeffs:
                    # Extract statistical features
                    stats = [
                        np.mean(coeff),
                        np.std(coeff),
                        np.max(coeff),
                        np.min(coeff),
                        np.median(coeff),
                        np.sum(np.abs(coeff)),  # Energy
                        np.mean(np.abs(coeff)),  # Mean absolute value
                        np.percentile(coeff, 75) - np.percentile(coeff, 25)  # IQR
                    ]
                    sample_features.extend(stats)
            
            all_features.append(sample_features)
        
        return np.array(all_features)


class EnsembleModel(nn.Module):
    """
    Ensemble model for coral bleaching prediction.
    
    Combines predictions from multiple base models with weighted voting.
    """
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        wavelet_dim: int,
        weights: Optional[Dict[str, float]] = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize ensemble model.
        
        Args:
            time_steps: Number of time steps in time series data
            num_features: Number of features in time series data
            wavelet_dim: Dimension of wavelet features
            weights: Optional dictionary of model weights
            device: Device to use for computation
        """
        super(EnsembleModel, self).__init__()
        
        self.time_steps = time_steps
        self.num_features = num_features
        self.wavelet_dim = wavelet_dim
        self.device = device
        
        # Initialize base models
        self.cnn_lstm = CNNLSTMModel(
            time_steps=time_steps,
            num_features=num_features,
            wavelet_dim=wavelet_dim
        ).to(device)
        
        self.transformer = DualTransformerModel(
            time_steps=time_steps,
            time_features=num_features
        ).to(device)
        
        self.tcn = TCNCoralModel(
            time_steps=time_steps,
            num_features=num_features
        ).to(device)
        
        # XGBoost model is handled separately since it's not a PyTorch model
        self.feature_extractor = FeatureExtractor()
        self.xgboost = XGBoostCoralModel(feature_extractor=self.feature_extractor)
        
        # Default weights if not provided
        if weights is None:
            self.weights = {
                'cnn_lstm': 0.3,
                'transformer': 0.3,
                'tcn': 0.2,
                'xgboost': 0.2
            }
        else:
            # Normalize weights
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
        
        # Track trained models
        self.trained_models = set()
    
    def forward(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through the ensemble model.
        
        Args:
            image: Image tensor of shape [batch_size, channels, height, width]
            time_series: Time series tensor of shape [batch_size, time_steps, num_features]
            wavelet: Optional wavelet features tensor of shape [batch_size, wavelet_dim]
            
        Returns:
            Tuple of (weighted ensemble prediction, individual model predictions)
        """
        # Move inputs to the correct device
        image = image.to(self.device)
        time_series = time_series.to(self.device)
        if wavelet is not None:
            wavelet = wavelet.to(self.device)
        
        # Initialize predictions dictionary
        predictions = {}
        
        # Get predictions from CNN-LSTM if trained
        if 'cnn_lstm' in self.trained_models:
            with torch.no_grad():
                cnn_lstm_pred, _ = self.cnn_lstm(image, time_series, wavelet)
                predictions['cnn_lstm'] = torch.sigmoid(cnn_lstm_pred)
        
        # Get predictions from Transformer if trained
        if 'transformer' in self.trained_models:
            with torch.no_grad():
                transformer_pred, _ = self.transformer(image, time_series)
                predictions['transformer'] = torch.sigmoid(transformer_pred)
        
        # Get predictions from TCN if trained
        if 'tcn' in self.trained_models:
            with torch.no_grad():
                tcn_pred, _ = self.tcn(image, time_series)
                predictions['tcn'] = torch.sigmoid(tcn_pred)
        
        # Get predictions from XGBoost if trained
        if 'xgboost' in self.trained_models:
            # Convert to numpy for XGBoost
            image_np = image.cpu().numpy()
            time_series_np = time_series.cpu().numpy()
            
            # Transpose image back to [batch_size, height, width, channels]
            image_np = np.transpose(image_np, (0, 2, 3, 1))
            
            # Get predictions
            xgb_pred_proba, _ = self.xgboost.predict(image_np, time_series_np)
            
            # Convert back to tensor
            predictions['xgboost'] = torch.FloatTensor(xgb_pred_proba).to(self.device)
        
        # Compute weighted ensemble prediction
        ensemble_pred = torch.zeros_like(next(iter(predictions.values())))
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        return ensemble_pred, predictions
    
    def load_model(self, model_name: str, model_path: str):
        """
        Load a specific model from disk.
        
        Args:
            model_name: Name of the model to load
            model_path: Path to the saved model
        """
        if model_name == 'cnn_lstm':
            self.cnn_lstm.load_state_dict(torch.load(model_path))
            self.cnn_lstm.eval()
            self.trained_models.add(model_name)
        elif model_name == 'transformer':
            self.transformer.load_state_dict(torch.load(model_path))
            self.transformer.eval()
            self.trained_models.add(model_name)
        elif model_name == 'tcn':
            self.tcn.load_state_dict(torch.load(model_path))
            self.tcn.eval()
            self.trained_models.add(model_name)
        elif model_name == 'xgboost':
            self.xgboost.load_model(model_path)
            self.trained_models.add(model_name)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def save_model(self, model_name: str, model_path: str):
        """
        Save a specific model to disk.
        
        Args:
            model_name: Name of the model to save
            model_path: Path to save the model
        """
        if model_name == 'cnn_lstm':
            torch.save(self.cnn_lstm.state_dict(), model_path)
        elif model_name == 'transformer':
            torch.save(self.transformer.state_dict(), model_path)
        elif model_name == 'tcn':
            torch.save(self.tcn.state_dict(), model_path)
        elif model_name == 'xgboost':
            self.xgboost.save_model(model_path)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
    
    def set_model_weights(self, weights: Dict[str, float]):
        """
        Set model weights for ensemble prediction.
        
        Args:
            weights: Dictionary of model weights
        """
        # Normalize weights
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def predict_with_uncertainty(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None,
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            image: Image tensor
            time_series: Time series tensor
            wavelet: Optional wavelet features tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, prediction variances, individual model uncertainties)
        """
        # Move inputs to the correct device
        image = image.to(self.device)
        time_series = time_series.to(self.device)
        if wavelet is not None:
            wavelet = wavelet.to(self.device)
        
        # Initialize uncertainties dictionary
        model_uncertainties = {}
        
        # Get predictions with uncertainty from CNN-LSTM if trained
        if 'cnn_lstm' in self.trained_models:
            self.cnn_lstm.enable_mc_dropout()
            cnn_lstm_mean, cnn_lstm_var = self.cnn_lstm.predict_with_uncertainty(
                image, time_series, wavelet, n_samples
            )
            model_uncertainties['cnn_lstm'] = cnn_lstm_var
            self.cnn_lstm.disable_mc_dropout()
        
        # Get predictions with uncertainty from Transformer if trained
        if 'transformer' in self.trained_models:
            self.transformer.enable_mc_dropout()
            transformer_mean, transformer_var = self.transformer.predict_with_uncertainty(
                image, time_series, n_samples
            )
            model_uncertainties['transformer'] = transformer_var
            self.transformer.disable_mc_dropout()
        
        # Get predictions with uncertainty from TCN if trained
        if 'tcn' in self.trained_models:
            self.tcn.enable_mc_dropout()
            tcn_mean, tcn_var = self.tcn.predict_with_uncertainty(
                image, time_series, n_samples
            )
            model_uncertainties['tcn'] = tcn_var
            self.tcn.disable_mc_dropout()
        
        # Get standard predictions from XGBoost if trained
        if 'xgboost' in self.trained_models:
            # XGBoost doesn't support MC dropout, so we use standard predictions
            # Convert to numpy for XGBoost
            image_np = image.cpu().numpy()
            time_series_np = time_series.cpu().numpy()
            
            # Transpose image back to [batch_size, height, width, channels]
            image_np = np.transpose(image_np, (0, 2, 3, 1))
            
            # Get predictions
            xgb_pred_proba, _ = self.xgboost.predict(image_np, time_series_np)
            
            # Set uncertainty to 0
            model_uncertainties['xgboost'] = torch.zeros_like(
                next(iter(model_uncertainties.values())) if model_uncertainties else 
                torch.FloatTensor(xgb_pred_proba).to(self.device)
            )
        
        # Compute weighted ensemble prediction and uncertainty
        ensemble_pred = torch.zeros_like(next(iter(model_uncertainties.values())))
        ensemble_var = torch.zeros_like(ensemble_pred)
        
        # Get predictions from all models
        _, predictions = self.forward(image, time_series, wavelet)
        
        # Compute weighted mean
        for model_name, pred in predictions.items():
            ensemble_pred += self.weights[model_name] * pred
        
        # Compute weighted variance
        for model_name, var in model_uncertainties.items():
            ensemble_var += (self.weights[model_name] ** 2) * var
        
        return ensemble_pred, ensemble_var, model_uncertainties
    
    def get_early_warning_signals(
        self, 
        images: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Detect early warning signals of coral bleaching.
        
        Args:
            images: Image tensors of shape [num_samples, time_steps, channels, height, width]
            time_series: Time series tensors of shape [num_samples, time_steps, num_features]
            wavelet: Optional wavelet features tensors
            window_size: Size of rolling window for signal detection
            
        Returns:
            Early warning signals of shape [num_samples, time_steps]
        """
        num_samples, time_steps = time_series.shape[:2]
        
        # Initialize early warning signals
        ews = np.zeros((num_samples, time_steps))
        
        # Process each time step
        for t in range(window_size, time_steps):
            # Get data up to current time step
            current_images = images[:, :t]
            current_time_series = time_series[:, :t]
            if wavelet is not None:
                current_wavelet = wavelet[:, :t]
            
            # Make predictions at each time step
            for i in range(num_samples):
                # Get last window for prediction
                window_images = current_images[i, -window_size:]
                window_time_series = current_time_series[i, -window_size:]
                if wavelet is not None:
                    window_wavelet = current_wavelet[i, -window_size:]
                
                # Make prediction
                with torch.no_grad():
                    if wavelet is not None:
                        pred, _ = self.forward(window_images, window_time_series, window_wavelet)
                    else:
                        pred, _ = self.forward(window_images, window_time_series)
                
                # Store prediction as early warning signal
                ews[i, t] = pred.cpu().numpy()
        
        return ews
    
    def get_feature_importance(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray
    ) -> Dict[str, Dict[str, float]]:
        """
        Get feature importance from different models.
        
        Args:
            images: Image data
            time_series: Time series data
            
        Returns:
            Dictionary of feature importances for each model
        """
        importance_dict = {}
        
        # Get feature importance from XGBoost if trained
        if 'xgboost' in self.trained_models:
            importance_dict['xgboost'] = self.xgboost.feature_importances
        
        # Other models require more complex methods for feature importance
        # Implementation depends on specific requirements
        
        return importance_dict


class EnsembleLightningModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for ensemble coral bleaching model.
    
    Handles training, evaluation, and prediction using multiple base models.
    """
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        wavelet_dim: int,
        weights: Optional[Dict[str, float]] = None,
        learning_rate: float = 0.001,
        use_wavelet: bool = True,
        optimize_weights: bool = True,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize Lightning module.
        
        Args:
            time_steps: Number of time steps in time series data
            num_features: Number of features in time series data
            wavelet_dim: Dimension of wavelet features
            weights: Optional dictionary of model weights
            learning_rate: Learning rate for optimizers
            use_wavelet: Whether to use wavelet features
            optimize_weights: Whether to optimize ensemble weights
            device: Device to use for computation
        """
        super(EnsembleLightningModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize ensemble model
        self.model = EnsembleModel(
            time_steps=time_steps,
            num_features=num_features,
            wavelet_dim=wavelet_dim,
            weights=weights,
            device=device
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Settings
        self.use_wavelet = use_wavelet
        self.optimize_weights = optimize_weights
        self.learning_rate = learning_rate
        
        # Model directory
        self.model_dir = "./models"
        os.makedirs(self.model_dir, exist_ok=True)
        
        # Metrics
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': precision_score,
            'recall': recall_score,
            'f1': f1_score,
            'auc': roc_auc_score
        }
        
        # Ensemble weights (learnable if optimize_weights is True)
        if optimize_weights:
            # Initialize weights as learnable parameters
            initial_weights = torch.tensor(
                [weights[m] if weights and m in weights else 0.25 for m in ['cnn_lstm', 'transformer', 'tcn', 'xgboost']],
                dtype=torch.float32
            )
            self.ensemble_weights = nn.Parameter(initial_weights)
        else:
            self.ensemble_weights = None
    
    def forward(self, image, time_series, wavelet=None):
        """Forward pass through the model."""
        # Get ensemble prediction
        ensemble_pred, individual_preds = self.model(image, time_series, wavelet)
        
        # Use learnable weights if enabled
        if self.optimize_weights and self.ensemble_weights is not None:
            # Apply softmax to ensure weights sum to 1
            weights = F.softmax(self.ensemble_weights, dim=0)
            
            # Compute weighted ensemble prediction
            ensemble_pred = torch.zeros_like(next(iter(individual_preds.values())))
            for i, (model_name, pred) in enumerate(['cnn_lstm', 'transformer', 'tcn', 'xgboost']):
                if model_name in individual_preds:
                    ensemble_pred += weights[i] * individual_preds[model_name]
        
        return ensemble_pred, individual_preds
    
    def train_model(
        self, 
        model_name: str, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10
    ) -> Dict[str, float]:
        """
        Train a specific model.
        
        Args:
            model_name: Name of the model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary of validation metrics
        """
        # Set up model and optimizer
        if model_name == 'cnn_lstm':
            model = self.model.cnn_lstm
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif model_name == 'transformer':
            model = self.model.transformer
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif model_name == 'tcn':
            model = self.model.tcn
            optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
        elif model_name == 'xgboost':
            # XGBoost is trained differently
            return self._train_xgboost(train_loader, val_loader)
        else:
            raise ValueError(f"Unknown model name: {model_name}")
        
        # Training loop
        model.train()
        best_val_loss = float('inf')
        best_metrics = {}
        
        for epoch in range(num_epochs):
            # Training step
            model.train()
            train_loss = 0.0
            
            for batch in train_loader:
                # Get data
                if self.use_wavelet:
                    image, time_series, wavelet, labels = batch
                    # Forward pass
                    if model_name == 'cnn_lstm':
                        outputs, _ = model(image, time_series, wavelet)
                    else:  # Transformer and TCN don't use wavelet features
                        outputs, _ = model(image, time_series)
                else:
                    image, time_series, labels = batch
                    # Forward pass
                    outputs, _ = model(image, time_series)
                
                # Calculate loss
                loss = self.criterion(outputs, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation step
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    # Get data
                    if self.use_wavelet:
                        image, time_series, wavelet, labels = batch
                        # Forward pass
                        if model_name == 'cnn_lstm':
                            outputs, _ = model(image, time_series, wavelet)
                        else:  # Transformer and TCN don't use wavelet features
                            outputs, _ = model(image, time_series)
                    else:
                        image, time_series, labels = batch
                        # Forward pass
                        outputs, _ = model(image, time_series)
                    
                    # Calculate loss
                    loss = self.criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    # Store predictions and labels for metrics calculation
                    val_preds.append(torch.sigmoid(outputs).cpu().numpy())
                    val_labels.append(labels.cpu().numpy())
            
            # Calculate average loss
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            # Calculate metrics
            val_preds = np.concatenate(val_preds)
            val_labels = np.concatenate(val_labels)
            
            metrics = {
                'accuracy': self.metrics['accuracy'](val_labels, (val_preds > 0.5).astype(int)),
                'precision': self.metrics['precision'](val_labels, (val_preds > 0.5).astype(int)),
                'recall': self.metrics['recall'](val_labels, (val_preds > 0.5).astype(int)),
                'f1': self.metrics['f1'](val_labels, (val_preds > 0.5).astype(int)),
                'auc': self.metrics['auc'](val_labels, val_preds)
            }
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Accuracy: {metrics['accuracy']:.4f} | "
                  f"Val AUC: {metrics['auc']:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = metrics
                
                # Save model
                model_path = os.path.join(self.model_dir, f"{model_name}_best.pt")
                if model_name in ['cnn_lstm', 'transformer', 'tcn']:
                    torch.save(model.state_dict(), model_path)
                
                # Mark model as trained
                self.model.trained_models.add(model_name)
        
        # Load best model
        if model_name in ['cnn_lstm', 'transformer', 'tcn']:
            model_path = os.path.join(self.model_dir, f"{model_name}_best.pt")
            model.load_state_dict(torch.load(model_path))
            model.eval()
        
        return best_metrics
    
    def _train_xgboost(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, float]:
        """
        Train the XGBoost model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary of validation metrics
        """
        # Get all training data
        train_images, train_time_series, train_labels = self._get_data_from_loader(train_loader)
        val_images, val_time_series, val_labels = self._get_data_from_loader(val_loader)
        
        # Train XGBoost model
        metrics = self.model.xgboost.train(
            images=train_images, 
            time_series=train_time_series, 
            labels=train_labels
        )
        
        # Mark XGBoost as trained
        self.model.trained_models.add('xgboost')
        
        # Save model
        model_path = os.path.join(self.model_dir, "xgboost_best.json")
        self.model.xgboost.save_model(model_path)
        
        # Evaluate on validation set
        val_pred_proba, val_pred = self.model.xgboost.predict(val_images, val_time_series)
        
        # Calculate metrics
        val_metrics = {
            'accuracy': self.metrics['accuracy'](val_labels, val_pred),
            'precision': self.metrics['precision'](val_labels, val_pred),
            'recall': self.metrics['recall'](val_labels, val_pred),
            'f1': self.metrics['f1'](val_labels, val_pred),
            'auc': self.metrics['auc'](val_labels, val_pred_proba)
        }
        
        return val_metrics
    
    def _get_data_from_loader(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract data from data loader.
        
        Args:
            loader: Data loader
            
        Returns:
            Tuple of (images, time_series, labels)
        """
        images, time_series, labels = [], [], []
        
        for batch in loader:
            if self.use_wavelet:
                img, ts, _, lbl = batch
            else:
                img, ts, lbl = batch
            
            # Convert to numpy and append
            images.append(img.cpu().numpy())
            time_series.append(ts.cpu().numpy())
            labels.append(lbl.cpu().numpy())
        
        # Concatenate
        images = np.concatenate(images)
        time_series = np.concatenate(time_series)
        labels = np.concatenate(labels)
        
        # Transpose images from [N, C, H, W] to [N, H, W, C]
        images = np.transpose(images, (0, 2, 3, 1))
        
        return images, time_series, labels
    
    def optimize_ensemble_weights(self, val_loader: DataLoader, num_epochs: int = 10) -> Dict[str, float]:
        """
        Optimize ensemble weights using validation data.
        
        Args:
            val_loader: Validation data loader
            num_epochs: Number of optimization epochs
            
        Returns:
            Dictionary of optimized weights
        """
        if not self.optimize_weights or self.ensemble_weights is None:
            return self.model.weights
        
        # Set up optimizer
        optimizer = torch.optim.Adam([self.ensemble_weights], lr=0.01)
        
        # Training loop
        best_val_loss = float('inf')
        best_weights = None
        
        for epoch in range(num_epochs):
            val_loss = 0.0
            
            for batch in val_loader:
                # Get data
                if self.use_wavelet:
                    image, time_series, wavelet, labels = batch
                else:
                    image, time_series, labels = batch
                    wavelet = None
                
                # Forward pass
                _, individual_preds = self.model(image, time_series, wavelet)
                
                # Apply softmax to ensure weights sum to 1
                weights = F.softmax(self.ensemble_weights, dim=0)
                
                # Compute weighted ensemble prediction
                ensemble_pred = torch.zeros_like(next(iter(individual_preds.values())))
                for i, model_name in enumerate(['cnn_lstm', 'transformer', 'tcn', 'xgboost']):
                    if model_name in individual_preds:
                        ensemble_pred += weights[i] * individual_preds[model_name]
                
                # Calculate loss
                loss = F.binary_cross_entropy(ensemble_pred, labels)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                val_loss += loss.item()
            
            # Calculate average loss
            val_loss /= len(val_loader)
            
            # Print progress
            print(f"Epoch {epoch+1}/{num_epochs} | Val Loss: {val_loss:.4f} | Weights: {F.softmax(self.ensemble_weights, dim=0).tolist()}")
            
            # Save best weights
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = F.softmax(self.ensemble_weights, dim=0).tolist()
        
        # Update model weights
        optimized_weights = {
            model_name: weight for model_name, weight in 
            zip(['cnn_lstm', 'transformer', 'tcn', 'xgboost'], best_weights)
        }
        self.model.set_model_weights(optimized_weights)
        
        return optimized_weights
    
    def train_all_models(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        num_epochs: int = 10
    ) -> Dict[str, Dict[str, float]]:
        """
        Train all models in the ensemble.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary of validation metrics for each model
        """
        all_metrics = {}
        
        # Train individual models
        for model_name in ['cnn_lstm', 'transformer', 'tcn', 'xgboost']:
            print(f"\nTraining {model_name} model...")
            metrics = self.train_model(model_name, train_loader, val_loader, num_epochs)
            all_metrics[model_name] = metrics
            print(f"{model_name} training complete. Validation metrics: {metrics}")
        
        # Optimize ensemble weights if enabled
        if self.optimize_weights:
            print("\nOptimizing ensemble weights...")
            optimized_weights = self.optimize_ensemble_weights(val_loader)
            print(f"Optimized weights: {optimized_weights}")
        
        return all_metrics
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the ensemble model on test data.
        
        Args:
            test_loader: Test data loader
            
        Returns:
            Dictionary of metrics for each model and the ensemble
        """
        # Initialize metrics
        metrics = {}
        all_labels = []
        predictions = {
            'ensemble': [],
            'cnn_lstm': [],
            'transformer': [],
            'tcn': [],
            'xgboost': []
        }
        
        # Set models to evaluation mode
        self.model.cnn_lstm.eval()
        self.model.transformer.eval()
        self.model.tcn.eval()
        
        # Evaluate on test set
        with torch.no_grad():
            for batch in test_loader:
                # Get data
                if self.use_wavelet:
                    image, time_series, wavelet, labels = batch
                else:
                    image, time_series, labels = batch
                    wavelet = None
                
                # Get predictions
                ensemble_pred, individual_preds = self.model(image, time_series, wavelet)
                
                # Store predictions and labels
                predictions['ensemble'].append(ensemble_pred.cpu().numpy())
                for model_name, pred in individual_preds.items():
                    predictions[model_name].append(pred.cpu().numpy())
                
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate predictions and labels
        all_labels = np.concatenate(all_labels)
        for model_name in predictions:
            if predictions[model_name]:
                predictions[model_name] = np.concatenate(predictions[model_name])
        
        # Calculate metrics for each model and ensemble
        for model_name, preds in predictions.items():
            if len(preds) > 0:
                metrics[model_name] = {
                    'accuracy': self.metrics['accuracy'](all_labels, (preds > 0.5).astype(int)),
                    'precision': self.metrics['precision'](all_labels, (preds > 0.5).astype(int)),
                    'recall': self.metrics['recall'](all_labels, (preds > 0.5).astype(int)),
                    'f1': self.metrics['f1'](all_labels, (preds > 0.5).astype(int)),
                    'auc': self.metrics['auc'](all_labels, preds)
                }
        
        return metrics
    
    def predict(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions using the ensemble model.
        
        Args:
            image: Image tensor
            time_series: Time series tensor
            wavelet: Optional wavelet features tensor
            
        Returns:
            Tuple of (ensemble predictions, individual model predictions)
        """
        return self.model(image, time_series, wavelet)
    
    def predict_with_uncertainty(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None,
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Make predictions with uncertainty estimation.
        
        Args:
            image: Image tensor
            time_series: Time series tensor
            wavelet: Optional wavelet features tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, prediction variances, individual model uncertainties)
        """
        return self.model.predict_with_uncertainty(image, time_series, wavelet, n_samples)
    
    def detect_early_warning_signals(
        self, 
        images: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: Optional[torch.Tensor] = None,
        window_size: int = 5
    ) -> np.ndarray:
        """
        Detect early warning signals of coral bleaching.
        
        Args:
            images: Image tensors
            time_series: Time series tensors
            wavelet: Optional wavelet features tensors
            window_size: Size of rolling window for signal detection
            
        Returns:
            Early warning signals
        """
        return self.model.get_early_warning_signals(images, time_series, wavelet, window_size)
    
    def save_all_models(self, model_dir: Optional[str] = None):
        """
        Save all trained models to disk.
        
        Args:
            model_dir: Directory to save models (defaults to self.model_dir)
        """
        if model_dir is None:
            model_dir = self.model_dir
        
        os.makedirs(model_dir, exist_ok=True)
        
        # Save individual models
        for model_name in self.model.trained_models:
            model_path = os.path.join(model_dir, f"{model_name}.pt")
            self.model.save_model(model_name, model_path)
        
        # Save ensemble weights
        weights_path = os.path.join(model_dir, "ensemble_weights.json")
        with open(weights_path, 'w') as f:
            json.dump(self.model.weights, f)
    
    def load_all_models(self, model_dir: Optional[str] = None):
        """
        Load all models from disk.
        
        Args:
            model_dir: Directory containing saved models (defaults to self.model_dir)
        """
        if model_dir is None:
            model_dir = self.model_dir
        
        # Load individual models if they exist
        for model_name in ['cnn_lstm', 'transformer', 'tcn', 'xgboost']:
            model_path = os.path.join(model_dir, f"{model_name}.pt")
            if os.path.exists(model_path):
                self.model.load_model(model_name, model_path)
                print(f"Loaded {model_name} model.")
        
        # Load ensemble weights if they exist
        weights_path = os.path.join(model_dir, "ensemble_weights.json")
        if os.path.exists(weights_path):
            with open(weights_path, 'r') as f:
                weights = json.load(f)
            self.model.set_model_weights(weights)
            print(f"Loaded ensemble weights: {weights}")


def load_data_polars(data_dir: str) -> Tuple[plrs.DataFrame, plrs.DataFrame, plrs.DataFrame]:
    """
    Load and process coral dataset using Polars.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        Tuple of (images DataFrame, time series DataFrame, labels DataFrame)
    """
    # Load image metadata
    image_df = plrs.read_csv(os.path.join(data_dir, 'processed/imagery_features/metadata.csv'))
    
    # Load time series data
    timeseries_df = plrs.read_csv(os.path.join(data_dir, 'processed/time_series_features/timeseries.csv'))
    
    # Load labels
    labels_df = plrs.read_csv(os.path.join(data_dir, 'processed/combined_dataset/labels.csv'))
    
    return image_df, timeseries_df, labels_df


def prepare_data(
    images: np.ndarray, 
    time_series: np.ndarray, 
    labels: np.ndarray, 
    use_wavelet: bool = True,
    test_size: float = 0.2,
    val_size: float = 0.2,
    batch_size: int = 32
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Prepare data loaders for training, validation, and testing.
    
    Args:
        images: Image data of shape [num_samples, height, width, channels]
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Labels of shape [num_samples]
        use_wavelet: Whether to extract and use wavelet features
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        batch_size: Batch size for data loaders
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from sklearn.model_selection import train_test_split
    
    # Extract wavelet features if needed
    wavelet_features = None
    if use_wavelet:
        wavelet_extractor = WaveletExtractor()
        wavelet_features = wavelet_extractor.extract_features(time_series)
    
    # Split data into train+val and test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(labels)), 
        test_size=test_size, 
        random_state=42, 
        stratify=labels
    )
    
    # Split train+val into train and val
    train_indices, val_indices = train_test_split(
        train_val_indices, 
        test_size=val_size, 
        random_state=42, 
        stratify=labels[train_val_indices]
    )
    
    # Create datasets
    if use_wavelet:
        train_dataset = CoralDataset(
            images[train_indices], 
            time_series[train_indices], 
            wavelet_features[train_indices], 
            labels[train_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        val_dataset = CoralDataset(
            images[val_indices], 
            time_series[val_indices], 
            wavelet_features[val_indices], 
            labels[val_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        test_dataset = CoralDataset(
            images[test_indices], 
            time_series[test_indices], 
            wavelet_features[test_indices], 
            labels[test_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    else:
        train_dataset = CoralDataset(
            images[train_indices], 
            time_series[train_indices], 
            labels[train_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        val_dataset = CoralDataset(
            images[val_indices], 
            time_series[val_indices], 
            labels[val_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
        
        test_dataset = CoralDataset(
            images[test_indices], 
            time_series[test_indices], 
            labels[test_indices],
            transform=transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    
    # Assume we have loaded the data
    images = np.random.randn(100, 224, 224, 3)  # 100 sample images
    time_series = np.random.randn(100, 24, 8)   # 100 samples, 24 time steps, 8 features
    labels = np.random.randint(0, 2, 100)       # Binary labels
    
    # Prepare data
    train_loader, val_loader, test_loader = prepare_data(
        images, time_series, labels, use_wavelet=True
    )
    
    # Get wavelet dimension from data (for example purposes)
    wavelet_extractor = WaveletExtractor()
    wavelet_features = wavelet_extractor.extract_features(time_series)
    wavelet_dim = wavelet_features.shape[1]
    
    # Initialize ensemble model
    ensemble_model = EnsembleLightningModel(
        time_steps=24,
        num_features=8,
        wavelet_dim=wavelet_dim,
        weights={'cnn_lstm': 0.3, 'transformer': 0.3, 'tcn': 0.2, 'xgboost': 0.2},
        use_wavelet=True,
        optimize_weights=True
    )
    
    # Train a specific model (for demonstration)
    print("Training CNN-LSTM model...")
    metrics = ensemble_model.train_model('cnn_lstm', train_loader, val_loader, num_epochs=2)
    print(f"CNN-LSTM validation metrics: {metrics}")
    
    # Or train all models (commented out to save time)
    # print("Training all models...")
    # all_metrics = ensemble_model.train_all_models(train_loader, val_loader, num_epochs=2)
    # print(f"All models validation metrics: {all_metrics}")
    
    # Make predictions on a sample
    batch = next(iter(test_loader))
    if ensemble_model.use_wavelet:
        image, time_series, wavelet, labels = batch
        ensemble_pred, individual_preds = ensemble_model.predict(image[:1], time_series[:1], wavelet[:1])
    else:
        image, time_series, labels = batch
        ensemble_pred, individual_preds = ensemble_model.predict(image[:1], time_series[:1])
    
    print(f"Ensemble prediction: {ensemble_pred}")
    print(f"Individual model predictions: {individual_preds}")
    
    # Save trained models
    ensemble_model.save_all_models()
    
    # Demonstrate uncertainty estimation
    if 'cnn_lstm' in ensemble_model.model.trained_models:
        print("Predicting with uncertainty...")
        if ensemble_model.use_wavelet:
            mean_pred, var_pred, model_uncertainties = ensemble_model.predict_with_uncertainty(
                image[:1], time_series[:1], wavelet[:1]
            )
        else:
            mean_pred, var_pred, model_uncertainties = ensemble_model.predict_with_uncertainty(
                image[:1], time_series[:1]
            )
        
        print(f"Mean prediction: {mean_pred}")
        print(f"Prediction variance: {var_pred}")