"""
Ensemble model for coral bleaching prediction.
Combines multiple model architectures for improved performance.
"""

import os
import numpy as np
import pandas as pd
import json
import pywt
import pickle
from typing import Tuple, Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pytorch_lightning as pl
# Import torchmetrics instead of pl.metrics
import torchmetrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import cv2

# Import models from other files
from cnn_lstm import CoralNet
from tcn import TCNCoralModel
from vit import DualTransformerModel


class FeatureExtractor:
    """
    Extract features from image and time series data.
    
    Parameters:
        wavelet (str): Wavelet type
        level (int): Decomposition level
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3
    ):
        self.wavelet = wavelet
        self.level = level
    
    def extract_statistical_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from time series.
        
        Args:
            time_series: Time series data of shape [batch_size, seq_len, features]
            
        Returns:
            Statistical features of shape [batch_size, features * num_stats]
        """
        batch_size, seq_len, num_features = time_series.shape
        
        # Define statistical functions
        stat_funcs = [
            np.mean, np.std, np.min, np.max, np.median,
            lambda x: np.percentile(x, 25),  # Q1
            lambda x: np.percentile(x, 75),  # Q3
        ]
        
        num_stats = len(stat_funcs)
        features = np.zeros((batch_size, num_features * num_stats))
        
        for i in range(batch_size):
            for j in range(num_features):
                ts = time_series[i, :, j]
                for k, func in enumerate(stat_funcs):
                    features[i, j * num_stats + k] = func(ts)
        
        return features
    
    def extract_wavelet_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features from time series.
        
        Args:
            time_series: Time series data of shape [batch_size, seq_len, features]
            
        Returns:
            Wavelet features
        """
        batch_size, seq_len, num_features = time_series.shape
        
        # Determine appropriate wavelet level based on sequence length
        max_level = int(np.log2(seq_len))
        adjusted_level = min(self.level, max_level - 1)
        adjusted_level = max(1, adjusted_level)  # Ensure at least level 1
        
        # Calculate feature dimension: 4 features (mean, std, energy, entropy) per level per time series feature
        feature_dim = num_features * (adjusted_level + 1) * 4
        features = np.zeros((batch_size, feature_dim))
        
        for i in range(batch_size):
            feature_idx = 0
            for j in range(num_features):
                ts = time_series[i, :, j]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(ts, self.wavelet, level=adjusted_level)
                
                # Extract features from coefficients
                for coeff in coeffs:
                    # Mean
                    features[i, feature_idx] = np.mean(coeff)
                    feature_idx += 1
                    
                    # Standard deviation
                    features[i, feature_idx] = np.std(coeff)
                    feature_idx += 1
                    
                    # Energy
                    features[i, feature_idx] = np.sum(coeff ** 2)
                    feature_idx += 1
                    
                    # Entropy
                    if np.sum(np.abs(coeff)) > 0:
                        p = np.abs(coeff) / np.sum(np.abs(coeff))
                        features[i, feature_idx] = -np.sum(p * np.log2(p + 1e-10))
                    feature_idx += 1
        
        return features
    
    def extract_all_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract all features from time series.
        
        Args:
            time_series: Time series data of shape [batch_size, seq_len, features]
            
        Returns:
            Combined features
        """
        stat_features = self.extract_statistical_features(time_series)
        wavelet_features = self.extract_wavelet_features(time_series)
        
        return np.hstack([stat_features, wavelet_features])


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model wrapper for ensemble.
    
    Parameters:
        input_channels (int): Number of input channels
        img_size (int): Image size
        time_input_dim (int): Input dimension for time series
        seq_len (int): Sequence length
        cnn_output_dim (int): CNN output dimension
        lstm_hidden_dim (int): LSTM hidden dimension
        lstm_layers (int): Number of LSTM layers
        output_dim (int): Output dimension
        dropout (float): Dropout probability
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        time_input_dim: int = 8,
        seq_len: int = 24,
        cnn_output_dim: int = 256,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        output_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(CNNLSTMModel, self).__init__()
        
        # Create CoralNet model with proper parameter naming (seq_len instead of time_steps)
        self.model = CoralNet(
            input_channels=input_channels,
            img_size=img_size,
            time_input_dim=time_input_dim,
            seq_len=seq_len,  # Fix: use seq_len instead of time_steps
            cnn_backbone='resnet18',
            cnn_output_dim=cnn_output_dim,
            lstm_hidden_dim=lstm_hidden_dim,
            lstm_layers=lstm_layers,
            lstm_output_dim=output_dim * 2,
            wavelet_output_dim=output_dim,
            fusion_output_dim=output_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
    
    def forward(self, images: torch.Tensor, time_series: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            time_series: Input time series
            
        Returns:
            Model output
        """
        outputs = self.model(images, time_series)
        
        # Handle potential attention output
        if isinstance(outputs, tuple):
            logits, _ = outputs
            return logits
        else:
            return outputs


class TCNModel(nn.Module):
    """
    TCN model wrapper for ensemble.
    
    Parameters:
        input_channels (int): Number of input channels
        img_size (int): Image size
        time_input_dim (int): Input dimension for time series
        seq_len (int): Sequence length
        output_dim (int): Output dimension
        hidden_dims (List[int]): Hidden dimensions
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        time_input_dim: int = 8,
        seq_len: int = 24,
        output_dim: int = 64,
        hidden_dims: List[int] = [64, 128, 256],
        dropout: float = 0.3
    ):
        super(TCNModel, self).__init__()
        
        # Create TCN model
        self.model = TCNCoralModel(
            input_channels=input_channels,
            img_size=img_size,
            time_input_dim=time_input_dim,
            sequence_length=seq_len,
            img_feature_dim=output_dim * 4,
            time_feature_dim=output_dim * 2,
            hidden_dims=hidden_dims,
            fusion_hidden_dim=output_dim * 2,
            fusion_output_dim=output_dim,
            cnn_backbone='resnet18',
            kernel_size=3,
            dropout=dropout
        )
    
    def forward(self, images: torch.Tensor, time_series: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            time_series: Input time series
            
        Returns:
            Model output
        """
        outputs = self.model(images, time_series)
        
        # Handle potential attention output
        if isinstance(outputs, tuple):
            logits, _ = outputs
            return logits
        else:
            return outputs


class TransformerModel(nn.Module):
    """
    Transformer model wrapper for ensemble.
    
    Parameters:
        input_channels (int): Number of input channels
        img_size (int): Image size
        time_input_dim (int): Input dimension for time series
        seq_len (int): Sequence length
        output_dim (int): Output dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        time_input_dim: int = 8,
        seq_len: int = 24,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super(TransformerModel, self).__init__()
        
        # Choose dimensions divisible by number of heads
        img_embed_dim = 768  # 12 * 64
        time_embed_dim = 512  # 8 * 64
        fusion_dim = 512  # 8 * 64
        
        # Create transformer model
        self.model = DualTransformerModel(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=input_channels,
            time_input_dim=time_input_dim,
            max_len=seq_len,
            img_embed_dim=img_embed_dim,
            time_embed_dim=time_embed_dim,
            fusion_dim=fusion_dim,
            img_num_heads=12,
            time_num_heads=8,
            fusion_num_heads=8,
            img_ff_dim=img_embed_dim * 4,
            time_ff_dim=time_embed_dim * 4,
            img_num_layers=6,
            time_num_layers=3,
            num_classes=1,  # Binary classification
            dropout=dropout,
            fusion_type='both'
        )
    
    def forward(self, images: torch.Tensor, time_series: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            time_series: Input time series
            
        Returns:
            Model output
        """
        outputs = self.model(images, time_series)
        
        # The transformer returns multiple outputs
        fused_logits, _, _, _, _, _, _ = outputs
        
        return fused_logits


class EnsembleModel(nn.Module):
    """
    Ensemble model combining multiple architectures.
    
    Parameters:
        input_channels (int): Number of input channels
        img_size (int): Image size
        time_input_dim (int): Input dimension for time series
        seq_len (int): Sequence length
        output_dim (int): Output dimension
        dropout (float): Dropout probability
        ensemble_type (str): Ensemble type ('weighted', 'stacking', 'boosting')
        use_cnn_lstm (bool): Whether to use CNN-LSTM model
        use_tcn (bool): Whether to use TCN model
        use_transformer (bool): Whether to use transformer model
        use_ml_models (bool): Whether to use traditional ML models
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        time_input_dim: int = 8,
        seq_len: int = 24,
        output_dim: int = 64,
        dropout: float = 0.3,
        ensemble_type: str = 'weighted',
        use_cnn_lstm: bool = True,
        use_tcn: bool = True,
        use_transformer: bool = True,
        use_ml_models: bool = True
    ):
        super(EnsembleModel, self).__init__()
        
        self.input_channels = input_channels
        self.img_size = img_size
        self.time_input_dim = time_input_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.ensemble_type = ensemble_type
        
        # Initialize models
        self.models = nn.ModuleDict()
        
        if use_cnn_lstm:
            self.cnn_lstm = CNNLSTMModel(
                input_channels=input_channels,
                img_size=img_size,
                time_input_dim=time_input_dim,
                seq_len=seq_len,  # Use seq_len consistently
                output_dim=output_dim,
                dropout=dropout
            )
            self.models['cnn_lstm'] = self.cnn_lstm
        
        if use_tcn:
            self.tcn = TCNModel(
                input_channels=input_channels,
                img_size=img_size,
                time_input_dim=time_input_dim,
                seq_len=seq_len,
                output_dim=output_dim,
                dropout=dropout
            )
            self.models['tcn'] = self.tcn
        
        if use_transformer:
            self.transformer = TransformerModel(
                input_channels=input_channels,
                img_size=img_size,
                time_input_dim=time_input_dim,
                seq_len=seq_len,
                output_dim=output_dim,
                dropout=dropout
            )
            self.models['transformer'] = self.transformer
        
        # Initialize ensemble weights
        if ensemble_type == 'weighted':
            num_models = sum([use_cnn_lstm, use_tcn, use_transformer])
            self.model_weights = nn.Parameter(torch.ones(num_models) / num_models)
        elif ensemble_type == 'stacking':
            # Feature extractor for traditional ML models
            self.feature_extractor = FeatureExtractor()
            
            # Stacking meta-learner
            model_output_dim = len(self.models)
            feature_dim = time_input_dim * 7 + time_input_dim * (min(3, int(np.log2(seq_len))) + 1) * 4
            combined_dim = model_output_dim + feature_dim
            
            self.meta_learner = nn.Sequential(
                nn.Linear(combined_dim, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(32, 1)
            )
        elif ensemble_type == 'boosting':
            # Initialize ML models
            if use_ml_models:
                self.rf_model = None
                self.gb_model = None
                self.xgb_model = None
            
            # Output combiner
            num_models = sum([use_cnn_lstm, use_tcn, use_transformer, use_ml_models])
            self.combiner = nn.Linear(num_models, 1)
    
    def init_ml_models(self, X_train, y_train):
        """
        Initialize and train ML models.
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        if self.ensemble_type != 'boosting':
            return
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.rf_model.fit(X_train, y_train)
        
        # Train Gradient Boosting
        self.gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.gb_model.fit(X_train, y_train)
        
        # Train XGBoost
        self.xgb_model = XGBClassifier(n_estimators=100, random_state=42)
        self.xgb_model.fit(X_train, y_train)
    
    def extract_ml_features(self, time_series):
        """
        Extract features for ML models.
        
        Args:
            time_series: Time series data
            
        Returns:
            Extracted features
        """
        if torch.is_tensor(time_series):
            time_series = time_series.cpu().numpy()
        
        return self.feature_extractor.extract_all_features(time_series)
    
    def predict_ml_models(self, features):
        """
        Get predictions from ML models.
        
        Args:
            features: Input features
            
        Returns:
            Predictions
        """
        if self.rf_model is None or self.gb_model is None or self.xgb_model is None:
            raise ValueError("ML models not initialized. Call init_ml_models first.")
        
        # Get predictions
        rf_preds = self.rf_model.predict_proba(features)[:, 1]
        gb_preds = self.gb_model.predict_proba(features)[:, 1]
        xgb_preds = self.xgb_model.predict_proba(features)[:, 1]
        
        # Combine predictions
        return np.column_stack([rf_preds, gb_preds, xgb_preds])
    
    def forward(
        self, 
        images: torch.Tensor, 
        time_series: torch.Tensor,
        ml_features: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            images: Input images
            time_series: Input time series
            ml_features: Features for ML models
            
        Returns:
            Model output
        """
        # Get predictions from deep learning models
        predictions = []
        for name, model in self.models.items():
            pred = model(images, time_series)
            predictions.append(pred)
        
        # Different ensemble methods
        if self.ensemble_type == 'weighted':
            # Apply softmax to weights
            weights = F.softmax(self.model_weights, dim=0)
            
            # Weighted average of predictions
            weighted_pred = torch.zeros_like(predictions[0])
            for i, pred in enumerate(predictions):
                weighted_pred += weights[i] * pred
            
            return weighted_pred
        
        elif self.ensemble_type == 'stacking':
            # Concatenate model predictions
            model_preds = torch.cat([p.view(p.size(0), -1) for p in predictions], dim=1)
            
            # Extract features from time series
            if ml_features is None:
                ml_features = self.extract_ml_features(time_series)
            ml_features = torch.FloatTensor(ml_features).to(images.device)
            
            # Combine predictions and features
            combined = torch.cat([model_preds, ml_features], dim=1)
            
            # Meta-learner prediction
            return self.meta_learner(combined)
        
        elif self.ensemble_type == 'boosting':
            # Get predictions from ML models
            if ml_features is None:
                ml_features = self.extract_ml_features(time_series)
            
            if hasattr(self, 'rf_model') and self.rf_model is not None:
                ml_preds = self.predict_ml_models(ml_features)
                ml_preds = torch.FloatTensor(ml_preds).to(images.device)
                
                # Concatenate all predictions
                all_preds = torch.cat([
                    torch.stack([p.view(-1) for p in predictions], dim=1),
                    ml_preds
                ], dim=1)
            else:
                # Only use deep learning models
                all_preds = torch.stack([p.view(-1) for p in predictions], dim=1)
            
            # Combine predictions
            return self.combiner(all_preds)
        
        else:
            # Simple averaging
            return torch.mean(torch.stack(predictions), dim=0)


class CoralDataset(Dataset):
    """
    Dataset for coral bleaching prediction.
    
    Parameters:
        image_paths (List[str]): List of image paths
        time_series (np.ndarray): Time series data
        labels (np.ndarray): Labels
        transform (Optional): Image transform
        extract_ml_features (bool): Whether to extract ML features
    """
    
    def __init__(
        self,
        image_paths: List[str],
        time_series: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Any] = None,
        extract_ml_features: bool = False
    ):
        self.image_paths = image_paths
        self.time_series = time_series
        self.labels = labels
        self.transform = transform
        self.extract_ml_features = extract_ml_features
        
        # Extract ML features if needed
        if extract_ml_features:
            self.feature_extractor = FeatureExtractor()
            self.ml_features = self.feature_extractor.extract_all_features(time_series)
        else:
            self.ml_features = None
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        
        # Get time series and label
        time_series = torch.FloatTensor(self.time_series[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        # Return ML features if extracted
        if self.ml_features is not None:
            return {
                'image': image,
                'time_series': time_series,
                'label': label,
                'ml_features': self.ml_features[idx]
            }
        else:
            return {
                'image': image,
                'time_series': time_series,
                'label': label
            }


class EnsembleLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for ensemble model.
    
    Parameters:
        model_params (Dict[str, Any]): Model parameters
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        pos_weight (float): Positive class weight
    """
    
    def __init__(
        self,
        model_params: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 2.0
    ):
        super(EnsembleLightningModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = EnsembleModel(**model_params)
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
        # Metrics
        # Use torchmetrics instead of pl.metrics
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
    
    def forward(self, images, time_series, ml_features=None):
        return self.model(images, time_series, ml_features)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Get ML features if available
        ml_features = batch.get('ml_features', None)
        
        # Forward pass
        logits = self(images, time_series, ml_features)
        
        # Reshape for loss calculation
        logits = logits.view(-1)
        labels = labels.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5
        self.train_acc(preds, labels.long())
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Get ML features if available
        ml_features = batch.get('ml_features', None)
        
        # Forward pass
        logits = self(images, time_series, ml_features)
        
        # Reshape for loss calculation
        logits = logits.view(-1)
        labels = labels.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        
        self.val_acc(preds, labels.long())
        self.val_auroc(probs, labels.long())
        self.val_f1(preds, labels.long())
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        self.log('val_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
        self.log('val_f1', self.val_f1, on_epoch=True, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Get ML features if available
        ml_features = batch.get('ml_features', None)
        
        # Forward pass
        logits = self(images, time_series, ml_features)
        
        # Reshape for loss calculation
        logits = logits.view(-1)
        labels = labels.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        
        self.test_acc(preds, labels.long())
        self.test_auroc(probs, labels.long())
        self.test_f1(preds, labels.long())
        
        # Log metrics
        self.log('test_loss', loss, on_epoch=True)
        self.log('test_acc', self.test_acc, on_epoch=True)
        self.log('test_auroc', self.test_auroc, on_epoch=True)
        self.log('test_f1', self.test_f1, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=True
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }


def get_transforms(img_size=224):
    """
    Get data transforms.
    
    Args:
        img_size (int): Image size
        
    Returns:
        Dictionary of transforms
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def load_data(
    image_dir: str,
    time_series_path: str,
    labels_path: str,
    img_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    extract_ml_features: bool = False
):
    """
    Load data for training.
    
    Args:
        image_dir (str): Directory containing images
        time_series_path (str): Path to time series data
        labels_path (str): Path to labels
        img_size (int): Image size
        batch_size (int): Batch size
        num_workers (int): Number of workers
        extract_ml_features (bool): Whether to extract ML features
        
    Returns:
        Dictionary with data loaders and metadata
    """
    # Load time series data
    time_series = np.load(time_series_path)
    
    # Load labels
    labels = np.load(labels_path)
    
    # Get image paths
    image_paths = sorted([
        os.path.join(image_dir, f)
        for f in os.listdir(image_dir)
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    # Ensure data dimensions match
    assert len(image_paths) == len(labels) == len(time_series), \
        "Mismatch in data dimensions"
    
    # Split data
    train_idx, temp_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.3,
        random_state=42,
        stratify=labels
    )
    
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.5,
        random_state=42,
        stratify=labels[temp_idx]
    )
    
    # Get transforms
    transforms_dict = get_transforms(img_size)
    
    # Create datasets
    train_dataset = CoralDataset(
        [image_paths[i] for i in train_idx],
        time_series[train_idx],
        labels[train_idx],
        transform=transforms_dict['train'],
        extract_ml_features=extract_ml_features
    )
    
    val_dataset = CoralDataset(
        [image_paths[i] for i in val_idx],
        time_series[val_idx],
        labels[val_idx],
        transform=transforms_dict['val'],
        extract_ml_features=extract_ml_features
    )
    
    test_dataset = CoralDataset(
        [image_paths[i] for i in test_idx],
        time_series[test_idx],
        labels[test_idx],
        transform=transforms_dict['test'],
        extract_ml_features=extract_ml_features
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get data dimensions
    seq_len, time_features = time_series.shape[1], time_series.shape[2]
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'seq_len': seq_len,
        'time_features': time_features,
        'train_idx': train_idx,
        'val_idx': val_idx,
        'test_idx': test_idx
    }


def init_ml_models(model, data):
    """
    Initialize ML models in the ensemble.
    
    Args:
        model (EnsembleModel): Model
        data (Dict): Data dictionary
        
    Returns:
        Initialized model
    """
    if model.ensemble_type != 'boosting':
        return model
    
    # Get training data
    train_time_series = data['time_series'][data['train_idx']]
    train_labels = data['labels'][data['train_idx']]
    
    # Extract features
    feature_extractor = FeatureExtractor()
    train_features = feature_extractor.extract_all_features(train_time_series)
    
    # Initialize ML models
    model.init_ml_models(train_features, train_labels)
    
    return model


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on data loader.
    
    Args:
        model (nn.Module): Model
        data_loader (DataLoader): Data loader
        device (torch.device): Device
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            time_series = batch['time_series'].to(device)
            labels = batch['label'].to(device)
            
            # Get ML features if available
            ml_features = batch.get('ml_features', None)
            if ml_features is not None:
                ml_features = ml_features.to(device)
            
            # Forward pass
            logits = model(images, time_series, ml_features)
            
            # Reshape logits
            logits = logits.view(-1)
            
            # Convert to probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Save for metrics calculation
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.view(-1).cpu().numpy())
    
    # Concatenate all predictions and labels
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds),
        'recall': recall_score(all_labels, all_preds),
        'f1': f1_score(all_labels, all_preds),
        'auc': roc_auc_score(all_labels, all_probs)
    }
    
    return metrics


def visualize_model_predictions(model, data_loader, num_samples=5, save_dir=None):
    """
    Visualize model predictions.
    
    Args:
        model (nn.Module): Model
        data_loader (DataLoader): Data loader
        num_samples (int): Number of samples to visualize
        save_dir (str): Directory to save visualizations
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    device = next(model.parameters()).device
    
    all_samples = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['image'].to(device)
            time_series = batch['time_series'].to(device)
            labels = batch['label'].to(device)
            
            # Get ML features if available
            ml_features = batch.get('ml_features', None)
            if ml_features is not None:
                ml_features = ml_features.to(device)
            
            # Forward pass
            logits = model(images, time_series, ml_features)
            
            # Reshape logits
            logits = logits.view(-1)
            
            # Convert to probabilities and predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            # Save samples
            for i in range(len(images)):
                all_samples.append({
                    'image': images[i].cpu().numpy(),
                    'time_series': time_series[i].cpu().numpy(),
                    'label': labels[i].item(),
                    'pred': preds[i].item(),
                    'prob': probs[i].item()
                })
                
                if len(all_samples) >= num_samples:
                    break
            
            if len(all_samples) >= num_samples:
                break
    
    # Visualize samples
    for i, sample in enumerate(all_samples):
        image = sample['image']
        time_series = sample['time_series']
        label = sample['label']
        pred = sample['pred']
        prob = sample['prob']
        
        # Transpose image back to HWC format
        image = np.transpose(image, (1, 2, 0))
        
        # Denormalize image
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot image
        plt.subplot(2, 1, 1)
        plt.imshow(image)
        plt.title(f"Image (Label: {int(label)}, Prediction: {int(pred)}, Probability: {prob:.2f})")
        
        # Plot time series
        plt.subplot(2, 1, 2)
        for j in range(min(3, time_series.shape[1])):  # Show first 3 features
            plt.plot(time_series[:, j], label=f'Feature {j+1}')
        
        plt.title('Time Series')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'sample_{i+1}.png'))
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    img_size = 224
    batch_size = 8
    max_epochs = 20
    
    try:
        # Model parameters
        model_params = {
            'input_channels': 3,
            'img_size': img_size,
            'time_input_dim': 8,
            'seq_len': 24,  # Use seq_len instead of time_steps
            'output_dim': 64,
            'dropout': 0.3,
            'ensemble_type': 'weighted',
            'use_cnn_lstm': True,
            'use_tcn': True,
            'use_transformer': True,
            'use_ml_models': False  # Set to False for simpler testing
        }
        
        # Create model for testing
        print("Creating ensemble model...")
        ensemble_model = EnsembleLightningModel(
            model_params=model_params,
            learning_rate=1e-4,
            weight_decay=1e-5,
            pos_weight=2.0
        )
        
        # Generate synthetic data for testing
        print("Generating synthetic data...")
        sample_images = torch.randn(2, 3, img_size, img_size)
        sample_time_series = torch.randn(2, 24, 8)
        
        # Forward pass
        print("Testing forward pass...")
        outputs = ensemble_model(sample_images, sample_time_series)
        print(f"Output shape: {outputs.shape}")
        
        print("Ensemble model created successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()