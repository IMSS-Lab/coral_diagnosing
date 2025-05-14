"""
CNN-LSTM with Attention model for coral bleaching prediction.
Combines image features with time series environmental data.

This model incorporates:
- EfficientNet backbone for image processing
- Bidirectional LSTM with self-attention for time series
- Wavelet feature processing for time-frequency analysis
- Multi-level feature fusion with attention mechanisms
- Uncertainty quantification via Monte Carlo Dropout
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import polars as plrs
import pywt
from typing import Tuple, Dict, List, Optional, Union
import pytorch_lightning as pl


class SelfAttention(nn.Module):
    """Self-attention module for time series data."""
    
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1):
        """
        Initialize self-attention module.
        
        Args:
            hidden_dim: Dimensionality of input features
            dropout_rate: Dropout probability
        """
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.scale_factor = torch.sqrt(torch.tensor(hidden_dim, dtype=torch.float32))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (output tensor with shape [batch_size, seq_len, hidden_dim],
                     attention weights with shape [batch_size, seq_len, seq_len])
        """
        # Apply layer normalization
        residual = x
        x = self.layer_norm(x)
        
        # Linear projections
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale_factor
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Apply residual connection
        output = residual + attention_output
        
        return output, attention_weights


class CNNModule(nn.Module):
    """CNN module for processing coral reef images using pre-trained backbone."""
    
    def __init__(
        self, 
        backbone: str = 'efficientnet_b0', 
        pretrained: bool = True, 
        output_dim: int = 256,
        dropout_rate: float = 0.3
    ):
        """
        Initialize CNN module.
        
        Args:
            backbone: Name of backbone CNN architecture
            pretrained: Whether to use pre-trained weights
            output_dim: Dimension of output features
            dropout_rate: Dropout probability
        """
        super(CNNModule, self).__init__()
        
        # Select backbone model
        if backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        elif backbone == 'efficientnet_b2':
            self.backbone = models.efficientnet_b2(pretrained=pretrained)
            feature_dim = 1408
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier with identity to get features
        if 'efficientnet' in backbone:
            self.backbone.classifier = nn.Identity()
        else:  # ResNet
            self.backbone.fc = nn.Identity()
        
        # Additional layers for feature processing
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image through CNN backbone.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Processed features of shape [batch_size, output_dim]
        """
        features = self.backbone(x)
        return self.feature_processor(features)


class LSTMModule(nn.Module):
    """LSTM module with self-attention for time series processing."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        num_layers: int = 2, 
        output_dim: int = 128,
        bidirectional: bool = True, 
        dropout_rate: float = 0.3
    ):
        """
        Initialize LSTM module.
        
        Args:
            input_dim: Number of input features
            hidden_dim: LSTM hidden dimension
            num_layers: Number of LSTM layers
            output_dim: Dimension of output features
            bidirectional: Whether to use bidirectional LSTM
            dropout_rate: Dropout probability
        """
        super(LSTMModule, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout=dropout_rate if num_layers > 1 else 0,
            batch_first=True
        )
        
        # Self-attention layer
        self.attention = SelfAttention(hidden_dim * self.num_directions, dropout_rate)
        
        # Output layers
        self.output_processor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * self.num_directions, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process time series through LSTM and attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (output features with shape [batch_size, output_dim],
                     attention weights with shape [batch_size, seq_len, seq_len])
        """
        batch_size = x.size(0)
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_dim).to(x.device)
        
        # Forward pass through LSTM
        output, _ = self.lstm(x, (h0, c0))  # output: [batch_size, seq_len, hidden_dim * num_directions]
        
        # Apply self-attention
        attended_output, attention_weights = self.attention(output)
        
        # Global max pooling across sequence dimension
        pooled_output, _ = torch.max(attended_output, dim=1)
        
        # Process through output layers
        final_output = self.output_processor(pooled_output)
        
        return final_output, attention_weights


class WaveletModule(nn.Module):
    """Module for processing wavelet features from time series data."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int = 64, 
        output_dim: int = 64,
        dropout_rate: float = 0.3
    ):
        """
        Initialize wavelet processing module.
        
        Args:
            input_dim: Number of input features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout_rate: Dropout probability
        """
        super(WaveletModule, self).__init__()
        
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate/2),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process wavelet features.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            
        Returns:
            Processed features of shape [batch_size, output_dim]
        """
        return self.feature_processor(x)


class FeatureFusion(nn.Module):
    """Module for fusing features from different modalities."""
    
    def __init__(
        self, 
        img_dim: int, 
        time_dim: int, 
        wavelet_dim: int, 
        hidden_dim: int = 128,
        output_dim: int = 64, 
        dropout_rate: float = 0.4
    ):
        """
        Initialize feature fusion module.
        
        Args:
            img_dim: Dimension of image features
            time_dim: Dimension of time series features
            wavelet_dim: Dimension of wavelet features
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension
            dropout_rate: Dropout probability
        """
        super(FeatureFusion, self).__init__()
        
        combined_dim = img_dim + time_dim + wavelet_dim
        
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        img_features: torch.Tensor, 
        time_features: torch.Tensor, 
        wavelet_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse features from different modalities.
        
        Args:
            img_features: Image features of shape [batch_size, img_dim]
            time_features: Time series features of shape [batch_size, time_dim]
            wavelet_features: Wavelet features of shape [batch_size, wavelet_dim]
            
        Returns:
            Fused features of shape [batch_size, output_dim]
        """
        combined = torch.cat([img_features, time_features, wavelet_features], dim=1)
        return self.fusion_layers(combined)


class CoralNet(nn.Module):
    """
    Combined model for coral bleaching prediction using image and time series data.
    
    This model fuses information from multiple modalities:
    1. Visual features from coral reef images
    2. Temporal patterns from environmental sensor data
    3. Wavelet-based time-frequency features
    
    It uses attention mechanisms for feature extraction and fusion.
    """
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        wavelet_dim: int,
        cnn_backbone: str = 'efficientnet_b0',
        img_feature_dim: int = 256,
        time_feature_dim: int = 128,
        wavelet_feature_dim: int = 64,
        lstm_hidden_dim: int = 64,
        fusion_hidden_dim: int = 128,
        fusion_output_dim: int = 64,
        dropout_rate: float = 0.3
    ):
        """
        Initialize coral bleaching prediction model.
        
        Args:
            time_steps: Number of time steps in time series data
            num_features: Number of features in time series data
            wavelet_dim: Dimension of wavelet features
            cnn_backbone: CNN backbone architecture
            img_feature_dim: Dimension of image features
            time_feature_dim: Dimension of time series features
            wavelet_feature_dim: Dimension of wavelet features
            lstm_hidden_dim: Hidden dimension of LSTM
            fusion_hidden_dim: Hidden dimension of fusion module
            fusion_output_dim: Output dimension of fusion module
            dropout_rate: Dropout probability
        """
        super(CoralNet, self).__init__()
        
        # CNN for image data
        self.cnn_module = CNNModule(
            backbone=cnn_backbone,
            pretrained=True,
            output_dim=img_feature_dim,
            dropout_rate=dropout_rate
        )
        
        # LSTM for time series data
        self.lstm_module = LSTMModule(
            input_dim=num_features,
            hidden_dim=lstm_hidden_dim,
            output_dim=time_feature_dim,
            bidirectional=True,
            dropout_rate=dropout_rate
        )
        
        # Wavelet feature processing
        self.wavelet_module = WaveletModule(
            input_dim=wavelet_dim,
            hidden_dim=wavelet_feature_dim,
            output_dim=wavelet_feature_dim,
            dropout_rate=dropout_rate
        )
        
        # Feature fusion
        self.fusion_module = FeatureFusion(
            img_dim=img_feature_dim,
            time_dim=time_feature_dim,
            wavelet_dim=wavelet_feature_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout_rate=dropout_rate
        )
        
        # Output layer
        self.output_layer = nn.Linear(fusion_output_dim, 1)
        
        # Store dropout rate for MC dropout during inference
        self.dropout_rate = dropout_rate
        self.mc_dropout = False
    
    def forward(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            image: Image tensor of shape [batch_size, channels, height, width]
            time_series: Time series tensor of shape [batch_size, time_steps, num_features]
            wavelet: Wavelet features tensor of shape [batch_size, wavelet_dim]
            
        Returns:
            Tuple of (prediction logits with shape [batch_size],
                     attention weights with shape [batch_size, time_steps, time_steps])
        """
        # Process image data
        image_features = self.cnn_module(image)
        
        # Process time series data
        time_features, attention_weights = self.lstm_module(time_series)
        
        # Process wavelet features
        wavelet_features = self.wavelet_module(wavelet)
        
        # Fuse features
        fused_features = self.fusion_module(image_features, time_features, wavelet_features)
        
        # Output layer
        output = self.output_layer(fused_features)
        
        return output.squeeze(), attention_weights
    
    def enable_mc_dropout(self):
        """Enable MC dropout for uncertainty estimation during inference."""
        self.mc_dropout = True
        self.train()  # Sets model to train mode but only dropout will be affected
    
    def disable_mc_dropout(self):
        """Disable MC dropout and return to normal inference mode."""
        self.mc_dropout = False
        self.eval()
    
    def predict_with_uncertainty(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor, 
        wavelet: torch.Tensor, 
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC dropout.
        
        Args:
            image: Image tensor
            time_series: Time series tensor
            wavelet: Wavelet features tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, prediction variances)
        """
        self.enable_mc_dropout()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred, _ = self(image, time_series, wavelet)
                pred = torch.sigmoid(pred)
                predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        self.disable_mc_dropout()
        return mean_pred, var_pred


class WaveletFeatureExtractor:
    """Utility class for extracting wavelet features from time series data."""
    
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        """
        Initialize wavelet feature extractor.
        
        Args:
            wavelet: Wavelet type to use
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


class CoralDataset(Dataset):
    """Dataset class for coral bleaching prediction."""
    
    def __init__(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        wavelet_features: np.ndarray, 
        labels: np.ndarray,
        transform = None
    ):
        """
        Initialize coral dataset.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            wavelet_features: Wavelet features of shape [num_samples, feature_dim]
            labels: Labels of shape [num_samples]
            transform: Transformations to apply to images
        """
        # Convert to PyTorch tensors
        # Ensure images are in the correct format (N, C, H, W)
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.time_series = torch.FloatTensor(time_series)
        self.wavelet_features = torch.FloatTensor(wavelet_features)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, time_series, wavelet_features, label)
        """
        image = self.images[idx]
        time_series = self.time_series[idx]
        wavelet = self.wavelet_features[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, time_series, wavelet, label


class CoralLightningModel(pl.LightningModule):
    """PyTorch Lightning module for coral bleaching prediction."""
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        wavelet_dim: int,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4,
        cnn_backbone: str = 'efficientnet_b0',
        **model_kwargs
    ):
        """
        Initialize Lightning module.
        
        Args:
            time_steps: Number of time steps in time series
            num_features: Number of features in time series
            wavelet_dim: Dimension of wavelet features
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
            cnn_backbone: CNN backbone architecture
            **model_kwargs: Additional arguments for the model
        """
        super(CoralLightningModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = CoralNet(
            time_steps=time_steps,
            num_features=num_features,
            wavelet_dim=wavelet_dim,
            cnn_backbone=cnn_backbone,
            **model_kwargs
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()
        
        # Metrics
        self.train_acc = pl.metrics.Accuracy(task='binary')
        self.val_acc = pl.metrics.Accuracy(task='binary')
        self.val_auroc = pl.metrics.AUROC(task='binary')
        self.val_f1 = pl.metrics.F1Score(task='binary')
        self.val_precision = pl.metrics.Precision(task='binary')
        self.val_recall = pl.metrics.Recall(task='binary')
    
    def forward(self, image, time_series, wavelet):
        """Forward pass through the model."""
        return self.model(image, time_series, wavelet)
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        image, time_series, wavelet, y = batch
        y_hat, _ = self(image, time_series, wavelet)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        acc = self.train_acc(torch.sigmoid(y_hat), y.int())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        image, time_series, wavelet, y = batch
        y_hat, _ = self(image, time_series, wavelet)
        val_loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        acc = self.val_acc(torch.sigmoid(y_hat), y.int())
        auroc = self.val_auroc(torch.sigmoid(y_hat), y.int())
        f1 = self.val_f1(torch.sigmoid(y_hat), y.int())
        precision = self.val_precision(torch.sigmoid(y_hat), y.int())
        recall = self.val_recall(torch.sigmoid(y_hat), y.int())
        
        # Log metrics
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_auroc', auroc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        self.log('val_precision', precision, prog_bar=True)
        self.log('val_recall', recall, prog_bar=True)
        
        return val_loss
    
    def test_step(self, batch, batch_idx):
        """Test step."""
        image, time_series, wavelet, y = batch
        y_hat, _ = self(image, time_series, wavelet)
        test_loss = self.criterion(y_hat, y)
        
        # Calculate metrics
        acc = self.val_acc(torch.sigmoid(y_hat), y.int())
        auroc = self.val_auroc(torch.sigmoid(y_hat), y.int())
        f1 = self.val_f1(torch.sigmoid(y_hat), y.int())
        precision = self.val_precision(torch.sigmoid(y_hat), y.int())
        recall = self.val_recall(torch.sigmoid(y_hat), y.int())
        
        # Log metrics
        self.log('test_loss', test_loss)
        self.log('test_acc', acc)
        self.log('test_auroc', auroc)
        self.log('test_f1', f1)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        
        return test_loss
    
    def predict_with_uncertainty(self, image, time_series, wavelet, n_samples=30):
        """Make predictions with uncertainty estimation."""
        return self.model.predict_with_uncertainty(image, time_series, wavelet, n_samples)
    
    def get_attention_weights(self, image, time_series, wavelet):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            _, attention_weights = self(image, time_series, wavelet)
        return attention_weights


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


def preprocess_data(
    images: np.ndarray,
    time_series: np.ndarray,
    labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data for model training.
    
    Args:
        images: Image data
        time_series: Time series data
        labels: Labels
        
    Returns:
        Tuple of (processed images, processed time series, wavelet features, labels)
    """
    # Extract wavelet features
    wavelet_extractor = WaveletFeatureExtractor(wavelet='db4', level=3)
    wavelet_features = wavelet_extractor.extract_features(time_series)
    
    return images, time_series, wavelet_features, labels


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from torch.utils.data import random_split, DataLoader
    
    # Assume we have loaded the data
    images = np.random.randn(100, 224, 224, 3)  # 100 sample images
    time_series = np.random.randn(100, 24, 8)   # 100 samples, 24 time steps, 8 features
    labels = np.random.randint(0, 2, 100)       # Binary labels
    
    # Preprocess data
    images, time_series, wavelet_features, labels = preprocess_data(images, time_series, labels)
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CoralDataset(images, time_series, wavelet_features, labels, transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = CoralNet(
        time_steps=24,
        num_features=8,
        wavelet_dim=wavelet_features.shape[1]
    )
    
    # Print model summary
    print(model)
    
    # Create Lightning module
    lightning_model = CoralLightningModel(
        time_steps=24,
        num_features=8,
        wavelet_dim=wavelet_features.shape[1]
    )
    
    # Example forward pass
    image_batch, ts_batch, wavelet_batch, _ = next(iter(train_loader))
    outputs, attention = model(image_batch, ts_batch, wavelet_batch)
    print(f"Output shape: {outputs.shape}")
    print(f"Attention shape: {attention.shape}")