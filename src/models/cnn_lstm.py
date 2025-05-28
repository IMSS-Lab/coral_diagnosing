"""
CNN-LSTM model with attention for coral bleaching prediction.
This model combines CNN, LSTM, and wavelet analysis with attention mechanisms.
"""

import os
import numpy as np
import pandas as pd
import json
import pywt
from typing import Tuple, Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
# Import torchmetrics instead of pl.metrics
import torchmetrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

from .wavelet import WaveletTransform


class SelfAttention(nn.Module):
    """
    Self-attention mechanism for sequence data.
    
    Parameters:
        embed_dim (int): Embedding dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.0
    ):
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for self-attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Tuple of (attended tensor, attention weights)
        """
        # Apply layer normalization first
        residual = x
        x = self.layer_norm(x)
        
        # Project to queries, keys, values
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        # Scaled dot-product attention
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        # Apply softmax to get attention weights
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention weights to values
        out = torch.matmul(attn, v)
        
        # Add residual connection
        out = out + residual
        
        return out, attn


class CNNModule(nn.Module):
    """
    CNN module for image feature extraction.
    
    Parameters:
        backbone (str): CNN backbone name ('resnet18', 'resnet50', 'efficientnet_b0')
        output_dim (int): Output dimension
        dropout (float): Dropout probability
        freeze_backbone (bool): Whether to freeze the backbone weights
    """
    
    def __init__(
        self,
        backbone: str = 'resnet18',
        output_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(CNNModule, self).__init__()
        
        # Initialize CNN backbone
        if backbone == 'resnet18':
            # Use the proper weights parameter instead of deprecated pretrained
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Remove classifier head
        self.backbone.fc = nn.Identity()
        if 'efficient' in backbone:
            self.backbone.classifier = nn.Identity()
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Feature processing layers
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for CNN module.
        
        Args:
            x: Input images of shape [batch_size, channels, height, width]
            
        Returns:
            Features of shape [batch_size, output_dim]
        """
        # Extract features
        features = self.backbone(x)
        
        # Process features
        output = self.feature_processor(features)
        
        return output


class LSTMModule(nn.Module):
    """
    LSTM module for time series processing.
    
    Parameters:
        input_size (int): Input feature dimension
        hidden_size (int): Hidden state dimension
        num_layers (int): Number of LSTM layers
        output_size (int): Output dimension
        dropout (float): Dropout probability
        bidirectional (bool): Whether to use bidirectional LSTM
    """
    
    def __init__(
        self,
        input_size: int = 8,
        hidden_size: int = 64,
        num_layers: int = 2,
        output_size: int = 128,
        dropout: float = 0.3,
        bidirectional: bool = True
    ):
        super(LSTMModule, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.bidirectional = bidirectional
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Attention mechanism
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = SelfAttention(
            embed_dim=lstm_output_dim,
            dropout=dropout
        )
        
        # Output processing
        self.output_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LSTM module.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Tuple of (output features, attention weights)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # [batch_size, seq_len, hidden_size*2]
        
        # Apply attention
        attended, attn_weights = self.attention(lstm_out)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(attended, dim=1)  # [batch_size, hidden_size*2]
        
        # Process for output
        output = self.output_processor(pooled)  # [batch_size, output_size]
        
        return output, attn_weights


class WaveletModule(nn.Module):
    """
    Wavelet feature extraction module.
    
    Parameters:
        input_size (int): Input dimension
        output_size (int): Output dimension
        wavelet (str): Wavelet family
        level (int): Decomposition level
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        input_size: int = 8,
        output_size: int = 64,
        wavelet: str = 'db4',
        level: int = 3,
        dropout: float = 0.3
    ):
        super(WaveletModule, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.wavelet = wavelet
        self.level = level
        
        # Define feature processor
        self.feature_processor = nn.Sequential(
            nn.Linear(input_size * (level + 1) * 4, 128),  # 4 features per level
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, output_size),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
    
    def extract_wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract wavelet features from time series.
        
        Args:
            x: Time series data of shape [batch_size, seq_len, input_size]
            
        Returns:
            Wavelet features of shape [batch_size, input_size*(level+1)*4]
        """
        batch_size, seq_len, input_size = x.shape
        
        # Ensure seq_len is suitable for wavelet decomposition
        x_np = x.cpu().numpy()
        
        # Determine appropriate wavelet level based on sequence length
        max_level = int(np.log2(seq_len))
        level = min(self.level, max_level - 1)
        level = max(1, level)  # Ensure at least level 1
        
        # Initialize features array
        features = np.zeros((batch_size, input_size * (level + 1) * 4))
        
        for b in range(batch_size):
            feature_idx = 0
            for i in range(input_size):
                ts = x_np[b, :, i]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(ts, self.wavelet, level=level)
                
                # Extract features from coefficients
                for level_idx, coeff in enumerate(coeffs):
                    # Statistical features from each level
                    if len(coeff) > 0:
                        features[b, feature_idx] = np.mean(coeff)
                        features[b, feature_idx + 1] = np.std(coeff)
                        features[b, feature_idx + 2] = np.sum(coeff ** 2)  # Energy
                        
                        # Entropy
                        if np.sum(np.abs(coeff)) > 0:
                            p = np.abs(coeff) / np.sum(np.abs(coeff))
                            features[b, feature_idx + 3] = -np.sum(p * np.log2(p + 1e-10))
                    
                    feature_idx += 4
        
        return torch.FloatTensor(features).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for wavelet module.
        
        Args:
            x: Time series data of shape [batch_size, seq_len, input_size]
            
        Returns:
            Processed features of shape [batch_size, output_size]
        """
        # Extract wavelet features
        wavelet_features = self.extract_wavelet_features(x)
        
        # Process features
        output = self.feature_processor(wavelet_features)
        
        return output


class FeatureFusion(nn.Module):
    """
    Feature fusion module.
    
    Parameters:
        cnn_dim (int): CNN feature dimension
        lstm_dim (int): LSTM feature dimension
        wavelet_dim (int): Wavelet feature dimension
        output_dim (int): Output dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        cnn_dim: int = 256,
        lstm_dim: int = 128,
        wavelet_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super(FeatureFusion, self).__init__()
        
        # Combined input dimension
        combined_dim = cnn_dim + lstm_dim + wavelet_dim
        
        # Fusion layers
        self.fusion_layers = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
    
    def forward(
        self, 
        cnn_features: torch.Tensor, 
        lstm_features: torch.Tensor, 
        wavelet_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for feature fusion.
        
        Args:
            cnn_features: CNN features of shape [batch_size, cnn_dim]
            lstm_features: LSTM features of shape [batch_size, lstm_dim]
            wavelet_features: Wavelet features of shape [batch_size, wavelet_dim]
            
        Returns:
            Fused features of shape [batch_size, output_dim]
        """
        # Concatenate features
        combined = torch.cat([cnn_features, lstm_features, wavelet_features], dim=1)
        
        # Apply fusion layers
        fused = self.fusion_layers(combined)
        
        return fused


class CoralNet(nn.Module):
    """
    Combined CNN-LSTM-Wavelet model for coral bleaching prediction.
    
    Parameters:
        input_channels (int): Number of input channels for images
        img_size (int): Input image size
        time_input_dim (int): Input dimension for time series
        seq_len (int): Sequence length for time series
        cnn_backbone (str): CNN backbone type
        cnn_output_dim (int): CNN output dimension
        lstm_hidden_dim (int): LSTM hidden dimension
        lstm_layers (int): Number of LSTM layers
        lstm_output_dim (int): LSTM output dimension
        wavelet_output_dim (int): Wavelet output dimension
        fusion_output_dim (int): Fusion output dimension
        dropout (float): Dropout probability
        bidirectional (bool): Whether LSTM is bidirectional
        freeze_backbone (bool): Whether to freeze CNN backbone
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        time_input_dim: int = 8,
        seq_len: int = 24,
        cnn_backbone: str = 'resnet18',
        cnn_output_dim: int = 256,
        lstm_hidden_dim: int = 64,
        lstm_layers: int = 2,
        lstm_output_dim: int = 128,
        wavelet_output_dim: int = 64,
        fusion_output_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
        freeze_backbone: bool = False
    ):
        super(CoralNet, self).__init__()
        
        # CNN module
        self.cnn_module = CNNModule(
            backbone=cnn_backbone,
            output_dim=cnn_output_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        # LSTM module
        self.lstm_module = LSTMModule(
            input_size=time_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            output_size=lstm_output_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        # Wavelet module
        self.wavelet_module = WaveletModule(
            input_size=time_input_dim,
            output_size=wavelet_output_dim,
            dropout=dropout
        )
        
        # Feature fusion
        self.fusion_module = FeatureFusion(
            cnn_dim=cnn_output_dim,
            lstm_dim=lstm_output_dim,
            wavelet_dim=wavelet_output_dim,
            output_dim=fusion_output_dim,
            dropout=dropout
        )
        
        # Output layer for binary classification
        self.output_layer = nn.Linear(fusion_output_dim, 1)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for fully connected layers."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self, 
        images: torch.Tensor, 
        time_series: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            images: Input images of shape [batch_size, channels, height, width]
            time_series: Input time series of shape [batch_size, seq_len, features]
            
        Returns:
            Model output or tuple of (output, attention weights)
        """
        # Process images through CNN
        cnn_features = self.cnn_module(images)
        
        # Process time series through LSTM
        lstm_features, attn_weights = self.lstm_module(time_series)
        
        # Process time series through wavelet module
        wavelet_features = self.wavelet_module(time_series)
        
        # Fuse features
        fused_features = self.fusion_module(cnn_features, lstm_features, wavelet_features)
        
        # Output layer
        output = self.output_layer(fused_features)
        
        # Return output and attention weights
        return output, {
            'attention_weights': attn_weights
        }


class CoralDataset(Dataset):
    """
    Dataset for coral bleaching prediction with image and time series data.
    
    Parameters:
        image_paths (List[str]): List of image file paths
        time_series (np.ndarray): Time series data
        labels (np.ndarray): Labels (0 or 1)
        transform (Optional): Image transform function
    """
    
    def __init__(
        self,
        image_paths: List[str],
        time_series: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Any] = None
    ):
        self.image_paths = image_paths
        self.time_series = time_series
        self.labels = labels
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0
        
        # Get time series and label
        time_series = torch.FloatTensor(self.time_series[idx])
        label = torch.FloatTensor([self.labels[idx]])
        
        return {
            'image': image,
            'time_series': time_series,
            'label': label
        }


class CoralLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for coral bleaching prediction.
    
    Parameters:
        model_params (Dict[str, Any]): Model parameters
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        pos_weight (float): Positive class weight for loss function
    """
    
    def __init__(
        self,
        model_params: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 2.0
    ):
        super(CoralLightningModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = CoralNet(**model_params)
        
        # Loss function with class weighting
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
    
    def forward(self, images, time_series):
        return self.model(images, time_series)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Forward pass
        outputs = self(images, time_series)
        
        # Extract logits and attention weights
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
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
        
        # Forward pass
        outputs = self(images, time_series)
        
        # Extract logits and attention weights
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
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
        
        # Forward pass
        outputs = self(images, time_series)
        
        # Extract logits and attention weights
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
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
    
    def get_attention_weights(self, images, time_series):
        """
        Get attention weights for visualization.
        
        Args:
            images: Input images
            time_series: Input time series
            
        Returns:
            Attention weights
        """
        with torch.no_grad():
            outputs = self(images, time_series)
        
        if isinstance(outputs, tuple):
            _, attention_dict = outputs
            return attention_dict['attention_weights']
        else:
            return None


def get_transforms(img_size=224):
    """
    Get data transforms for training and validation.
    
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
    num_workers: int = 4
):
    """
    Load data for training.
    
    Args:
        image_dir (str): Directory containing images
        time_series_path (str): Path to time series data
        labels_path (str): Path to labels
        img_size (int): Image size
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
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
    
    # Split into train, validation, and test sets
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
        transform=transforms_dict['train']
    )
    
    val_dataset = CoralDataset(
        [image_paths[i] for i in val_idx],
        time_series[val_idx],
        labels[val_idx],
        transform=transforms_dict['val']
    )
    
    test_dataset = CoralDataset(
        [image_paths[i] for i in test_idx],
        time_series[test_idx],
        labels[test_idx],
        transform=transforms_dict['test']
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
        'time_features': time_features
    }


def visualize_attention(model, dataloader, num_samples=5, save_dir=None):
    """
    Visualize attention weights.
    
    Args:
        model (CoralNet): Model
        dataloader (DataLoader): Data loader
        num_samples (int): Number of samples to visualize
        save_dir (str): Directory to save visualizations
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get batches
    all_samples = []
    for batch in dataloader:
        images = batch['image'].to(device)
        time_series = batch['time_series'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            outputs = model(images, time_series)
        
        if isinstance(outputs, tuple):
            logits, attn_dict = outputs
            attn_weights = attn_dict['attention_weights']
            
            # Convert to numpy for visualization
            logits_np = logits.cpu().numpy()
            attn_weights_np = attn_weights.cpu().numpy()
            time_series_np = time_series.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(len(logits)):
                all_samples.append({
                    'logit': logits_np[i, 0],
                    'attn_weights': attn_weights_np[i],
                    'time_series': time_series_np[i],
                    'label': labels_np[i, 0]
                })
                
                if len(all_samples) >= num_samples:
                    break
        
        if len(all_samples) >= num_samples:
            break
    
    # Visualize samples
    for i, sample in enumerate(all_samples[:num_samples]):
        logit = sample['logit']
        attn_weights = sample['attn_weights']
        time_series = sample['time_series']
        label = sample['label']
        
        prob = 1 / (1 + np.exp(-logit))  # Sigmoid
        pred = 1 if prob > 0.5 else 0
        
        plt.figure(figsize=(12, 8))
        
        # Plot time series
        plt.subplot(2, 1, 1)
        for j in range(min(3, time_series.shape[1])):  # Show first 3 features
            plt.plot(time_series[:, j], label=f'Feature {j+1}')
        
        plt.title(f'Time Series (Label: {int(label)}, Prediction: {pred}, Probability: {prob:.2f})')
        plt.xlabel('Time Step')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Plot attention weights
        plt.subplot(2, 1, 2)
        plt.imshow(attn_weights, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight')
        plt.title('Attention Weights')
        plt.xlabel('Time Step')
        plt.ylabel('Time Step')
        
        plt.tight_layout()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'attention_{i+1}.png'))
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Configuration
    img_size = 224
    batch_size = 16
    max_epochs = 30
    
    # Model parameters
    model_params = {
        'input_channels': 3,
        'img_size': img_size,
        'time_input_dim': 8,
        'seq_len': 24,
        'cnn_backbone': 'efficientnet_b0',
        'cnn_output_dim': 256,
        'lstm_hidden_dim': 64,
        'lstm_layers': 2,
        'lstm_output_dim': 128,
        'wavelet_output_dim': 64,
        'fusion_output_dim': 64,
        'dropout': 0.3,
        'bidirectional': True,
        'freeze_backbone': False
    }
    
    # Create model for testing
    try:
        model = CoralNet(**model_params)
        print(model)
        
        # Create synthetic data
        sample_images = torch.randn(2, 3, img_size, img_size)
        sample_time_series = torch.randn(2, 24, 8)
        
        # Forward pass
        outputs, attention_dict = model(sample_images, sample_time_series)
        print(f"Output shape: {outputs.shape}")
        print(f"Attention weights shape: {attention_dict['attention_weights'].shape}")
        
        # Create PyTorch Lightning model
        lightning_model = CoralLightningModel(
            model_params=model_params,
            learning_rate=1e-4,
            weight_decay=1e-5,
            pos_weight=2.0
        )
        
        print("Model created successfully!")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()