"""
Temporal Convolutional Network (TCN) for coral bleaching prediction.
Combines CNN-based image features with temporal convolutions for time series data.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
# Import torchmetrics instead of using pl.metrics
import torchmetrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

from .wavelet import WaveletTransform


class Chomp1d(nn.Module):
    """
    Module to ensure causal convolutions by chomping the padding.
    
    This removes the "future" padding to ensure convolutions only look at the past.
    
    Parameters:
        chomp_size (int): Number of elements to remove from the end
    """
    
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Remove padding from the end of the sequence."""
        # Remove padding from the end
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block with dilated convolutions for TCN.
    
    Parameters:
        n_inputs (int): Number of input channels
        n_outputs (int): Number of output channels
        kernel_size (int): Kernel size
        stride (int): Stride
        dilation (int): Dilation factor
        padding (int): Padding
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)  # Remove padding to ensure causal convolution
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)  # Remove padding to ensure causal convolution
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection if dimensions don't match
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the temporal block."""
        # First conv block
        out = self.conv1(x)
        out = self.chomp1(out)  # Remove padding
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.chomp2(out)  # Remove padding
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure residual and output have the same size for the sequence dimension
        # This is important when the sequence length changes due to dilation and padding
        if res.size(2) != out.size(2):
            # Adjust residual to match output size
            if res.size(2) > out.size(2):
                res = res[:, :, :out.size(2)]
            else:
                out = out[:, :, :res.size(2)]
        
        out = out + res
        
        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network for time series processing.
    
    Parameters:
        num_inputs (int): Number of input channels
        num_channels (List[int]): Number of channels in each layer
        kernel_size (int): Kernel size
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2
    ):
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Add temporal block with increasing dilation
            # For causal convolution: ensure receptive field only covers past values
            # For a kernel of size k and dilation d, we need padding of (k-1)*d
            # to maintain the output size
            padding = (kernel_size-1) * dilation_size
            
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=padding,
                    dropout=dropout
                )
            )
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the TCN."""
        return self.network(x)


class SequenceAttention(nn.Module):
    """
    Attention mechanism for sequence data.
    
    Parameters:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads (defaults to 1)
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        dropout: float = 0.0
    ):
        super(SequenceAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Query, key, value projections
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the attention mechanism.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Tuple of (attended tensor, attention weights)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Project queries, keys, values
        q = self.query(x)  # [batch_size, seq_len, embed_dim]
        k = self.key(x)    # [batch_size, seq_len, embed_dim]
        v = self.value(x)  # [batch_size, seq_len, embed_dim]
        
        # Compute scaled dot-product attention
        # [batch_size, seq_len, seq_len]
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [batch_size, seq_len, embed_dim]
        attn_output = torch.bmm(attn_weights, v)
        
        # Project to output dimension
        attn_output = self.output_proj(attn_output)
        
        return attn_output, attn_weights


class CNNModule(nn.Module):
    """
    CNN module for image feature extraction.
    
    Parameters:
        backbone_name (str): Name of the CNN backbone
        output_dim (int): Dimension of the output features
        freeze_backbone (bool): Whether to freeze the backbone
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        backbone_name: str = 'resnet18',
        output_dim: int = 256,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super(CNNModule, self).__init__()
        
        # Load the CNN backbone
        if backbone_name == 'resnet18':
            # Use proper weights parameter instead of deprecated pretrained
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = 512
        elif backbone_name == 'resnet34':
            self.backbone = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
            feature_dim = 512
        elif backbone_name == 'resnet50':
            self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            feature_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Replace the classifier with an identity
        self.backbone.fc = nn.Identity()
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)  # Smaller dropout after ReLU
        )
        
        # Freeze the backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the CNN module.
        
        Args:
            x: Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
            CNN features of shape [batch_size, output_dim]
        """
        # Extract features
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # Project to output dimension
        projected = self.projection(features)  # [batch_size, output_dim]
        
        return projected


class TCNModule(nn.Module):
    """
    TCN module for time series processing.
    
    Parameters:
        input_dim (int): Dimension of input features
        hidden_dims (List[int]): Dimensions of hidden layers
        output_dim (int): Dimension of output features
        kernel_size (int): Kernel size for TCN
        dropout (float): Dropout probability
        use_attention (bool): Whether to use attention
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int],
        output_dim: int,
        kernel_size: int = 3,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        super(TCNModule, self).__init__()
        
        # TCN for time series processing
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        self.use_attention = use_attention
        
        # Attention mechanism
        if use_attention:
            self.attention = SequenceAttention(
                embed_dim=hidden_dims[-1],
                dropout=dropout
            )
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)  # Smaller dropout after ReLU
        )
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for the TCN module.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            TCN features of shape [batch_size, output_dim] or
            Tuple of (features, attention weights)
        """
        # TCN expects [batch_size, channels, seq_len]
        x = x.transpose(1, 2)
        
        # Apply TCN
        tcn_out = self.tcn(x)  # [batch_size, hidden_dims[-1], seq_len]
        
        # Apply attention if specified
        attn_weights = None
        if self.use_attention:
            # Transpose to [batch_size, seq_len, channels]
            tcn_out = tcn_out.transpose(1, 2)
            
            # Apply attention
            attended, attn_weights = self.attention(tcn_out)
            
            # Global pool over sequence dimension
            pooled = attended.mean(dim=1)  # [batch_size, hidden_dims[-1]]
        else:
            # Global pool over sequence dimension
            pooled = tcn_out.mean(dim=2)  # [batch_size, hidden_dims[-1]]
        
        # Project to output dimension
        output = self.projection(pooled)  # [batch_size, output_dim]
        
        if self.use_attention and attn_weights is not None:
            return output, attn_weights
        else:
            return output


class FeatureFusion(nn.Module):
    """
    Feature fusion module for combining image and time series features.
    
    Parameters:
        img_dim (int): Dimension of image features
        time_dim (int): Dimension of time series features
        hidden_dim (int): Dimension of hidden layers
        output_dim (int): Dimension of output features
        dropout (float): Dropout probability
    """
    
    def __init__(
        self,
        img_dim: int,
        time_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super(FeatureFusion, self).__init__()
        
        # Feature projections to common dimension
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        # Context-aware gating
        # After projections and concatenation, the context dimension is hidden_dim + hidden_dim + (hidden_dim * 2) = hidden_dim * 4
        context_dim = hidden_dim * 4
        self.img_gate = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        self.time_gate = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Final fusion network
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU()
        )
        
    def forward(
        self, 
        img_features: torch.Tensor, 
        time_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass for the fusion module.
        
        Args:
            img_features: Image features of shape [batch_size, img_dim]
            time_features: Time series features of shape [batch_size, time_dim]
            
        Returns:
            Fused features of shape [batch_size, output_dim]
        """
        # Project features to common dimension
        img_proj = self.img_proj(img_features)   # [batch_size, hidden_dim]
        time_proj = self.time_proj(time_features)  # [batch_size, hidden_dim]
        
        # Concatenate features
        concat_features = torch.cat([img_proj, time_proj], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Create context vector for gating
        # The context is [img_proj, time_proj, concat_features] which is [hidden_dim, hidden_dim, hidden_dim*2]
        # Total dimension: hidden_dim * 4
        context = torch.cat([img_proj, time_proj, concat_features], dim=1)  # [batch_size, hidden_dim * 4]
        
        # Apply gating
        img_gate = self.img_gate(context)
        time_gate = self.time_gate(context)
        
        # Apply gates to features
        gated_img = img_proj * img_gate
        gated_time = time_proj * time_gate
        
        # Combine gated features
        combined = torch.cat([gated_img, gated_time], dim=1)  # [batch_size, hidden_dim * 2]
        
        # Final fusion
        fused = self.fusion(combined)  # [batch_size, output_dim]
        
        return fused


class TCNCoralModel(nn.Module):
    """
    TCN-based model for coral bleaching prediction.
    
    Parameters:
        input_channels (int): Number of input channels for image
        img_size (int): Size of the input image
        time_input_dim (int): Input dimension for time series
        sequence_length (int): Length of the time series
        img_feature_dim (int): Dimension of image features
        time_feature_dim (int): Dimension of time series features
        hidden_dims (List[int]): Dimensions of hidden layers for TCN
        fusion_hidden_dim (int): Hidden dimension for fusion
        fusion_output_dim (int): Output dimension for fusion
        cnn_backbone (str): Name of the CNN backbone
        kernel_size (int): Kernel size for TCN
        dropout (float): Dropout probability
        freeze_backbone (bool): Whether to freeze the CNN backbone
        use_attention (bool): Whether to use attention in TCN
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        img_size: int = 224,
        time_input_dim: int = 8,
        sequence_length: int = 24,
        img_feature_dim: int = 256,
        time_feature_dim: int = 128,
        hidden_dims: List[int] = [64, 128, 256],
        fusion_hidden_dim: int = 128,
        fusion_output_dim: int = 64,
        cnn_backbone: str = 'resnet18',
        kernel_size: int = 3,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
        use_attention: bool = True
    ):
        super(TCNCoralModel, self).__init__()
        
        # CNN module for image feature extraction
        self.cnn_module = CNNModule(
            backbone_name=cnn_backbone,
            output_dim=img_feature_dim,
            freeze_backbone=freeze_backbone,
            dropout=dropout
        )
        
        # TCN module for time series processing
        self.tcn_module = TCNModule(
            input_dim=time_input_dim,
            hidden_dims=hidden_dims,
            output_dim=time_feature_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            use_attention=use_attention
        )
        
        # Feature fusion module
        self.fusion_module = FeatureFusion(
            img_dim=img_feature_dim,
            time_dim=time_feature_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=dropout
        )
        
        # Output layer for binary classification
        self.output_layer = nn.Linear(fusion_output_dim, 1)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(
        self, 
        images: torch.Tensor, 
        time_series: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass for the coral bleaching prediction model.
        
        Args:
            images: Input images of shape [batch_size, channels, height, width]
            time_series: Input time series of shape [batch_size, seq_len, features]
            
        Returns:
            Model output or tuple of (output, attention maps)
        """
        # Extract image features
        img_features = self.cnn_module(images)  # [batch_size, img_feature_dim]
        
        # Process time series
        tcn_out = self.tcn_module(time_series)
        
        # Check if TCN returned attention weights
        attn_weights = None
        if isinstance(tcn_out, tuple):
            time_features, attn_weights = tcn_out
        else:
            time_features = tcn_out
            
        # Debug print statements to check shapes
        # print(f"Image features shape: {img_features.shape}")
        # print(f"Time features shape: {time_features.shape}")
        
        # Fuse features
        fused_features = self.fusion_module(img_features, time_features)  # [batch_size, fusion_output_dim]
        
        # Output layer
        output = self.output_layer(fused_features)  # [batch_size, 1]
        
        if attn_weights is not None:
            return output, {
                'attention_weights': attn_weights
            }
        else:
            return output


class CoralDataset(Dataset):
    """
    Dataset for coral bleaching prediction with images and time series.
    
    Parameters:
        image_paths (List[str]): List of paths to image files
        time_series (np.ndarray): Time series data
        labels (np.ndarray): Labels (0 or 1 for binary classification)
        transform (Optional[Any]): Transform to apply to images
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
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transform if specified
        if self.transform:
            image = self.transform(image)
        else:
            # Default transform
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get time series
        time_series = torch.from_numpy(self.time_series[idx]).float()
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.float)
        
        return {
            'image': image,
            'time_series': time_series,
            'label': label
        }


class TCNLightningModel(pl.LightningModule):
    """
    PyTorch Lightning module for TCN coral bleaching model.
    
    Parameters:
        model_params (Dict[str, Any]): Parameters for the TCN model
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
        super(TCNLightningModel, self).__init__()
        
        # Save hyperparameters
        self.save_hyperparameters()
        
        # Create model
        self.model = TCNCoralModel(**model_params)
        
        # Create loss function with class weighting
        pos_weight_tensor = torch.tensor([pos_weight])
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        
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
        
        # Handle outputs with attention
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
        logits = logits.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate accuracy
        preds = torch.sigmoid(logits) > 0.5
        self.train_acc(preds, labels.int())
        
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
        
        # Handle outputs with attention
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
        logits = logits.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        
        self.val_acc(preds, labels.int())
        self.val_auroc(probs, labels.int())
        self.val_f1(preds, labels.int())
        
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
        
        # Handle outputs with attention
        if isinstance(outputs, tuple):
            logits, _ = outputs
        else:
            logits = outputs
        
        # Reshape logits to match labels
        logits = logits.view(-1)
        
        # Calculate loss
        loss = self.criterion(logits, labels)
        
        # Calculate metrics
        probs = torch.sigmoid(logits)
        preds = probs > 0.5
        
        self.test_acc(preds, labels.int())
        self.test_auroc(probs, labels.int())
        self.test_f1(preds, labels.int())
        
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
    Get data transforms for images.
    
    Args:
        img_size (int): Size of the input image
        
    Returns:
        Dictionary of transforms for train and validation
    """
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms without augmentation
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return {
        'train': train_transform,
        'val': val_transform
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
    Load and prepare data for training.
    
    Args:
        image_dir (str): Directory containing images
        time_series_path (str): Path to time series data
        labels_path (str): Path to labels data
        img_size (int): Size of the input image
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        
    Returns:
        Dictionary of data loaders and data dimensions
    """
    # Load time series data
    time_series = np.load(time_series_path)
    
    # Load labels
    labels = np.load(labels_path)
    
    # Get list of image paths
    image_paths = sorted([
        os.path.join(image_dir, f) 
        for f in os.listdir(image_dir) 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    
    # Check dimensions
    assert len(image_paths) == len(labels) == len(time_series), \
        "Mismatch in dataset dimensions"
    
    # Get transforms
    transforms_dict = get_transforms(img_size)
    
    # Split data into train, validation, and test sets
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
        transform=transforms_dict['val']  # Use val transform for test set
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
    Visualize attention weights from the model.
    
    Args:
        model (TCNCoralModel): Trained model
        dataloader (DataLoader): Data loader
        num_samples (int): Number of samples to visualize
        save_dir (str): Directory to save plots
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    samples = []
    for batch in dataloader:
        samples.append(batch)
        if len(samples) * dataloader.batch_size >= num_samples:
            break
    
    fig_idx = 0
    for batch in samples:
        images = batch['image'].to(device)
        time_series = batch['time_series'].to(device)
        labels = batch['label'].to(device)
        
        batch_size = images.size(0)
        
        with torch.no_grad():
            outputs = model(images, time_series)
            
            # Check if model returned attention weights
            if isinstance(outputs, tuple):
                logits, attention_dict = outputs
                attention_weights = attention_dict['attention_weights']
                
                # Plot attention weights for each sample in the batch
                for i in range(min(batch_size, num_samples - fig_idx)):
                    if fig_idx >= num_samples:
                        break
                    
                    # Get logits and true label
                    sample_logit = logits[i].item()
                    sample_prob = torch.sigmoid(logits[i]).item()
                    sample_pred = 1 if sample_prob > 0.5 else 0
                    sample_label = labels[i].item()
                    
                    # Get attention weights and time series
                    sample_attn = attention_weights[i].cpu().numpy()
                    sample_ts = time_series[i].cpu().numpy()
                    
                    # Plot attention
                    plt.figure(figsize=(12, 5))
                    
                    # Time series plot
                    plt.subplot(1, 2, 1)
                    seq_len, num_features = sample_ts.shape
                    for j in range(num_features):
                        plt.plot(range(seq_len), sample_ts[:, j], label=f'Feature {j+1}')
                    plt.xlabel('Time')
                    plt.ylabel('Value')
                    plt.title('Time Series Features')
                    plt.legend()
                    
                    # Attention plot
                    plt.subplot(1, 2, 2)
                    plt.imshow(sample_attn, cmap='viridis', aspect='auto')
                    plt.colorbar()
                    plt.xlabel('Sequence')
                    plt.ylabel('Attention Head')
                    plt.title(f'Attention Weights (Label: {sample_label}, Pred: {sample_pred}, Prob: {sample_prob:.2f})')
                    
                    plt.tight_layout()
                    
                    if save_dir:
                        plt.savefig(os.path.join(save_dir, f'attention_{fig_idx}.png'))
                        plt.close()
                    else:
                        plt.show()
                    
                    fig_idx += 1
            else:
                print("Model does not return attention weights.")
                break


if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Temporal Convolutional Network (TCN) for Coral Bleaching Prediction")
    print("=================================================================")
    
    # Configuration
    img_size = 224
    batch_size = 4
    
    # Model parameters
    model_params = {
        'input_channels': 3,              # RGB images
        'img_size': img_size,             # Image size
        'time_input_dim': 8,              # Number of time series features
        'sequence_length': 24,            # Number of time steps
        'img_feature_dim': 256,           # Image feature dimension
        'time_feature_dim': 128,          # Time series feature dimension
        'hidden_dims': [64, 128, 256],    # TCN hidden dimensions
        'fusion_hidden_dim': 128,         # Fusion hidden dimension
        'fusion_output_dim': 64,          # Fusion output dimension
        'cnn_backbone': 'resnet18',       # CNN backbone
        'kernel_size': 3,                 # TCN kernel size
        'dropout': 0.3,                   # Dropout probability
        'freeze_backbone': False,         # Don't freeze CNN backbone
        'use_attention': True             # Use attention in TCN
    }
    
    # Create model for testing
    print("\n1. Creating model...")
    model = TCNCoralModel(**model_params)
    print(model)
    
    # Move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    model = model.to(device)
    
    try:
        # Generate synthetic data for testing
        print("\n2. Generating synthetic test data...")
        
        # Synthetic images: [batch_size, channels, height, width]
        images = torch.randn(batch_size, 3, img_size, img_size).to(device)
        print(f"   Images shape: {images.shape}")
        
        # Synthetic time series: [batch_size, sequence_length, time_input_dim]
        time_series = torch.randn(batch_size, 24, 8).to(device)
        print(f"   Time series shape: {time_series.shape}")
        
        # Enable debugging mode for detailed shape information
        debug_mode = True
        
        # Set model to evaluation mode
        model.eval()
        
        # Forward pass
        print("\n3. Performing forward pass...")
        with torch.no_grad():
            # Debug step: Extract features and print shapes
            if debug_mode:
                img_features = model.cnn_module(images)
                print(f"   Image features shape: {img_features.shape}")
                
                tcn_out = model.tcn_module(time_series)
                if isinstance(tcn_out, tuple):
                    time_features, _ = tcn_out
                else:
                    time_features = tcn_out
                print(f"   Time features shape: {time_features.shape}")
                
                # Debug fusion module
                img_proj = model.fusion_module.img_proj(img_features)
                time_proj = model.fusion_module.time_proj(time_features)
                print(f"   Projected image features shape: {img_proj.shape}")
                print(f"   Projected time features shape: {time_proj.shape}")
                
                # Calculate context
                concat_features = torch.cat([img_proj, time_proj], dim=1)
                context = torch.cat([img_proj, time_proj, concat_features], dim=1)
                print(f"   Context shape: {context.shape}")
                print(f"   Context dimension: {context.shape[1]}")
                
                # Now run the full model
                print("   Running full model forward pass...")
                
            # Full forward pass
            outputs = model(images, time_series)
        
        # Process outputs
        if isinstance(outputs, tuple):
            logits, attention_dict = outputs
            has_attention = True
            attention_weights = attention_dict['attention_weights']
            print(f"   Model returned logits with shape {logits.shape} and attention weights")
        else:
            logits = outputs
            has_attention = False
            print(f"   Model returned logits with shape {logits.shape}")
        
        # Convert logits to probabilities and predictions
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        
        # Create synthetic labels
        labels = torch.randint(0, 2, (batch_size, 1)).float().to(device)
        
        # Print results
        print("\n4. Model predictions:")
        for i in range(batch_size):
            print(f"   Sample {i+1}: Logit={logits[i].item():.4f}, Probability={probs[i].item():.4f}, " +
                f"Prediction={preds[i].item():.0f}, Label={labels[i].item():.0f}")
        
        # Visualize attention if available
        if has_attention:
            print("\n5. Visualizing attention for first sample...")
            plt.figure(figsize=(12, 6))
            
            # Get attention weights for first sample
            sample_attn = attention_weights[0].cpu().numpy()
            sample_ts = time_series[0].cpu().numpy()
            
            # Plot attention
            plt.subplot(1, 2, 1)
            # Time series visualization
            for j in range(sample_ts.shape[1]):
                plt.plot(range(sample_ts.shape[0]), sample_ts[:, j], label=f'Feature {j+1}' if j < 3 else "")
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.title('Time Series Features')
            plt.legend(loc='upper right')
            
            plt.subplot(1, 2, 2)
            # Attention visualization
            im = plt.imshow(sample_attn, cmap='viridis', aspect='auto')
            plt.colorbar(im)
            plt.xlabel('Time Step')
            plt.ylabel('Attention Weight')
            plt.title('TCN Attention Weights')
            
            # Save figure
            plt.tight_layout()
            plt.savefig('tcn_attention_visualization.png')
            print("   Attention visualization saved to 'tcn_attention_visualization.png'")
        
        # Create PyTorch Lightning model
        print("\n6. Creating PyTorch Lightning model...")
        lightning_model = TCNLightningModel(
            model_params=model_params,
            learning_rate=1e-4,
            weight_decay=1e-5,
            pos_weight=2.0
        )
        
        print("\nExample completed successfully!")
        
    except Exception as e:
        print(f"\nError during execution: {str(e)}")
        import traceback
        traceback.print_exc()