"""
Temporal Convolutional Network (TCN) model for coral bleaching prediction.
Combines CNN for images with TCN for time series environmental data.

This model incorporates:
- ResNet or EfficientNet for image processing
- TCN for temporal patterns with dilated convolutions
- Sequence-aware attention mechanism
- Cross-modal feature fusion
- Built-in feature importance assessment
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import polars as plrs
from typing import Tuple, Dict, List, Optional, Union
import pytorch_lightning as pl


class TemporalBlock(nn.Module):
    """
    Basic temporal convolutional block with residual connection.
    
    Each block consists of two dilated causal convolutions with normalization,
    activation, and dropout, followed by a residual connection.
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
        """
        Initialize temporal block.
        
        Args:
            n_inputs: Number of input channels
            n_outputs: Number of output channels
            kernel_size: Size of the convolutional kernel
            stride: Stride of the convolution
            dilation: Dilation factor
            padding: Padding size
            dropout: Dropout probability
        """
        super(TemporalBlock, self).__init__()
        
        # First dilated convolution
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # Second dilated convolution
        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection if input and output dimensions differ
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        # Weight initialization
        self.init_weights()
    
    def init_weights(self):
        """Initialize the weights of convolutional layers."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the temporal block.
        
        Args:
            x: Input tensor of shape [batch_size, n_inputs, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, n_outputs, seq_len]
        """
        # Save original input for residual connection
        residual = x
        
        # First dilated convolution
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # Second dilated convolution
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        # Residual connection
        if self.downsample is not None:
            residual = self.downsample(residual)
        
        # Add residual connection
        out = out + residual
        
        return out


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for time series processing.
    
    Consists of multiple stacked temporal blocks with increasing dilation.
    """
    
    def __init__(
        self, 
        num_inputs: int, 
        num_channels: List[int], 
        kernel_size: int = 3, 
        dropout: float = 0.2
    ):
        """
        Initialize Temporal Convolutional Network.
        
        Args:
            num_inputs: Number of input channels
            num_channels: List specifying the number of channels in each layer
            kernel_size: Kernel size for all convolutions
            dropout: Dropout probability
        """
        super(TemporalConvNet, self).__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Exponentially increasing dilation
            
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Ensure causality by using padding = (kernel_size-1) * dilation
            padding = (kernel_size-1) * dilation_size
            
            # Add temporal block
            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=padding, dropout=dropout
            )]
        
        # Sequential container of temporal blocks
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN.
        
        Args:
            x: Input tensor of shape [batch_size, num_inputs, seq_len]
            
        Returns:
            Output tensor of shape [batch_size, num_channels[-1], seq_len]
        """
        return self.network(x)


class SequenceAttention(nn.Module):
    """
    Sequence attention mechanism for emphasizing important time steps.
    
    Applies attention along the sequence dimension.
    """
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        """
        Initialize sequence attention.
        
        Args:
            hidden_dim: Dimension of input features
            dropout: Dropout probability
        """
        super(SequenceAttention, self).__init__()
        
        # Query, key, value projections
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([hidden_dim]))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply sequence attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, hidden_dim]
            
        Returns:
            Tuple of (output tensor of shape [batch_size, seq_len, hidden_dim],
                     attention weights of shape [batch_size, seq_len, seq_len])
        """
        batch_size, seq_len, hidden_dim = x.shape
        
        # Ensure scale is on the correct device
        scale = self.scale.to(x.device)
        
        # Linear projections
        Q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        K = self.key(x)    # [batch_size, seq_len, hidden_dim]
        V = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # Calculate attention scores
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # [batch_size, seq_len, seq_len]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch_size, seq_len, seq_len]
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        context = torch.matmul(attn_weights, V)  # [batch_size, seq_len, hidden_dim]
        
        # Output projection
        output = self.output_proj(context)  # [batch_size, seq_len, hidden_dim]
        
        return output, attn_weights


class CNNModule(nn.Module):
    """
    CNN module for processing image data using pre-trained backbones.
    
    Extracts visual features from coral reef images.
    """
    
    def __init__(
        self, 
        backbone: str = 'resnet18', 
        pretrained: bool = True, 
        output_dim: int = 256,
        dropout: float = 0.3
    ):
        """
        Initialize CNN module.
        
        Args:
            backbone: Name of backbone CNN architecture
            pretrained: Whether to use pre-trained weights
            output_dim: Dimension of output features
            dropout: Dropout probability
        """
        super(CNNModule, self).__init__()
        
        # Select backbone model
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            feature_dim = 512
        elif backbone == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            feature_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Replace classifier/fully connected layer with identity to get features
        if 'resnet' in backbone:
            self.backbone.fc = nn.Identity()
        else:  # EfficientNet
            self.backbone.classifier = nn.Identity()
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Process image through CNN backbone.
        
        Args:
            x: Image tensor of shape [batch_size, channels, height, width]
            
        Returns:
            Feature tensor of shape [batch_size, output_dim]
        """
        # Extract features
        features = self.backbone(x)  # [batch_size, feature_dim]
        
        # Project to output dimension
        output = self.projection(features)  # [batch_size, output_dim]
        
        return output


class TCNModule(nn.Module):
    """
    TCN module for processing time series data.
    
    Processes environmental time series with dilated convolutions.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int], 
        output_dim: int = 128,
        kernel_size: int = 3, 
        dropout: float = 0.2,
        attention: bool = True
    ):
        """
        Initialize TCN module.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden dimensions for each TCN layer
            output_dim: Dimension of output features
            kernel_size: Kernel size for TCN
            dropout: Dropout probability
            attention: Whether to use sequence attention
        """
        super(TCNModule, self).__init__()
        
        self.use_attention = attention
        
        # TCN network
        self.tcn = TemporalConvNet(
            num_inputs=input_dim,
            num_channels=hidden_dims,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # Sequence attention (optional)
        if attention:
            self.attention = SequenceAttention(hidden_dims[-1], dropout)
        
        # Output projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_dims[-1], output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Process time series through TCN.
        
        Args:
            x: Time series tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (output features of shape [batch_size, output_dim],
                     attention weights if attention is used)
        """
        # Transpose for TCN (expects [batch_size, input_dim, seq_len])
        x = x.transpose(1, 2)  # [batch_size, input_dim, seq_len]
        
        # Apply TCN
        out = self.tcn(x)  # [batch_size, hidden_dims[-1], seq_len]
        
        # Transpose back
        out = out.transpose(1, 2)  # [batch_size, seq_len, hidden_dims[-1]]
        
        # Apply attention if enabled
        attn_weights = None
        if self.use_attention:
            out, attn_weights = self.attention(out)
        
        # Global pooling (mean across sequence dimension)
        pooled = torch.mean(out, dim=1)  # [batch_size, hidden_dims[-1]]
        
        # Apply projection
        output = self.projection(pooled)  # [batch_size, output_dim]
        
        return output, attn_weights


class FeatureFusion(nn.Module):
    """
    Feature fusion module for combining image and time series features.
    
    Uses gating mechanism to control information flow from each modality.
    """
    
    def __init__(
        self, 
        img_dim: int, 
        time_dim: int, 
        hidden_dim: int = 128, 
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        """
        Initialize feature fusion module.
        
        Args:
            img_dim: Dimension of image features
            time_dim: Dimension of time series features
            hidden_dim: Hidden dimension for fusion layers
            output_dim: Output dimension
            dropout: Dropout probability
        """
        super(FeatureFusion, self).__init__()
        
        # Projection for each modality
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        # Gating mechanisms
        self.img_gate = nn.Sequential(
            nn.Linear(img_dim + time_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.time_gate = nn.Sequential(
            nn.Linear(img_dim + time_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Fusion layers
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
        Fuse image and time series features.
        
        Args:
            img_features: Image features of shape [batch_size, img_dim]
            time_features: Time series features of shape [batch_size, time_dim]
            
        Returns:
            Fused features of shape [batch_size, output_dim]
        """
        # Concatenate features for gating
        concat_features = torch.cat([img_features, time_features], dim=1)
        
        # Apply gating mechanisms
        img_gate_values = self.img_gate(concat_features)
        time_gate_values = self.time_gate(concat_features)
        
        # Apply projections
        img_proj = self.img_proj(img_features)
        time_proj = self.time_proj(time_features)
        
        # Apply gates
        gated_img = img_proj * img_gate_values
        gated_time = time_proj * time_gate_values
        
        # Concatenate gated features
        gated_concat = torch.cat([gated_img, gated_time], dim=1)
        
        # Apply fusion layers
        fused = self.fusion(gated_concat)
        
        return fused


class TCNCoralModel(nn.Module):
    """
    Combined TCN model for coral bleaching prediction.
    
    Processes both image and time series data with specialized sub-networks,
    then fuses the features for final prediction.
    """
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        cnn_backbone: str = 'resnet18',
        cnn_output_dim: int = 256,
        tcn_hidden_dims: List[int] = [64, 128, 256],
        tcn_output_dim: int = 128,
        tcn_kernel_size: int = 3,
        fusion_hidden_dim: int = 128,
        fusion_output_dim: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True
    ):
        """
        Initialize the TCN coral bleaching prediction model.
        
        Args:
            time_steps: Number of time steps in time series data
            num_features: Number of features in time series data
            cnn_backbone: CNN backbone architecture name
            cnn_output_dim: Dimension of CNN output features
            tcn_hidden_dims: Hidden dimensions for TCN layers
            tcn_output_dim: Dimension of TCN output features
            tcn_kernel_size: Kernel size for TCN
            fusion_hidden_dim: Hidden dimension for fusion layers
            fusion_output_dim: Output dimension of fusion module
            dropout: Dropout probability
            use_attention: Whether to use sequence attention in TCN
        """
        super(TCNCoralModel, self).__init__()
        
        # CNN for image processing
        self.cnn_module = CNNModule(
            backbone=cnn_backbone,
            pretrained=True,
            output_dim=cnn_output_dim,
            dropout=dropout
        )
        
        # TCN for time series processing
        self.tcn_module = TCNModule(
            input_dim=num_features,
            hidden_dims=tcn_hidden_dims,
            output_dim=tcn_output_dim,
            kernel_size=tcn_kernel_size,
            dropout=dropout,
            attention=use_attention
        )
        
        # Feature fusion
        self.fusion_module = FeatureFusion(
            img_dim=cnn_output_dim,
            time_dim=tcn_output_dim,
            hidden_dim=fusion_hidden_dim,
            output_dim=fusion_output_dim,
            dropout=dropout
        )
        
        # Output layer
        self.output_layer = nn.Linear(fusion_output_dim, 1)
        
        # Store configuration
        self.dropout = dropout
        self.mc_dropout = False
    
    def forward(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            image: Image tensor of shape [batch_size, channels, height, width]
            time_series: Time series tensor of shape [batch_size, time_steps, num_features]
            
        Returns:
            Tuple of (prediction logits of shape [batch_size],
                     attention weights if attention is used)
        """
        # Process image data
        img_features = self.cnn_module(image)
        
        # Process time series data
        time_features, attn_weights = self.tcn_module(time_series)
        
        # Fuse features
        fused_features = self.fusion_module(img_features, time_features)
        
        # Output layer
        output = self.output_layer(fused_features)
        
        return output.squeeze(), attn_weights
    
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
        n_samples: int = 30
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions with uncertainty estimation using MC dropout.
        
        Args:
            image: Image tensor
            time_series: Time series tensor
            n_samples: Number of MC samples
            
        Returns:
            Tuple of (mean predictions, prediction variances)
        """
        self.enable_mc_dropout()
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred, _ = self(image, time_series)
                pred = torch.sigmoid(pred)
                predictions.append(pred.unsqueeze(0))
        
        predictions = torch.cat(predictions, dim=0)
        mean_pred = torch.mean(predictions, dim=0)
        var_pred = torch.var(predictions, dim=0)
        
        self.disable_mc_dropout()
        return mean_pred, var_pred
    
    def get_feature_importance(
        self, 
        time_series: torch.Tensor, 
        target: Optional[torch.Tensor] = None,
        method: str = 'gradient'
    ) -> torch.Tensor:
        """
        Calculate feature importance for time series data.
        
        Args:
            time_series: Time series tensor of shape [batch_size, time_steps, num_features]
            target: Optional target tensor (used for gradient-based methods)
            method: Method for calculating importance ('gradient', 'integrated_gradients', 'attention')
            
        Returns:
            Feature importance tensor of shape [batch_size, num_features]
        """
        batch_size, time_steps, num_features = time_series.shape
        
        if method == 'attention' and hasattr(self.tcn_module, 'attention'):
            # Use attention weights to calculate importance
            with torch.no_grad():
                _, attn_weights = self.tcn_module(time_series)
                
                # Average attention across sequence dimension
                importance = attn_weights.mean(dim=1).mean(dim=1)  # [batch_size, time_steps]
                
                # Calculate feature importance
                feature_importance = torch.zeros(batch_size, num_features, device=time_series.device)
                
                for b in range(batch_size):
                    for t in range(time_steps):
                        for f in range(num_features):
                            feature_importance[b, f] += importance[b, t] * time_series[b, t, f]
                
                # Normalize
                feature_importance = feature_importance / feature_importance.sum(dim=1, keepdim=True)
                
                return feature_importance
                
        elif method == 'gradient' and target is not None:
            # Use gradient with respect to input
            time_series.requires_grad_(True)
            
            # Forward pass
            output, _ = self(time_series)
            
            # Backward pass
            gradients = torch.autograd.grad(
                outputs=output,
                inputs=time_series,
                grad_outputs=torch.ones_like(output),
                create_graph=True
            )[0]
            
            # Calculate feature importance as mean absolute gradient
            feature_importance = gradients.abs().mean(dim=1)  # [batch_size, num_features]
            
            # Normalize
            feature_importance = feature_importance / feature_importance.sum(dim=1, keepdim=True)
            
            return feature_importance
            
        else:
            raise ValueError(f"Unsupported importance method: {method}")


class CoralDataset(Dataset):
    """Dataset class for coral bleaching prediction."""
    
    def __init__(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        labels: np.ndarray,
        transform = None
    ):
        """
        Initialize coral dataset.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            labels: Labels of shape [num_samples]
            transform: Transformations to apply to images
        """
        # Convert to PyTorch tensors
        # Ensure images are in the correct format (N, C, H, W)
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.time_series = torch.FloatTensor(time_series)
        self.labels = torch.FloatTensor(labels)
        self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, time_series, label)
        """
        image = self.images[idx]
        time_series = self.time_series[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, time_series, label


class TCNLightningModel(pl.LightningModule):
    """PyTorch Lightning module for coral bleaching prediction with TCN."""
    
    def __init__(
        self,
        time_steps: int,
        num_features: int,
        cnn_backbone: str = 'resnet18',
        cnn_output_dim: int = 256,
        tcn_hidden_dims: List[int] = [64, 128, 256],
        tcn_output_dim: int = 128,
        tcn_kernel_size: int = 3,
        fusion_hidden_dim: int = 128,
        fusion_output_dim: int = 64,
        dropout: float = 0.3,
        use_attention: bool = True,
        learning_rate: float = 0.001,
        weight_decay: float = 1e-4
    ):
        """
        Initialize Lightning module.
        
        Args:
            time_steps: Number of time steps in time series
            num_features: Number of features in time series
            cnn_backbone: CNN backbone architecture
            cnn_output_dim: Dimension of CNN output features
            tcn_hidden_dims: Hidden dimensions for TCN layers
            tcn_output_dim: Dimension of TCN output features
            tcn_kernel_size: Kernel size for TCN
            fusion_hidden_dim: Hidden dimension for fusion layers
            fusion_output_dim: Output dimension of fusion module
            dropout: Dropout probability
            use_attention: Whether to use sequence attention in TCN
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super(TCNLightningModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = TCNCoralModel(
            time_steps=time_steps,
            num_features=num_features,
            cnn_backbone=cnn_backbone,
            cnn_output_dim=cnn_output_dim,
            tcn_hidden_dims=tcn_hidden_dims,
            tcn_output_dim=tcn_output_dim,
            tcn_kernel_size=tcn_kernel_size,
            fusion_hidden_dim=fusion_hidden_dim,
            fusion_output_dim=fusion_output_dim,
            dropout=dropout,
            use_attention=use_attention
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
    
    def forward(self, image, time_series):
        """Forward pass through the model."""
        return self.model(image, time_series)
    
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
        image, time_series, y = batch
        y_hat, _ = self(image, time_series)
        loss = self.criterion(y_hat, y)
        
        # Calculate accuracy
        acc = self.train_acc(torch.sigmoid(y_hat), y.int())
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        image, time_series, y = batch
        y_hat, _ = self(image, time_series)
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
        image, time_series, y = batch
        y_hat, _ = self(image, time_series)
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
    
    def predict_with_uncertainty(self, image, time_series, n_samples=30):
        """Make predictions with uncertainty estimation."""
        return self.model.predict_with_uncertainty(image, time_series, n_samples)
    
    def get_attention_weights(self, image, time_series):
        """Get attention weights for interpretability."""
        with torch.no_grad():
            _, attention_weights = self(image, time_series)
        return attention_weights
    
    def get_feature_importance(self, time_series, target=None, method='attention'):
        """Get feature importance scores."""
        return self.model.get_feature_importance(time_series, target, method)


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


def detect_early_warning_signals(
    time_series: np.ndarray, 
    window_size: int = 5, 
    thresholds: Dict[str, float] = None
) -> np.ndarray:
    """
    Detect early warning signals of coral bleaching in time series data.
    
    Uses statistical indicators like variance, autocorrelation, and skewness
    to identify critical transitions before they occur.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        window_size: Size of rolling window for calculating indicators
        thresholds: Dictionary of threshold values for each indicator
        
    Returns:
        Early warning signals of shape [num_samples, time_steps]
    """
    num_samples, time_steps, num_features = time_series.shape
    
    # Initialize thresholds if not provided
    if thresholds is None:
        thresholds = {
            'variance': 1.5,
            'autocorrelation': 0.8,
            'skewness': 2.0
        }
    
    # Initialize early warning signals
    ews = np.zeros((num_samples, time_steps))
    
    for i in range(num_samples):
        for t in range(window_size, time_steps):
            # Extract window
            window = time_series[i, t-window_size:t, :]
            
            # Calculate indicators
            variance = np.var(window, axis=0)
            
            # Calculate lag-1 autocorrelation
            autocorr = np.zeros(num_features)
            for f in range(num_features):
                signal = window[:, f]
                # Normalize signal
                signal = (signal - np.mean(signal)) / np.std(signal)
                # Calculate autocorrelation
                autocorr[f] = np.corrcoef(signal[:-1], signal[1:])[0, 1]
            
            # Calculate skewness
            skewness = np.zeros(num_features)
            for f in range(num_features):
                signal = window[:, f]
                # Normalize signal
                signal = (signal - np.mean(signal)) / np.std(signal)
                # Calculate skewness
                skewness[f] = np.mean(signal**3)
            
            # Check if indicators exceed thresholds
            var_exceed = np.mean(variance) > thresholds['variance']
            autocorr_exceed = np.mean(autocorr) > thresholds['autocorrelation']
            skew_exceed = np.mean(np.abs(skewness)) > thresholds['skewness']
            
            # Combine indicators
            ews[i, t] = var_exceed + autocorr_exceed + skew_exceed
    
    return ews


if __name__ == "__main__":
    # Example usage
    import matplotlib.pyplot as plt
    from torch.utils.data import random_split, DataLoader
    
    # Assume we have loaded the data
    images = np.random.randn(100, 224, 224, 3)  # 100 sample images
    time_series = np.random.randn(100, 24, 8)   # 100 samples, 24 time steps, 8 features
    labels = np.random.randint(0, 2, 100)       # Binary labels
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    dataset = CoralDataset(images, time_series, labels, transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)
    
    # Initialize model
    model = TCNCoralModel(
        time_steps=24,
        num_features=8,
        tcn_hidden_dims=[64, 128, 256]
    )
    
    # Print model summary
    print(model)
    
    # Create Lightning module
    lightning_model = TCNLightningModel(
        time_steps=24,
        num_features=8,
        tcn_hidden_dims=[64, 128, 256]
    )
    
    # Example forward pass
    image_batch, ts_batch, _ = next(iter(train_loader))
    outputs, attention = model(image_batch, ts_batch)
    print(f"Output shape: {outputs.shape}")
    
    # Detect early warning signals
    ews = detect_early_warning_signals(time_series)
    print(f"Early warning signals shape: {ews.shape}")