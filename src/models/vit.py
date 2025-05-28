"""
Vision Transformer (ViT) implementation for coral bleaching prediction.
Combines image and time series data through a dual-transformer architecture.
"""

import os
import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional, Union, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import pytorch_lightning as pl
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import cv2

from .wavelet import WaveletTransform


class PatchEmbedding(nn.Module):
    """
    Split image into patches and embed them.
    
    Parameters:
        img_size (int): Size of the input image (assumes square image)
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection to embedding dimension
        self.proj = nn.Conv2d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to convert image to patch embeddings.
        
        Args:
            x: Input image of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Patch embeddings of shape [batch_size, num_patches, embed_dim]
        """
        # [batch_size, in_channels, img_size, img_size] -> [batch_size, embed_dim, patches_h, patches_w]
        x = self.proj(x)
        
        # [batch_size, embed_dim, patches_h, patches_w] -> [batch_size, embed_dim, num_patches]
        x = x.flatten(2)
        
        # [batch_size, embed_dim, num_patches] -> [batch_size, num_patches, embed_dim]
        x = x.transpose(1, 2)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Parameters:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Ensure embedding dimension is divisible by number of heads
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for queries, keys, values
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = self.head_dim ** -0.5
        
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for multi-head attention.
        
        Args:
            query: Query tensor of shape [batch_size, seq_len_q, embed_dim]
            key: Key tensor of shape [batch_size, seq_len_k, embed_dim]
            value: Value tensor of shape [batch_size, seq_len_v, embed_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size = query.size(0)
        
        # Linear projections
        q = self.q_proj(query)  # [batch_size, seq_len_q, embed_dim]
        k = self.k_proj(key)    # [batch_size, seq_len_k, embed_dim]
        v = self.v_proj(value)  # [batch_size, seq_len_v, embed_dim]
        
        # Reshape for multi-head attention
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_q, head_dim]
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_k, head_dim]
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [batch_size, num_heads, seq_len_v, head_dim]
        
        # Compute scaled dot-product attention
        # [batch_size, num_heads, seq_len_q, seq_len_k]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
        
        # Apply softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        # [batch_size, num_heads, seq_len_q, head_dim]
        output = torch.matmul(attn_weights, v)
        
        # Reshape back
        # [batch_size, seq_len_q, num_heads, head_dim] -> [batch_size, seq_len_q, embed_dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        # Final linear projection
        output = self.out_proj(output)
        
        return output, attn_weights


class PositionwiseFeedForward(nn.Module):
    """
    Position-wise feed-forward network for transformer.
    
    Parameters:
        embed_dim (int): Embedding dimension
        ff_dim (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        ff_dim: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for position-wise feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer.
    
    Parameters:
        embed_dim (int): Embedding dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        max_len: int = 5000, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter but should be saved and moved with the model)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add positional encoding.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single layer of transformer encoder.
    
    Parameters:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = PositionwiseFeedForward(embed_dim, ff_dim, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for transformer encoder layer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Self-attention with residual connection and layer normalization
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.self_attn(x, x, x, attn_mask)
        x = residual + self.dropout1(x)
        
        # Feed-forward with residual connection and layer normalization
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + self.dropout2(x)
        
        return x, attn_weights


class TransformerEncoder(nn.Module):
    """
    Transformer encoder with multiple layers.
    
    Parameters:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        num_layers (int): Number of encoder layers
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        num_layers: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(
        self, 
        x: torch.Tensor, 
        attn_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for transformer encoder.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            attn_mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor, list of attention weights from each layer)
        """
        attn_weights_list = []
        
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask)
            attn_weights_list.append(attn_weights)
        
        return x, attn_weights_list


class TimeSeriesEmbedding(nn.Module):
    """
    Embedding for time series data.
    
    Parameters:
        input_dim (int): Input dimension (number of features)
        embed_dim (int): Embedding dimension
        max_len (int): Maximum sequence length
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        max_len: int = 1000, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Linear projection for feature embedding
        self.feature_embed = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len, dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to embed time series data.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Embedded tensor of shape [batch_size, seq_len, embed_dim]
        """
        # Project features to embedding dimension
        x = self.feature_embed(x)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        return x


class ImageTransformer(nn.Module):
    """
    Transformer for image data.
    
    Parameters:
        img_size (int): Size of the input image
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        num_layers (int): Number of transformer layers
        num_classes (int): Number of classes for classification
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        embed_dim: int = 768, 
        num_heads: int = 12, 
        ff_dim: int = 3072, 
        num_layers: int = 12, 
        num_classes: int = 2, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.num_patches = self.patch_embedding.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for the transformer."""
        # Initialize patch embedding
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize transformer layers
        self.apply(self._init_transformer_weights)
        
    def _init_transformer_weights(self, m):
        """Initialize transformer layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for image transformer.
        
        Args:
            x: Input tensor of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Tuple of (classification logits, final hidden state, attention weights)
        """
        batch_size = x.size(0)
        
        # Convert image to patches and embed
        x = self.patch_embedding(x)  # [batch_size, num_patches, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, num_patches + 1, embed_dim]
        
        # Add position embedding
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Apply transformer encoder
        x, attn_weights = self.transformer(x)
        
        # Classification using the [CLS] token
        x_cls = self.norm(x[:, 0])  # Use only the classification token
        logits = self.classifier(x_cls)
        
        return logits, x, attn_weights


class TimeSeriesTransformer(nn.Module):
    """
    Transformer for time series data.
    
    Parameters:
        input_dim (int): Input dimension (number of features)
        embed_dim (int): Embedding dimension
        max_len (int): Maximum sequence length
        num_heads (int): Number of attention heads
        ff_dim (int): Feed-forward dimension
        num_layers (int): Number of transformer layers
        num_classes (int): Number of classes for classification
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        input_dim: int, 
        embed_dim: int, 
        max_len: int = 1000, 
        num_heads: int = 8, 
        ff_dim: int = 2048, 
        num_layers: int = 6, 
        num_classes: int = 2, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Make sure embed_dim is divisible by num_heads
        # If not, adjust embed_dim to the nearest value divisible by num_heads
        if embed_dim % num_heads != 0:
            new_embed_dim = ((embed_dim // num_heads) + 1) * num_heads
            print(f"Warning: Adjusting embed_dim from {embed_dim} to {new_embed_dim} to be divisible by {num_heads} heads")
            embed_dim = new_embed_dim
            
        self.embedding = TimeSeriesEmbedding(input_dim, embed_dim, max_len, dropout)
        
        # Global class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer encoder
        self.transformer = TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        
        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for the transformer."""
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize transformer layers
        self.apply(self._init_transformer_weights)
        
    def _init_transformer_weights(self, m):
        """Initialize transformer layer weights."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass for time series transformer.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (classification logits, final hidden state, attention weights)
        """
        batch_size = x.size(0)
        
        # Embed time series
        x = self.embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Add classification token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # [batch_size, seq_len + 1, embed_dim]
        
        # Apply transformer encoder
        x, attn_weights = self.transformer(x)
        
        # Classification using the [CLS] token
        x_cls = self.norm(x[:, 0])  # Use only the classification token
        logits = self.classifier(x_cls)
        
        return logits, x, attn_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing image and time series features.
    
    Parameters:
        embed_dim (int): Embedding dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        dropout: float = 0.0
    ):
        super().__init__()
        
        # Make sure embed_dim is divisible by num_heads
        # If not, adjust embed_dim to the nearest value divisible by num_heads
        if embed_dim % num_heads != 0:
            new_embed_dim = ((embed_dim // num_heads) + 1) * num_heads
            print(f"Warning: Adjusting embed_dim from {embed_dim} to {new_embed_dim} to be divisible by {num_heads} heads")
            embed_dim = new_embed_dim
            
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross-attention: image to time and time to image
        self.img2time_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.time2img_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        # Layer normalization
        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_time = nn.LayerNorm(embed_dim)
        
        # Feed-forward networks
        self.ff_img = PositionwiseFeedForward(embed_dim, embed_dim * 4, dropout)
        self.ff_time = PositionwiseFeedForward(embed_dim, embed_dim * 4, dropout)
        
        # Output layer normalization
        self.norm_img_out = nn.LayerNorm(embed_dim)
        self.norm_time_out = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        img_features: torch.Tensor, 
        time_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-modal attention.
        
        Args:
            img_features: Image features of shape [batch_size, img_seq_len, embed_dim]
            time_features: Time series features of shape [batch_size, time_seq_len, embed_dim]
            
        Returns:
            Tuple of (fused features, image attention weights, time attention weights)
        """
        # Normalize inputs
        img_norm = self.norm_img(img_features)
        time_norm = self.norm_time(time_features)
        
        # Cross-attention: image attends to time
        img2time, img2time_weights = self.img2time_attn(img_norm, time_norm, time_norm)
        
        # Cross-attention: time attends to image
        time2img, time2img_weights = self.time2img_attn(time_norm, img_norm, img_norm)
        
        # Residual connections
        img_features = img_features + self.dropout(img2time)
        time_features = time_features + self.dropout(time2img)
        
        # Feed-forward networks with residual connections
        img_ff = self.norm_img_out(img_features)
        time_ff = self.norm_time_out(time_features)
        
        img_features = img_features + self.dropout(self.ff_img(img_ff))
        time_features = time_features + self.dropout(self.ff_time(time_ff))
        
        return img_features, time_features, img2time_weights, time2img_weights


class FusionModule(nn.Module):
    """
    Module for fusing image and time series features.
    
    Parameters:
        img_dim (int): Image feature dimension
        time_dim (int): Time series feature dimension
        output_dim (int): Output dimension
        num_heads (int): Number of attention heads
        dropout (float): Dropout probability
        fusion_type (str): Type of fusion - 'concat', 'attention', or 'both'
    """
    
    def __init__(
        self, 
        img_dim: int, 
        time_dim: int, 
        output_dim: int, 
        num_heads: int = 8, 
        dropout: float = 0.0, 
        fusion_type: str = 'both'
    ):
        super().__init__()
        
        self.fusion_type = fusion_type
        
        # Make sure output_dim is divisible by num_heads for attention-based fusion
        if fusion_type in ['attention', 'both'] and output_dim % num_heads != 0:
            new_output_dim = ((output_dim // num_heads) + 1) * num_heads
            print(f"Warning: Adjusting output_dim from {output_dim} to {new_output_dim} to be divisible by {num_heads} heads")
            output_dim = new_output_dim
            
        # Projection layers to common dimension
        self.img_proj = nn.Linear(img_dim, output_dim)
        self.time_proj = nn.Linear(time_dim, output_dim)
        
        if fusion_type in ['attention', 'both']:
            # Cross-modal attention
            self.cross_attention = CrossModalAttention(output_dim, num_heads, dropout)
        
        if fusion_type in ['concat', 'both']:
            # Concatenation fusion
            self.concat_proj = nn.Linear(output_dim * 2, output_dim)
        else:
            # Averaging fusion for attention-only
            self.agg_proj = nn.Linear(output_dim, output_dim)
        
        # Final layer normalization
        self.norm = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, 
        img_features: torch.Tensor, 
        time_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass for feature fusion.
        
        Args:
            img_features: Image features of shape [batch_size, img_seq_len, img_dim]
            time_features: Time series features of shape [batch_size, time_seq_len, time_dim]
            
        Returns:
            Tuple of (fused features, optional image-to-time attention weights, optional time-to-image attention weights)
        """
        # Project to common dimension
        img_proj = self.img_proj(img_features)  # [batch_size, img_seq_len, output_dim]
        time_proj = self.time_proj(time_features)  # [batch_size, time_seq_len, output_dim]
        
        img2time_weights = None
        time2img_weights = None
        
        if self.fusion_type in ['attention', 'both']:
            # Cross-modal attention
            img_attn, time_attn, img2time_weights, time2img_weights = self.cross_attention(img_proj, time_proj)
        else:
            img_attn, time_attn = img_proj, time_proj
        
        # CLS token features (first token)
        img_cls = img_attn[:, 0]  # [batch_size, output_dim]
        time_cls = time_attn[:, 0]  # [batch_size, output_dim]
        
        if self.fusion_type in ['concat', 'both']:
            # Concatenate CLS tokens
            concat_features = torch.cat([img_cls, time_cls], dim=-1)  # [batch_size, output_dim * 2]
            fused_features = self.concat_proj(concat_features)  # [batch_size, output_dim]
        else:
            # Average CLS tokens
            avg_features = (img_cls + time_cls) / 2.0
            fused_features = self.agg_proj(avg_features)
            
        # Final normalization and dropout
        fused_features = self.norm(fused_features)
        fused_features = self.dropout(fused_features)
        
        return fused_features, img2time_weights, time2img_weights


class DualTransformerModel(nn.Module):
    """
    Dual transformer model for coral bleaching prediction using both image and time series data.
    
    Parameters:
        img_size (int): Size of the input image
        patch_size (int): Size of each patch
        in_channels (int): Number of input channels for image
        time_input_dim (int): Input dimension for time series
        max_len (int): Maximum sequence length for time series
        img_embed_dim (int): Embedding dimension for image
        time_embed_dim (int): Embedding dimension for time series
        fusion_dim (int): Dimension for fusion features
        img_num_heads (int): Number of attention heads for image transformer
        time_num_heads (int): Number of attention heads for time series transformer
        fusion_num_heads (int): Number of attention heads for fusion
        img_ff_dim (int): Feed-forward dimension for image transformer
        time_ff_dim (int): Feed-forward dimension for time series transformer
        img_num_layers (int): Number of transformer layers for image
        time_num_layers (int): Number of transformer layers for time series
        num_classes (int): Number of classes for classification
        dropout (float): Dropout probability
        fusion_type (str): Type of fusion - 'concat', 'attention', or 'both'
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        time_input_dim: int = 8,
        max_len: int = 100,
        img_embed_dim: int = 768,
        time_embed_dim: int = 512,
        fusion_dim: int = 512,
        img_num_heads: int = 12,
        time_num_heads: int = 8,
        fusion_num_heads: int = 8,
        img_ff_dim: int = 3072,
        time_ff_dim: int = 2048,
        img_num_layers: int = 12,
        time_num_layers: int = 6,
        num_classes: int = 2,
        dropout: float = 0.1,
        fusion_type: str = 'both'
    ):
        super().__init__()
        
        # Make sure embedding dimensions are divisible by respective number of heads
        if img_embed_dim % img_num_heads != 0:
            new_img_embed_dim = ((img_embed_dim // img_num_heads) + 1) * img_num_heads
            print(f"Warning: Adjusting img_embed_dim from {img_embed_dim} to {new_img_embed_dim} to be divisible by {img_num_heads} heads")
            img_embed_dim = new_img_embed_dim
            
        if time_embed_dim % time_num_heads != 0:
            new_time_embed_dim = ((time_embed_dim // time_num_heads) + 1) * time_num_heads
            print(f"Warning: Adjusting time_embed_dim from {time_embed_dim} to {new_time_embed_dim} to be divisible by {time_num_heads} heads")
            time_embed_dim = new_time_embed_dim
            
        if fusion_dim % fusion_num_heads != 0:
            new_fusion_dim = ((fusion_dim // fusion_num_heads) + 1) * fusion_num_heads
            print(f"Warning: Adjusting fusion_dim from {fusion_dim} to {new_fusion_dim} to be divisible by {fusion_num_heads} heads")
            fusion_dim = new_fusion_dim
        
        # Image transformer
        self.image_transformer = ImageTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=img_embed_dim,
            num_heads=img_num_heads,
            ff_dim=img_ff_dim,
            num_layers=img_num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Time series transformer
        self.time_transformer = TimeSeriesTransformer(
            input_dim=time_input_dim,
            embed_dim=time_embed_dim,
            max_len=max_len,
            num_heads=time_num_heads,
            ff_dim=time_ff_dim,
            num_layers=time_num_layers,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Feature fusion
        self.fusion_module = FusionModule(
            img_dim=img_embed_dim,
            time_dim=time_embed_dim,
            output_dim=fusion_dim,
            num_heads=fusion_num_heads,
            dropout=dropout,
            fusion_type=fusion_type
        )
        
        # Final classifier
        self.classifier = nn.Linear(fusion_dim, num_classes)
        
    def forward(
        self,
        images: torch.Tensor,
        time_series: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Forward pass for dual transformer model.
        
        Args:
            images: Input images of shape [batch_size, in_channels, img_size, img_size]
            time_series: Input time series of shape [batch_size, seq_len, time_input_dim]
            
        Returns:
            Tuple of (fused predictions, image predictions, time predictions, 
                     image attention weights, time attention weights, 
                     image-to-time attention weights, time-to-image attention weights)
        """
        # Process images
        img_logits, img_features, img_attn_weights = self.image_transformer(images)
        
        # Process time series
        time_logits, time_features, time_attn_weights = self.time_transformer(time_series)
        
        # Fuse features
        fused_features, img2time_weights, time2img_weights = self.fusion_module(img_features, time_features)
        
        # Final classification
        fused_logits = self.classifier(fused_features)
        
        return fused_logits, img_logits, time_logits, img_attn_weights, time_attn_weights, img2time_weights, time2img_weights


class CoralDataset(Dataset):
    """
    Dataset for coral bleaching prediction combining imagery and time series data.
    
    Parameters:
        image_paths (List[str]): List of paths to image files
        time_series (np.ndarray): Time series data
        labels (np.ndarray): Labels
        image_transform (callable, optional): Transform to apply to images
    """
    
    def __init__(
        self,
        image_paths: List[str],
        time_series: np.ndarray,
        labels: np.ndarray,
        image_transform: Optional[Any] = None
    ):
        self.image_paths = image_paths
        self.time_series = time_series
        self.labels = labels
        self.image_transform = image_transform
        
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load image
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms if provided
        if self.image_transform:
            image = self.image_transform(image)
        else:
            # Default transform: convert to tensor and normalize
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
        
        # Get time series
        time_series = torch.from_numpy(self.time_series[idx]).float()
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'image': image,
            'time_series': time_series,
            'label': label
        }


class CoralTransformerLightning(pl.LightningModule):
    """
    PyTorch Lightning module for training the coral bleaching prediction model.
    
    Parameters:
        model_config (Dict): Configuration for the model
        learning_rate (float): Learning rate
        weight_decay (float): Weight decay
        fusion_weight (float): Weight for fusion loss
        img_weight (float): Weight for image loss
        time_weight (float): Weight for time series loss
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        fusion_weight: float = 0.6,
        img_weight: float = 0.2,
        time_weight: float = 0.2
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # Create model
        self.model = DualTransformerModel(**model_config)
        
        # Loss weights
        self.fusion_weight = fusion_weight
        self.img_weight = img_weight
        self.time_weight = time_weight
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, images, time_series):
        return self.model(images, time_series)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Forward pass
        fused_logits, img_logits, time_logits, _, _, _, _ = self.model(images, time_series)
        
        # Calculate losses
        fusion_loss = self.criterion(fused_logits, labels)
        img_loss = self.criterion(img_logits, labels)
        time_loss = self.criterion(time_logits, labels)
        
        # Weighted combined loss
        loss = (
            self.fusion_weight * fusion_loss +
            self.img_weight * img_loss +
            self.time_weight * time_loss
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_fusion_loss', fusion_loss, on_step=True, on_epoch=True)
        self.log('train_img_loss', img_loss, on_step=True, on_epoch=True)
        self.log('train_time_loss', time_loss, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        time_series = batch['time_series']
        labels = batch['label']
        
        # Forward pass
        fused_logits, img_logits, time_logits, _, _, _, _ = self.model(images, time_series)
        
        # Calculate losses
        fusion_loss = self.criterion(fused_logits, labels)
        img_loss = self.criterion(img_logits, labels)
        time_loss = self.criterion(time_logits, labels)
        
        # Weighted combined loss
        loss = (
            self.fusion_weight * fusion_loss +
            self.img_weight * img_loss +
            self.time_weight * time_loss
        )
        
        # Calculate accuracy
        fused_preds = torch.argmax(fused_logits, dim=1)
        accuracy = (fused_preds == labels).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_fusion_loss', fusion_loss, on_epoch=True)
        self.log('val_img_loss', img_loss, on_epoch=True)
        self.log('val_time_loss', time_loss, on_epoch=True)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True)
        
        return {
            'val_loss': loss,
            'val_accuracy': accuracy,
            'labels': labels,
            'fused_preds': fused_preds,
            'fused_logits': fused_logits
        }
    
    def validation_epoch_end(self, outputs):
        # Collect all predictions and labels
        all_labels = torch.cat([x['labels'] for x in outputs])
        all_preds = torch.cat([x['fused_preds'] for x in outputs])
        all_logits = torch.cat([x['fused_logits'] for x in outputs])
        
        # Calculate metrics
        accuracy = accuracy_score(all_labels.cpu(), all_preds.cpu())
        precision = precision_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        recall = recall_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        f1 = f1_score(all_labels.cpu(), all_preds.cpu(), average='weighted')
        
        # Calculate AUC if binary classification
        if all_logits.shape[1] == 2:
            probas = F.softmax(all_logits, dim=1)[:, 1].cpu()
            auc = roc_auc_score(all_labels.cpu(), probas)
            self.log('val_auc', auc, prog_bar=True)
        
        # Log metrics
        self.log('val_acc_epoch', accuracy, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        # Same as validation step
        return self.validation_step(batch, batch_idx)
    
    def test_epoch_end(self, outputs):
        # Same as validation epoch end
        return self.validation_epoch_end(outputs)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=10,  # Adjust based on your dataset size and batch size
            eta_min=1e-6
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }
    
    def get_attention_maps(self, images, time_series):
        """
        Get attention maps for visualization.
        
        Args:
            images: Input images
            time_series: Input time series
            
        Returns:
            Tuple of attention maps
        """
        with torch.no_grad():
            _, _, _, img_attn_weights, time_attn_weights, img2time_weights, time2img_weights = self.model(images, time_series)
        
        return {
            'img_attention': img_attn_weights,
            'time_attention': time_attn_weights,
            'img2time_attention': img2time_weights,
            'time2img_attention': time2img_weights
        }


def visualize_attention(model, dataloader, save_dir='attention_maps'):
    """
    Visualize attention maps from the model.
    
    Args:
        model: Trained model
        dataloader: Dataloader with samples
        save_dir: Directory to save visualizations
    """
    os.makedirs(save_dir, exist_ok=True)
    
    device = next(model.parameters()).device
    model.eval()
    
    # Get a batch of data
    batch = next(iter(dataloader))
    images = batch['image'].to(device)
    time_series = batch['time_series'].to(device)
    labels = batch['label'].to(device)
    
    # Get attention maps
    attention_maps = model.get_attention_maps(images, time_series)
    
    # Visualize image attention
    if attention_maps['img_attention']:
        img_attention = attention_maps['img_attention'][-1][0].cpu().numpy()  # Last layer, first head
        img_attention = img_attention[0]  # First example in batch
        
        # Visualize attention for the CLS token to all patches
        cls_to_patches = img_attention[0, 1:]  # CLS token's attention to patches
        
        # Reshape to spatial grid
        h = w = int(np.sqrt(cls_to_patches.shape[0]))
        attn_map = cls_to_patches.reshape(h, w)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.imshow(attn_map, cmap='viridis')
        plt.colorbar()
        plt.title('CLS token attention to image patches')
        plt.savefig(os.path.join(save_dir, 'img_attention.png'))
        plt.close()
    
    # Visualize time attention
    if attention_maps['time_attention']:
        time_attention = attention_maps['time_attention'][-1][0].cpu().numpy()  # Last layer, first head
        time_attention = time_attention[0]  # First example in batch
        
        # Visualize attention for the CLS token to all time steps
        cls_to_time = time_attention[0, 1:]  # CLS token's attention to time steps
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(cls_to_time)
        plt.title('CLS token attention to time steps')
        plt.xlabel('Time step')
        plt.ylabel('Attention weight')
        plt.grid(True)
        plt.savefig(os.path.join(save_dir, 'time_attention.png'))
        plt.close()
    
    # Visualize cross-modal attention: image to time
    if attention_maps['img2time_attention'] is not None:
        img2time = attention_maps['img2time_attention'][0].cpu().numpy()  # First head
        img2time = img2time[0]  # First example in batch
        
        # Average across heads if necessary
        if len(img2time.shape) > 2:
            img2time = img2time.mean(axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(img2time, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Image to Time Series Attention')
        plt.xlabel('Time steps')
        plt.ylabel('Image patches (including CLS)')
        plt.savefig(os.path.join(save_dir, 'img2time_attention.png'))
        plt.close()
    
    # Visualize cross-modal attention: time to image
    if attention_maps['time2img_attention'] is not None:
        time2img = attention_maps['time2img_attention'][0].cpu().numpy()  # First head
        time2img = time2img[0]  # First example in batch
        
        # Average across heads if necessary
        if len(time2img.shape) > 2:
            time2img = time2img.mean(axis=0)
        
        plt.figure(figsize=(12, 8))
        plt.imshow(time2img, cmap='viridis', aspect='auto')
        plt.colorbar()
        plt.title('Time Series to Image Attention')
        plt.xlabel('Image patches (including CLS)')
        plt.ylabel('Time steps')
        plt.savefig(os.path.join(save_dir, 'time2img_attention.png'))
        plt.close()


def get_transforms(img_size=224):
    """
    Get train and validation transforms for images.
    
    Args:
        img_size: Size of the input image
        
    Returns:
        Dictionary of transforms for train and validation
    """
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
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
    Load and prepare data for training.
    
    Args:
        image_dir: Directory containing images
        time_series_path: Path to time series data
        labels_path: Path to labels
        img_size: Size of the input image
        batch_size: Batch size
        num_workers: Number of workers for data loading
        
    Returns:
        Dictionary of data loaders and data information
    """
    # Load time series data
    time_series = np.load(time_series_path)
    
    # Load labels
    labels = np.load(labels_path)
    
    # Get image paths
    image_paths = [os.path.join(image_dir, f"img_{i}.jpg") for i in range(len(labels))]
    
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
        image_transform=transforms_dict['train']
    )
    
    val_dataset = CoralDataset(
        [image_paths[i] for i in val_idx],
        time_series[val_idx],
        labels[val_idx],
        image_transform=transforms_dict['val']
    )
    
    test_dataset = CoralDataset(
        [image_paths[i] for i in test_idx],
        time_series[test_idx],
        labels[test_idx],
        image_transform=transforms_dict['test']
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
    time_steps, time_features = time_series.shape[1], time_series.shape[2]
    num_classes = len(np.unique(labels))
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'time_steps': time_steps,
        'time_features': time_features,
        'num_classes': num_classes
    }


if __name__ == "__main__":
    # Example usage
    img_size = 224
    patch_size = 16
    in_channels = 3
    time_input_dim = 8  # Number of time series features
    time_steps = 100    # Number of time steps
    
    # Adjust embedding dimensions to be divisible by the number of heads
    img_num_heads = 12
    time_num_heads = 8
    fusion_num_heads = 8
    
    # Ensure dimensions are divisible by number of heads
    img_embed_dim = img_num_heads * 64    # 12 * 64 = 768
    time_embed_dim = time_num_heads * 64  # 8 * 64 = 512
    fusion_dim = fusion_num_heads * 64    # 8 * 64 = 512
    
    # Define model configuration
    model_config = {
        'img_size': img_size,
        'patch_size': patch_size,
        'in_channels': in_channels,
        'time_input_dim': time_input_dim,
        'max_len': time_steps,
        'img_embed_dim': img_embed_dim,
        'time_embed_dim': time_embed_dim,
        'fusion_dim': fusion_dim,
        'img_num_heads': img_num_heads,
        'time_num_heads': time_num_heads,
        'fusion_num_heads': fusion_num_heads,
        'img_ff_dim': img_embed_dim * 4,
        'time_ff_dim': time_embed_dim * 4,
        'img_num_layers': 6,     # Reduced from 12 for faster training
        'time_num_layers': 3,    # Reduced from 6 for faster training
        'num_classes': 2,        # Binary classification
        'dropout': 0.1,
        'fusion_type': 'both'
    }
    
    # Create model
    model = DualTransformerModel(**model_config)
    
    # Test with random input data
    batch_size = 2
    random_images = torch.randn(batch_size, in_channels, img_size, img_size)
    random_time_series = torch.randn(batch_size, time_steps, time_input_dim)
    
    # Forward pass
    outputs = model(random_images, random_time_series)
    
    print("Model output shapes:")
    print(f"Fused logits: {outputs[0].shape}")
    print(f"Image logits: {outputs[1].shape}")
    print(f"Time logits: {outputs[2].shape}")
    print("Model test successful!")