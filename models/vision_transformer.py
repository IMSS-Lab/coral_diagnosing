"""
Vision Transformer + Temporal Transformer model for coral bleaching prediction.
Combines visual and temporal transformers for processing image and time series data.

This model incorporates:
- Vision Transformer (ViT) for image processing
- Temporal Transformer for time series data
- Cross-modal attention for feature fusion
- Positional encodings for both modalities
- Uncertainty quantification methods
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import polars as plrs
from typing import Tuple, Dict, List, Optional, Union, Any
import pytorch_lightning as pl


class PositionalEncoding(nn.Module):
    """
    Positional encoding for transformer models.
    
    Adds positional information to input embeddings.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and transpose
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Register buffer (persistent but not a parameter)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor of shape [seq_len, batch_size, d_model]
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Convert images into patches and embed them.
    
    This is the first step in Vision Transformer.
    """
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        """
        Initialize patch embedding.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels
            embed_dim: Embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Convolutional layer for patch extraction and embedding
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert images to patch embeddings.
        
        Args:
            x: Image tensor of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Patch embeddings of shape [batch_size, num_patches, embed_dim]
        """
        # Apply projection (B, C, H, W) -> (B, E, H//P, W//P)
        x = self.proj(x)
        
        # Flatten patches and transpose
        # (B, E, H//P, W//P) -> (B, E, N) -> (B, N, E)
        batch_size = x.shape[0]
        x = x.flatten(2).transpose(1, 2)
        
        return x


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.
    
    Allows model to attend to information from different positions
    with different attention patterns.
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        """
        Initialize multi-head attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Linear projections for queries, keys, and values
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor of shape [batch_size, seq_len, embed_dim],
                     attention weights of shape [batch_size, num_heads, seq_len, seq_len])
        """
        batch_size, seq_len, _ = x.shape
        
        # Ensure scale is on the correct device
        self.scale = self.scale.to(x.device)
        
        # Project input to queries, keys, and values
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Calculate attention scores
        # [batch_size, num_heads, seq_len, seq_len]
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        # [batch_size, num_heads, seq_len, head_dim]
        context = torch.matmul(attn_weights, v)
        
        # Reshape and project
        # [batch_size, seq_len, embed_dim]
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.reshape(batch_size, seq_len, self.embed_dim)
        
        output = self.output_proj(context)
        output = self.output_dropout(output)
        
        return output, attn_weights


class FeedForward(nn.Module):
    """
    Feed-forward network used in transformer blocks.
    
    Consists of two linear transformations with a ReLU or GELU activation in between.
    """
    
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.0, activation: str = 'gelu'):
        """
        Initialize feed-forward network.
        
        Args:
            embed_dim: Input and output dimension
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            activation: Activation function ('relu' or 'gelu')
        """
        super(FeedForward, self).__init__()
        
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        self.activation = F.relu if activation == 'relu' else F.gelu
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feed-forward network.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            
        Returns:
            Output tensor of shape [batch_size, seq_len, embed_dim]
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """
    Transformer block consisting of multi-head attention and feed-forward network.
    
    Uses pre-layer normalization architecture.
    """
    
    def __init__(
        self, 
        embed_dim: int, 
        num_heads: int, 
        ff_dim: int, 
        dropout: float = 0.0, 
        activation: str = 'gelu'
    ):
        """
        Initialize transformer block.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function for feed-forward network
        """
        super(TransformerBlock, self).__init__()
        
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ff = FeedForward(embed_dim, ff_dim, dropout, activation)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply transformer block.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, embed_dim]
            mask: Optional attention mask
            
        Returns:
            Tuple of (output tensor of shape [batch_size, seq_len, embed_dim],
                     attention weights)
        """
        # Pre-layer normalization
        norm_x = self.norm1(x)
        
        # Self-attention
        attn_output, attn_weights = self.attn(norm_x, mask)
        
        # Add residual connection
        x = x + self.dropout(attn_output)
        
        # Pre-layer normalization
        norm_x = self.norm2(x)
        
        # Feed-forward network
        ff_output = self.ff(norm_x)
        
        # Add residual connection
        x = x + self.dropout(ff_output)
        
        return x, attn_weights


class VisionTransformer(nn.Module):
    """
    Vision Transformer for processing image data.
    
    Converts image into patches, applies transformer blocks to process them.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        num_classes: int = 1000,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ff_dim: int = 3072,
        dropout: float = 0.1,
        activation: str = 'gelu',
        representation_size: Optional[int] = None,
        include_top: bool = True
    ):
        """
        Initialize Vision Transformer.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function for feed-forward network
            representation_size: If specified, adds a linear layer after transformer blocks
            include_top: Whether to include classification head
        """
        super(VisionTransformer, self).__init__()
        
        self.include_top = include_top
        
        # Patch embedding
        self.embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.embedding.num_patches
        
        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, activation)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Additional representation layer (optional)
        if representation_size is not None:
            self.representation = nn.Sequential(
                nn.Linear(embed_dim, representation_size),
                nn.Tanh()
            )
        else:
            self.representation = None
        
        # Classification head (optional)
        if include_top:
            self.head = nn.Linear(representation_size if representation_size is not None else embed_dim, num_classes)
        else:
            self.head = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize patch embedding projection
        nn.init.normal_(self.embedding.proj.weight, std=0.02)
        nn.init.zeros_(self.embedding.proj.bias)
        
        # Initialize class token
        nn.init.normal_(self.cls_token, std=0.02)
        
        # Initialize position embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)
        
        # Initialize transformer blocks
        for block in self.blocks:
            nn.init.normal_(block.attn.qkv_proj.weight, std=0.02)
            nn.init.zeros_(block.attn.qkv_proj.bias)
            nn.init.normal_(block.attn.output_proj.weight, std=0.02)
            nn.init.zeros_(block.attn.output_proj.bias)
            
            nn.init.normal_(block.ff.fc1.weight, std=0.02)
            nn.init.zeros_(block.ff.fc1.bias)
            nn.init.normal_(block.ff.fc2.weight, std=0.02)
            nn.init.zeros_(block.ff.fc2.bias)
        
        # Initialize layer normalization
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
        # Initialize representation layer if it exists
        if self.representation is not None:
            nn.init.normal_(self.representation[0].weight, std=0.02)
            nn.init.zeros_(self.representation[0].bias)
        
        # Initialize classification head if it exists
        if self.head is not None:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply Vision Transformer.
        
        Args:
            x: Image tensor of shape [batch_size, in_channels, img_size, img_size]
            
        Returns:
            Tuple of (output tensor, list of attention weights)
        """
        batch_size = x.shape[0]
        
        # Create patch embeddings
        x = self.embedding(x)  # [batch_size, num_patches, embed_dim]
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, 1 + num_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Store attention weights from each block
        attention_weights = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Use CLS token representation
        x = x[:, 0]
        
        # Apply additional representation layer if it exists
        if self.representation is not None:
            x = self.representation(x)
        
        # Apply classification head if it exists
        if self.include_top and self.head is not None:
            x = self.head(x)
        
        return x, attention_weights


class TemporalTransformer(nn.Module):
    """
    Transformer for processing time series data.
    
    Uses positional encoding and transformer blocks to process sequential data.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        depth: int = 4,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu',
        include_top: bool = True,
        output_dim: Optional[int] = None
    ):
        """
        Initialize Temporal Transformer.
        
        Args:
            input_dim: Dimension of input features
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            ff_dim: Feed-forward hidden dimension
            dropout: Dropout probability
            activation: Activation function for feed-forward network
            include_top: Whether to include output head
            output_dim: Dimension of output (if include_top is True)
        """
        super(TemporalTransformer, self).__init__()
        
        self.include_top = include_top
        
        # Input embedding
        self.input_embedding = nn.Linear(input_dim, embed_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, dropout=dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout, activation)
            for _ in range(depth)
        ])
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Output head (optional)
        if include_top and output_dim is not None:
            self.head = nn.Linear(embed_dim, output_dim)
        else:
            self.head = None
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize input embedding
        nn.init.normal_(self.input_embedding.weight, std=0.02)
        nn.init.zeros_(self.input_embedding.bias)
        
        # Initialize transformer blocks (already initialized in TransformerBlock)
        
        # Initialize layer normalization
        nn.init.ones_(self.norm.weight)
        nn.init.zeros_(self.norm.bias)
        
        # Initialize output head if it exists
        if self.head is not None:
            nn.init.normal_(self.head.weight, std=0.02)
            nn.init.zeros_(self.head.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Apply Temporal Transformer.
        
        Args:
            x: Time series tensor of shape [batch_size, seq_len, input_dim]
            
        Returns:
            Tuple of (output tensor, list of attention weights)
        """
        # Project input to embedding dimension
        x = self.input_embedding(x)  # [batch_size, seq_len, embed_dim]
        
        # Transpose for positional encoding (expects seq_len first)
        x = x.transpose(0, 1)  # [seq_len, batch_size, embed_dim]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        
        # Transpose back
        x = x.transpose(0, 1)  # [batch_size, seq_len, embed_dim]
        
        # Store attention weights from each block
        attention_weights = []
        
        # Apply transformer blocks
        for block in self.blocks:
            x, attn_weights = block(x)
            attention_weights.append(attn_weights)
        
        # Apply layer normalization
        x = self.norm(x)
        
        # Global pooling (mean across sequence dimension)
        x = x.mean(dim=1)  # [batch_size, embed_dim]
        
        # Apply output head if it exists
        if self.include_top and self.head is not None:
            x = self.head(x)
        
        return x, attention_weights


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for fusing information from image and time series modalities.
    
    Allows each modality to attend to the other.
    """
    
    def __init__(self, img_dim: int, time_dim: int, output_dim: int, num_heads: int = 4, dropout: float = 0.1):
        """
        Initialize cross-modal attention.
        
        Args:
            img_dim: Dimension of image features
            time_dim: Dimension of time series features
            output_dim: Dimension of output features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(CrossModalAttention, self).__init__()
        
        # Project image and time features to common dimension
        self.img_proj = nn.Linear(img_dim, output_dim)
        self.time_proj = nn.Linear(time_dim, output_dim)
        
        # Multi-head attention for cross-modality
        self.img2time_attn = MultiHeadAttention(output_dim, num_heads, dropout)
        self.time2img_attn = MultiHeadAttention(output_dim, num_heads, dropout)
        
        # Feed-forward networks
        self.img_ff = FeedForward(output_dim, output_dim * 4, dropout)
        self.time_ff = FeedForward(output_dim, output_dim * 4, dropout)
        
        # Layer normalization
        self.img_norm1 = nn.LayerNorm(output_dim)
        self.img_norm2 = nn.LayerNorm(output_dim)
        self.time_norm1 = nn.LayerNorm(output_dim)
        self.time_norm2 = nn.LayerNorm(output_dim)
        
        # Output fusion
        self.fusion = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize projections
        nn.init.normal_(self.img_proj.weight, std=0.02)
        nn.init.zeros_(self.img_proj.bias)
        nn.init.normal_(self.time_proj.weight, std=0.02)
        nn.init.zeros_(self.time_proj.bias)
        
        # Initialize fusion layer
        nn.init.normal_(self.fusion[0].weight, std=0.02)
        nn.init.zeros_(self.fusion[0].bias)
    
    def forward(
        self, 
        img_features: torch.Tensor, 
        time_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Apply cross-modal attention.
        
        Args:
            img_features: Image features of shape [batch_size, img_dim]
            time_features: Time series features of shape [batch_size, time_dim]
            
        Returns:
            Tuple of (fused features of shape [batch_size, output_dim],
                     tuple of attention weights)
        """
        batch_size = img_features.shape[0]
        
        # Project features to common dimension
        img_proj = self.img_proj(img_features)  # [batch_size, output_dim]
        time_proj = self.time_proj(time_features)  # [batch_size, output_dim]
        
        # Add sequence dimension for attention
        img_proj = img_proj.unsqueeze(1)  # [batch_size, 1, output_dim]
        time_proj = time_proj.unsqueeze(1)  # [batch_size, 1, output_dim]
        
        # Cross-modal attention
        # Image attends to time
        img_norm = self.img_norm1(img_proj)
        img_attend_time, img2time_attn = self.img2time_attn(img_norm, time_proj)
        img_proj = img_proj + img_attend_time
        
        # Time attends to image
        time_norm = self.time_norm1(time_proj)
        time_attend_img, time2img_attn = self.time2img_attn(time_norm, img_proj)
        time_proj = time_proj + time_attend_img
        
        # Feed-forward networks
        img_norm = self.img_norm2(img_proj)
        img_ff_out = self.img_ff(img_norm)
        img_proj = img_proj + img_ff_out
        
        time_norm = self.time_norm2(time_proj)
        time_ff_out = self.time_ff(time_norm)
        time_proj = time_proj + time_ff_out
        
        # Remove sequence dimension
        img_proj = img_proj.squeeze(1)  # [batch_size, output_dim]
        time_proj = time_proj.squeeze(1)  # [batch_size, output_dim]
        
        # Fusion
        fused = torch.cat([img_proj, time_proj], dim=1)  # [batch_size, output_dim * 2]
        fused = self.fusion(fused)  # [batch_size, output_dim]
        
        return fused, (img2time_attn, time2img_attn)


class DualTransformerModel(nn.Module):
    """
    Complete dual transformer model for coral bleaching prediction.
    
    Combines Vision Transformer for images and Temporal Transformer for time series,
    with cross-modal attention for feature fusion.
    """
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        time_steps: int = 24,
        time_features: int = 8,
        embed_dim: int = 384,
        vision_depth: int = 6,
        temporal_depth: int = 4,
        num_heads: int = 6,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Initialize dual transformer model.
        
        Args:
            img_size: Size of input image (assumed square)
            patch_size: Size of each patch (assumed square)
            in_channels: Number of input channels
            time_steps: Number of time steps in time series
            time_features: Number of features in time series
            embed_dim: Embedding dimension for transformers
            vision_depth: Number of transformer blocks in vision transformer
            temporal_depth: Number of transformer blocks in temporal transformer
            num_heads: Number of attention heads
            fusion_dim: Dimension of fused features
            dropout: Dropout probability
            activation: Activation function for feed-forward networks
        """
        super(DualTransformerModel, self).__init__()
        
        # Vision Transformer for image processing
        self.vision_transformer = VisionTransformer(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=vision_depth,
            num_heads=num_heads,
            ff_dim=embed_dim * 4,
            dropout=dropout,
            activation=activation,
            include_top=False  # No output head, we'll use our own
        )
        
        # Temporal Transformer for time series processing
        self.temporal_transformer = TemporalTransformer(
            input_dim=time_features,
            embed_dim=embed_dim,
            depth=temporal_depth,
            num_heads=num_heads,
            ff_dim=embed_dim * 4,
            dropout=dropout,
            activation=activation,
            include_top=False  # No output head, we'll use our own
        )
        
        # Cross-modal attention for feature fusion
        self.cross_attention = CrossModalAttention(
            img_dim=embed_dim,
            time_dim=embed_dim,
            output_dim=fusion_dim,
            num_heads=num_heads // 2,
            dropout=dropout
        )
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, 1)
        )
        
        # Bayesian dropout rate for MC dropout
        self.dropout_rate = dropout
        self.mc_dropout = False
    
    def forward(
        self, 
        image: torch.Tensor, 
        time_series: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Forward pass through the model.
        
        Args:
            image: Image tensor of shape [batch_size, in_channels, img_size, img_size]
            time_series: Time series tensor of shape [batch_size, time_steps, time_features]
            
        Returns:
            Tuple of (output logits, dictionary with attention weights)
        """
        # Process image through Vision Transformer
        img_features, img_attention = self.vision_transformer(image)
        
        # Process time series through Temporal Transformer
        time_features, time_attention = self.temporal_transformer(time_series)
        
        # Apply cross-modal attention
        fused_features, cross_attention = self.cross_attention(img_features, time_features)
        
        # Output head
        output = self.output_head(fused_features)
        
        # Store attention weights
        attention_dict = {
            'image_attention': img_attention,
            'time_attention': time_attention,
            'cross_attention': cross_attention
        }
        
        return output.squeeze(), attention_dict
    
    def enable_mc_dropout(self):
        """Enable MC dropout for uncertainty estimation during inference."""
        self.mc_dropout = True
        
        # Set model to train mode but only for dropout layers
        self.train()
        
        # Set all non-dropout layers to eval mode
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                module.eval()
    
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


class DualTransformerLightningModel(pl.LightningModule):
    """PyTorch Lightning module for coral bleaching prediction."""
    
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        time_steps: int = 24,
        time_features: int = 8,
        embed_dim: int = 384,
        vision_depth: int = 6,
        temporal_depth: int = 4,
        num_heads: int = 6,
        fusion_dim: int = 128,
        dropout: float = 0.1,
        activation: str = 'gelu',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize Lightning module.
        
        Args:
            img_size: Size of input image
            patch_size: Size of each patch
            in_channels: Number of input channels
            time_steps: Number of time steps in time series
            time_features: Number of features in time series
            embed_dim: Embedding dimension
            vision_depth: Depth of vision transformer
            temporal_depth: Depth of temporal transformer
            num_heads: Number of attention heads
            fusion_dim: Dimension of fused features
            dropout: Dropout probability
            activation: Activation function
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for optimizer
        """
        super(DualTransformerLightningModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize model
        self.model = DualTransformerModel(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            time_steps=time_steps,
            time_features=time_features,
            embed_dim=embed_dim,
            vision_depth=vision_depth,
            temporal_depth=temporal_depth,
            num_heads=num_heads,
            fusion_dim=fusion_dim,
            dropout=dropout,
            activation=activation
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
        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        # Create scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=10,
            eta_min=self.hparams.learning_rate / 10
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
    
    def get_attention_maps(self, image, time_series):
        """Get attention maps for interpretability."""
        with torch.no_grad():
            _, attention_dict = self(image, time_series)
        return attention_dict


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
    model = DualTransformerModel(
        img_size=224,
        patch_size=16,
        in_channels=3,
        time_steps=24,
        time_features=8,
        embed_dim=384,
        vision_depth=6,
        temporal_depth=4,
        num_heads=6,
        fusion_dim=128,
        dropout=0.1
    )
    
    # Print model summary
    print(model)
    
    # Create Lightning module
    lightning_model = DualTransformerLightningModel(
        img_size=224,
        patch_size=16,
        in_channels=3,
        time_steps=24,
        time_features=8,
        embed_dim=384,
        vision_depth=6,
        temporal_depth=4,
        num_heads=6,
        fusion_dim=128,
        dropout=0.1
    )
    
    # Example forward pass
    image_batch, ts_batch, _ = next(iter(train_loader))
    outputs, attention_dict = model(image_batch, ts_batch)
    print(f"Output shape: {outputs.shape}")
    print(f"Attention dict keys: {attention_dict.keys()}")