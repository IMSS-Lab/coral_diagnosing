"""
Wavelet transform module for time series feature extraction.
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List, Optional, Union, Any


class WaveletTransform(nn.Module):
    """
    Wavelet transform module for time series feature extraction.
    
    Parameters:
        wavelet (str): Wavelet type
        level (int): Decomposition level
        mode (str): Signal extension mode
    """
    
    def __init__(
        self,
        wavelet: str = 'db4',
        level: int = 3,
        mode: str = 'symmetric'
    ):
        super(WaveletTransform, self).__init__()
        
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for wavelet transform.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, features]
            
        Returns:
            Wavelet features
        """
        batch_size, seq_len, num_features = x.shape
        
        # Convert to numpy for wavelet transform
        x_np = x.cpu().numpy()
        
        # Determine appropriate wavelet level based on sequence length
        max_level = int(np.log2(seq_len))
        adjusted_level = min(self.level, max_level - 1)
        adjusted_level = max(1, adjusted_level)  # Ensure at least level 1
        
        # Initialize features array
        features = np.zeros((batch_size, num_features * (adjusted_level + 1) * 4))
        
        for b in range(batch_size):
            feature_idx = 0
            for i in range(num_features):
                ts = x_np[b, :, i]
                
                # Wavelet decomposition
                coeffs = pywt.wavedec(ts, self.wavelet, level=adjusted_level, mode=self.mode)
                
                # Extract features from coefficients
                for coeff in coeffs:
                    # Mean
                    features[b, feature_idx] = np.mean(coeff)
                    feature_idx += 1
                    
                    # Standard deviation
                    features[b, feature_idx] = np.std(coeff)
                    feature_idx += 1
                    
                    # Energy
                    features[b, feature_idx] = np.sum(coeff ** 2)
                    feature_idx += 1
                    
                    # Entropy
                    if np.sum(np.abs(coeff)) > 0:
                        p = np.abs(coeff) / np.sum(np.abs(coeff))
                        features[b, feature_idx] = -np.sum(p * np.log2(p + 1e-10))
                    feature_idx += 1
        
        # Convert back to tensor
        return torch.FloatTensor(features).to(x.device) 