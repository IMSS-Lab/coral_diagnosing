"""
Combined models.py for coral bleaching prediction.
This file consolidates multiple model architectures and utilities.
"""

import os
import math
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
import torchvision.models as models
import torchvision.transforms as transforms
import pytorch_lightning as pl
import torchmetrics # Import torchmetrics directly

from einops import rearrange, repeat # For ViT
from einops.layers.torch import Rearrange # For ViT

# XGBoost specific imports
try:
    import xgboost as xgb
    from xgboost import DMatrix # Explicitly import DMatrix
except ImportError:
    print("Warning: XGBoost not installed or not properly installed. XGBoostModel will not be available.")
    xgb = None # Placeholder if XGBoost is not available
    DMatrix = None

# Scikit-learn specific imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# SHAP for XGBoost interpretability
try:
    import shap
except ImportError:
    print("Warning: SHAP library not installed. SHAP-based interpretability will not be available for XGBoost.")
    shap = None

# Polars for XGBoost data loading example (optional)
try:
    import polars as plrs
except ImportError:
    print("Warning: Polars library not installed. load_data_polars will not be available.")
    plrs = None


import matplotlib.pyplot as plt
import cv2

# --- Content from wavelet.py ---
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
        try:
            wavelet_obj = pywt.Wavelet(self.wavelet)
            max_level = pywt.dwt_max_level(seq_len, wavelet_obj)
        except ValueError: 
             max_level = int(np.log2(seq_len)) -1 if seq_len > 1 else 0


        adjusted_level = min(self.level, max_level)
        adjusted_level = max(1, adjusted_level)
        
        num_coeffs_sets = adjusted_level + 1
        features_per_ts = num_coeffs_sets * 4
        
        features_np_array = np.zeros((batch_size, num_features * features_per_ts))
        
        for b in range(batch_size):
            current_batch_channel_offset = 0
            for i in range(num_features): # For each input time series channel
                ts = x_np[b, :, i]
                coeffs = pywt.wavedec(ts, self.wavelet, level=adjusted_level, mode=self.mode)
                
                # Start index for this channel's features in the flat array for sample b
                channel_feature_start_idx = i * features_per_ts

                for coeff_set_idx, coeff in enumerate(coeffs):
                    # Start index for this coefficient set's stats
                    stat_start_idx = channel_feature_start_idx + coeff_set_idx * 4

                    if len(coeff) == 0: 
                        features_np_array[b, stat_start_idx : stat_start_idx + 4] = 0
                        continue

                    features_np_array[b, stat_start_idx + 0] = np.mean(coeff)
                    features_np_array[b, stat_start_idx + 1] = np.std(coeff)
                    features_np_array[b, stat_start_idx + 2] = np.sum(coeff ** 2)
                    if np.sum(np.abs(coeff)) > 0:
                        p = np.abs(coeff) / np.sum(np.abs(coeff))
                        features_np_array[b, stat_start_idx + 3] = -np.sum(p * np.log2(p + 1e-10))
                    else:
                        features_np_array[b, stat_start_idx + 3] = 0
        
        return torch.FloatTensor(features_np_array).to(x.device)


# --- Content from cnn_lstm.py ---
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
        residual = x
        x = self.layer_norm(x)
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k ** 0.5)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out + residual
        
        return out, attn


class CNN_LSTM_CNNModule(nn.Module): 
    """
    CNN module for image feature extraction (specific to CNN-LSTM).
    """
    def __init__(
        self,
        backbone: str = 'resnet18',
        output_dim: int = 256,
        dropout: float = 0.3,
        freeze_backbone: bool = False
    ):
        super(CNN_LSTM_CNNModule, self).__init__()
        
        if backbone == 'resnet18':
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
        
        self.backbone.fc = nn.Identity()
        if 'efficient' in backbone: 
            self.backbone.classifier = nn.Identity()
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        output = self.feature_processor(features)
        return output


class LSTMModule(nn.Module):
    """
    LSTM module for time series processing.
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
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0, 
            bidirectional=bidirectional
        )
        
        lstm_output_dim = hidden_size * 2 if bidirectional else hidden_size
        self.attention = SelfAttention(
            embed_dim=lstm_output_dim,
            dropout=dropout
        )
        
        self.output_processor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, output_size),
            nn.BatchNorm1d(output_size),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        lstm_out, _ = self.lstm(x)
        attended, attn_weights = self.attention(lstm_out)
        pooled = torch.mean(attended, dim=1)
        output = self.output_processor(pooled)
        return output, attn_weights


class WaveletModule(nn.Module): 
    """
    Wavelet feature extraction module (for CNN-LSTM).
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
        self.level = level # Configured level, used for sizing Linear layer
        
        # Linear layer is sized based on configured self.level
        wavelet_feature_dim_config = input_size * (self.level + 1) * 4

        self.feature_processor = nn.Sequential(
            nn.Linear(wavelet_feature_dim_config, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout/2),
            nn.Linear(64, output_size), 
            nn.BatchNorm1d(output_size), 
            nn.ReLU()
        )
    
    def extract_wavelet_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract wavelet features from time series.
        
        Args:
            x: Time series data of shape [batch_size, seq_len, input_size]
            
        Returns:
            Wavelet features of shape [batch_size, self.input_size*(self.level+1)*4] (padded with zeros if actual_level < self.level)
        """
        batch_size, seq_len, input_size_runtime = x.shape
        if input_size_runtime != self.input_size:
             raise ValueError(f"Runtime input_size {input_size_runtime} does not match WaveletModule initialized input_size {self.input_size}")

        x_np = x.cpu().numpy()
        
        try:
            wavelet_obj = pywt.Wavelet(self.wavelet)
            max_level_possible = pywt.dwt_max_level(seq_len, wavelet_obj)
        except ValueError:
             max_level_possible = int(np.log2(seq_len)) -1 if seq_len > 1 else 0

        actual_level_for_decomp = min(self.level, max_level_possible)
        actual_level_for_decomp = max(1, actual_level_for_decomp)
        
        dim_expected_by_linear = self.input_size * (self.level + 1) * 4
        features_np_array = np.zeros((batch_size, dim_expected_by_linear))
        
        num_coeffs_sets_configured_per_channel = self.level + 1

        for b in range(batch_size):
            for i in range(self.input_size): 
                ts = x_np[b, :, i]
                coeffs = pywt.wavedec(ts, self.wavelet, level=actual_level_for_decomp)
                
                channel_base_idx = i * num_coeffs_sets_configured_per_channel * 4

                for coeff_set_idx, coeff_set in enumerate(coeffs): 
                    stat_base_idx = channel_base_idx + coeff_set_idx * 4
                    
                    if len(coeff_set) > 0:
                        features_np_array[b, stat_base_idx + 0] = np.mean(coeff_set)
                        features_np_array[b, stat_base_idx + 1] = np.std(coeff_set)
                        features_np_array[b, stat_base_idx + 2] = np.sum(coeff_set ** 2)
                        
                        if np.sum(np.abs(coeff_set)) > 0:
                            p = np.abs(coeff_set) / np.sum(np.abs(coeff_set))
                            features_np_array[b, stat_base_idx + 3] = -np.sum(p * np.log2(p + 1e-10))
                        else:
                            features_np_array[b, stat_base_idx + 3] = 0.0
                    else:
                        features_np_array[b, stat_base_idx : stat_base_idx + 4] = 0.0
        
        return torch.FloatTensor(features_np_array).to(x.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        wavelet_features = self.extract_wavelet_features(x)
        output = self.feature_processor(wavelet_features)
        return output


class CNN_LSTM_FeatureFusion(nn.Module): 
    """
    Feature fusion module (for CNN-LSTM).
    """
    def __init__(
        self,
        cnn_dim: int = 256,
        lstm_dim: int = 128,
        wavelet_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super(CNN_LSTM_FeatureFusion, self).__init__()
        
        combined_dim = cnn_dim + lstm_dim + wavelet_dim
        
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
        combined = torch.cat([cnn_features, lstm_features, wavelet_features], dim=1)
        fused = self.fusion_layers(combined)
        return fused


class CoralNet(nn.Module): 
    """
    Combined CNN-LSTM-Wavelet model for coral bleaching prediction.
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
        wavelet_level: int = 3, 
        fusion_output_dim: int = 64,
        dropout: float = 0.3,
        bidirectional: bool = True,
        freeze_backbone: bool = False
    ):
        super(CoralNet, self).__init__()
        
        self.cnn_module = CNN_LSTM_CNNModule( 
            backbone=cnn_backbone,
            output_dim=cnn_output_dim,
            dropout=dropout,
            freeze_backbone=freeze_backbone
        )
        
        self.lstm_module = LSTMModule(
            input_size=time_input_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            output_size=lstm_output_dim,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.wavelet_module = WaveletModule( 
            input_size=time_input_dim,
            output_size=wavelet_output_dim, 
            wavelet='db4', 
            level=wavelet_level, 
            dropout=dropout
        )
        
        self.fusion_module = CNN_LSTM_FeatureFusion( 
            cnn_dim=cnn_output_dim,
            lstm_dim=lstm_output_dim,
            wavelet_dim=wavelet_output_dim, 
            output_dim=fusion_output_dim,
            dropout=dropout
        )
        
        self.output_layer = nn.Linear(fusion_output_dim, 1) 
        
        self._init_weights()
    
    def _init_weights(self):
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
        cnn_features = self.cnn_module(images)
        lstm_features, attn_weights = self.lstm_module(time_series)
        wavelet_features = self.wavelet_module(time_series)
        
        fused_features = self.fusion_module(cnn_features, lstm_features, wavelet_features)
        output = self.output_layer(fused_features)
        
        return output, {'attention_weights': attn_weights}


class CNN_LSTM_CoralDataset(Dataset): 
    """
    Dataset for CNN-LSTM coral bleaching prediction.
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
        img_path = self.image_paths[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image.transpose(2, 0, 1)) / 255.0 
        
        ts_data = torch.FloatTensor(self.time_series[idx])
        label = torch.FloatTensor([self.labels[idx]]) 
        
        return {'image': image, 'time_series': ts_data, 'label': label}


class CoralLightningModel(pl.LightningModule): 
    """
    PyTorch Lightning module for CNN-LSTM coral bleaching prediction.
    """
    def __init__(
        self,
        model_params: Dict[str, Any],
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        pos_weight: float = 2.0
    ):
        super(CoralLightningModel, self).__init__()
        self.save_hyperparameters()
        
        self.model = CoralNet(**model_params) 
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
    
    def forward(self, images, time_series):
        return self.model(images, time_series)
    
    def _common_step(self, batch, batch_idx, stage: str):
        images, time_series, labels = batch['image'], batch['time_series'], batch['label_float']  # Use float labels for BCE
        outputs = self(images, time_series)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        logits = logits.view(-1) 
        labels = labels.view(-1) 

        loss = self.criterion(logits, labels)
        
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5) 

        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)

        if stage == 'train':
            self.train_acc(preds, labels.long())
            self.log(f'{stage}_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == 'val':
            self.val_acc(preds, labels.long())
            self.val_auroc(probs, labels.long())
            self.val_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.val_acc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_f1', self.val_f1, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_acc(preds, labels.long())
            self.test_auroc(probs, labels.long())
            self.test_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.test_acc, on_epoch=True)
            self.log(f'{stage}_auroc', self.test_auroc, on_epoch=True)
            self.log(f'{stage}_f1', self.test_f1, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'train')

    def validation_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        return self._common_step(batch, batch_idx, 'test')
            
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6 
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def get_attention_weights(self, images, time_series):
        self.eval() 
        with torch.no_grad():
            outputs = self(images, time_series)
        if isinstance(outputs, tuple) and 'attention_weights' in outputs[1]:
            return outputs[1]['attention_weights']
        return None


def cnn_lstm_get_transforms(img_size=224): 
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
    return {'train': train_transform, 'val': val_transform, 'test': val_transform}


def cnn_lstm_load_data( 
    image_dir: str, time_series_path: str, labels_path: str,
    img_size: int = 224, batch_size: int = 32, num_workers: int = 4
):
    time_series_data = np.load(time_series_path)
    labels_data = np.load(labels_path)
    image_paths = sorted([
        os.path.join(image_dir, f) for f in os.listdir(image_dir) 
        if f.endswith(('.jpg', '.png', '.jpeg'))
    ])
    assert len(image_paths) == len(labels_data) == len(time_series_data), "Data dimension mismatch"

    indices = np.arange(len(labels_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_data)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_data[temp_idx])
    
    transforms_dict = cnn_lstm_get_transforms(img_size)
    
    train_dataset = CNN_LSTM_CoralDataset([image_paths[i] for i in train_idx], time_series_data[train_idx], labels_data[train_idx], transforms_dict['train'])
    val_dataset = CNN_LSTM_CoralDataset([image_paths[i] for i in val_idx], time_series_data[val_idx], labels_data[val_idx], transforms_dict['val'])
    test_dataset = CNN_LSTM_CoralDataset([image_paths[i] for i in test_idx], time_series_data[test_idx], labels_data[test_idx], transforms_dict['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    seq_len, time_features = time_series_data.shape[1], time_series_data.shape[2]
    
    return {
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'seq_len': seq_len, 'time_features': time_features
    }

def cnn_lstm_visualize_attention(model, dataloader, num_samples=5, save_dir=None): 
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    all_samples_data = []
    count = 0
    for batch in dataloader:
        if count >= num_samples: break
        images = batch['image'].to(device)
        time_series = batch['time_series'].to(device)
        labels = batch['label'] 

        attn_weights = model.get_attention_weights(images, time_series)
        if attn_weights is None:
            print("No attention weights returned by the model.")
            return

        with torch.no_grad():
            outputs = model(images, time_series)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        logits_np = logits.cpu().numpy()
        attn_weights_np = attn_weights.cpu().numpy()
        time_series_np = time_series.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(logits_np)):
            if count >= num_samples: break
            all_samples_data.append({
                'logit': logits_np[i, 0],
                'attn_weights': attn_weights_np[i],
                'time_series': time_series_np[i],
                'label': labels_np[i, 0]
            })
            count +=1
            
    for i, sample in enumerate(all_samples_data):
        logit, attn_weights, time_series_s, label = sample['logit'], sample['attn_weights'], sample['time_series'], sample['label']
        prob = 1 / (1 + np.exp(-logit))
        pred = 1 if prob > 0.5 else 0
        
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 1, 1)
        for j in range(min(3, time_series_s.shape[1])):
            plt.plot(time_series_s[:, j], label=f'Feature {j+1}')
        plt.title(f'Time Series (Label: {int(label)}, Pred: {pred}, Prob: {prob:.2f})')
        plt.xlabel('Time Step'); plt.ylabel('Value'); plt.legend(); plt.grid(True)
        
        plt.subplot(2, 1, 2)
        plt.imshow(attn_weights, aspect='auto', cmap='viridis')
        plt.colorbar(label='Attention Weight'); plt.title('Attention Weights')
        plt.xlabel('Time Step'); plt.ylabel('Time Step') 
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, f'cnn_lstm_attention_{i+1}.png')); plt.close()
        else:
            plt.show()


# --- Content from tcn.py ---
class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int, stride: int, dilation: int, padding: int, dropout: float = 0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs) 
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU() 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if res.size(2) != out.size(2):
             res = res[:, :, -out.size(2):] 
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs: int, num_channels: List[int], kernel_size: int = 3, dropout: float = 0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size 
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding, dropout=dropout))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class SequenceAttention(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int = 1, dropout: float = 0.0):
        super(SequenceAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads 
        
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.size()
        q, k, v = self.query(x), self.key(x), self.value(x)
        
        attn_weights = torch.bmm(q, k.transpose(1, 2)) / (self.embed_dim ** 0.5)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.bmm(attn_weights, v)
        attn_output = self.output_proj(attn_output)
        return attn_output, attn_weights

class TCN_CNNModule(nn.Module): 
    def __init__(self, backbone_name: str = 'resnet18', output_dim: int = 256, freeze_backbone: bool = False, dropout: float = 0.3):
        super(TCN_CNNModule, self).__init__()
        if backbone_name == 'resnet18':
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
        
        self.backbone.fc = nn.Identity()
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        projected = self.projection(features)
        return projected

class TCNModule(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, kernel_size: int = 3, dropout: float = 0.3, use_attention: bool = True):
        super(TCNModule, self).__init__()
        self.tcn = TemporalConvNet(num_inputs=input_dim, num_channels=hidden_dims, kernel_size=kernel_size, dropout=dropout)
        self.use_attention = use_attention
        tcn_output_dim = hidden_dims[-1]

        if use_attention:
            self.attention = SequenceAttention(embed_dim=tcn_output_dim, dropout=dropout)
        
        self.projection = nn.Sequential(
            nn.Linear(tcn_output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout/2)
        )
        
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x = x.transpose(1, 2) 
        tcn_out = self.tcn(x) 
        
        attn_weights = None
        if self.use_attention:
            tcn_out_permuted = tcn_out.transpose(1, 2) 
            attended, attn_weights = self.attention(tcn_out_permuted)
            pooled = attended.mean(dim=1) 
        else:
            pooled = tcn_out.mean(dim=2) 
        
        output = self.projection(pooled)
        
        return (output, attn_weights) if self.use_attention and attn_weights is not None else output


class TCN_FeatureFusion(nn.Module): 
    def __init__(self, img_dim: int, time_dim: int, hidden_dim: int = 128, output_dim: int = 64, dropout: float = 0.3):
        super(TCN_FeatureFusion, self).__init__()
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        
        context_dim = hidden_dim * 4 
        
        self.img_gate = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Sigmoid())
        self.time_gate = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.Sigmoid())
        
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim), nn.ReLU()
        )
        
    def forward(self, img_features: torch.Tensor, time_features: torch.Tensor) -> torch.Tensor:
        img_proj = self.img_proj(img_features)
        time_proj = self.time_proj(time_features)
        
        concat_proj_features = torch.cat([img_proj, time_proj], dim=1) 
        context = torch.cat([img_proj, time_proj, concat_proj_features], dim=1)

        img_gate_vals = self.img_gate(context)
        time_gate_vals = self.time_gate(context)
        
        gated_img = img_proj * img_gate_vals
        gated_time = time_proj * time_gate_vals
        
        combined_gated = torch.cat([gated_img, gated_time], dim=1)
        fused = self.fusion(combined_gated)
        return fused


class TCNCoralModel(nn.Module): 
    def __init__(
        self, input_channels: int = 3, img_size: int = 224, time_input_dim: int = 8, 
        sequence_length: int = 24, img_feature_dim: int = 256, time_feature_dim: int = 128,
        hidden_dims: List[int] = [64, 128], 
        fusion_hidden_dim: int = 128, fusion_output_dim: int = 64,
        cnn_backbone: str = 'resnet18', kernel_size: int = 3, dropout: float = 0.3,
        freeze_backbone: bool = False, use_attention: bool = True
    ):
        super(TCNCoralModel, self).__init__()
        self.cnn_module = TCN_CNNModule(cnn_backbone, img_feature_dim, freeze_backbone, dropout)
        self.tcn_module = TCNModule(time_input_dim, hidden_dims, time_feature_dim, kernel_size, dropout, use_attention)
        self.fusion_module = TCN_FeatureFusion(img_feature_dim, time_feature_dim, fusion_hidden_dim, fusion_output_dim, dropout)
        self.output_layer = nn.Linear(fusion_output_dim, 1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Conv1d, nn.Conv2d)): 
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        
    def forward(self, images: torch.Tensor, time_series: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        img_features = self.cnn_module(images)
        tcn_out = self.tcn_module(time_series)
        
        attn_weights = None
        if isinstance(tcn_out, tuple):
            time_features, attn_weights = tcn_out
        else:
            time_features = tcn_out
            
        fused_features = self.fusion_module(img_features, time_features)
        output = self.output_layer(fused_features)
        
        return (output, {'attention_weights': attn_weights}) if attn_weights is not None else output


class TCN_CoralDataset(Dataset): 
    def __init__(self, image_paths: List[str], time_series: np.ndarray, labels: np.ndarray, transform: Optional[Any] = None):
        self.image_paths = image_paths
        self.time_series = time_series
        self.labels = labels
        self.transform = transform
        
    def __len__(self) -> int: return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform: image = self.transform(image)
        else: image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0 
            
        ts_data = torch.from_numpy(self.time_series[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.float32).unsqueeze(0) 
        
        return {'image': image, 'time_series': ts_data, 'label': label}


class TCNLightningModel(pl.LightningModule):
    def __init__(self, model_params: Dict[str, Any], learning_rate: float = 1e-4, weight_decay: float = 1e-5, pos_weight: float = 2.0):
        super(TCNLightningModel, self).__init__()
        self.save_hyperparameters()
        self.model = TCNCoralModel(**model_params)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))

        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')
        
    def forward(self, images, time_series): return self.model(images, time_series)
    
    def _common_step(self, batch, batch_idx, stage: str):
        images, time_series, labels = batch['image'], batch['time_series'], batch['label_float']  # Use float labels for BCE
        outputs = self(images, time_series)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs
        
        logits = logits.view(-1)
        labels = labels.view(-1)

        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)
        preds = probs > 0.5

        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        if stage == 'train':
            self.train_acc(preds, labels.long())
            self.log(f'{stage}_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == 'val':
            self.val_acc(preds, labels.long())
            self.val_auroc(probs, labels.long())
            self.val_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.val_acc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_f1', self.val_f1, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_acc(preds, labels.long())
            self.test_auroc(probs, labels.long())
            self.test_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.test_acc, on_epoch=True)
            self.log(f'{stage}_auroc', self.test_auroc, on_epoch=True)
            self.log(f'{stage}_f1', self.test_f1, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6) 
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def get_attention_weights(self, images, time_series): 
        self.eval()
        with torch.no_grad():
            outputs = self(images, time_series)
        if isinstance(outputs, tuple) and 'attention_weights' in outputs[1]:
            return outputs[1]['attention_weights']
        return None

def tcn_get_transforms(img_size=224): 
    train_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {'train': train_transform, 'val': val_transform, 'test': val_transform} 

def tcn_load_data(image_dir: str, time_series_path: str, labels_path: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4): 
    time_series_data = np.load(time_series_path)
    labels_data = np.load(labels_path)
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    assert len(image_paths) == len(labels_data) == len(time_series_data), "Data dimension mismatch"

    indices = np.arange(len(labels_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_data)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_data[temp_idx])
    
    transforms_dict = tcn_get_transforms(img_size)
    
    train_dataset = TCN_CoralDataset([image_paths[i] for i in train_idx], time_series_data[train_idx], labels_data[train_idx], transforms_dict['train'])
    val_dataset = TCN_CoralDataset([image_paths[i] for i in val_idx], time_series_data[val_idx], labels_data[val_idx], transforms_dict['val'])
    test_dataset = TCN_CoralDataset([image_paths[i] for i in test_idx], time_series_data[test_idx], labels_data[test_idx], transforms_dict['test']) 
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    seq_len, time_features = time_series_data.shape[1], time_series_data.shape[2]
    return {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader, 'seq_len': seq_len, 'time_features': time_features}

def tcn_visualize_attention(model, dataloader, num_samples=5, save_dir=None): 
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    fig_idx = 0
    for batch_idx, batch in enumerate(dataloader):
        if fig_idx >= num_samples: break
        images = batch['image'].to(device)
        time_series = batch['time_series'].to(device)
        labels = batch['label'] 

        attn_weights_tensor = model.get_attention_weights(images, time_series)
        if attn_weights_tensor is None:
            print("TCN model does not return attention weights or use_attention is False.")
            return

        with torch.no_grad():
            outputs = model(images, time_series)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

        logits_np = logits.cpu().numpy()
        attn_weights_np = attn_weights_tensor.cpu().numpy()
        time_series_np = time_series.cpu().numpy()
        labels_np = labels.cpu().numpy()

        for i in range(len(logits_np)):
            if fig_idx >= num_samples: break
            
            sample_logit = logits_np[i, 0]
            sample_prob = 1 / (1 + np.exp(-sample_logit))
            sample_pred = 1 if sample_prob > 0.5 else 0
            sample_label = labels_np[i, 0]
            sample_attn = attn_weights_np[i] 
            sample_ts = time_series_np[i]    
            
            plt.figure(figsize=(12, 10))
            
            plt.subplot(2, 1, 1)
            for j in range(min(3, sample_ts.shape[1])):
                plt.plot(sample_ts[:, j], label=f'Feature {j+1}')
            plt.xlabel('Time Step'); plt.ylabel('Value')
            plt.title(f'Time Series (Label: {int(sample_label)}, Pred: {sample_pred}, Prob: {sample_prob:.2f})')
            plt.legend(); plt.grid(True)
            
            plt.subplot(2, 1, 2)
            im = plt.imshow(sample_attn, cmap='viridis', aspect='auto')
            plt.colorbar(im, label='Attention Weight')
            plt.xlabel('Key Time Step'); plt.ylabel('Query Time Step')
            plt.title('TCN Self-Attention Weights')
            
            plt.tight_layout()
            if save_dir: plt.savefig(os.path.join(save_dir, f'tcn_attention_{fig_idx}.png')); plt.close()
            else: plt.show()
            fig_idx += 1


# --- Content from vit.py ---
class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).flatten(2).transpose(1, 2) 
        return x

class ViT_MultiHeadAttention(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False) 
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_mask is not None: attn = attn.masked_fill(attn_mask == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn_weights_viz = attn 
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_weights_viz


class ViT_PositionwiseFeedForward(nn.Module): 
    def __init__(self, embed_dim: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.dropout(self.act(self.fc1(x)))))

class ViT_PositionalEncoding(nn.Module): 
    def __init__(self, embed_dim: int, max_len: int = 5000, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0)) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class ViT_TransformerEncoderLayer(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = ViT_MultiHeadAttention(embed_dim, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = ViT_PositionwiseFeedForward(embed_dim, ff_dim, dropout=dropout)
        self.dropout = nn.Dropout(dropout) 

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        h, attn_weights = self.attn(self.norm1(x), attn_mask=attn_mask)
        x = x + self.dropout(h) 
        x = x + self.dropout(self.mlp(self.norm2(x))) 
        return x, attn_weights

class ViT_TransformerEncoder(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, num_layers: int, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            ViT_TransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attn_weights_list = []
        for layer in self.layers:
            x, attn_weights = layer(x, attn_mask)
            attn_weights_list.append(attn_weights)
        return x, attn_weights_list

class TimeSeriesEmbedding(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, max_len: int = 1000, dropout: float = 0.0):
        super().__init__()
        self.feature_embed = nn.Linear(input_dim, embed_dim)
        self.pos_encoding = ViT_PositionalEncoding(embed_dim, max_len, dropout) 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_embed(x)
        x = self.pos_encoding(x)
        return x

class ImageTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, 
                 embed_dim: int = 768, num_heads: int = 12, ff_dim: int = 3072, 
                 num_layers: int = 12, num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embedding.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        self.transformer = ViT_TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_transformer_weights_module) 

    def _init_transformer_weights_module(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        x = self.patch_embedding(x) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x = x + self.pos_embedding
        x = self.dropout(x)
        x, attn_weights = self.transformer(x) 
        
        x_cls = self.norm(x[:, 0]) 
        logits = self.classifier(x_cls)
        return logits, x, attn_weights 

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, max_len: int = 1000, num_heads: int = 8, 
                 ff_dim: int = 2048, num_layers: int = 6, num_classes: int = 2, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + (embed_dim % num_heads > 0)) * num_heads

        self.embedding = TimeSeriesEmbedding(input_dim, embed_dim, max_len, dropout)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.transformer = ViT_TransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_transformer_weights_module) 

    def _init_transformer_weights_module(self, m): 
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        B = x.shape[0]
        x = self.embedding(x) 
        cls_tokens = self.cls_token.expand(B, -1, -1) 
        x = torch.cat((cls_tokens, x), dim=1) 
        x, attn_weights = self.transformer(x)
        x_cls = self.norm(x[:, 0])
        logits = self.classifier(x_cls)
        return logits, x, attn_weights 

class CrossModalAttention(nn.Module): 
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            embed_dim = ((embed_dim // num_heads) + (embed_dim % num_heads > 0)) * num_heads

        self.img_to_time_attn = ViT_MultiHeadAttention(embed_dim, num_heads, dropout)
        self.time_to_img_attn = ViT_MultiHeadAttention(embed_dim, num_heads, dropout)

        self.norm_img1 = nn.LayerNorm(embed_dim)
        self.norm_time1 = nn.LayerNorm(embed_dim)
        self.norm_img2 = nn.LayerNorm(embed_dim) 
        self.norm_time2 = nn.LayerNorm(embed_dim) 

        self.ff_img = ViT_PositionwiseFeedForward(embed_dim, embed_dim * 4, dropout) 
        self.ff_time = ViT_PositionwiseFeedForward(embed_dim, embed_dim * 4, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features: torch.Tensor, time_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        img_norm_q = self.norm_img1(img_features)
        time_norm_kv = self.norm_time1(time_features) 
        img_attended_by_time, i2t_weights = self.img_to_time_attn(img_norm_q, time_norm_kv, time_norm_kv)
        img_out = img_features + self.dropout(img_attended_by_time)
        img_out_norm_ff = self.norm_img2(img_out) 
        img_out = img_out + self.dropout(self.ff_img(img_out_norm_ff))

        time_norm_q = self.norm_time1(time_features) 
        img_norm_kv = self.norm_img1(img_features)
        time_attended_by_img, t2i_weights = self.time_to_img_attn(time_norm_q, img_norm_kv, img_norm_kv)
        time_out = time_features + self.dropout(time_attended_by_img)
        time_out_norm_ff = self.norm_time2(time_out) 
        time_out = time_out + self.dropout(self.ff_time(time_out_norm_ff))
        
        return img_out, time_out, i2t_weights, t2i_weights


class ViT_FusionModule(nn.Module): 
    def __init__(self, img_dim: int, time_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.0, fusion_type: str = 'both'):
        super().__init__()
        self.fusion_type = fusion_type
        
        if fusion_type in ['attention', 'both'] and output_dim % num_heads != 0:
            output_dim = ((output_dim // num_heads) + (output_dim % num_heads > 0)) * num_heads

        self.img_proj = nn.Linear(img_dim, output_dim)
        self.time_proj = nn.Linear(time_dim, output_dim)
        
        if fusion_type in ['attention', 'both']:
            self.cross_attention = CrossModalAttention(output_dim, num_heads, dropout)
        
        if fusion_type in ['concat', 'both']:
            self.cls_fusion_proj = nn.Linear(output_dim * 2, output_dim) 
        elif fusion_type == 'attention': 
             self.cls_fusion_proj = nn.Linear(output_dim * 2, output_dim) 
        
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, img_features: torch.Tensor, time_features: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        img_proj = self.img_proj(img_features) 
        time_proj = self.time_proj(time_features) 
        
        i2t_weights, t2i_weights = None, None
        
        if self.fusion_type in ['attention', 'both']:
            img_fused_seq, time_fused_seq, i2t_weights, t2i_weights = self.cross_attention(img_proj, time_proj)
        else: 
            img_fused_seq, time_fused_seq = img_proj, time_proj
            
        img_cls = img_fused_seq[:, 0] 
        time_cls = time_fused_seq[:, 0] 
        
        if self.fusion_type in ['concat', 'both', 'attention']: 
            fused_cls_tokens = torch.cat([img_cls, time_cls], dim=-1) 
            fused_features = self.cls_fusion_proj(fused_cls_tokens)
        else: 
            raise ValueError(f"Unknown fusion_type: {self.fusion_type}")

        fused_features = self.norm(fused_features)
        fused_features = self.dropout(fused_features)
        return fused_features, i2t_weights, t2i_weights


class DualTransformerModel(nn.Module): 
    def __init__(
        self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, time_input_dim: int = 8,
        max_len: int = 100, img_embed_dim: int = 768, time_embed_dim: int = 512, fusion_dim: int = 512,
        img_num_heads: int = 12, time_num_heads: int = 8, fusion_num_heads: int = 8,
        img_ff_dim: int = 3072, time_ff_dim: int = 2048, img_num_layers: int = 12,
        time_num_layers: int = 6, num_classes: int = 2, dropout: float = 0.1, fusion_type: str = 'both'
    ):
        super().__init__()
        if img_embed_dim % img_num_heads != 0: img_embed_dim = ((img_embed_dim // img_num_heads) + (img_embed_dim % img_num_heads > 0)) * img_num_heads
        if time_embed_dim % time_num_heads != 0: time_embed_dim = ((time_embed_dim // time_num_heads) + (time_embed_dim % time_num_heads > 0)) * time_num_heads
        if fusion_dim % fusion_num_heads != 0: fusion_dim = ((fusion_dim // fusion_num_heads) + (fusion_dim % fusion_num_heads > 0)) * fusion_num_heads

        self.image_transformer = ImageTransformer(img_size, patch_size, in_channels, img_embed_dim, img_num_heads, img_ff_dim, img_num_layers, num_classes, dropout)
        self.time_transformer = TimeSeriesTransformer(time_input_dim, time_embed_dim, max_len, time_num_heads, time_ff_dim, time_num_layers, num_classes, dropout)
        self.fusion_module = ViT_FusionModule(img_embed_dim, time_embed_dim, fusion_dim, fusion_num_heads, dropout, fusion_type)
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, images: torch.Tensor, time_series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        img_logits, img_full_features, img_attn_weights = self.image_transformer(images)
        time_logits, time_full_features, time_attn_weights = self.time_transformer(time_series)
        
        fused_features, i2t_weights, t2i_weights = self.fusion_module(img_full_features, time_full_features)
        fused_logits = self.classifier(fused_features)
        
        return fused_logits, img_logits, time_logits, img_attn_weights, time_attn_weights, i2t_weights, t2i_weights

class ViT_CoralDataset(Dataset): 
    def __init__(self, image_paths: List[str], time_series: np.ndarray, labels: np.ndarray, image_transform: Optional[Any] = None):
        self.image_paths = image_paths
        self.time_series = time_series
        self.labels = labels
        self.image_transform = image_transform
        
    def __len__(self) -> int: return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_transform: img = self.image_transform(img)
        else: img = torch.from_numpy(img.transpose(2,0,1)).float() / 255.0
        
        ts_data = torch.from_numpy(self.time_series[idx]).float()
        label = torch.tensor(self.labels[idx], dtype=torch.long) 
        return {'image': img, 'time_series': ts_data, 'label': label}

class CoralTransformerLightning(pl.LightningModule): 
    def __init__(self, model_config: Dict[str, Any], learning_rate: float = 1e-4, weight_decay: float = 1e-4, 
                 fusion_weight: float = 0.6, img_weight: float = 0.2, time_weight: float = 0.2):
        super().__init__()
        self.save_hyperparameters() 
        self.model = DualTransformerModel(**model_config) 
        self.criterion = nn.CrossEntropyLoss() 

        num_classes = self.hparams.model_config.get('num_classes', 2)
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
        
        auroc_task = 'binary' if num_classes == 2 else 'multiclass'
        self.val_auroc = torchmetrics.AUROC(task=auroc_task, num_classes=num_classes if num_classes > 2 else None)
        self.test_auroc = torchmetrics.AUROC(task=auroc_task, num_classes=num_classes if num_classes > 2 else None)

        self.val_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')
        self.test_f1 = torchmetrics.F1Score(task='multiclass', num_classes=num_classes, average='macro')


    def forward(self, images, time_series): return self.model(images, time_series)
    
    def _common_step(self, batch, batch_idx, stage: str):
        images, time_series, labels = batch['image'], batch['time_series'], batch['label_long']  # Use long labels for CrossEntropy
        fused_logits, img_logits, time_logits, _, _, _, _ = self(images, time_series)
        
        loss_fused = self.criterion(fused_logits, labels)
        loss_img = self.criterion(img_logits, labels)
        loss_time = self.criterion(time_logits, labels)
        
        loss = self.hparams.fusion_weight * loss_fused + \
               self.hparams.img_weight * loss_img + \
               self.hparams.time_weight * loss_time
        
        probs = F.softmax(fused_logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)
        self.log(f'{stage}_loss_fused', loss_fused, on_step=(stage=='train'), on_epoch=True)

        num_classes = self.hparams.model_config.get('num_classes', 2)

        if stage == 'train':
            self.train_acc(preds, labels)
            self.log(f'{stage}_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == 'val':
            self.val_acc(preds, labels)
            self.val_f1(preds, labels)
            self.log(f'{stage}_acc', self.val_acc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_f1', self.val_f1, on_epoch=True, prog_bar=True)
            if num_classes == 2: self.val_auroc(probs[:,1], labels) 
            else: self.val_auroc(probs, labels)
            self.log(f'{stage}_auroc', self.val_auroc, on_epoch=True, prog_bar=True)

        elif stage == 'test':
            self.test_acc(preds, labels)
            self.test_f1(preds, labels)
            self.log(f'{stage}_acc', self.test_acc, on_epoch=True)
            self.log(f'{stage}_f1', self.test_f1, on_epoch=True)
            if num_classes == 2: self.test_auroc(probs[:,1], labels)
            else: self.test_auroc(probs, labels)
            self.log(f'{stage}_auroc', self.test_auroc, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        T_max_epochs = self.trainer.max_epochs if self.trainer else 10 
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max_epochs, eta_min=1e-6)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}

    def get_attention_maps(self, images, time_series):
        self.eval()
        with torch.no_grad():
            _, _, _, img_attn, time_attn, i2t_attn, t2i_attn = self(images, time_series)
        return {
            'img_attention': img_attn, 'time_attention': time_attn,
            'img2time_attention': i2t_attn, 'time2img_attention': t2i_attn
        }

def vit_visualize_attention(model, dataloader, num_samples=1, save_dir='attention_maps_vit'): 
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device
    model.eval()
    
    batch = next(iter(dataloader)) 
    images = batch['image'][:num_samples].to(device)
    time_series = batch['time_series'][:num_samples].to(device)
    
    attns = model.get_attention_maps(images, time_series)

    for i in range(num_samples):
        if attns['img_attention'] and len(attns['img_attention']) > 0:
            img_self_attn = attns['img_attention'][-1][i, :, 0, 1:].cpu().numpy() 
            img_self_attn_avg_heads = img_self_attn.mean(axis=0)
            num_patches_side = int(np.sqrt(img_self_attn_avg_heads.shape[0]))
            if num_patches_side * num_patches_side == img_self_attn_avg_heads.shape[0]: 
                plt.figure(figsize=(6,6))
                plt.imshow(img_self_attn_avg_heads.reshape(num_patches_side, num_patches_side), cmap='viridis')
                plt.title(f'ViT Img Self-Attention (CLS to Patches) Sample {i}')
                if save_dir: plt.savefig(os.path.join(save_dir, f'vit_img_self_attn_s{i}.png')); plt.close()
                else: plt.show()

        if attns['time_attention'] and len(attns['time_attention']) > 0:
            ts_self_attn = attns['time_attention'][-1][i, :, 0, 1:].cpu().numpy() 
            ts_self_attn_avg_heads = ts_self_attn.mean(axis=0)
            plt.figure(figsize=(10,4))
            plt.plot(ts_self_attn_avg_heads)
            plt.title(f'ViT TS Self-Attention (CLS to Time Steps) Sample {i}')
            if save_dir: plt.savefig(os.path.join(save_dir, f'vit_ts_self_attn_s{i}.png')); plt.close()
            else: plt.show()

        if attns['img2time_attention'] is not None:
            i2t_attn_map = attns['img2time_attention'][i, :, 0, :].cpu().numpy() 
            i2t_attn_map_avg_heads = i2t_attn_map.mean(axis=0)
            plt.figure(figsize=(10,4))
            plt.plot(i2t_attn_map_avg_heads)
            plt.title(f'ViT Img-to-TS Cross-Attention (CLS_img to Time Steps) Sample {i}')
            if save_dir: plt.savefig(os.path.join(save_dir, f'vit_i2t_cross_attn_s{i}.png')); plt.close()
            else: plt.show()


def vit_get_transforms(img_size=224): 
    train_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {'train': train_transform, 'val': val_transform, 'test': val_transform} 

def vit_load_data(image_dir: str, time_series_path: str, labels_path: str, img_size: int = 224, batch_size: int = 32, num_workers: int = 4): 
    time_series_data = np.load(time_series_path)
    labels_data = np.load(labels_path)
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    image_paths.sort() 
    
    assert len(image_paths) == len(labels_data) == len(time_series_data), "Data dimension mismatch"

    indices = np.arange(len(labels_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_data)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_data[temp_idx])
    
    transforms_dict = vit_get_transforms(img_size)
    
    train_dataset = ViT_CoralDataset([image_paths[i] for i in train_idx], time_series_data[train_idx], labels_data[train_idx], transforms_dict['train'])
    val_dataset = ViT_CoralDataset([image_paths[i] for i in val_idx], time_series_data[val_idx], labels_data[val_idx], transforms_dict['val'])
    test_dataset = ViT_CoralDataset([image_paths[i] for i in test_idx], time_series_data[test_idx], labels_data[test_idx], transforms_dict['test'])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    time_steps, time_features = time_series_data.shape[1], time_series_data.shape[2]
    num_classes = len(np.unique(labels_data))
    return {'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader, 
            'time_steps': time_steps, 'time_features': time_features, 'num_classes': num_classes}


# --- Content from xgb.py ---
class XGB_FeatureExtractor: 
    def __init__(self, pretrained_backbone: str = 'resnet18', wavelet_name: str = 'db4', wavelet_level: int = 3):
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.initialize_image_extractor(pretrained_backbone)
        
        self.temporal_feature_names_template = [
            'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr',
            'skew', 'kurtosis', 'trend', 'autocorr_lag1', 'autocorr_lag2',
            'entropy', 'energy', 'peak_frequency'
        ]
        self.wavelet_feature_names_template = ['mean', 'std', 'energy', 'entropy'] 

    def initialize_image_extractor(self, backbone_name: str):
        if backbone_name == 'resnet18':
            self.cnn_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.feature_dim = 512
        elif backbone_name == 'resnet50':
            self.cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_dim = 2048
        elif backbone_name == 'efficientnet_b0':
            self.cnn_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        if 'resnet' in backbone_name:
            self.cnn_model = torch.nn.Sequential(*(list(self.cnn_model.children())[:-1]))
        else: 
            self.cnn_model.classifier = torch.nn.Identity()
        
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def extract_image_features(self, images_np: np.ndarray) -> np.ndarray: 
        if images_np.shape[-1] == 3: images_np = np.transpose(images_np, (0, 3, 1, 2)) 
        images_np = images_np.astype(np.float32) / 255.0
            
        images_tensor = torch.from_numpy(images_np).to(self.device)
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        images_tensor = normalize(images_tensor)
        
        all_features = []
        with torch.no_grad():
            for i in range(0, len(images_tensor), 32): 
                batch = images_tensor[i:i+32]
                batch_features = self.cnn_model(batch)
                if len(batch_features.shape) > 2: 
                    batch_features = batch_features.view(batch_features.size(0), -1)
                all_features.append(batch_features.cpu().numpy())
        return np.vstack(all_features)

    def extract_temporal_features(self, time_series: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        num_samples, time_steps, num_ts_features = time_series.shape
        num_stat_features = len(self.temporal_feature_names_template)
        extracted_features = np.zeros((num_samples, num_ts_features * num_stat_features))
        feature_names = []

        for i in range(num_samples):
            for j in range(num_ts_features):
                ts = time_series[i, :, j]
                f_idx_offset = j * num_stat_features
                
                extracted_features[i, f_idx_offset + 0] = np.mean(ts)
                extracted_features[i, f_idx_offset + 1] = np.std(ts) if np.std(ts) > 1e-9 else 0
                extracted_features[i, f_idx_offset + 2] = np.min(ts)
                extracted_features[i, f_idx_offset + 3] = np.max(ts)
                extracted_features[i, f_idx_offset + 4] = np.median(ts)
                extracted_features[i, f_idx_offset + 5] = np.percentile(ts, 25)
                extracted_features[i, f_idx_offset + 6] = np.percentile(ts, 75)
                extracted_features[i, f_idx_offset + 7] = extracted_features[i, f_idx_offset + 6] - extracted_features[i, f_idx_offset + 5] 

                if np.std(ts) > 1e-9:
                    ts_norm = (ts - np.mean(ts)) / np.std(ts)
                    extracted_features[i, f_idx_offset + 8] = np.mean(ts_norm**3) 
                    extracted_features[i, f_idx_offset + 9] = np.mean(ts_norm**4) - 3 
                    extracted_features[i, f_idx_offset + 10] = np.polyfit(np.arange(time_steps), ts, 1)[0] if time_steps > 1 else 0 
                    extracted_features[i, f_idx_offset + 11] = np.corrcoef(ts[:-1], ts[1:])[0, 1] if time_steps > 1 else 0 
                    extracted_features[i, f_idx_offset + 12] = np.corrcoef(ts[:-2], ts[2:])[0, 1] if time_steps > 2 else 0 
                else: 
                    extracted_features[i, f_idx_offset + 8 : f_idx_offset + 13] = 0

                p = ts / (np.sum(ts) + 1e-9) 
                p = p[p>0] 
                extracted_features[i, f_idx_offset + 13] = -np.sum(p * np.log2(p + 1e-10)) if len(p) > 0 else 0 
                extracted_features[i, f_idx_offset + 14] = np.sum(ts**2) 
                
                if time_steps > 1:
                    fft_vals = np.abs(np.fft.rfft(ts))[1:] 
                    extracted_features[i, f_idx_offset + 15] = np.argmax(fft_vals) + 1 if len(fft_vals) > 0 else 0 
                else:
                    extracted_features[i, f_idx_offset + 15] = 0
        
        for j_name in range(num_ts_features):
            for stat_name in self.temporal_feature_names_template:
                feature_names.append(f"ts{j_name}_{stat_name}")
        return extracted_features, feature_names
    
    def extract_wavelet_features(self, time_series: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        num_samples, time_steps, num_ts_features = time_series.shape
        
        try:
            wavelet_obj = pywt.Wavelet(self.wavelet_name)
            max_level = pywt.dwt_max_level(time_steps, wavelet_obj)
        except ValueError:
            max_level = int(np.log2(time_steps)) -1 if time_steps > 1 else 0
        
        actual_level = min(self.wavelet_level, max_level); actual_level = max(1, actual_level)
        
        num_coeff_sets = actual_level + 1 
        num_wavelet_stat_features = len(self.wavelet_feature_names_template)
        total_wavelet_features_per_ts = num_coeff_sets * num_wavelet_stat_features
        
        extracted_features = np.zeros((num_samples, num_ts_features * total_wavelet_features_per_ts))
        feature_names = []

        for i in range(num_samples):
            for j in range(num_ts_features):
                ts = time_series[i, :, j]
                coeffs = pywt.wavedec(ts, self.wavelet_name, level=actual_level)
                f_idx_offset = j * total_wavelet_features_per_ts 

                for k_level, coeff_set in enumerate(coeffs): 
                    c_idx_offset = f_idx_offset + k_level * num_wavelet_stat_features 
                    if len(coeff_set) == 0:
                        extracted_features[i, c_idx_offset : c_idx_offset + num_wavelet_stat_features] = 0
                        continue
                    
                    extracted_features[i, c_idx_offset + 0] = np.mean(coeff_set)
                    extracted_features[i, c_idx_offset + 1] = np.std(coeff_set) if np.std(coeff_set) > 1e-9 else 0
                    extracted_features[i, c_idx_offset + 2] = np.sum(coeff_set**2) 
                    
                    p_wavelet = np.abs(coeff_set) / (np.sum(np.abs(coeff_set)) + 1e-9)
                    p_wavelet = p_wavelet[p_wavelet>0]
                    extracted_features[i, c_idx_offset + 3] = -np.sum(p_wavelet * np.log2(p_wavelet + 1e-10)) if len(p_wavelet) > 0 else 0 
        
        for j_name in range(num_ts_features):
            for k_level_name_idx in range(num_coeff_sets):
                level_name = "cA" if k_level_name_idx == 0 else f"cD{k_level_name_idx}" 
                for stat_name in self.wavelet_feature_names_template:
                    feature_names.append(f"ts{j_name}_wavelet_{level_name}_{stat_name}")
        return extracted_features, feature_names

    def extract_spatial_features(self, images_np: np.ndarray) -> Tuple[np.ndarray, List[str]]: 
        num_samples = images_np.shape[0]
        num_spatial_stat_features = 12 
        extracted_features = np.zeros((num_samples, num_spatial_stat_features))
        feature_names = [
            "R_mean", "G_mean", "B_mean", "R_std", "G_std", "B_std",
            "RG_ratio", "RB_ratio", "GB_ratio",
            "grad_mean", "grad_std", "grad_p90"
        ]

        for i in range(num_samples):
            img_rgb = images_np[i] 
            img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

            for c in range(3): 
                extracted_features[i, c] = np.mean(img_rgb[:,:,c])
                extracted_features[i, c+3] = np.std(img_rgb[:,:,c])
            
            extracted_features[i, 6] = extracted_features[i,0] / (extracted_features[i,1] + 1e-9) 
            extracted_features[i, 7] = extracted_features[i,0] / (extracted_features[i,2] + 1e-9) 
            extracted_features[i, 8] = extracted_features[i,1] / (extracted_features[i,2] + 1e-9) 

            sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(sobelx**2 + sobely**2)
            extracted_features[i, 9] = np.mean(grad_mag)
            extracted_features[i, 10] = np.std(grad_mag)
            extracted_features[i, 11] = np.percentile(grad_mag, 90)
            
        return extracted_features, feature_names

    def extract_all_features(self, images_np: np.ndarray, time_series: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        all_feats_list = []
        all_names_list = []

        cnn_feats = self.extract_image_features(images_np)
        all_feats_list.append(cnn_feats)
        all_names_list.extend([f"cnn_{k}" for k in range(cnn_feats.shape[1])])

        temporal_feats, temporal_names = self.extract_temporal_features(time_series)
        all_feats_list.append(temporal_feats)
        all_names_list.extend(temporal_names)

        wavelet_feats, wavelet_names = self.extract_wavelet_features(time_series)
        all_feats_list.append(wavelet_feats)
        all_names_list.extend(wavelet_names)

        spatial_feats, spatial_names = self.extract_spatial_features(images_np)
        all_feats_list.append(spatial_feats)
        all_names_list.extend(spatial_names)
        
        combined_features = np.hstack(all_feats_list)
        
        assert combined_features.shape[1] == len(all_names_list), \
            f"Feature data columns ({combined_features.shape[1]}) and names length ({len(all_names_list)}) mismatch!"
        
        return combined_features, all_names_list


class XGBoostCoralModel:
    def __init__(self, feature_extractor: Optional[XGB_FeatureExtractor] = None, xgb_params: Optional[Dict[str, Any]] = None,
                 n_estimators: int = 100, learning_rate: float = 0.1, max_depth: int = 5,
                 subsample: float = 0.8, colsample_bytree: float = 0.8,
                 early_stopping_rounds: int = 10, feature_selection_threshold: float = 0.01, cv_folds: int = 5):
        if xgb is None: raise ImportError("XGBoost library is required for XGBoostCoralModel.")
        self.feature_extractor = feature_extractor if feature_extractor else XGB_FeatureExtractor()
        
        default_params = {
            'objective': 'binary:logistic', 'eval_metric': 'auc', 
            # 'n_estimators' is handled by num_boost_round in xgb.train
            'learning_rate': learning_rate, 'max_depth': max_depth, 'subsample': subsample,
            'colsample_bytree': colsample_bytree, 'tree_method': 'hist', 'use_label_encoder': False,
            'verbosity': 0 
        }
        # n_estimators from xgb_params takes precedence if provided, else from constructor arg
        self.xgb_params = {**default_params, **(xgb_params if xgb_params else {})} 
        if 'n_estimators' in self.xgb_params: # n_estimators is for sklearn API, not xgb.train
             self.num_boost_round = self.xgb_params.pop('n_estimators')
        else:
             self.num_boost_round = n_estimators


        self.early_stopping_rounds = early_stopping_rounds
        self.feature_selection_threshold = feature_selection_threshold
        self.cv_folds = cv_folds
        self.model = None
        self.feature_names_fitted = None 
        self.selected_features_indices = None 
        self.scaler = StandardScaler()

    def preprocess_and_select_features(self, features_raw: np.ndarray, feature_names_raw: List[str], train_mode: bool = True) -> np.ndarray:
        if train_mode:
            scaled_features = self.scaler.fit_transform(features_raw)
            self.feature_names_fitted = feature_names_raw 
            
            if self.model and self.feature_selection_threshold > 0: 
                importances = self.model.get_score(importance_type='gain')
                imp_map = {name: score for name, score in importances.items()}
                norm_importances = np.array([imp_map.get(name, 0) for name in self.feature_names_fitted])
                if np.sum(norm_importances) > 0: norm_importances = norm_importances / np.sum(norm_importances)
                
                self.selected_features_indices = [
                    i for i, imp in enumerate(norm_importances) if imp > self.feature_selection_threshold
                ]
                if not self.selected_features_indices: 
                    self.selected_features_indices = list(range(len(self.feature_names_fitted)))

                return scaled_features[:, self.selected_features_indices]
            else: 
                self.selected_features_indices = list(range(len(self.feature_names_fitted)))
                return scaled_features
        else: 
            scaled_features = self.scaler.transform(features_raw)
            if self.selected_features_indices is not None:
                return scaled_features[:, self.selected_features_indices]
            else: 
                return scaled_features

    def train(self, images_np: np.ndarray, time_series: np.ndarray, labels: np.ndarray, use_cv: bool = True) -> Dict[str, float]:
        features_raw, feature_names_raw = self.feature_extractor.extract_all_features(images_np, time_series)
        
        X_scaled_all_feats = self.scaler.fit_transform(features_raw) 
        self.feature_names_fitted = feature_names_raw 

        if use_cv:
            return self._train_with_cv_and_selection(X_scaled_all_feats, labels, feature_names_raw)

        X_train_all, X_val_all, y_train, y_val = train_test_split(X_scaled_all_feats, labels, test_size=0.2, random_state=42, stratify=labels)
        
        dtrain = DMatrix(X_train_all, label=y_train, feature_names=self.feature_names_fitted)
        dval = DMatrix(X_val_all, label=y_val, feature_names=self.feature_names_fitted)
        
        self.model = xgb.train(
            self.xgb_params, dtrain,
            num_boost_round=self.num_boost_round, 
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100 if self.xgb_params.get('verbosity',0) > 0 else False
        )

        self._perform_feature_selection(self.feature_names_fitted)
        
        y_pred_proba_val = self.model.predict(dval)
        y_pred_binary_val = (y_pred_proba_val > 0.5).astype(int)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_binary_val),
            'precision': precision_score(y_val, y_pred_binary_val, zero_division=0),
            'recall': recall_score(y_val, y_pred_binary_val, zero_division=0),
            'f1': f1_score(y_val, y_pred_binary_val, zero_division=0),
            'auc': roc_auc_score(y_val, y_pred_proba_val)
        }
        return metrics

    def _perform_feature_selection(self, current_feature_names: List[str]):
        if not self.model or not current_feature_names: return

        importances = self.model.get_score(importance_type='gain')
        
        booster_feature_names = self.model.feature_names
        if booster_feature_names is None: 
            booster_feature_names = [f'f{i}' for i in range(len(current_feature_names))] # XGB default if names not set
            
        imp_map = {}
        for f_idx_str, score in importances.items():
            try: 
                idx = int(f_idx_str[1:]) # Assumes 'fDDDD' format from booster if names not set
                if 0 <= idx < len(booster_feature_names): # Check index validity
                    actual_name_from_booster = booster_feature_names[idx]
                    imp_map[actual_name_from_booster] = score
            except (ValueError, TypeError): # If f_idx_str is not 'f'+number, assume it's an actual name
                 if f_idx_str in current_feature_names: # Check if it's one of the provided full names
                    imp_map[f_idx_str] = score
        
        full_importances_values = np.array([imp_map.get(name, 0.0) for name in current_feature_names])


        if np.sum(full_importances_values) > 0:
            norm_importances = full_importances_values / np.sum(full_importances_values)
        else: 
            norm_importances = np.zeros_like(full_importances_values)

        self.selected_features_indices = [
            i for i, imp_score in enumerate(norm_importances) if imp_score > self.feature_selection_threshold
        ]
        if not self.selected_features_indices and len(current_feature_names) > 0: 
            if np.any(full_importances_values > 0):
                 self.selected_features_indices = [np.argmax(full_importances_values)]
            else: 
                 self.selected_features_indices = [0] 


    def _train_with_cv_and_selection(self, X_scaled_all_feats: np.ndarray, labels: np.ndarray, feature_names_raw: List[str]) -> Dict[str, float]:
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_metrics_agg = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'auc': []}

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_scaled_all_feats, labels)):
            X_train_fold, X_val_fold = X_scaled_all_feats[train_idx], X_scaled_all_feats[val_idx]
            y_train_fold, y_val_fold = labels[train_idx], labels[val_idx]

            dtrain_fold = DMatrix(X_train_fold, label=y_train_fold, feature_names=feature_names_raw)
            dval_fold = DMatrix(X_val_fold, label=y_val_fold, feature_names=feature_names_raw)

            model_fold = xgb.train(
                self.xgb_params, dtrain_fold,
                num_boost_round=self.num_boost_round,
                evals=[(dtrain_fold, 'train'), (dval_fold, 'val')],
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
            
            y_pred_proba_fold = model_fold.predict(dval_fold)
            y_pred_binary_fold = (y_pred_proba_fold > 0.5).astype(int)

            cv_metrics_agg['accuracy'].append(accuracy_score(y_val_fold, y_pred_binary_fold))
            cv_metrics_agg['precision'].append(precision_score(y_val_fold, y_pred_binary_fold, zero_division=0))
            cv_metrics_agg['recall'].append(recall_score(y_val_fold, y_pred_binary_fold, zero_division=0))
            cv_metrics_agg['f1'].append(f1_score(y_val_fold, y_pred_binary_fold, zero_division=0))
            cv_metrics_agg['auc'].append(roc_auc_score(y_val_fold, y_pred_proba_fold))
        
        dtrain_full = DMatrix(X_scaled_all_feats, label=labels, feature_names=feature_names_raw)
        self.model = xgb.train(
            self.xgb_params, dtrain_full,
            num_boost_round=self.num_boost_round, 
            verbose_eval=False
        )
        
        self._perform_feature_selection(feature_names_raw)
        
        final_metrics = {metric: np.mean(scores) for metric, scores in cv_metrics_agg.items()}
        final_metrics.update({f"{metric}_std": np.std(scores) for metric, scores in cv_metrics_agg.items()})
        return final_metrics

    def predict(self, images_np: np.ndarray, time_series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.model: raise ValueError("Model not trained yet.")
        # Use the feature extractor associated with this model instance
        features_raw, current_feature_names_from_extraction = self.feature_extractor.extract_all_features(images_np, time_series)
        
        X_scaled = self.scaler.transform(features_raw) # Scaler was fit on self.feature_names_fitted
        
        # The DMatrix for prediction must match the features the model was trained on.
        # If feature selection was done, selected_features_indices refers to indices in self.feature_names_fitted.
        # The model itself (self.model) was trained with feature names `self.feature_names_fitted`.
        
        X_to_predict = X_scaled # By default, use all scaled features
        names_for_dmatrix = self.feature_names_fitted # Names model was trained with

        if self.selected_features_indices is not None and self.feature_names_fitted is not None:
            # This implies we want to predict using only the selected subset,
            # which is only valid if the model was *retrained* on this subset.
            # For now, assume self.model is trained on ALL features in self.feature_names_fitted.
            # If self.model was retrained on selected features, then selected_names_for_dmatrix
            # would be [self.feature_names_fitted[i] for i in self.selected_features_indices]
            # and X_to_predict would be X_scaled[:, self.selected_features_indices].
            # The current code trains self.model on all features, so predict should use all.
            pass # Keep X_to_predict as X_scaled and names_for_dmatrix as self.feature_names_fitted

        if names_for_dmatrix is None:
            raise ValueError("feature_names_fitted is None. Model might not have been trained correctly.")
        if X_to_predict.shape[1] != len(names_for_dmatrix):
            # This would indicate a severe internal inconsistency.
             raise ValueError(f"Data columns ({X_to_predict.shape[1]}) don't match names length ({len(names_for_dmatrix)}) for prediction.")


        dtest = DMatrix(X_to_predict, feature_names=names_for_dmatrix)
        y_pred_proba = self.model.predict(dtest)
        y_pred_binary = (y_pred_proba > 0.5).astype(int)
        return y_pred_proba, y_pred_binary

    def get_feature_importances(self) -> Optional[Dict[str, float]]:
        if not self.model or not self.feature_names_fitted: return None
        importances_raw = self.model.get_score(importance_type='gain')
        
        imp_dict = {name: 0.0 for name in self.feature_names_fitted}
        
        booster_feature_names = self.model.feature_names
        if booster_feature_names is None: 
            booster_feature_names = [f'f{i}' for i in range(len(self.feature_names_fitted))]

        for f_idx_str, score in importances_raw.items():
            try:
                actual_name_from_booster = booster_feature_names[int(f_idx_str[1:])] if f_idx_str.startswith('f') else f_idx_str
                if actual_name_from_booster in imp_dict: # Make sure the name is in our list of fitted features
                     imp_dict[actual_name_from_booster] = score
            except (ValueError, IndexError, TypeError): 
                 if f_idx_str in imp_dict: 
                      imp_dict[f_idx_str] = score

        total_importance = sum(imp_dict.values())
        if total_importance > 0:
            return {name: score / total_importance for name, score in imp_dict.items() if name in self.feature_names_fitted}
        return {name: 0.0 for name in self.feature_names_fitted} # Return all with 0 if no importance


    def get_shap_values(self, images_np: np.ndarray, time_series: np.ndarray, sample_size: int = 100) -> Optional[Dict[str, Any]]:
        if not self.model or shap is None: return None
        features_raw, _ = self.feature_extractor.extract_all_features(images_np, time_series)
        X_scaled = self.scaler.transform(features_raw)

        # SHAP values should be calculated on the features the model was trained with.
        # If feature selection was only for analysis, use all features.
        # If a model using only selected features is desired, it should be retrained.
        # For this instance (self.model), it was trained on self.feature_names_fitted.
        X_for_shap = X_scaled
        names_for_shap = self.feature_names_fitted

        if self.selected_features_indices is not None and self.feature_names_fitted is not None:
            # If the *intent* is to get SHAP for a model conceptually using selected features,
            # but self.model is trained on all, this is complex.
            # For now, assume SHAP is for self.model (trained on all features).
             pass # X_for_shap remains X_scaled, names_for_shap remains self.feature_names_fitted

        if names_for_shap is None:
             # This might happen if the model was trained without feature names in DMatrix
             # Fallback to generic names for SHAP if absolutely necessary
             names_for_shap = [f"f{i}" for i in range(X_for_shap.shape[1])]


        if sample_size < X_for_shap.shape[0]:
            indices = np.random.choice(X_for_shap.shape[0], sample_size, replace=False)
            X_sample = X_for_shap[indices]
        else:
            X_sample = X_for_shap
        
        dmatrix_for_shap = DMatrix(X_sample, feature_names=names_for_shap)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(dmatrix_for_shap) 
        
        return {'shap_values': shap_values, 'feature_names': names_for_shap, 'data': X_sample}

    def save_model(self, model_path: str):
        if not self.model: raise ValueError("Model not trained.")
        dir_name = os.path.dirname(model_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        self.model.save_model(model_path)
        
        metadata = {
            'feature_names_fitted': self.feature_names_fitted,
            'selected_features_indices': self.selected_features_indices,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') and self.scaler.mean_ is not None else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') and self.scaler.scale_ is not None else None,
            'xgb_params': self.xgb_params,
            'num_boost_round': self.num_boost_round # Save this as well
        }
        with open(os.path.splitext(model_path)[0] + "_metadata.json", 'w') as f:
            json.dump(metadata, f)

    def load_model(self, model_path: str):
        if xgb is None: raise ImportError("XGBoost library is required to load XGBoostCoralModel.")
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        with open(os.path.splitext(model_path)[0] + "_metadata.json", 'r') as f:
            metadata = json.load(f)
        self.feature_names_fitted = metadata.get('feature_names_fitted')
        self.selected_features_indices = metadata.get('selected_features_indices')
        self.scaler = StandardScaler() 
        if metadata.get('scaler_mean') and metadata.get('scaler_scale'):
            self.scaler.mean_ = np.array(metadata['scaler_mean'])
            self.scaler.scale_ = np.array(metadata['scaler_scale'])
            self.scaler.n_features_in_ = len(self.scaler.mean_) if self.scaler.mean_ is not None else 0
        self.xgb_params = metadata.get('xgb_params', self.xgb_params) 
        self.num_boost_round = metadata.get('num_boost_round', self.num_boost_round if hasattr(self, 'num_boost_round') else 100)


def load_data_polars(data_dir: str) -> Optional[Tuple[Any, Any, Any]]: 
    if plrs is None:
        print("Polars not available. Cannot load data with load_data_polars.")
        return None
    try:
        image_df = plrs.read_csv(os.path.join(data_dir, 'processed/imagery_features/metadata.csv'))
        timeseries_df = plrs.read_csv(os.path.join(data_dir, 'processed/time_series_features/timeseries.csv'))
        labels_df = plrs.read_csv(os.path.join(data_dir, 'processed/combined_dataset/labels.csv'))
        return image_df, timeseries_df, labels_df
    except Exception as e:
        print(f"Error loading data with Polars: {e}")
        return None


# --- Content from ensemble.py ---
class EnsembleTimeSeriesFeatureExtractor: 
    def __init__(self, wavelet: str = 'db4', level: int = 3):
        self.wavelet_name = wavelet 
        self.level = level
    
    def extract_statistical_features(self, time_series_np: np.ndarray) -> np.ndarray:
        batch_size, _, num_features = time_series_np.shape
        stat_funcs = [np.mean, np.std, np.min, np.max, np.median, 
                      lambda x: np.percentile(x, 25), lambda x: np.percentile(x, 75)]
        num_stats = len(stat_funcs)
        features = np.zeros((batch_size, num_features * num_stats))
        
        for i in range(batch_size):
            for j in range(num_features):
                ts = time_series_np[i, :, j]
                for k, func in enumerate(stat_funcs):
                    features[i, j * num_stats + k] = func(ts) if len(ts) > 0 else 0
        return features
    
    def extract_wavelet_features(self, time_series_np: np.ndarray) -> np.ndarray:
        batch_size, seq_len, num_features = time_series_np.shape
        try:
            wavelet_obj = pywt.Wavelet(self.wavelet_name)
            max_level = pywt.dwt_max_level(seq_len, wavelet_obj)
        except ValueError:
             max_level = int(np.log2(seq_len)) -1 if seq_len > 1 else 0

        actual_level = min(self.level, max_level); actual_level = max(1, actual_level)
        
        num_coeff_sets = actual_level + 1
        wavelet_stats_per_set = 4 
        feature_dim = num_features * num_coeff_sets * wavelet_stats_per_set
        features = np.zeros((batch_size, feature_dim))
        
        for i in range(batch_size):
            for j in range(num_features): 
                ts = time_series_np[i, :, j]
                coeffs = pywt.wavedec(ts, self.wavelet_name, level=actual_level)
                
                channel_base_idx = j * num_coeff_sets * wavelet_stats_per_set

                for coeff_set_idx, coeff_set in enumerate(coeffs): 
                    stat_base_idx = channel_base_idx + coeff_set_idx * wavelet_stats_per_set

                    if len(coeff_set) == 0:
                         features[i, stat_base_idx : stat_base_idx + wavelet_stats_per_set] = 0
                         continue

                    features[i, stat_base_idx + 0] = np.mean(coeff_set)
                    features[i, stat_base_idx + 1] = np.std(coeff_set)
                    features[i, stat_base_idx + 2] = np.sum(coeff_set**2)
                    p = np.abs(coeff_set) / (np.sum(np.abs(coeff_set)) + 1e-10)
                    p = p[p>0]
                    features[i, stat_base_idx + 3] = -np.sum(p * np.log2(p + 1e-10)) if len(p) > 0 else 0
        return features
            
    def extract_all_features(self, time_series_np: np.ndarray) -> np.ndarray:
        stat_features = self.extract_statistical_features(time_series_np)
        wavelet_features = self.extract_wavelet_features(time_series_np)
        return np.hstack([stat_features, wavelet_features])

class EnsembleModel(nn.Module):
    def __init__(
        self,
        cnn_lstm_params: Dict[str, Any], 
        tcn_params: Dict[str, Any],
        transformer_params: Dict[str, Any],
        ensemble_type: str = 'weighted', 
        use_cnn_lstm: bool = True, use_tcn: bool = True, use_transformer: bool = True,
        use_ml_models_for_boosting: bool = False, 
        dropout: float = 0.3,
        seq_len: int = 24, 
        time_input_dim: int = 8 
    ):
        super(EnsembleModel, self).__init__()
        self.ensemble_type = ensemble_type
        self.models_dict = nn.ModuleDict() 

        if use_cnn_lstm: self.models_dict['cnn_lstm'] = CoralNet(**cnn_lstm_params)
        if use_tcn: self.models_dict['tcn'] = TCNCoralModel(**tcn_params)
        if use_transformer: self.models_dict['transformer'] = DualTransformerModel(**transformer_params)
        
        self.ml_models = {} 
        self.use_ml_models_for_boosting = use_ml_models_for_boosting

        if ensemble_type == 'weighted':
            num_pt_models = len(self.models_dict)
            self.model_weights = nn.Parameter(torch.ones(num_pt_models) / num_pt_models if num_pt_models > 0 else torch.empty(0))
        
        elif ensemble_type == 'stacking':
            self.feature_extractor_for_ml = EnsembleTimeSeriesFeatureExtractor() 
            meta_input_dim = len(self.models_dict) 
            dummy_ts = np.zeros((1, seq_len, time_input_dim)) 
            ml_feat_dim = self.feature_extractor_for_ml.extract_all_features(dummy_ts).shape[1]
            meta_input_dim += ml_feat_dim
            
            self.meta_learner = nn.Sequential(
                nn.Linear(meta_input_dim, 64), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(64, 1) 
            )
            
        elif ensemble_type == 'boosting': 
            num_outputs_to_combine = len(self.models_dict)
            if self.use_ml_models_for_boosting: num_outputs_to_combine += 3 
            self.combiner = nn.Linear(num_outputs_to_combine, 1) if num_outputs_to_combine > 0 else None


    def init_ml_models(self, X_train_ml_feats: np.ndarray, y_train: np.ndarray):
        if self.ensemble_type != 'boosting' or not self.use_ml_models_for_boosting: return
        if X_train_ml_feats is None or y_train is None:
            return

        self.ml_models['rf'] = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train_ml_feats, y_train)
        self.ml_models['gb'] = GradientBoostingClassifier(n_estimators=100, random_state=42).fit(X_train_ml_feats, y_train)
        if xgb: self.ml_models['xgb'] = xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss').fit(X_train_ml_feats, y_train)

    def _get_pt_model_predictions(self, images, time_series):
        pt_preds = []
        for model_name, model_instance in self.models_dict.items():
            out = model_instance(images, time_series)
            if model_name == 'transformer': 
                 fused_logits = out[0] 
                 if fused_logits.shape[1] != 1: # Transformer for ensemble MUST output 1 logit
                      raise ValueError(f"Transformer in ensemble expected single logit for BCE, got {fused_logits.shape}. Adjust transformer_params['num_classes']=1.")
                 pred_logit = fused_logits
            else: 
                pred_logit = out[0] if isinstance(out, tuple) else out 
            pt_preds.append(pred_logit)
        return pt_preds 

    def forward(self, images: torch.Tensor, time_series: torch.Tensor, ml_features_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        pt_model_outputs = self._get_pt_model_predictions(images, time_series) 

        if self.ensemble_type == 'weighted':
            if not pt_model_outputs: return torch.zeros(images.size(0), 1, device=images.device) 
            stacked_preds = torch.cat(pt_model_outputs, dim=1) 
            weights = F.softmax(self.model_weights, dim=0)
            final_pred = (stacked_preds * weights).sum(dim=1, keepdim=True) 
            return final_pred

        elif self.ensemble_type == 'stacking':
            if not pt_model_outputs and ml_features_input is None : return torch.zeros(images.size(0), 1, device=images.device)
            
            meta_learner_inputs = []
            if pt_model_outputs:
                meta_learner_inputs.append(torch.cat(pt_model_outputs, dim=1)) 
            
            if ml_features_input is not None:
                meta_learner_inputs.append(ml_features_input.to(images.device))
            
            if not meta_learner_inputs: return torch.zeros(images.size(0), 1, device=images.device)

            combined_for_meta = torch.cat(meta_learner_inputs, dim=1)
            return self.meta_learner(combined_for_meta) 

        elif self.ensemble_type == 'boosting':
            all_logit_sources = []
            if pt_model_outputs:
                all_logit_sources.append(torch.cat(pt_model_outputs, dim=1))

            if self.use_ml_models_for_boosting and ml_features_input is not None and self.ml_models:
                ml_preds_list = []
                ml_features_np = ml_features_input.cpu().numpy()
                if self.ml_models.get('rf'): ml_preds_list.append(self.ml_models['rf'].predict_proba(ml_features_np)[:, 1])
                if self.ml_models.get('gb'): ml_preds_list.append(self.ml_models['gb'].predict_proba(ml_features_np)[:, 1])
                if self.ml_models.get('xgb') and xgb: ml_preds_list.append(self.ml_models['xgb'].predict_proba(ml_features_np)[:, 1])
                
                if ml_preds_list:
                    ml_preds_np = np.stack(ml_preds_list, axis=1)
                    ml_logits_np = np.log((ml_preds_np + 1e-9) / (1 - ml_preds_np + 1e-9))
                    all_logit_sources.append(torch.from_numpy(ml_logits_np).float().to(images.device))

            if not all_logit_sources or self.combiner is None: return torch.zeros(images.size(0), 1, device=images.device)
            
            combined_logits = torch.cat(all_logit_sources, dim=1)
            return self.combiner(combined_logits) 
        
        else: 
            if not pt_model_outputs: return torch.zeros(images.size(0), 1, device=images.device)
            return torch.mean(torch.stack(pt_model_outputs, dim=0), dim=0) 


class Ensemble_CoralDataset(Dataset): 
    def __init__(self, image_paths: List[str], time_series_np: np.ndarray, labels_np: np.ndarray,
                 transform: Optional[Any] = None, extract_ml_features_flag: bool = False, 
                 seq_len: Optional[int] = None, time_input_dim: Optional[int] = None): 
        self.image_paths = image_paths
        self.time_series_np = time_series_np
        self.labels_np = labels_np
        self.transform = transform
        self.extract_ml_features_flag = extract_ml_features_flag
        
        if extract_ml_features_flag:
            if seq_len is None or time_input_dim is None:
                seq_len = time_series_np.shape[1]
                time_input_dim = time_series_np.shape[2]
            self.ml_feature_extractor = EnsembleTimeSeriesFeatureExtractor()
            self.ml_features_extracted = self.ml_feature_extractor.extract_all_features(time_series_np)
        else:
            self.ml_features_extracted = None
            
    def __len__(self) -> int: return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, np.ndarray]]:
        img = cv2.imread(self.image_paths[idx]); img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform: img = self.transform(img)
        else: img = torch.FloatTensor(img.transpose(2,0,1)) / 255.0
            
        ts_data = torch.FloatTensor(self.time_series_np[idx])
        label = torch.FloatTensor([self.labels_np[idx]])
        
        item = {'image': img, 'time_series': ts_data, 'label': label}
        if self.ml_features_extracted is not None:
            item['ml_features'] = torch.FloatTensor(self.ml_features_extracted[idx])
        return item


class EnsembleLightningModel(pl.LightningModule):
    def __init__(self, model_params: Dict[str, Any], learning_rate: float = 1e-4, weight_decay: float = 1e-5, pos_weight: float = 2.0):
        super(EnsembleLightningModel, self).__init__()
        self.save_hyperparameters()
        
        self.model = EnsembleModel(**model_params)
        
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
        self.train_acc = torchmetrics.Accuracy(task='binary')
        self.val_acc = torchmetrics.Accuracy(task='binary')
        self.test_acc = torchmetrics.Accuracy(task='binary')
        self.val_auroc = torchmetrics.AUROC(task='binary')
        self.test_auroc = torchmetrics.AUROC(task='binary')
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

    def forward(self, images, time_series, ml_features=None):
        return self.model(images, time_series, ml_features_input=ml_features)

    def _common_step(self, batch, batch_idx, stage: str):
        images, time_series, labels = batch['image'], batch['time_series'], batch['label_float']  # Use float labels for BCE
        ml_features = batch.get('ml_features', None)
        
        logits = self(images, time_series, ml_features)
        logits = logits.view(-1); labels = labels.view(-1)
        loss = self.criterion(logits, labels)
        
        probs = torch.sigmoid(logits); preds = probs > 0.5
        self.log(f'{stage}_loss', loss, on_step=(stage=='train'), on_epoch=True, prog_bar=True)

        if stage == 'train':
            self.train_acc(preds, labels.long())
            self.log(f'{stage}_acc', self.train_acc, on_step=True, on_epoch=True, prog_bar=True)
        elif stage == 'val':
            self.val_acc(preds, labels.long()); self.val_auroc(probs, labels.long()); self.val_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.val_acc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_auroc', self.val_auroc, on_epoch=True, prog_bar=True)
            self.log(f'{stage}_f1', self.val_f1, on_epoch=True, prog_bar=True)
        elif stage == 'test':
            self.test_acc(preds, labels.long()); self.test_auroc(probs, labels.long()); self.test_f1(preds, labels.long())
            self.log(f'{stage}_acc', self.test_acc, on_epoch=True)
            self.log(f'{stage}_auroc', self.test_auroc, on_epoch=True)
            self.log(f'{stage}_f1', self.test_f1, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'train')
    def validation_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'val')
    def test_step(self, batch, batch_idx): return self._common_step(batch, batch_idx, 'test')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6) 
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def on_train_start(self):
        if self.model.ensemble_type == 'boosting' and self.model.use_ml_models_for_boosting:
            pass


def ensemble_get_transforms(img_size=224): 
    return cnn_lstm_get_transforms(img_size)


def ensemble_load_data(image_dir: str, time_series_path: str, labels_path: str,
                       img_size: int = 224, batch_size: int = 32, num_workers: int = 4,
                       extract_ml_features_in_dataset: bool = False): 
    time_series_data = np.load(time_series_path)
    labels_data = np.load(labels_path)
    image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    assert len(image_paths) == len(labels_data) == len(time_series_data), "Data dimension mismatch"

    indices = np.arange(len(labels_data))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels_data)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=labels_data[temp_idx])
    
    transforms_dict = ensemble_get_transforms(img_size) 
    
    seq_len, time_features = time_series_data.shape[1], time_series_data.shape[2]

    train_dataset = Ensemble_CoralDataset(
        [image_paths[i] for i in train_idx], time_series_data[train_idx], labels_data[train_idx],
        transform=transforms_dict['train'], extract_ml_features_flag=extract_ml_features_in_dataset,
        seq_len=seq_len, time_input_dim=time_features
    )
    val_dataset = Ensemble_CoralDataset(
        [image_paths[i] for i in val_idx], time_series_data[val_idx], labels_data[val_idx],
        transform=transforms_dict['val'], extract_ml_features_flag=extract_ml_features_in_dataset,
        seq_len=seq_len, time_input_dim=time_features
    )
    test_dataset = Ensemble_CoralDataset(
        [image_paths[i] for i in test_idx], time_series_data[test_idx], labels_data[test_idx],
        transform=transforms_dict['test'], extract_ml_features_flag=extract_ml_features_in_dataset,
        seq_len=seq_len, time_input_dim=time_features
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        
    return {
        'train_loader': train_loader, 'val_loader': val_loader, 'test_loader': test_loader,
        'seq_len': seq_len, 'time_features': time_features,
        'train_ml_features_if_extracted': train_dataset.ml_features_extracted if extract_ml_features_in_dataset else None,
        'train_labels_for_ml': labels_data[train_idx] if extract_ml_features_in_dataset else None
    }

def ensemble_init_ml_boosting_models(ensemble_lightning_model: EnsembleLightningModel, 
                                     train_ml_features: np.ndarray, train_labels: np.ndarray):
    if ensemble_lightning_model.model.ensemble_type == 'boosting' and \
       ensemble_lightning_model.model.use_ml_models_for_boosting:
        print("Initializing ML models for ensemble boosting...")
        ensemble_lightning_model.model.init_ml_models(train_ml_features, train_labels)
        print("ML models initialized.")


def ensemble_visualize_model_predictions(lightning_model, dataloader, num_samples=5, save_dir=None):
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    device = next(lightning_model.parameters()).device
    lightning_model.eval()
    
    samples_data = []
    count = 0
    for batch in dataloader:
        if count >= num_samples: break
        images = batch['image'].to(device)
        time_series_data = batch['time_series'].to(device)
        labels_cpu = batch['label'].cpu().numpy()
        ml_features = batch.get('ml_features', None)
        if ml_features is not None: ml_features = ml_features.to(device)

        with torch.no_grad():
            logits = lightning_model(images, time_series_data, ml_features) 
        
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        
        images_cpu = images.cpu().numpy()
        time_series_cpu = time_series_data.cpu().numpy()

        for i in range(len(images_cpu)):
            if count >= num_samples: break
            samples_data.append({
                'image': images_cpu[i], 'time_series': time_series_cpu[i],
                'label': labels_cpu[i,0], 'pred': preds[i,0], 'prob': probs[i,0]
            })
            count += 1
            
    mean_norm = np.array([0.485, 0.456, 0.406])
    std_norm = np.array([0.229, 0.224, 0.225])

    for i, sample in enumerate(samples_data):
        img_norm, ts_s, lbl, prd, prb = sample['image'], sample['time_series'], sample['label'], sample['pred'], sample['prob']
        
        img_denorm = np.transpose(img_norm, (1,2,0)) 
        img_denorm = std_norm * img_denorm + mean_norm
        img_denorm = np.clip(img_denorm, 0, 1)
        
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1); plt.imshow(img_denorm)
        plt.title(f"Image (Label: {int(lbl)}, Pred: {int(prd)}, Prob: {prb:.2f})")
        
        plt.subplot(2,1,2)
        for j in range(min(3, ts_s.shape[1])): plt.plot(ts_s[:,j], label=f'TS Feat {j+1}')
        plt.title('Time Series'); plt.xlabel('Time Step'); plt.ylabel('Value'); plt.legend(); plt.grid(True)
        
        plt.tight_layout()
        if save_dir: plt.savefig(os.path.join(save_dir, f'ensemble_pred_sample_{i+1}.png')); plt.close()
        else: plt.show()

# --- Main execution block (for testing) ---
if __name__ == "__main__":
    # Example: Testing CNN-LSTM Model
    print("Testing CNN-LSTM Model...")
    cnn_lstm_model_params = {
        'input_channels': 3, 'img_size': 224, 'time_input_dim': 8, 'seq_len': 24,
        'cnn_backbone': 'resnet18', 'cnn_output_dim': 128, 'lstm_hidden_dim': 64,
        'lstm_layers': 1, 'lstm_output_dim': 64, 'wavelet_output_dim': 32, 'wavelet_level': 2,
        'fusion_output_dim': 64, 'dropout': 0.2
    }
    test_cnn_lstm_model = CoralLightningModel(model_params=cnn_lstm_model_params)
    test_images = torch.randn(2, 3, 224, 224)
    test_ts = torch.randn(2, 24, 8) 
    out_cnn_lstm, _ = test_cnn_lstm_model(test_images, test_ts) 
    print(f"CNN-LSTM output shape: {out_cnn_lstm.shape}")

    # Example: Testing TCN Model
    print("\nTesting TCN Model...")
    tcn_model_params = {
        'input_channels': 3, 'img_size': 224, 'time_input_dim': 8, 'sequence_length': 24,
        'img_feature_dim': 128, 'time_feature_dim': 64, 'hidden_dims': [32, 32], 
        'fusion_hidden_dim': 64, 'fusion_output_dim': 32, 'cnn_backbone': 'resnet18',
        'kernel_size': 3, 'dropout': 0.2, 'use_attention': False
    }
    test_tcn_model = TCNLightningModel(model_params=tcn_model_params)
    out_tcn = test_tcn_model(test_images, test_ts)
    print(f"TCN output shape: {out_tcn.shape if isinstance(out_tcn, torch.Tensor) else out_tcn[0].shape}")

    # Example: Testing ViT Model
    print("\nTesting ViT Model...")
    vit_model_config = {
        'img_size': 224, 'patch_size': 16, 'in_channels': 3, 'time_input_dim': 8, 'max_len': 24,
        'img_embed_dim': 192, 'time_embed_dim': 128, 'fusion_dim': 128, 
        'img_num_heads': 3, 'time_num_heads': 2, 'fusion_num_heads': 2,
        'img_ff_dim': 192*2, 'time_ff_dim': 128*2, 'img_num_layers': 3, 'time_num_layers': 2,
        'num_classes': 2, 'dropout': 0.1, 'fusion_type': 'concat' 
    }
    
    test_vit_model = CoralTransformerLightning(model_config=vit_model_config)
    class MockTrainer: max_epochs = 10
    test_vit_model.trainer = MockTrainer()
    test_vit_model.configure_optimizers() 

    fused_logits_vit, _, _, _, _, _, _ = test_vit_model(test_images, test_ts)
    print(f"ViT fused output shape: {fused_logits_vit.shape}")


    if xgb:
        print("\nTesting XGBoost Model...")
        images_np_dummy = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
        ts_np_dummy = np.random.randn(10, 24, 8) 
        labels_np_dummy = np.random.randint(0, 2, 10)       
        
        xgb_model_inst = XGBoostCoralModel(n_estimators=10, max_depth=3, 
                                           feature_extractor=XGB_FeatureExtractor(wavelet_level=2)) 
        try:
            xgb_model_inst.train(images_np_dummy, ts_np_dummy, labels_np_dummy, use_cv=False)
            print("XGBoost model trained.")
            probs, preds = xgb_model_inst.predict(images_np_dummy[:2], ts_np_dummy[:2])
            print(f"XGBoost prediction probabilities: {probs.shape}, predictions: {preds.shape}")
        except Exception as e:
            print(f"Error testing XGBoost: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nXGBoost not available, skipping XGBoost test.")
    
    print("\nTesting Ensemble Model...")
    
    ens_cnn_lstm_params = cnn_lstm_model_params.copy() 
    ens_tcn_params = tcn_model_params.copy()
    ens_vit_params_for_ensemble = vit_model_config.copy()
    # CRITICAL FIX for Ensemble compatibility with BCEWithLogitsLoss:
    # ViT (DualTransformerModel) must output a single logit if other models do.
    # Original ViT Lightning uses CrossEntropyLoss (num_classes=2 for binary).
    # If ensemble target is single logit for BCE, ViT config must be num_classes=1.
    ens_vit_params_for_ensemble['num_classes'] = 1 

    ensemble_model_params = {
        'cnn_lstm_params': ens_cnn_lstm_params,
        'tcn_params': ens_tcn_params,
        'transformer_params': ens_vit_params_for_ensemble, 
        'ensemble_type': 'weighted',
        'use_cnn_lstm': True, 'use_tcn': True, 'use_transformer': True,
        'use_ml_models_for_boosting': False,
        'dropout': 0.2, 'seq_len': 24, 'time_input_dim': 8
    }
    test_ensemble_model = EnsembleLightningModel(model_params=ensemble_model_params)
    test_ensemble_model.trainer = MockTrainer() 
    test_ensemble_model.configure_optimizers()

    out_ensemble = test_ensemble_model(test_images, test_ts)
    print(f"Ensemble output shape: {out_ensemble.shape}")

    print("\nAll basic model tests completed.")