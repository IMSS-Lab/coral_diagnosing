"""
XGBoost model for coral bleaching prediction.
Combines feature engineering with gradient boosting.

This model incorporates:
- Extraction of engineered features from images and time series
- Wavelet decomposition for time-frequency analysis
- Feature importance and interpretability capabilities
- Robust performance on limited data
- Early stopping and model optimization
"""

import os
import json  # Added missing import
import numpy as np
import pandas as pd
import polars as plrs
from typing import Tuple, Dict, List, Optional, Union, Any
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
# Import xgboost properly ensuring we get the DMatrix class
try:
    import xgboost as xgb
    from xgboost import DMatrix  # Explicitly import DMatrix
except ImportError:
    raise ImportError("XGBoost not properly installed. Try: pip install --upgrade xgboost")
import pywt
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.inspection import permutation_importance
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import shap


class FeatureExtractor:
    """
    Extract engineered features from both image and time series data.
    
    This class extracts meaningful features from raw data that can be
    used by the XGBoost model for prediction.
    """
    
    def __init__(
        self,
        pretrained_backbone: str = 'resnet18',
        wavelet_name: str = 'db4',
        wavelet_level: int = 3
    ):
        """
        Initialize feature extractor.
        
        Args:
            pretrained_backbone: Name of pre-trained CNN backbone for image features
            wavelet_name: Name of wavelet for time-frequency analysis
            wavelet_level: Decomposition level for wavelet transform
        """
        self.wavelet_name = wavelet_name
        self.wavelet_level = wavelet_level
        
        # Set device first before initializing CNN
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize CNN feature extractor
        self.initialize_image_extractor(pretrained_backbone)
        
        # Feature names
        self.temporal_feature_names = [
            'mean', 'std', 'min', 'max', 'median', 'q25', 'q75', 'iqr',
            'skew', 'kurtosis', 'trend', 'autocorr_lag1', 'autocorr_lag2',
            'entropy', 'energy', 'peak_frequency'
        ]
        
        self.wavelet_feature_names = [
            'wavelet_mean', 'wavelet_std', 'wavelet_energy', 'wavelet_entropy'
        ]
    
    def initialize_image_extractor(self, backbone_name: str):
        """
        Initialize CNN model for image feature extraction.
        
        Args:
            backbone_name: Name of pre-trained CNN backbone
        """
        # Load pre-trained model - fixed deprecated parameter usage
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
        
        # Remove the fully connected layer to get features
        if 'resnet' in backbone_name:
            self.cnn_model = torch.nn.Sequential(*(list(self.cnn_model.children())[:-1]))
        else:  # EfficientNet
            self.cnn_model.classifier = torch.nn.Identity()
        
        # Move to GPU if available
        self.cnn_model = self.cnn_model.to(self.device)
        self.cnn_model.eval()
    
    def extract_image_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images using pre-trained CNN.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            
        Returns:
            Image features of shape [num_samples, feature_dim]
        """
        # Convert to PyTorch tensor and move to device
        # Ensure images are in the correct format (N, C, H, W)
        if images.shape[-1] == 3:  # If in format [N, H, W, C]
            images = np.transpose(images, (0, 3, 1, 2))
        
        # Ensure images are float32 for PyTorch
        if images.dtype != np.float32:
            images = images.astype(np.float32)
            
        # Normalize if images are in range [0, 255]
        if images.max() > 1.0:
            images = images / 255.0
            
        images_tensor = torch.FloatTensor(images).to(self.device)
        
        # Apply normalization
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        images_tensor = normalize(images_tensor)
        
        # Extract features
        features = []
        with torch.no_grad():
            for i in range(0, len(images_tensor), 32):  # Process in batches to avoid OOM
                batch = images_tensor[i:i+32]
                batch_features = self.cnn_model(batch)
                if len(batch_features.shape) == 4:  # For ResNet
                    batch_features = batch_features.reshape(batch_features.size(0), -1)
                features.append(batch_features.cpu().numpy())
        
        # Concatenate all batches
        features = np.vstack(features)
        
        return features
    
    def extract_temporal_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from time series data.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Temporal features of shape [num_samples, num_features * num_temporal_features]
        """
        num_samples, time_steps, num_features = time_series.shape
        num_temporal_features = len(self.temporal_feature_names)
        
        # Initialize feature array
        features = np.zeros((num_samples, num_features * num_temporal_features))
        
        for i in range(num_samples):
            for j in range(num_features):
                # Extract time series
                ts = time_series[i, :, j]
                feature_idx = j * num_temporal_features
                
                # Basic statistics
                features[i, feature_idx] = np.mean(ts)
                features[i, feature_idx + 1] = np.std(ts)
                features[i, feature_idx + 2] = np.min(ts)
                features[i, feature_idx + 3] = np.max(ts)
                features[i, feature_idx + 4] = np.median(ts)
                features[i, feature_idx + 5] = np.percentile(ts, 25)
                features[i, feature_idx + 6] = np.percentile(ts, 75)
                features[i, feature_idx + 7] = np.percentile(ts, 75) - np.percentile(ts, 25)
                
                # Skewness and kurtosis
                if np.std(ts) > 0:
                    ts_norm = (ts - np.mean(ts)) / np.std(ts)
                    features[i, feature_idx + 8] = np.mean(ts_norm**3)  # Skewness
                    features[i, feature_idx + 9] = np.mean(ts_norm**4) - 3  # Kurtosis
                
                # Trend (linear regression slope)
                t = np.arange(time_steps)
                if np.std(ts) > 0:
                    features[i, feature_idx + 10] = np.polyfit(t, ts, 1)[0]
                
                # Autocorrelation
                if np.std(ts) > 0:
                    # Lag-1 autocorrelation
                    features[i, feature_idx + 11] = np.corrcoef(ts[:-1], ts[1:])[0, 1]
                    # Lag-2 autocorrelation
                    if time_steps > 2:
                        features[i, feature_idx + 12] = np.corrcoef(ts[:-2], ts[2:])[0, 1]
                
                # Entropy and energy
                if np.min(ts) >= 0 and np.sum(ts) > 0:
                    p = ts / np.sum(ts)
                    features[i, feature_idx + 13] = -np.sum(p * np.log2(p + 1e-10))  # Entropy
                
                features[i, feature_idx + 14] = np.sum(ts**2)  # Energy
                
                # Peak frequency (using FFT)
                if time_steps > 1:
                    fft = np.abs(np.fft.rfft(ts))
                    if len(fft) > 0:
                        features[i, feature_idx + 15] = np.argmax(fft)
        
        return features
    
    # Fix wavelet level calculation 
    def extract_wavelet_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract wavelet features from time series data.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Wavelet features of shape [num_samples, num_features * num_wavelet_features * (wavelet_level+1)]
        """
        num_samples, time_steps, num_features = time_series.shape
        
        # Determine appropriate wavelet level based on data length
        # For wavelet transform, max_level = floor(log2(signal_length))
        max_possible_level = int(np.log2(time_steps))
        # Subtract 1 to avoid boundary effects warning
        adjusted_wavelet_level = min(self.wavelet_level, max_possible_level - 1)
        # Ensure at least level 1
        adjusted_wavelet_level = max(1, adjusted_wavelet_level)
        
        print(f"Using wavelet level: {adjusted_wavelet_level} (original: {self.wavelet_level}, max possible: {max_possible_level})")
        
        num_wavelet_features = len(self.wavelet_feature_names)
        
        # Initialize feature array
        features = np.zeros((num_samples, num_features * num_wavelet_features * (adjusted_wavelet_level + 1)))
        
        for i in range(num_samples):
            feature_idx = 0
            for j in range(num_features):
                # Extract time series
                ts = time_series[i, :, j]
                
                # Perform wavelet decomposition
                coeffs = pywt.wavedec(ts, self.wavelet_name, level=adjusted_wavelet_level)
                
                # Extract features from each coefficient level
                for level, coeff in enumerate(coeffs):
                    # Mean
                    features[i, feature_idx] = np.mean(coeff)
                    feature_idx += 1
                    
                    # Standard deviation
                    features[i, feature_idx] = np.std(coeff)
                    feature_idx += 1
                    
                    # Energy
                    features[i, feature_idx] = np.sum(coeff**2)
                    feature_idx += 1
                    
                    # Entropy
                    if np.min(np.abs(coeff)) >= 0 and np.sum(np.abs(coeff)) > 0:
                        p = np.abs(coeff) / np.sum(np.abs(coeff))
                        features[i, feature_idx] = -np.sum(p * np.log2(p + 1e-10))
                    feature_idx += 1
        
        return features
    
    def extract_spatial_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract handcrafted spatial features from images.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            
        Returns:
            Spatial features of shape [num_samples, num_spatial_features]
        """
        num_samples = images.shape[0]
        
        # Define number of spatial features
        num_spatial_features = 12  # Adjust based on features extracted
        
        # Initialize feature array
        features = np.zeros((num_samples, num_spatial_features))
        
        for i in range(num_samples):
            # Get image
            img = images[i].copy()
            
            # Convert to appropriate format for OpenCV (uint8 if not already)
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # Convert to grayscale if needed
            if img.shape[-1] == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            else:
                gray = img
            
            # Color features (if image is color)
            if img.shape[-1] == 3:
                # Mean color values
                features[i, 0] = np.mean(img[:, :, 0])  # Red channel mean
                features[i, 1] = np.mean(img[:, :, 1])  # Green channel mean
                features[i, 2] = np.mean(img[:, :, 2])  # Blue channel mean
                
                # Color standard deviations
                features[i, 3] = np.std(img[:, :, 0])  # Red channel std
                features[i, 4] = np.std(img[:, :, 1])  # Green channel std
                features[i, 5] = np.std(img[:, :, 2])  # Blue channel std
                
                # Color ratios
                r_g_ratio = np.mean(img[:, :, 0]) / (np.mean(img[:, :, 1]) + 1e-10)
                r_b_ratio = np.mean(img[:, :, 0]) / (np.mean(img[:, :, 2]) + 1e-10)
                g_b_ratio = np.mean(img[:, :, 1]) / (np.mean(img[:, :, 2]) + 1e-10)
                
                features[i, 6] = r_g_ratio
                features[i, 7] = r_b_ratio
                features[i, 8] = g_b_ratio
            
            # Texture features from grayscale image
            # GLCM (Gray-Level Co-occurrence Matrix) features
            if gray.shape[0] > 1 and gray.shape[1] > 1:  # Check if image is valid
                # Normalize and convert to uint8
                gray_norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                # Calculate gradient (approximation of texture)
                sobelx = cv2.Sobel(gray_norm, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray_norm, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # Gradient statistics
                features[i, 9] = np.mean(gradient_magnitude)  # Mean gradient
                features[i, 10] = np.std(gradient_magnitude)  # Std gradient
                features[i, 11] = np.percentile(gradient_magnitude, 90)  # 90th percentile
        
        return features
    
    def extract_all_features(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features from images and time series data.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Tuple of (combined features, feature names)
        """
        # Extract features
        cnn_features = self.extract_image_features(images)
        temporal_features = self.extract_temporal_features(time_series)
        wavelet_features = self.extract_wavelet_features(time_series)
        spatial_features = self.extract_spatial_features(images)
        
        # Combine all features
        combined_features = np.hstack([
            cnn_features, 
            temporal_features, 
            wavelet_features, 
            spatial_features
        ])
        
        # Generate feature names
        cnn_feature_names = [f"cnn_{i}" for i in range(cnn_features.shape[1])]
        
        _, _, num_ts_features = time_series.shape
        temporal_feature_names = []
        for f in range(num_ts_features):
            feature_base = f"feature_{f}"
            temporal_feature_names.extend([f"{feature_base}_{name}" for name in self.temporal_feature_names])
        
        # Determine appropriate wavelet level based on data length
        max_possible_level = int(np.log2(time_series.shape[1]))
        adjusted_wavelet_level = min(self.wavelet_level, max_possible_level - 1)
        adjusted_wavelet_level = max(1, adjusted_wavelet_level)
        
        wavelet_feature_names = []
        for f in range(num_ts_features):
            feature_base = f"feature_{f}"
            for level in range(adjusted_wavelet_level + 1):  # +1 for approximation coefficients
                wavelet_feature_names.extend([f"{feature_base}_wavelet_level{level}_{name}" for name in self.wavelet_feature_names])
        
        spatial_feature_names = [
            "red_mean", "green_mean", "blue_mean",
            "red_std", "green_std", "blue_std",
            "r_g_ratio", "r_b_ratio", "g_b_ratio",
            "gradient_mean", "gradient_std", "gradient_p90"
        ]
        
        # Combine all feature names
        all_feature_names = cnn_feature_names + temporal_feature_names + wavelet_feature_names + spatial_feature_names
        
        return combined_features, all_feature_names


# Rest of the code remains the same...
class XGBoostCoralModel:
    """
    XGBoost model for coral bleaching prediction.
    
    Combines feature engineering with gradient boosting for
    accurate and interpretable predictions.
    """
    
    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        xgb_params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 10,
        feature_selection_threshold: float = 0.01,
        cv_folds: int = 5
    ):
        """
        Initialize XGBoost coral bleaching prediction model.
        
        Args:
            feature_extractor: Feature extractor instance
            xgb_params: XGBoost parameters
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for boosting
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            early_stopping_rounds: Number of rounds for early stopping
            feature_selection_threshold: Threshold for feature importance-based selection
            cv_folds: Number of cross-validation folds
        """
        # Initialize feature extractor if not provided
        if feature_extractor is None:
            self.feature_extractor = FeatureExtractor()
        else:
            self.feature_extractor = feature_extractor
        
        # Initialize XGBoost parameters
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'auc',
                'n_estimators': n_estimators,
                'learning_rate': learning_rate,
                'max_depth': max_depth,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'tree_method': 'hist',  # For faster training
                'use_label_encoder': False,  # Removed the deprecated parameter
                'verbosity': 1
            }
        else:
            self.xgb_params = xgb_params
        
        self.early_stopping_rounds = early_stopping_rounds
        self.feature_selection_threshold = feature_selection_threshold
        self.cv_folds = cv_folds
        
        # Model and feature information
        self.model = None
        self.feature_names = None
        self.selected_features = None
        self.feature_importances = None
        self.scaler = StandardScaler()
    
    def preprocess_features(
        self, 
        features: np.ndarray, 
        feature_names: List[str], 
        train: bool = True
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess features by scaling and selection.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            train: Whether to fit or transform using scaler
            
        Returns:
            Tuple of (preprocessed features, selected feature names)
        """
        # Scale features
        if train:
            scaled_features = self.scaler.fit_transform(features)
        else:
            scaled_features = self.scaler.transform(features)
        
        # Feature selection if model is already trained and in inference mode
        if not train and self.selected_features is not None:
            # Select only important features
            selected_indices = [i for i, name in enumerate(feature_names) if name in self.selected_features]
            return scaled_features[:, selected_indices], [feature_names[i] for i in selected_indices]
        
        return scaled_features, feature_names
    
    def train(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        labels: np.ndarray, 
        use_cv: bool = True
    ) -> Dict[str, float]:
        """
        Train the XGBoost model on extracted features.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            labels: Labels of shape [num_samples]
            use_cv: Whether to use cross-validation
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Extract features
        features, feature_names = self.feature_extractor.extract_all_features(images, time_series)
        self.feature_names = feature_names
        
        # Preprocess features
        X, _ = self.preprocess_features(features, feature_names)
        y = labels
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train with cross-validation
        if use_cv:
            return self._train_with_cv(X, y)
        
        # Train with validation set
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Train the model
        evals = [(dtrain, 'train'), (dval, 'val')]
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.xgb_params['n_estimators'],
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=100
        )
        
        # Feature selection based on importance
        self._select_important_features()
        
        # Evaluate on validation set
        y_pred = self.model.predict(dval)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred_binary),
            'precision': precision_score(y_val, y_pred_binary),
            'recall': recall_score(y_val, y_pred_binary),
            'f1': f1_score(y_val, y_pred_binary),
            'auc': roc_auc_score(y_val, y_pred)
        }
        
        return metrics
    
    def _train_with_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Train with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Initialize stratified k-fold
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        
        # Initialize arrays for predictions and metrics
        cv_scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'auc': []
        }
        
        # Perform cross-validation
        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create DMatrix
            dtrain = DMatrix(X_train, label=y_train)
            dval = DMatrix(X_val, label=y_val)
            
            # Train the model
            evals = [(dtrain, 'train'), (dval, 'val')]
            model = xgb.train(
                self.xgb_params,
                dtrain,
                num_boost_round=self.xgb_params['n_estimators'],
                evals=evals,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=False
            )
            
            # Evaluate on validation set
            y_pred = model.predict(dval)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            # Calculate metrics
            cv_scores['accuracy'].append(accuracy_score(y_val, y_pred_binary))
            cv_scores['precision'].append(precision_score(y_val, y_pred_binary))
            cv_scores['recall'].append(recall_score(y_val, y_pred_binary))
            cv_scores['f1'].append(f1_score(y_val, y_pred_binary))
            cv_scores['auc'].append(roc_auc_score(y_val, y_pred))
        
        # Train the final model on all data
        dtrain_all = DMatrix(X, label=y)
        self.model = xgb.train(
            self.xgb_params,
            dtrain_all,
            num_boost_round=self.xgb_params['n_estimators']
        )
        
        # Feature selection based on importance
        self._select_important_features()
        
        # Calculate mean and std of metrics
        metrics = {
            f"{metric}": np.mean(scores) for metric, scores in cv_scores.items()
        }
        metrics.update({
            f"{metric}_std": np.std(scores) for metric, scores in cv_scores.items()
        })
        
        return metrics
    
    def _select_important_features(self):
        """
        Select important features based on feature importance.
        """
        if self.model is None or self.feature_names is None:
            return
        
        # Get feature importance
        importance_scores = self.model.get_score(importance_type='gain')
        
        # Convert to array and normalize
        importance_array = np.zeros(len(self.feature_names))
        for feature, score in importance_scores.items():
            # Find feature index
            if feature in self.feature_names:
                idx = self.feature_names.index(feature)
                importance_array[idx] = score
        
        # Normalize importance
        if np.sum(importance_array) > 0:
            importance_array = importance_array / np.sum(importance_array)
        
        # Store feature importances
        self.feature_importances = {
            self.feature_names[i]: importance_array[i] for i in range(len(self.feature_names))
        }
        
        # Select features above threshold
        self.selected_features = [
            self.feature_names[i] for i in range(len(self.feature_names))
            if importance_array[i] > self.feature_selection_threshold
        ]
    
    def predict(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            images: Image data of shape [num_samples, height, width, channels]
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Tuple of (predicted probabilities, predicted labels)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features, feature_names = self.feature_extractor.extract_all_features(images, time_series)
        
        # Preprocess features
        X, _ = self.preprocess_features(features, feature_names, train=False)
        
        # Create DMatrix
        dtest = DMatrix(X)
        
        # Make predictions
        y_pred_proba = self.model.predict(dtest)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        return y_pred_proba, y_pred
    
    def calculate_permutation_importance(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        labels: np.ndarray, 
        n_repeats: int = 10
    ) -> Dict[str, np.ndarray]:
        """
        Calculate permutation importance of features.
        
        Args:
            images: Image data
            time_series: Time series data
            labels: Ground truth labels
            n_repeats: Number of repetitions
            
        Returns:
            Dictionary of permutation importance results
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features, feature_names = self.feature_extractor.extract_all_features(images, time_series)
        
        # Preprocess features
        X, selected_feature_names = self.preprocess_features(features, feature_names, train=False)
        
        # Define prediction function for sklearn's permutation_importance
        def predict_fn(X_subset):
            return self.model.predict(DMatrix(X_subset))
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            predict_fn, X, labels, 
            n_repeats=n_repeats, 
            random_state=42
        )
        
        # Create dictionary of results
        result = {
            'importances_mean': perm_importance.importances_mean,
            'importances_std': perm_importance.importances_std,
            'feature_names': selected_feature_names
        }
        
        return result
    
    def get_shap_values(
        self, 
        images: np.ndarray, 
        time_series: np.ndarray, 
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate SHAP values for feature interpretation.
        
        Args:
            images: Image data
            time_series: Time series data
            sample_size: Number of samples to use for SHAP calculation
            
        Returns:
            Dictionary with SHAP values and feature names
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features, feature_names = self.feature_extractor.extract_all_features(images, time_series)
        
        # Preprocess features
        X, selected_feature_names = self.preprocess_features(features, feature_names, train=False)
        
        # Subsample if necessary
        if sample_size < X.shape[0]:
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create DMatrix
        dmatrix = DMatrix(X_sample)
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_sample)
        
        result = {
            'shap_values': shap_values,
            'feature_names': selected_feature_names,
            'data': X_sample
        }
        
        return result
    
    def detect_early_warning_signals(
        self, 
        time_series: np.ndarray, 
        window_size: int = 5
    ) -> np.ndarray:
        """
        Detect early warning signals of coral bleaching in time series data.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            window_size: Size of rolling window for calculating indicators
            
        Returns:
            Early warning signals of shape [num_samples, time_steps]
        """
        # Extract temporal features for each time window
        num_samples, time_steps, num_features = time_series.shape
        ews = np.zeros((num_samples, time_steps))
        
        for i in range(num_samples):
            for t in range(window_size, time_steps):
                # Extract window
                window_data = np.expand_dims(time_series[i, t-window_size:t, :], axis=0)
                
                # Extract features from window
                window_features, _ = self.feature_extractor.extract_temporal_features(window_data)
                window_features = np.hstack([
                    window_features,
                    self.feature_extractor.extract_wavelet_features(window_data)
                ])
                
                # Preprocess features
                X, _ = self.preprocess_features(window_features, self.feature_names, train=False)
                
                # Make prediction on window
                dmatrix = DMatrix(X)
                pred = self.model.predict(dmatrix)[0]
                
                # Store prediction as early warning signal
                ews[i, t] = pred
        
        return ews
    
    def save_model(self, model_path: str):
        """
        Save the model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        # Save XGBoost model
        self.model.save_model(model_path)
        
        # Save additional information
        metadata = {
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances,
            'xgb_params': self.xgb_params,
            'scaler_mean': self.scaler.mean_.tolist() if hasattr(self.scaler, 'mean_') else None,
            'scaler_scale': self.scaler.scale_.tolist() if hasattr(self.scaler, 'scale_') else None
        }
        
        # Save metadata
        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    def load_model(self, model_path: str):
        """
        Load the model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        # Load XGBoost model
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load additional information
        metadata_path = os.path.splitext(model_path)[0] + "_metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        self.feature_names = metadata['feature_names']
        self.selected_features = metadata['selected_features']
        self.feature_importances = metadata['feature_importances']
        self.xgb_params = metadata['xgb_params']
        
        # Restore scaler
        if metadata['scaler_mean'] and metadata['scaler_scale']:
            self.scaler.mean_ = np.array(metadata['scaler_mean'])
            self.scaler.scale_ = np.array(metadata['scaler_scale'])
            self.scaler.n_features_in_ = len(self.scaler.mean_)


class XGBoostLightningModel(pl.LightningModule):
    """PyTorch Lightning wrapper for XGBoost coral bleaching model."""
    
    def __init__(
        self,
        xgb_params: Optional[Dict[str, Any]] = None,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 5,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        feature_selection_threshold: float = 0.01
    ):
        """
        Initialize Lightning module.
        
        Args:
            xgb_params: XGBoost parameters
            n_estimators: Number of boosting rounds
            learning_rate: Learning rate for boosting
            max_depth: Maximum tree depth
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of columns
            feature_selection_threshold: Threshold for feature importance-based selection
        """
        super(XGBoostLightningModel, self).__init__()
        self.save_hyperparameters()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor()
        
        # Initialize XGBoost model
        self.model = XGBoostCoralModel(
            feature_extractor=self.feature_extractor,
            xgb_params=xgb_params,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            feature_selection_threshold=feature_selection_threshold
        )
    
    def setup(self, stage=None):
        """Setup stage."""
        pass
    
    def train_dataloader(self):
        """Train dataloader."""
        pass
    
    def val_dataloader(self):
        """Validation dataloader."""
        pass
    
    def fit(self, images, time_series, labels, use_cv=True):
        """
        Train the model on the provided data.
        
        Args:
            images: Image data
            time_series: Time series data
            labels: Ground truth labels
            use_cv: Whether to use cross-validation
            
        Returns:
            Dictionary of metrics
        """
        metrics = self.model.train(images, time_series, labels, use_cv=use_cv)
        return metrics
    
    def predict(self, images, time_series):
        """
        Make predictions on the provided data.
        
        Args:
            images: Image data
            time_series: Time series data
            
        Returns:
            Tuple of (predicted probabilities, predicted labels)
        """
        return self.model.predict(images, time_series)
    
    def get_feature_importance(self):
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary of feature importances
        """
        if self.model.feature_importances is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.feature_importances
    
    def get_shap_values(self, images, time_series, sample_size=100):
        """
        Calculate SHAP values for feature interpretation.
        
        Args:
            images: Image data
            time_series: Time series data
            sample_size: Number of samples to use for SHAP calculation
            
        Returns:
            Dictionary with SHAP values and feature names
        """
        return self.model.get_shap_values(images, time_series, sample_size)
    
    def detect_early_warning_signals(self, time_series, window_size=5):
        """
        Detect early warning signals of coral bleaching in time series data.
        
        Args:
            time_series: Time series data
            window_size: Size of rolling window for calculating indicators
            
        Returns:
            Early warning signals
        """
        return self.model.detect_early_warning_signals(time_series, window_size)
    
    def save_model(self, model_path):
        """Save the model to disk."""
        self.model.save_model(model_path)
    
    def load_model(self, model_path):
        """Load the model from disk."""
        self.model.load_model(model_path)


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
    # Update the example code to handle XGBoost properly
    # Example usage with proper data type handling
    import matplotlib.pyplot as plt
    
    print("Starting coral bleaching prediction model test")
    
    # Create sample data with proper types for the image data
    # Images should be uint8 for OpenCV operations
    print("Generating sample data...")
    images = np.random.randint(0, 256, (100, 224, 224, 3), dtype=np.uint8)
    time_series = np.random.randn(100, 32, 8)   # Using 32 timesteps to better handle wavelet decomposition
    labels = np.random.randint(0, 2, 100)       # Binary labels
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    feature_extractor = FeatureExtractor(wavelet_level=2)  # Lower wavelet level to avoid warnings
    
    # Extract features
    print("Extracting features...")
    features, feature_names = feature_extractor.extract_all_features(images, time_series)
    print(f"Extracted {features.shape[1]} features")
    
    # Create and train XGBoost model
    print("Training XGBoost model...")
    model = XGBoostCoralModel(feature_extractor=feature_extractor)
    
    # Use a smaller subset for faster testing if needed
    test_size = 50
    metrics = model.train(images[:test_size], time_series[:test_size], labels[:test_size], use_cv=False)  # Set use_cv=False for faster testing
    
    print("Training metrics:", metrics)
    
    # Make predictions
    print("Making predictions...")
    y_pred_proba, y_pred = model.predict(images[:5], time_series[:5])
    print("Predictions:", y_pred)
    
    # Get feature importances
    print("Getting feature importances...")
    importances = model.feature_importances
    
    # Plot feature importances
    if importances:
        print("Plotting feature importances...")
        # Sort importances
        importance_items = list(importances.items())
        sorted_idx = np.argsort([imp for _, imp in importance_items])
        top_n = 10  # Show only top 10 features for clarity
        sorted_features = [importance_items[i][0] for i in sorted_idx[-top_n:]]
        sorted_importances = [importance_items[i][1] for i in sorted_idx[-top_n:]]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(sorted_features)), sorted_importances)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.savefig('feature_importances.png')  # Save to file instead of showing
        print("Feature importance plot saved to 'feature_importances.png'")
    
    print("Testing completed successfully")