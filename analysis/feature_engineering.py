"""
Feature engineering module for coral bleaching prediction.

This module handles feature extraction and selection for the Duke Coral Health project.
It includes functions for:
- Extracting wavelet features from time series data
- Extracting image features using pre-trained networks
- Feature selection based on importance and correlation
- Dimensionality reduction
- Feature visualization and analysis
"""

import os
import numpy as np
import pandas as pd
import polars as plrs
import torch
import torch.nn as nn
from typing import Tuple, Dict, List, Optional, Union, Any
import pywt
import cv2
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torchvision.models as models
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import entropy
from scipy import signal


class WaveletFeatureExtractor:
    """
    Extract wavelet-based features from time series data.
    
    Performs wavelet decomposition to capture time-frequency characteristics.
    """
    
    def __init__(
        self, 
        wavelet: str = 'db4', 
        level: int = 3, 
        features: List[str] = ['mean', 'std', 'energy', 'entropy']
    ):
        """
        Initialize wavelet feature extractor.
        
        Args:
            wavelet: Wavelet type to use (e.g., 'db4', 'sym5', 'coif3')
            level: Decomposition level
            features: List of features to extract from wavelet coefficients
        """
        self.wavelet = wavelet
        self.level = level
        self.features = features
    
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
                
                # Extract features from each coefficient level
                for coeff in coeffs:
                    # Calculate statistical features
                    feature_values = []
                    
                    if 'mean' in self.features:
                        feature_values.append(np.mean(coeff))
                    
                    if 'std' in self.features:
                        feature_values.append(np.std(coeff))
                    
                    if 'min' in self.features:
                        feature_values.append(np.min(coeff))
                    
                    if 'max' in self.features:
                        feature_values.append(np.max(coeff))
                    
                    if 'median' in self.features:
                        feature_values.append(np.median(coeff))
                    
                    if 'iqr' in self.features:
                        feature_values.append(np.percentile(coeff, 75) - np.percentile(coeff, 25))
                    
                    if 'energy' in self.features:
                        feature_values.append(np.sum(coeff**2))
                    
                    if 'entropy' in self.features:
                        # Calculate wavelet entropy (spectral entropy)
                        abs_coeff = np.abs(coeff) + 1e-10  # Add small constant to avoid log(0)
                        norm_coeff = abs_coeff / np.sum(abs_coeff)
                        feature_values.append(-np.sum(norm_coeff * np.log2(norm_coeff)))
                    
                    if 'kurtosis' in self.features:
                        # Kurtosis measures "peakedness" of distribution
                        m4 = np.mean((coeff - np.mean(coeff))**4)
                        m2 = np.mean((coeff - np.mean(coeff))**2)
                        if m2 != 0:
                            kurt = m4 / (m2**2) - 3  # Excess kurtosis (normal = 0)
                        else:
                            kurt = 0
                        feature_values.append(kurt)
                    
                    if 'skewness' in self.features:
                        # Skewness measures asymmetry of distribution
                        m3 = np.mean((coeff - np.mean(coeff))**3)
                        m2 = np.mean((coeff - np.mean(coeff))**2)
                        if m2 != 0:
                            skew = m3 / (m2**1.5)
                        else:
                            skew = 0
                        feature_values.append(skew)
                    
                    # Add features to sample features
                    sample_features.extend(feature_values)
            
            all_features.append(sample_features)
        
        return np.array(all_features)
    
    def visualize_decomposition(self, time_series: np.ndarray, sample_idx: int = 0, feature_idx: int = 0) -> None:
        """
        Visualize wavelet decomposition for a single time series.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            sample_idx: Index of sample to visualize
            feature_idx: Index of feature to visualize
        """
        # Extract time series
        ts = time_series[sample_idx, :, feature_idx]
        
        # Perform wavelet decomposition
        coeffs = pywt.wavedec(ts, self.wavelet, level=self.level)
        
        # Plot original time series and wavelet coefficients
        fig, axes = plt.subplots(len(coeffs) + 1, 1, figsize=(12, 8), sharex=False)
        
        # Plot original signal
        axes[0].plot(ts)
        axes[0].set_title('Original Signal')
        axes[0].set_xlim(0, len(ts))
        
        # Plot wavelet coefficients
        for i, coeff in enumerate(coeffs):
            axes[i+1].plot(coeff)
            if i == 0:
                axes[i+1].set_title(f'Approximation Coefficients (Level {self.level})')
            else:
                axes[i+1].set_title(f'Detail Coefficients (Level {self.level-i+1})')
            axes[i+1].set_xlim(0, len(coeff))
        
        plt.tight_layout()
        plt.show()


class TimeSeriesFeatureExtractor:
    """
    Extract statistical and spectral features from time series data.
    
    Calculates various statistical measures, frequency domain features, and temporal patterns.
    """
    
    def __init__(
        self, 
        features: List[str] = [
            'mean', 'std', 'min', 'max', 'median', 'iqr',
            'skewness', 'kurtosis', 'trend', 'seasonality', 
            'spectral_power', 'spectral_entropy', 'peak_frequency',
            'autocorr_lag1', 'autocorr_lag2', 'autocorr_lag3'
        ]
    ):
        """
        Initialize time series feature extractor.
        
        Args:
            features: List of features to extract
        """
        self.features = features
    
    def extract_features(self, time_series: np.ndarray) -> np.ndarray:
        """
        Extract features from time series data.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            
        Returns:
            Extracted features of shape [num_samples, feature_dim]
        """
        num_samples, time_steps, num_features = time_series.shape
        
        # Calculate number of features per time series
        num_ts_features = 0
        for feature in self.features:
            if feature in ['spectral_power', 'spectral_centroid']:
                # These return multiple values (one per frequency band)
                num_ts_features += 5  # Using 5 frequency bands
            else:
                num_ts_features += 1
        
        # Initialize features array
        feature_array = np.zeros((num_samples, num_features * num_ts_features))
        
        for i in range(num_samples):
            feature_idx = 0
            
            for f in range(num_features):
                # Get time series for this feature
                ts = time_series[i, :, f]
                
                # Extract basic statistical features
                for feature in self.features:
                    if feature == 'mean':
                        feature_array[i, f * num_ts_features + feature_idx] = np.mean(ts)
                        feature_idx += 1
                    
                    elif feature == 'std':
                        feature_array[i, f * num_ts_features + feature_idx] = np.std(ts)
                        feature_idx += 1
                    
                    elif feature == 'min':
                        feature_array[i, f * num_ts_features + feature_idx] = np.min(ts)
                        feature_idx += 1
                    
                    elif feature == 'max':
                        feature_array[i, f * num_ts_features + feature_idx] = np.max(ts)
                        feature_idx += 1
                    
                    elif feature == 'median':
                        feature_array[i, f * num_ts_features + feature_idx] = np.median(ts)
                        feature_idx += 1
                    
                    elif feature == 'iqr':
                        feature_array[i, f * num_ts_features + feature_idx] = np.percentile(ts, 75) - np.percentile(ts, 25)
                        feature_idx += 1
                    
                    elif feature == 'skewness':
                        # Calculate skewness
                        m3 = np.mean((ts - np.mean(ts))**3)
                        m2 = np.mean((ts - np.mean(ts))**2)
                        if m2 != 0:
                            skew = m3 / (m2**1.5)
                        else:
                            skew = 0
                        feature_array[i, f * num_ts_features + feature_idx] = skew
                        feature_idx += 1
                    
                    elif feature == 'kurtosis':
                        # Calculate kurtosis
                        m4 = np.mean((ts - np.mean(ts))**4)
                        m2 = np.mean((ts - np.mean(ts))**2)
                        if m2 != 0:
                            kurt = m4 / (m2**2) - 3  # Excess kurtosis
                        else:
                            kurt = 0
                        feature_array[i, f * num_ts_features + feature_idx] = kurt
                        feature_idx += 1
                    
                    elif feature == 'trend':
                        # Calculate linear trend
                        x = np.arange(len(ts))
                        # Use polynomial fit to get trend
                        if len(ts) > 1:
                            trend = np.polyfit(x, ts, 1)[0]  # Slope of the trend line
                        else:
                            trend = 0
                        feature_array[i, f * num_ts_features + feature_idx] = trend
                        feature_idx += 1
                    
                    elif feature == 'seasonality':
                        # Calculate seasonality using autocorrelation
                        if len(ts) > 1:
                            # Detrend the time series
                            detrended = signal.detrend(ts)
                            # Calculate autocorrelation
                            acf = np.correlate(detrended, detrended, mode='full')
                            acf = acf[len(acf)//2:]  # Keep only the positive lags
                            # Normalize
                            acf /= acf[0]
                            # Find first peak after lag 0
                            peaks, _ = signal.find_peaks(acf)
                            if len(peaks) > 0:
                                seasonality = peaks[0]  # Lag of first peak
                            else:
                                seasonality = 0
                        else:
                            seasonality = 0
                        feature_array[i, f * num_ts_features + feature_idx] = seasonality
                        feature_idx += 1
                    
                    elif feature == 'spectral_power':
                        # Calculate power in different frequency bands
                        if len(ts) > 1:
                            # Compute power spectral density
                            freqs, psd = signal.welch(ts, fs=1.0, nperseg=min(len(ts), 16))
                            # Define frequency bands
                            bands = [
                                (0, 0.1),     # Very low frequency
                                (0.1, 0.2),   # Low frequency
                                (0.2, 0.3),   # Medium-low frequency
                                (0.3, 0.4),   # Medium-high frequency
                                (0.4, 0.5)     # High frequency
                            ]
                            # Calculate power in each band
                            for band in bands:
                                band_power = np.sum(psd[(freqs >= band[0]) & (freqs < band[1])])
                                feature_array[i, f * num_ts_features + feature_idx] = band_power
                                feature_idx += 1
                        else:
                            for _ in range(5):  # 5 frequency bands
                                feature_array[i, f * num_ts_features + feature_idx] = 0
                                feature_idx += 1
                    
                    elif feature == 'spectral_entropy':
                        # Calculate spectral entropy
                        if len(ts) > 1:
                            # Compute power spectral density
                            _, psd = signal.welch(ts, fs=1.0, nperseg=min(len(ts), 16))
                            # Normalize PSD
                            psd_norm = psd / np.sum(psd)
                            # Calculate entropy
                            spec_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
                        else:
                            spec_entropy = 0
                        feature_array[i, f * num_ts_features + feature_idx] = spec_entropy
                        feature_idx += 1
                    
                    elif feature == 'peak_frequency':
                        # Find dominant frequency
                        if len(ts) > 1:
                            # Compute power spectral density
                            freqs, psd = signal.welch(ts, fs=1.0, nperseg=min(len(ts), 16))
                            # Find peak frequency
                            peak_freq = freqs[np.argmax(psd)]
                        else:
                            peak_freq = 0
                        feature_array[i, f * num_ts_features + feature_idx] = peak_freq
                        feature_idx += 1
                    
                    elif feature.startswith('autocorr_lag'):
                        # Calculate autocorrelation at specific lag
                        lag = int(feature.split('lag')[1])
                        if len(ts) > lag:
                            # Calculate autocorrelation
                            autocorr = np.corrcoef(ts[:-lag], ts[lag:])[0, 1]
                        else:
                            autocorr = 0
                        feature_array[i, f * num_ts_features + feature_idx] = autocorr
                        feature_idx += 1
        
        return feature_array
    
    def visualize_features(
        self, 
        time_series: np.ndarray, 
        labels: np.ndarray,
        sample_indices: Optional[List[int]] = None,
        feature_idx: int = 0
    ) -> None:
        """
        Visualize time series features for selected samples.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            labels: Labels of shape [num_samples]
            sample_indices: Indices of samples to visualize (default: select a few from each class)
            feature_idx: Index of environmental feature to visualize
        """
        # If no sample indices provided, select a few from each class
        if sample_indices is None:
            class_0_indices = np.where(labels == 0)[0]
            class_1_indices = np.where(labels == 1)[0]
            
            # Select up to 3 samples from each class
            sample_indices = []
            if len(class_0_indices) > 0:
                sample_indices.extend(class_0_indices[:min(3, len(class_0_indices))])
            if len(class_1_indices) > 0:
                sample_indices.extend(class_1_indices[:min(3, len(class_1_indices))])
        
        # Plot time series
        plt.figure(figsize=(12, 6))
        
        for idx in sample_indices:
            ts = time_series[idx, :, feature_idx]
            label = 'Healthy' if labels[idx] == 0 else 'Bleached'
            plt.plot(ts, label=f"Sample {idx} ({label})")
        
        plt.title(f"Time Series Feature {feature_idx}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # Plot spectrum
        plt.figure(figsize=(12, 6))
        
        for idx in sample_indices:
            ts = time_series[idx, :, feature_idx]
            label = 'Healthy' if labels[idx] == 0 else 'Bleached'
            
            if len(ts) > 1:
                freqs, psd = signal.welch(ts, fs=1.0, nperseg=min(len(ts), 16))
                plt.semilogy(freqs, psd, label=f"Sample {idx} ({label})")
        
        plt.title(f"Power Spectrum of Feature {feature_idx}")
        plt.xlabel("Frequency")
        plt.ylabel("Power Spectral Density")
        plt.legend()
        plt.grid(True)
        plt.show()


class ImageFeatureExtractor:
    """
    Extract features from images using pre-trained networks.
    
    Uses CNN backbones like ResNet, EfficientNet, etc. for feature extraction.
    """
    
    def __init__(
        self,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        output_dim: int = 1280,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize image feature extractor.
        
        Args:
            backbone: CNN backbone to use
            pretrained: Whether to use pretrained weights
            output_dim: Dimension of output features
            device: Device to use for computation
        """
        self.device = device
        self.output_dim = output_dim
        
        # Initialize CNN backbone
        if backbone == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            self.model.fc = nn.Identity()  # Remove classification layer
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif backbone == 'resnet50':
            self.model = models.resnet50(pretrained=pretrained)
            self.model.fc = nn.Identity()  # Remove classification layer
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif backbone == 'efficientnet_b0':
            self.model = models.efficientnet_b0(pretrained=pretrained)
            self.model.classifier = nn.Identity()  # Remove classification layer
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif backbone == 'densenet121':
            self.model = models.densenet121(pretrained=pretrained)
            self.model.classifier = nn.Identity()  # Remove classification layer
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        elif backbone == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            self.model.classifier = nn.Sequential(*list(self.model.classifier.children())[:-1])  # Remove last layer
            self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Move model to device
        self.model = self.model.to(device)
        self.model.eval()
    
    def extract_features(self, images: np.ndarray) -> np.ndarray:
        """
        Extract features from images.
        
        Args:
            images: Images of shape [num_samples, height, width, channels] or [num_samples, channels, height, width]
            
        Returns:
            Extracted features of shape [num_samples, output_dim]
        """
        # Ensure images are in the correct format [num_samples, channels, height, width]
        if images.shape[1] != 3 and images.shape[3] == 3:
            # Convert [num_samples, height, width, channels] to [num_samples, channels, height, width]
            images = np.transpose(images, (0, 3, 1, 2))
        
        # Convert to torch tensor
        images_tensor = torch.tensor(images, dtype=torch.float32).to(self.device)
        
        # Apply preprocessing
        images_tensor = self.preprocess(images_tensor)
        
        # Extract features in batches to avoid OOM errors
        batch_size = 32
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(0, len(images_tensor), batch_size), desc="Extracting image features"):
                batch = images_tensor[i:i+batch_size]
                batch_features = self.model(batch)
                features.append(batch_features.cpu().numpy())
        
        # Concatenate features
        features = np.concatenate(features)
        
        return features


class FeatureSelector:
    """
    Feature selection using various statistical methods.
    
    Selects most informative features based on various criteria.
    """
    
    def __init__(self, method: str = 'mutual_info', k: int = 10):
        """
        Initialize feature selector.
        
        Args:
            method: Feature selection method ('mutual_info', 'f_classif', or 'correlation')
            k: Number of features to select
        """
        self.method = method
        self.k = k
        self.selected_indices = None
        self.selector = None
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit feature selector and select best features.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            labels: Labels of shape [num_samples]
            
        Returns:
            Feature importance scores of shape [num_features]
        """
        if self.method == 'mutual_info':
            self.selector = SelectKBest(mutual_info_classif, k=self.k)
        elif self.method == 'f_classif':
            self.selector = SelectKBest(f_classif, k=self.k)
        elif self.method == 'correlation':
            # Correlation-based feature selection
            self.selected_indices = self._correlation_selector(features, labels, k=self.k)
            return self._get_importance_scores(features)
        else:
            raise ValueError(f"Unsupported feature selection method: {self.method}")
        
        self.selector.fit(features, labels)
        self.selected_indices = np.where(self.selector.get_support())[0]
        
        return self.selector.scores_
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform features by selecting only the best features.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            
        Returns:
            Selected features of shape [num_samples, k]
        """
        if self.selected_indices is None:
            raise ValueError("Feature selector not fitted yet!")
        
        if self.method in ['mutual_info', 'f_classif']:
            return self.selector.transform(features)
        else:
            return features[:, self.selected_indices]
    
    def fit_transform(self, features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Fit feature selector and transform features.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            labels: Labels of shape [num_samples]
            
        Returns:
            Selected features of shape [num_samples, k]
        """
        self.fit(features, labels)
        return self.transform(features)
    
    def _correlation_selector(self, features: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
        """
        Select features based on correlation with target.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            labels: Labels of shape [num_samples]
            k: Number of features to select
            
        Returns:
            Indices of selected features
        """
        # Calculate correlation between each feature and the target
        correlations = np.zeros(features.shape[1])
        
        for i in range(features.shape[1]):
            correlations[i] = np.abs(np.corrcoef(features[:, i], labels)[0, 1])
        
        # Select top k features with highest correlation
        selected_indices = np.argsort(correlations)[-k:]
        
        return selected_indices
    
    def _get_importance_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Get feature importance scores.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            
        Returns:
            Feature importance scores of shape [num_features]
        """
        if self.method == 'correlation':
            # Calculate correlation between each feature and the target
            correlations = np.zeros(features.shape[1])
            
            for i in range(features.shape[1]):
                correlations[i] = np.abs(np.corrcoef(features[:, i], self.labels)[0, 1])
            
            return correlations
        else:
            return self.selector.scores_


class DimensionalityReducer:
    """
    Dimensionality reduction for feature visualization and analysis.
    
    Reduces high-dimensional feature vectors to 2D for visualization.
    """
    
    def __init__(self, method: str = 'pca', n_components: int = 2):
        """
        Initialize dimensionality reducer.
        
        Args:
            method: Dimensionality reduction method ('pca', 'tsne', or 'umap')
            n_components: Number of components to reduce to
        """
        self.method = method
        self.n_components = n_components
        self.reducer = None
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """
        Fit the reducer and transform the features.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            
        Returns:
            Reduced features of shape [num_samples, n_components]
        """
        if self.method == 'pca':
            self.reducer = PCA(n_components=self.n_components)
            return self.reducer.fit_transform(features)
        elif self.method == 'tsne':
            self.reducer = TSNE(n_components=self.n_components, random_state=42)
            return self.reducer.fit_transform(features)
        elif self.method == 'umap':
            try:
                import umap
                self.reducer = umap.UMAP(n_components=self.n_components, random_state=42)
                return self.reducer.fit_transform(features)
            except ImportError:
                print("UMAP not available. Install with 'pip install umap-learn'.")
                print("Falling back to PCA.")
                self.method = 'pca'
                self.reducer = PCA(n_components=self.n_components)
                return self.reducer.fit_transform(features)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {self.method}")
    
    def visualize(self, features: np.ndarray, labels: np.ndarray, title: str = "Feature Visualization") -> None:
        """
        Visualize features in 2D.
        
        Args:
            features: Feature matrix of shape [num_samples, num_features]
            labels: Labels of shape [num_samples]
            title: Plot title
        """
        # Reduce dimensionality to 2D
        reduced_features = self.fit_transform(features)
        
        # Create scatter plot
        plt.figure(figsize=(10, 8))
        
        # Plot each class with a different color
        for label in np.unique(labels):
            mask = labels == label
            class_name = 'Healthy' if label == 0 else 'Bleached'
            plt.scatter(
                reduced_features[mask, 0],
                reduced_features[mask, 1],
                label=class_name,
                alpha=0.7
            )
        
        plt.title(title)
        plt.xlabel(f"{self.method.upper()} Component 1")
        plt.ylabel(f"{self.method.upper()} Component 2")
        plt.legend()
        plt.grid(True)
        plt.show()


def extract_all_features(
    images: np.ndarray,
    timeseries: np.ndarray
) -> Tuple[np.ndarray, Dict[str, List[str]]]:
    """
    Extract all features from image and time series data.
    
    Args:
        images: Images of shape [num_samples, height, width, channels]
        timeseries: Time series of shape [num_samples, time_steps, num_features]
        
    Returns:
        Tuple of (extracted features, feature names dictionary)
    """
    print("Extracting features...")
    
    # Initialize feature extractors
    image_extractor = ImageFeatureExtractor(backbone='efficientnet_b0')
    ts_extractor = TimeSeriesFeatureExtractor()
    wavelet_extractor = WaveletFeatureExtractor()
    
    # Extract features
    image_features = image_extractor.extract_features(images)
    ts_features = ts_extractor.extract_features(timeseries)
    wavelet_features = wavelet_extractor.extract_features(timeseries)
    
    # Combine features
    combined_features = np.hstack([image_features, ts_features, wavelet_features])
    
    # Create feature names
    feature_names = {
        'image': [f"img_{i}" for i in range(image_features.shape[1])],
        'timeseries': [f"ts_{i}" for i in range(ts_features.shape[1])],
        'wavelet': [f"wav_{i}" for i in range(wavelet_features.shape[1])]
    }
    
    print(f"Extracted features: {combined_features.shape[1]} total features")
    print(f"  - Image features: {image_features.shape[1]}")
    print(f"  - Time series features: {ts_features.shape[1]}")
    print(f"  - Wavelet features: {wavelet_features.shape[1]}")
    
    return combined_features, feature_names


def select_best_features(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Dict[str, List[str]],
    k: int = 100,
    method: str = 'mutual_info'
) -> Tuple[np.ndarray, List[str]]:
    """
    Select the best features for classification.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        labels: Labels of shape [num_samples]
        feature_names: Dictionary of feature names
        k: Number of features to select
        method: Feature selection method
        
    Returns:
        Tuple of (selected features, selected feature names)
    """
    print(f"Selecting {k} best features using {method}...")
    
    # Initialize feature selector
    selector = FeatureSelector(method=method, k=k)
    
    # Fit and transform
    selected_features = selector.fit_transform(features, labels)
    
    # Get selected feature names
    all_feature_names = []
    for category_names in feature_names.values():
        all_feature_names.extend(category_names)
    
    selected_indices = selector.selected_indices
    selected_names = [all_feature_names[i] for i in selected_indices]
    
    print(f"Selected {len(selected_names)} features.")
    
    return selected_features, selected_names


def visualize_feature_importance(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: Dict[str, List[str]],
    method: str = 'mutual_info',
    top_n: int = 20
) -> List[Tuple[str, float]]:
    """
    Visualize feature importance.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        labels: Labels of shape [num_samples]
        feature_names: Dictionary of feature names
        method: Feature selection method
        top_n: Number of top features to display
        
    Returns:
        List of (feature name, importance) tuples
    """
    print(f"Calculating feature importance using {method}...")
    
    # Initialize feature selector
    selector = FeatureSelector(method=method, k=features.shape[1])  # Select all features
    
    # Fit and get importance scores
    importance_scores = selector.fit(features, labels)
    
    # Get all feature names
    all_feature_names = []
    for category_names in feature_names.values():
        all_feature_names.extend(category_names)
    
    # Sort features by importance
    feature_importance = list(zip(all_feature_names, importance_scores))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N features
    top_features = feature_importance[:top_n]
    
    # Visualize
    plt.figure(figsize=(12, 8))
    plt.barh([f[0] for f in top_features][::-1], [f[1] for f in top_features][::-1])
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Feature Importance')
    plt.tight_layout()
    plt.show()
    
    return feature_importance


def save_features(
    features: np.ndarray,
    feature_names: List[str],
    labels: np.ndarray,
    save_path: str
) -> None:
    """
    Save extracted features to disk.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        feature_names: List of feature names
        labels: Labels of shape [num_samples]
        save_path: Path to save features
    """
    print(f"Saving features to {save_path}...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Create dataframe
    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df['label'] = labels
    
    # Save to CSV
    feature_df.to_csv(save_path, index=False)
    print(f"Saved features to {save_path}")


if __name__ == "__main__":
    # Example usage
    import preprocessing
    
    # Load data
    data_dir = "./data/processed"
    train_data, val_data, test_data, _ = preprocessing.load_processed_data(data_dir)
    
    # Extract features
    train_features, feature_names = extract_all_features(
        train_data['images'],
        train_data['timeseries']
    )
    
    # Select best features
    selected_features, selected_names = select_best_features(
        train_features,
        train_data['labels'],
        feature_names,
        k=100
    )
    
    # Visualize feature importance
    feature_importance = visualize_feature_importance(
        train_features,
        train_data['labels'],
        feature_names
    )
    
    # Save features
    save_features(
        selected_features,
        selected_names,
        train_data['labels'],
        save_path="./data/processed/selected_features.csv"
    )