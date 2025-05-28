"""
Data preprocessing module for coral bleaching prediction.

This module handles data loading, cleaning, and preprocessing for the Duke Coral Health project.
It includes functions for:
- Loading raw image and time series data
- Data normalization and standardization
- Handling missing values
- Splitting data into train, validation, and test sets
- Data augmentation for images
"""

import os
import numpy as np
import pandas as pd
import polars as plrs
import torch
from typing import Tuple, Dict, List, Optional, Union, Any
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm


def load_raw_data(data_dir: str) -> Tuple[Dict[str, Any], Dict[str, Any], plrs.DataFrame]:
    """
    Load raw image and time series data from directories.
    
    Args:
        data_dir: Path to the data directory
        
    Returns:
        Tuple of (image data dict, time series data dict, labels dataframe)
    """
    print("Loading raw data...")
    
    # Initialize data containers
    image_data = {}
    timeseries_data = {}
    
    # Load image files
    image_dir = os.path.join(data_dir, 'raw/imagery')
    if os.path.exists(image_dir):
        print(f"Loading images from {image_dir}...")
        image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.tif'))]
        
        for image_file in tqdm(image_files, desc="Loading images"):
            # Extract ID from filename
            file_id = os.path.splitext(image_file)[0]
            
            # Load image
            img_path = os.path.join(image_dir, image_file)
            image = cv2.imread(img_path)
            
            # Convert BGR to RGB
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image_data[file_id] = image
            else:
                print(f"Warning: Could not load image {img_path}")
    else:
        print(f"Warning: Image directory {image_dir} does not exist.")
    
    # Load time series data
    ts_dir = os.path.join(data_dir, 'raw/environmental')
    if os.path.exists(ts_dir):
        print(f"Loading time series data from {ts_dir}...")
        ts_files = [f for f in os.listdir(ts_dir) if f.endswith(('.csv', '.txt'))]
        
        for ts_file in tqdm(ts_files, desc="Loading time series"):
            # Extract ID from filename
            file_id = os.path.splitext(ts_file)[0]
            
            # Load time series data
            ts_path = os.path.join(ts_dir, ts_file)
            try:
                # Try using polars for faster loading
                ts_data = plrs.read_csv(ts_path)
                timeseries_data[file_id] = ts_data
            except Exception as e:
                print(f"Warning: Could not load time series {ts_path} with polars: {e}")
                try:
                    # Fall back to pandas
                    ts_data = pd.read_csv(ts_path)
                    timeseries_data[file_id] = ts_data
                except Exception as e:
                    print(f"Warning: Could not load time series {ts_path} with pandas: {e}")
    else:
        print(f"Warning: Time series directory {ts_dir} does not exist.")
    
    # Load labels
    labels_path = os.path.join(data_dir, 'raw/labels/coral_health_labels.csv')
    if os.path.exists(labels_path):
        print(f"Loading labels from {labels_path}...")
        try:
            labels_df = plrs.read_csv(labels_path)
        except Exception as e:
            print(f"Warning: Could not load labels with polars: {e}")
            labels_df = pd.read_csv(labels_path)
            # Convert pandas to polars
            labels_df = plrs.from_pandas(labels_df)
    else:
        print(f"Warning: Labels file {labels_path} does not exist.")
        # Create empty labels dataframe
        labels_df = plrs.DataFrame()
    
    return image_data, timeseries_data, labels_df


def clean_image_data(image_data: Dict[str, np.ndarray], target_size: Tuple[int, int] = (224, 224)) -> Dict[str, np.ndarray]:
    """
    Clean and preprocess image data.
    
    Args:
        image_data: Dictionary of raw image data
        target_size: Target size for images (height, width)
        
    Returns:
        Dictionary of cleaned image data
    """
    print("Cleaning image data...")
    
    cleaned_images = {}
    
    for img_id, img in tqdm(image_data.items(), desc="Cleaning images"):
        # Skip if image is None
        if img is None:
            print(f"Warning: Image {img_id} is None.")
            continue
        
        # Check if image is valid
        if len(img.shape) < 2:
            print(f"Warning: Image {img_id} has invalid shape {img.shape}.")
            continue
        
        # Resize image
        if img.shape[:2] != target_size:
            img = cv2.resize(img, (target_size[1], target_size[0]))
        
        # Ensure image has 3 channels
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] > 3:
            img = img[:, :, :3]  # Take first 3 channels
        
        # Convert to float
        img = img.astype(np.float32) / 255.0
        
        # Store cleaned image
        cleaned_images[img_id] = img
    
    return cleaned_images


def clean_timeseries_data(
    timeseries_data: Dict[str, Union[plrs.DataFrame, pd.DataFrame]],
    required_features: Optional[List[str]] = None,
    target_steps: int = 24
) -> Dict[str, np.ndarray]:
    """
    Clean and preprocess time series data.
    
    Args:
        timeseries_data: Dictionary of raw time series data
        required_features: List of required feature names (optional)
        target_steps: Target number of time steps
        
    Returns:
        Dictionary of cleaned time series data as numpy arrays
    """
    print("Cleaning time series data...")
    
    # Default required features if not provided
    if required_features is None:
        required_features = [
            'temperature', 'salinity', 'ph', 'dissolved_oxygen',
            'turbidity', 'chlorophyll', 'nitrate', 'phosphate'
        ]
    
    cleaned_timeseries = {}
    
    for ts_id, ts_df in tqdm(timeseries_data.items(), desc="Cleaning time series"):
        # Convert to pandas if it's a polars dataframe
        if isinstance(ts_df, plrs.DataFrame):
            ts_df = ts_df.to_pandas()
        
        # Check if dataframe is valid
        if ts_df.empty:
            print(f"Warning: Time series {ts_id} is empty.")
            continue
        
        # Check for date/time column
        date_col = None
        for col in ts_df.columns:
            if 'date' in col.lower() or 'time' in col.lower():
                date_col = col
                break
        
        # Sort by date if date column exists
        if date_col is not None:
            try:
                ts_df[date_col] = pd.to_datetime(ts_df[date_col])
                ts_df = ts_df.sort_values(by=date_col)
            except Exception as e:
                print(f"Warning: Could not parse dates in {ts_id}: {e}")
        
        # Extract required features
        feature_data = []
        for feature in required_features:
            # Find feature column
            feature_col = None
            for col in ts_df.columns:
                if feature.lower() in col.lower():
                    feature_col = col
                    break
            
            # Extract feature data
            if feature_col is not None:
                feature_data.append(ts_df[feature_col].values)
            else:
                # Use zeros if feature is missing
                print(f"Warning: Feature {feature} not found in {ts_id}.")
                feature_data.append(np.zeros(len(ts_df)))
        
        # Stack features
        ts_array = np.column_stack(feature_data)
        
        # Resample to target length
        if len(ts_array) != target_steps:
            # Interpolate to target length
            indices = np.linspace(0, len(ts_array) - 1, target_steps)
            ts_array_resampled = np.zeros((target_steps, ts_array.shape[1]))
            
            for i in range(ts_array.shape[1]):
                ts_array_resampled[:, i] = np.interp(indices, np.arange(len(ts_array)), ts_array[:, i])
            
            ts_array = ts_array_resampled
        
        # Handle NaN values
        ts_array = np.nan_to_num(ts_array, nan=0.0)
        
        # Store cleaned time series
        cleaned_timeseries[ts_id] = ts_array
    
    return cleaned_timeseries


def align_data(
    image_data: Dict[str, np.ndarray],
    timeseries_data: Dict[str, np.ndarray],
    labels_df: plrs.DataFrame,
    id_column: str = 'sample_id',
    label_column: str = 'bleaching_status'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Align image, time series, and label data.
    
    Args:
        image_data: Dictionary of cleaned image data
        timeseries_data: Dictionary of cleaned time series data
        labels_df: DataFrame with labels
        id_column: Name of ID column in labels dataframe
        label_column: Name of label column in labels dataframe
        
    Returns:
        Tuple of (aligned images, aligned time series, aligned labels)
    """
    print("Aligning data...")
    
    # Convert polars to pandas
    if isinstance(labels_df, plrs.DataFrame):
        labels_df = labels_df.to_pandas()
    
    # Get IDs with both image and time series data
    valid_ids = set(image_data.keys()) & set(timeseries_data.keys())
    print(f"Found {len(valid_ids)} samples with both image and time series data.")
    
    # Filter labels to only include valid IDs
    valid_labels = labels_df[labels_df[id_column].isin(valid_ids)]
    print(f"Found {len(valid_labels)} samples with labels.")
    
    # Check if we have any valid samples
    if len(valid_labels) == 0:
        print("Warning: No samples have both data and labels.")
        # Create empty arrays
        aligned_images = np.array([])
        aligned_timeseries = np.array([])
        aligned_labels = np.array([])
        return aligned_images, aligned_timeseries, aligned_labels
    
    # Get final list of valid IDs
    final_ids = valid_labels[id_column].values
    
    # Initialize arrays
    first_img = next(iter(image_data.values()))
    first_ts = next(iter(timeseries_data.values()))
    aligned_images = np.zeros((len(final_ids), *first_img.shape), dtype=np.float32)
    aligned_timeseries = np.zeros((len(final_ids), *first_ts.shape), dtype=np.float32)
    aligned_labels = np.zeros(len(final_ids), dtype=np.int32)
    
    # Fill arrays
    for i, sample_id in enumerate(tqdm(final_ids, desc="Aligning data")):
        # Get data
        img = image_data[sample_id]
        ts = timeseries_data[sample_id]
        label = labels_df.loc[labels_df[id_column] == sample_id, label_column].values[0]
        
        # Convert label to integer
        if isinstance(label, str):
            if label.lower() in ['healthy', 'no', 'normal']:
                label = 0
            elif label.lower() in ['bleached', 'yes', 'abnormal']:
                label = 1
        
        # Store data
        aligned_images[i] = img
        aligned_timeseries[i] = ts
        aligned_labels[i] = label
    
    return aligned_images, aligned_timeseries, aligned_labels


def normalize_data(
    images: np.ndarray,
    timeseries: np.ndarray,
    image_norm_type: str = 'minmax',
    ts_norm_type: str = 'standard'
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Normalize image and time series data.
    
    Args:
        images: Array of images
        timeseries: Array of time series
        image_norm_type: Type of normalization for images ('minmax' or 'standard')
        ts_norm_type: Type of normalization for time series ('minmax' or 'standard')
        
    Returns:
        Tuple of (normalized images, normalized time series, normalization parameters)
    """
    print("Normalizing data...")
    
    norm_params = {}
    
    # Normalize images
    if image_norm_type == 'minmax':
        # Images are already normalized to [0, 1]
        norm_images = images
        norm_params['image_norm'] = {'type': 'minmax', 'min': 0, 'max': 1}
    elif image_norm_type == 'standard':
        # Compute mean and std across all images
        mean = np.mean(images, axis=(0, 1, 2))
        std = np.std(images, axis=(0, 1, 2))
        
        # Standard normalization
        norm_images = (images - mean) / (std + 1e-8)
        
        norm_params['image_norm'] = {'type': 'standard', 'mean': mean.tolist(), 'std': std.tolist()}
    else:
        raise ValueError(f"Unknown image normalization type: {image_norm_type}")
    
    # Normalize time series
    num_samples, time_steps, num_features = timeseries.shape
    norm_timeseries = np.zeros_like(timeseries)
    
    ts_norm_params = []
    
    for f in range(num_features):
        feature_data = timeseries[:, :, f].flatten()
        
        if ts_norm_type == 'minmax':
            # Min-max normalization
            scaler = MinMaxScaler()
            norm_data = scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
            ts_norm_params.append({
                'type': 'minmax',
                'min': scaler.data_min_[0],
                'max': scaler.data_max_[0]
            })
        elif ts_norm_type == 'standard':
            # Standard normalization
            scaler = StandardScaler()
            norm_data = scaler.fit_transform(feature_data.reshape(-1, 1)).flatten()
            ts_norm_params.append({
                'type': 'standard',
                'mean': scaler.mean_[0],
                'std': scaler.scale_[0]
            })
        else:
            raise ValueError(f"Unknown time series normalization type: {ts_norm_type}")
        
        # Reshape normalized data back to original shape
        norm_timeseries[:, :, f] = norm_data.reshape(num_samples, time_steps)
    
    norm_params['ts_norm'] = ts_norm_params
    
    return norm_images, norm_timeseries, norm_params


def split_data(
    images: np.ndarray,
    timeseries: np.ndarray,
    labels: np.ndarray,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 42
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Split data into training, validation, and test sets.
    
    Args:
        images: Array of images
        timeseries: Array of time series
        labels: Array of labels
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train data dict, val data dict, test data dict)
    """
    print("Splitting data...")
    
    # First split into train+val and test
    train_val_indices, test_indices = train_test_split(
        np.arange(len(labels)),
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Then split train+val into train and val
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=val_size,
        random_state=random_state,
        stratify=labels[train_val_indices]
    )
    
    # Create data dictionaries
    train_data = {
        'images': images[train_indices],
        'timeseries': timeseries[train_indices],
        'labels': labels[train_indices]
    }
    
    val_data = {
        'images': images[val_indices],
        'timeseries': timeseries[val_indices],
        'labels': labels[val_indices]
    }
    
    test_data = {
        'images': images[test_indices],
        'timeseries': timeseries[test_indices],
        'labels': labels[test_indices]
    }
    
    # Print data split statistics
    print(f"Train set: {len(train_indices)} samples")
    print(f"  Class distribution: {np.bincount(labels[train_indices])}")
    print(f"Validation set: {len(val_indices)} samples")
    print(f"  Class distribution: {np.bincount(labels[val_indices])}")
    print(f"Test set: {len(test_indices)} samples")
    print(f"  Class distribution: {np.bincount(labels[test_indices])}")
    
    return train_data, val_data, test_data


class CoralDataset(Dataset):
    """PyTorch Dataset for coral bleaching data."""
    
    def __init__(
        self,
        images: np.ndarray,
        timeseries: np.ndarray,
        labels: np.ndarray,
        transform: Optional[Any] = None
    ):
        """
        Initialize dataset.
        
        Args:
            images: Array of images
            timeseries: Array of time series
            labels: Array of labels
            transform: Optional image transforms
        """
        self.images = torch.tensor(images, dtype=torch.float32).permute(0, 3, 1, 2)
        self.timeseries = torch.tensor(timeseries, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)
        self.transform = transform
    
    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get dataset item.
        
        Args:
            idx: Index of item
            
        Returns:
            Tuple of (image, time series, label)
        """
        image = self.images[idx]
        timeseries = self.timeseries[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, timeseries, label


def create_data_loaders(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create PyTorch data loaders.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        batch_size: Batch size
        num_workers: Number of worker threads
        
    Returns:
        Tuple of (train loader, val loader, test loader)
    """
    print("Creating data loaders...")
    
    # Define image transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = CoralDataset(
        train_data['images'], 
        train_data['timeseries'], 
        train_data['labels'],
        transform=train_transform
    )
    
    val_dataset = CoralDataset(
        val_data['images'], 
        val_data['timeseries'], 
        val_data['labels'],
        transform=val_transform
    )
    
    test_dataset = CoralDataset(
        test_data['images'], 
        test_data['timeseries'], 
        test_data['labels'],
        transform=val_transform
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
    
    return train_loader, val_loader, test_loader


def save_processed_data(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    norm_params: Dict[str, Any],
    save_dir: str
) -> None:
    """
    Save processed data to disk.
    
    Args:
        train_data: Training data dictionary
        val_data: Validation data dictionary
        test_data: Test data dictionary
        norm_params: Normalization parameters
        save_dir: Directory to save data
    """
    print(f"Saving processed data to {save_dir}...")
    
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Save data
    np.save(os.path.join(save_dir, 'train_images.npy'), train_data['images'])
    np.save(os.path.join(save_dir, 'train_timeseries.npy'), train_data['timeseries'])
    np.save(os.path.join(save_dir, 'train_labels.npy'), train_data['labels'])
    
    np.save(os.path.join(save_dir, 'val_images.npy'), val_data['images'])
    np.save(os.path.join(save_dir, 'val_timeseries.npy'), val_data['timeseries'])
    np.save(os.path.join(save_dir, 'val_labels.npy'), val_data['labels'])
    
    np.save(os.path.join(save_dir, 'test_images.npy'), test_data['images'])
    np.save(os.path.join(save_dir, 'test_timeseries.npy'), test_data['timeseries'])
    np.save(os.path.join(save_dir, 'test_labels.npy'), test_data['labels'])
    
    # Save normalization parameters
    np.save(os.path.join(save_dir, 'norm_params.npy'), norm_params)
    
    print("Data saved successfully.")


def load_processed_data(load_dir: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Load processed data from disk.
    
    Args:
        load_dir: Directory containing processed data
        
    Returns:
        Tuple of (train data dict, val data dict, test data dict, norm params)
    """
    print(f"Loading processed data from {load_dir}...")
    
    # Load data
    train_data = {
        'images': np.load(os.path.join(load_dir, 'train_images.npy')),
        'timeseries': np.load(os.path.join(load_dir, 'train_timeseries.npy')),
        'labels': np.load(os.path.join(load_dir, 'train_labels.npy'))
    }
    
    val_data = {
        'images': np.load(os.path.join(load_dir, 'val_images.npy')),
        'timeseries': np.load(os.path.join(load_dir, 'val_timeseries.npy')),
        'labels': np.load(os.path.join(load_dir, 'val_labels.npy'))
    }
    
    test_data = {
        'images': np.load(os.path.join(load_dir, 'test_images.npy')),
        'timeseries': np.load(os.path.join(load_dir, 'test_timeseries.npy')),
        'labels': np.load(os.path.join(load_dir, 'test_labels.npy'))
    }
    
    # Load normalization parameters
    norm_params = np.load(os.path.join(load_dir, 'norm_params.npy'), allow_pickle=True).item()
    
    print("Data loaded successfully.")
    print(f"Train set: {len(train_data['labels'])} samples")
    print(f"  Class distribution: {np.bincount(train_data['labels'].astype(int))}")
    print(f"Validation set: {len(val_data['labels'])} samples")
    print(f"  Class distribution: {np.bincount(val_data['labels'].astype(int))}")
    print(f"Test set: {len(test_data['labels'])} samples")
    print(f"  Class distribution: {np.bincount(test_data['labels'].astype(int))}")
    
    return train_data, val_data, test_data, norm_params


def preprocess_pipeline(
    data_dir: str,
    save_dir: Optional[str] = None,
    target_img_size: Tuple[int, int] = (224, 224),
    target_ts_steps: int = 24,
    required_features: Optional[List[str]] = None,
    test_size: float = 0.2,
    val_size: float = 0.2
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Full preprocessing pipeline.
    
    Args:
        data_dir: Directory containing raw data
        save_dir: Directory to save processed data (optional)
        target_img_size: Target image size
        target_ts_steps: Target number of time steps
        required_features: List of required feature names (optional)
        test_size: Fraction of data to use for testing
        val_size: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train data dict, val data dict, test data dict, norm params)
    """
    # Load raw data
    image_data, timeseries_data, labels_df = load_raw_data(data_dir)
    
    # Clean data
    clean_images = clean_image_data(image_data, target_img_size)
    clean_timeseries = clean_timeseries_data(timeseries_data, required_features, target_ts_steps)
    
    # Align data
    aligned_images, aligned_timeseries, aligned_labels = align_data(clean_images, clean_timeseries, labels_df)
    
    # Normalize data
    norm_images, norm_timeseries, norm_params = normalize_data(aligned_images, aligned_timeseries)
    
    # Split data
    train_data, val_data, test_data = split_data(norm_images, norm_timeseries, aligned_labels, test_size, val_size)
    
    # Save processed data if requested
    if save_dir is not None:
        save_processed_data(train_data, val_data, test_data, norm_params, save_dir)
    
    return train_data, val_data, test_data, norm_params


if __name__ == "__main__":
    # Example usage
    data_dir = "./data"
    save_dir = "./data/processed"
    
    # Run preprocessing pipeline
    train_data, val_data, test_data, norm_params = preprocess_pipeline(
        data_dir=data_dir,
        save_dir=save_dir
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data
    )
    
    # Example: Preview a batch
    images, timeseries, labels = next(iter(train_loader))
    print(f"Batch shapes: images={images.shape}, timeseries={timeseries.shape}, labels={labels.shape}")