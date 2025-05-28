"""
Visualization module for coral bleaching prediction.

This module provides comprehensive visualization functions for the Duke Coral Health project.
It includes visualizations for:
- Time series data and patterns
- Model performance and comparison
- Feature importance and relationships
- Early warning signals
- Environmental correlations
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from matplotlib.colors import ListedColormap
from scipy import signal
import torch
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cv2
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator


def set_plotting_style():
    """Set consistent plotting style for all visualizations."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 16


def visualize_time_series(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    sample_indices: Optional[List[int]] = None,
    save_dir: Optional[str] = None
):
    """
    Visualize time series data.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        sample_indices: Indices of samples to visualize (default: select a few from each class)
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
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
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Visualize each feature
    for f in range(min(len(feature_names), time_series.shape[2])):
        feature_name = feature_names[f]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot time series for each sample
        for idx in sample_indices:
            label_text = 'Healthy' if labels[idx] == 0 else 'Bleached'
            color = 'green' if labels[idx] == 0 else 'red'
            plt.plot(time_series[idx, :, f], color=color, alpha=0.7, label=f"Sample {idx} ({label_text})")
        
        plt.title(f"Time Series: {feature_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"timeseries_{feature_name}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Visualize average patterns
    class_0_mean = np.mean(time_series[labels == 0], axis=0)
    class_1_mean = np.mean(time_series[labels == 1], axis=0)
    
    for f in range(min(len(feature_names), time_series.shape[2])):
        feature_name = feature_names[f]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot average patterns
        plt.plot(class_0_mean[:, f], 'g-', linewidth=2, label='Healthy (Mean)')
        plt.plot(class_1_mean[:, f], 'r-', linewidth=2, label='Bleached (Mean)')
        
        # Plot confidence intervals
        if np.sum(labels == 0) > 1:
            class_0_std = np.std(time_series[labels == 0, :, f], axis=0)
            plt.fill_between(
                range(len(class_0_mean[:, f])),
                class_0_mean[:, f] - class_0_std,
                class_0_mean[:, f] + class_0_std,
                color='green',
                alpha=0.2,
                label='Healthy (±1 SD)'
            )
        
        if np.sum(labels == 1) > 1:
            class_1_std = np.std(time_series[labels == 1, :, f], axis=0)
            plt.fill_between(
                range(len(class_1_mean[:, f])),
                class_1_mean[:, f] - class_1_std,
                class_1_mean[:, f] + class_1_std,
                color='red',
                alpha=0.2,
                label='Bleached (±1 SD)'
            )
        
        plt.title(f"Average Patterns: {feature_name}")
        plt.xlabel("Time Step")
        plt.ylabel("Value")
        plt.grid(True)
        plt.legend()
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"avg_pattern_{feature_name}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    """
    Visualize feature importance.
    
    Args:
        importance_df: DataFrame with feature importance
        top_n: Number of top features to show
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Check if dataframe has required columns
    if 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
        if 'feature' in importance_df.columns and any(col.endswith('_importance') for col in importance_df.columns):
            # Try to find importance column
            importance_col = next(col for col in importance_df.columns if col.endswith('_importance'))
            importance_df = importance_df.rename(columns={importance_col: 'importance'})
        else:
            raise ValueError("Importance dataframe must have 'feature' and 'importance' columns")
    
    # Sort by importance and get top N
    plot_df = importance_df.sort_values('importance', ascending=False).head(top_n)
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, max(6, 0.3 * len(plot_df))))
    ax = sns.barplot(data=plot_df, y='feature', x='importance', orient='h')
    
    # Add value labels
    for i, v in enumerate(plot_df['importance']):
        ax.text(v + 0.01 * ax.get_xlim()[1], i, f"{v:.3f}", va='center')
    
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    top_features: Optional[List[str]] = None,
    save_dir: Optional[str] = None
):
    """
    Visualize feature distributions by class.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        top_features: List of top feature names to visualize (optional)
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Determine features to visualize
    if top_features:
        # Find indices of top features
        indices = [feature_names.index(name) for name in top_features if name in feature_names]
        vis_features = [feature_names[i] for i in indices]
    else:
        # Use all features
        indices = list(range(min(10, len(feature_names))))  # Limit to 10 features by default
        vis_features = [feature_names[i] for i in indices]
    
    # Visualize each feature
    for i, feature_name in zip(indices, vis_features):
        if i >= features.shape[1]:
            continue
        
        # Extract feature values
        feature_values = features[:, i]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        # Create dataframe for seaborn
        df = pd.DataFrame({
            'value': feature_values,
            'class': ['Healthy' if label == 0 else 'Bleached' for label in labels]
        })
        
        # Plot distributions
        sns.histplot(data=df, x='value', hue='class', element='step', stat='density', common_norm=False, alpha=0.6)
        
        plt.title(f"Distribution: {feature_name}")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"distribution_{feature_name}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Create boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='class', y='value')
        sns.stripplot(data=df, x='class', y='value', color='black', alpha=0.2, jitter=True)
        
        plt.title(f"Boxplot: {feature_name}")
        plt.xlabel("Class")
        plt.ylabel("Value")
        plt.grid(True, axis='y')
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"boxplot_{feature_name}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_correlation_matrix(
    features: np.ndarray,
    feature_names: List[str],
    top_features: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize correlation matrix between features.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        feature_names: List of feature names
        top_features: List of top feature names to visualize (optional)
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Determine features to visualize
    if top_features:
        # Find indices of top features
        indices = [feature_names.index(name) for name in top_features if name in feature_names]
        vis_features = [feature_names[i] for i in indices]
        
        # Extract feature subset
        feature_subset = features[:, indices]
    else:
        # Use all features (up to a limit)
        max_features = min(20, len(feature_names))
        indices = list(range(max_features))
        vis_features = [feature_names[i] for i in indices]
        
        # Extract feature subset
        feature_subset = features[:, :max_features]
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(feature_subset.T)
    
    # Create heatmap
    plt.figure(figsize=(max(10, len(vis_features) * 0.5), max(8, len(vis_features) * 0.5)))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        mask=mask,
        xticklabels=vis_features,
        yticklabels=vis_features
    )
    
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_dimensionality_reduction(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    perplexity: int = 30,
    n_components: int = 2,
    random_state: int = 42,
    save_path: Optional[str] = None
):
    """
    Visualize dimensionality reduction of features.
    
    Args:
        features: Feature matrix of shape [num_samples, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        perplexity: Perplexity for t-SNE
        n_components: Number of components
        random_state: Random seed
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        dr = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
        embedding = dr.fit_transform(features)
        title = f"t-SNE Visualization (perplexity={perplexity})"
    elif method.lower() == 'pca':
        dr = PCA(n_components=n_components, random_state=random_state)
        embedding = dr.fit_transform(features)
        explained_var = dr.explained_variance_ratio_ * 100
        title = f"PCA Visualization (explained variance: {explained_var[0]:.1f}%, {explained_var[1]:.1f}%)"
    elif method.lower() == 'umap':
        try:
            import umap
            dr = umap.UMAP(n_components=n_components, random_state=random_state)
            embedding = dr.fit_transform(features)
            title = "UMAP Visualization"
        except ImportError:
            print("UMAP not available. Install with 'pip install umap-learn'.")
            print("Falling back to t-SNE.")
            dr = TSNE(n_components=n_components, perplexity=perplexity, random_state=random_state)
            embedding = dr.fit_transform(features)
            title = f"t-SNE Visualization (perplexity={perplexity})"
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    
    # Create scatter plot
    plt.figure(figsize=(10, 8))
    
    # Create dataframe for seaborn
    df = pd.DataFrame({
        'x': embedding[:, 0],
        'y': embedding[:, 1],
        'class': ['Healthy' if label == 0 else 'Bleached' for label in labels]
    })
    
    # Plot scatter
    sns.scatterplot(data=df, x='x', y='y', hue='class', style='class', s=100, alpha=0.7)
    
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.grid(True)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_learning_curves(
    history_df: pd.DataFrame,
    metrics: List[str] = ['loss', 'accuracy'],
    model_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize learning curves for model training.
    
    Args:
        history_df: DataFrame with training history
        metrics: List of metrics to visualize
        model_names: List of model names (optional)
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Determine number of plots
    n_plots = len(metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]  # Make axes iterable for single plot
    
    # Identify model columns in dataframe
    if model_names is None:
        # Try to automatically detect models from column names
        model_names = set()
        for col in history_df.columns:
            if any(col.startswith(f"{prefix}_") for prefix in ['train', 'val', 'test']):
                model_name = col.split('_', 2)[1]  # Split by underscore and take middle part
                model_names.add(model_name)
        model_names = sorted(list(model_names))
    
    # Create a colormap for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for j, model_name in enumerate(model_names):
            train_col = f"train_{model_name}_{metric}"
            val_col = f"val_{model_name}_{metric}"
            
            # Check if columns exist
            if train_col in history_df.columns:
                ax.plot(history_df['epoch'], history_df[train_col], color=colors[j], linestyle='-', 
                         label=f"{model_name} (Train)")
            
            if val_col in history_df.columns:
                ax.plot(history_df['epoch'], history_df[val_col], color=colors[j], linestyle='--',
                         label=f"{model_name} (Val)")
        
        ax.set_title(f"{metric.capitalize()} Curves")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(metric.capitalize())
        ax.legend(loc='best')
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_model_comparison(
    metrics_df: pd.DataFrame,
    metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    save_path: Optional[str] = None
):
    """
    Visualize model comparison based on evaluation metrics.
    
    Args:
        metrics_df: DataFrame with model metrics
        metrics: List of metrics to visualize
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Check if dataframe has required format
    if 'model' not in metrics_df.columns:
        metrics_df = metrics_df.reset_index().rename(columns={'index': 'model'})
    
    # Determine number of plots
    n_metrics = len(metrics)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]  # Make axes iterable for single plot
    
    # Plot each metric
    for i, metric in enumerate(metrics):
        if metric in metrics_df.columns:
            ax = axes[i]
            
            # Sort by metric value
            sorted_df = metrics_df.sort_values(metric, ascending=False)
            
            # Create bar plot
            sns.barplot(data=sorted_df, x='model', y=metric, ax=ax)
            
            # Add value labels
            for j, v in enumerate(sorted_df[metric]):
                ax.text(j, v + 0.01, f"{v:.3f}", ha='center')
            
            ax.set_title(f"{metric.capitalize()}")
            ax.set_xlabel("Model")
            ax.set_ylabel(metric.capitalize())
            ax.set_ylim(0, 1.05)  # Assuming metrics are between 0 and 1
            ax.tick_params(axis='x', rotation=45)
        else:
            print(f"Metric '{metric}' not found in dataframe columns.")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_confusion_matrices(
    confusion_matrices: Dict[str, np.ndarray],
    save_path: Optional[str] = None
):
    """
    Visualize confusion matrices for multiple models.
    
    Args:
        confusion_matrices: Dictionary of {model_name: confusion_matrix}
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Determine grid size
    n_models = len(confusion_matrices)
    n_cols = min(3, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    
    # Make axes iterable if there's only one subplot
    if n_models == 1:
        axes = np.array([axes])
    
    # Flatten axes if there are multiple rows and columns
    if n_rows > 1 and n_cols > 1:
        axes = axes.flatten()
    
    # Plot each confusion matrix
    for i, (model_name, cm) in enumerate(confusion_matrices.items()):
        if i < len(axes):
            ax = axes[i]
            
            # Plot confusion matrix
            sns.heatmap(
                cm, 
                annot=True, 
                fmt='d', 
                cmap='Blues',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                ax=ax
            )
            
            ax.set_title(f"{model_name} Confusion Matrix")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
    
    # Hide empty subplots
    for i in range(n_models, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_roc_curves(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None
):
    """
    Visualize ROC curves for multiple models.
    
    Args:
        roc_data: Dictionary of {model_name: {'fpr': fpr_array, 'tpr': tpr_array, 'auc': auc_value}}
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a colormap for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(roc_data)))
    
    # Plot each ROC curve
    for i, (model_name, data) in enumerate(roc_data.items()):
        plt.plot(
            data['fpr'], 
            data['tpr'], 
            color=colors[i],
            lw=2, 
            label=f"{model_name} (AUC = {data['auc']:.3f})"
        )
    
    # Plot diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_pr_curves(
    pr_data: Dict[str, Dict[str, np.ndarray]],
    save_path: Optional[str] = None
):
    """
    Visualize Precision-Recall curves for multiple models.
    
    Args:
        pr_data: Dictionary of {model_name: {'precision': precision_array, 'recall': recall_array, 'ap': average_precision}}
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create figure
    plt.figure(figsize=(10, 8))
    
    # Create a colormap for models
    colors = plt.cm.tab10(np.linspace(0, 1, len(pr_data)))
    
    # Plot each PR curve
    for i, (model_name, data) in enumerate(pr_data.items()):
        plt.plot(
            data['recall'], 
            data['precision'], 
            color=colors[i],
            lw=2, 
            label=f"{model_name} (AP = {data['ap']:.3f})"
        )
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend(loc='lower left')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_feature_changes_over_time(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    top_features: List[str],
    save_dir: Optional[str] = None
):
    """
    Visualize how features change over time before bleaching events.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        top_features: List of top feature names to visualize
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get feature indices
    feature_indices = [feature_names.index(name) for name in top_features if name in feature_names]
    
    # Get bleached and healthy samples
    bleached_indices = np.where(labels == 1)[0]
    healthy_indices = np.where(labels == 0)[0]
    
    # Calculate average patterns
    bleached_avg = np.mean(time_series[bleached_indices], axis=0)  # [time_steps, num_features]
    healthy_avg = np.mean(time_series[healthy_indices], axis=0)  # [time_steps, num_features]
    
    # Calculate standard deviation
    bleached_std = np.std(time_series[bleached_indices], axis=0)
    healthy_std = np.std(time_series[healthy_indices], axis=0)
    
    # Time steps for x-axis
    time_steps = np.arange(time_series.shape[1])
    
    # Create a summary figure for all top features
    fig, axes = plt.subplots(len(feature_indices), 2, figsize=(15, 5 * len(feature_indices)))
    
    for i, idx in enumerate(feature_indices):
        feature_name = feature_names[idx]
        
        # Pattern comparison plot (left subplot)
        ax1 = axes[i, 0] if len(feature_indices) > 1 else axes[0]
        
        # Plot average patterns
        ax1.plot(time_steps, bleached_avg[:, idx], 'r-', linewidth=2, label='Bleached (Mean)')
        ax1.plot(time_steps, healthy_avg[:, idx], 'g-', linewidth=2, label='Healthy (Mean)')
        
        # Plot confidence intervals
        ax1.fill_between(
            time_steps,
            bleached_avg[:, idx] - bleached_std[:, idx],
            bleached_avg[:, idx] + bleached_std[:, idx],
            color='red',
            alpha=0.2,
            label='Bleached (±1 SD)'
        )
        
        ax1.fill_between(
            time_steps,
            healthy_avg[:, idx] - healthy_std[:, idx],
            healthy_avg[:, idx] + healthy_std[:, idx],
            color='green',
            alpha=0.2,
            label='Healthy (±1 SD)'
        )
        
        ax1.set_title(f"Average Pattern: {feature_name}")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value")
        ax1.legend()
        ax1.grid(True)
        
        # Difference plot (right subplot)
        ax2 = axes[i, 1] if len(feature_indices) > 1 else axes[1]
        
        # Calculate and plot difference
        diff = bleached_avg[:, idx] - healthy_avg[:, idx]
        ax2.plot(time_steps, diff, 'b-', linewidth=2)
        
        # Highlight significant differences
        sig_threshold = np.std(diff) * 1.5
        significant_points = np.abs(diff) > sig_threshold
        
        if np.any(significant_points):
            ax2.scatter(
                time_steps[significant_points],
                diff[significant_points],
                color='red',
                s=50,
                zorder=5,
                label='Significant Difference'
            )
        
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.set_title(f"Difference Pattern: {feature_name}")
        ax2.set_xlabel("Time Step")
        ax2.set_ylabel("Difference (Bleached - Healthy)")
        if np.any(significant_points):
            ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_changes_summary.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create individual plots for each feature
    for idx in feature_indices:
        feature_name = feature_names[idx]
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot average patterns
        plt.plot(time_steps, bleached_avg[:, idx], 'r-', linewidth=2, label='Bleached (Mean)')
        plt.plot(time_steps, healthy_avg[:, idx], 'g-', linewidth=2, label='Healthy (Mean)')
        
        # Plot confidence intervals
        plt.fill_between(
            time_steps,
            bleached_avg[:, idx] - bleached_std[:, idx],
            bleached_avg[:, idx] + bleached_std[:, idx],
            color='red',
            alpha=0.2,
            label='Bleached (±1 SD)'
        )
        
        plt.fill_between(
            time_steps,
            healthy_avg[:, idx] - healthy_std[:, idx],
            healthy_avg[:, idx] + healthy_std[:, idx],
            color='green',
            alpha=0.2,
            label='Healthy (±1 SD)'
        )
        
        # Calculate percentage change
        pct_change = (bleached_avg[:, idx] - healthy_avg[:, idx]) / np.abs(healthy_avg[:, idx]) * 100
        
        # Add a second y-axis for percentage change
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.plot(time_steps, pct_change, 'b--', linewidth=1.5, label='% Change')
        ax2.set_ylabel('Percentage Change (%)')
        
        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        
        plt.title(f"Pattern Comparison: {feature_name}")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Value")
        ax1.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"feature_change_{feature_name}.png"), dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_early_warning_signals(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    ews_results: Dict[str, np.ndarray],
    save_dir: Optional[str] = None
):
    """
    Visualize early warning signals for coral bleaching.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        ews_results: Dictionary of early warning signal results
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Get bleached samples
    bleached_indices = np.where(labels == 1)[0]
    
    # Time steps for x-axis
    time_steps = np.arange(time_series.shape[1])
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot early warning signals for variance
    if 'variance' in ews_results:
        plt.plot(time_steps, ews_results['variance'].mean(axis=0), 'r-', linewidth=2, label='Variance')
    
    # Plot early warning signals for autocorrelation
    if 'autocorrelation' in ews_results:
        plt.plot(time_steps, ews_results['autocorrelation'].mean(axis=0), 'g-', linewidth=2, label='Autocorrelation')
    
    # Plot early warning signals for skewness
    if 'skewness' in ews_results:
        plt.plot(time_steps, ews_results['skewness'].mean(axis=0), 'b-', linewidth=2, label='Skewness')
    
    # Plot combined signal
    if 'combined' in ews_results:
        plt.plot(time_steps, ews_results['combined'].mean(axis=0), 'k-', linewidth=3, label='Combined Signal')
    
    # Highlight threshold for warning
    if 'threshold' in ews_results:
        plt.axhline(y=ews_results['threshold'], color='r', linestyle='--', label='Warning Threshold')
    
    plt.title("Early Warning Signals for Coral Bleaching")
    plt.xlabel("Time Step")
    plt.ylabel("Signal Strength")
    plt.legend()
    plt.grid(True)
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "early_warning_signals.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create heatmap of early warning signals across samples and time
    if 'combined' in ews_results:
        plt.figure(figsize=(12, 8))
        
        # Plot heatmap
        sns.heatmap(
            ews_results['combined'],
            cmap='YlOrRd',
            xticklabels=10,  # Show every 10th time step
            yticklabels=False
        )
        
        plt.title("Early Warning Signals Heatmap")
        plt.xlabel("Time Step")
        plt.ylabel("Sample")
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "early_warning_heatmap.png"), dpi=300, bbox_inches='tight')
        
        plt.show()
    
    # Visualize distribution of warning times
    if 'warning_times' in ews_results:
        plt.figure(figsize=(10, 6))
        
        # Plot histogram of warning times
        sns.histplot(ews_results['warning_times'], bins=10, kde=True)
        
        plt.title("Distribution of Early Warning Times")
        plt.xlabel("Time Steps Before Bleaching")
        plt.ylabel("Frequency")
        plt.grid(True)
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, "warning_times_distribution.png"), dpi=300, bbox_inches='tight')
        
        plt.show()


def visualize_feature_ranking(
    feature_ranking: pd.DataFrame,
    top_n: int = 20,
    highlight_early_warning: bool = True,
    save_path: Optional[str] = None
):
    """
    Visualize ranking of features by importance.
    
    Args:
        feature_ranking: DataFrame with feature ranking information
        top_n: Number of top features to display
        highlight_early_warning: Whether to highlight early warning features
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Ensure required columns exist
    required_cols = ['feature']
    if not all(col in feature_ranking.columns for col in required_cols):
        raise ValueError(f"Feature ranking dataframe must have columns: {required_cols}")
    
    # Select ranking column
    if 'avg_rank' in feature_ranking.columns:
        rank_col = 'avg_rank'
    elif 'importance' in feature_ranking.columns:
        rank_col = 'importance'
    elif 'score' in feature_ranking.columns:
        rank_col = 'score'
    else:
        # Use the first numeric column if none of the expected ones are found
        numeric_cols = feature_ranking.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            rank_col = numeric_cols[0]
        else:
            raise ValueError("No numeric column found for ranking features")
    
    # Get early warning flag column
    early_warning_col = None
    for col in ['early_warning', 'is_early_warning', 'warning_feature']:
        if col in feature_ranking.columns:
            early_warning_col = col
            break
    
    # Sort by ranking and get top N
    if rank_col == 'avg_rank':  # Lower is better
        plot_df = feature_ranking.sort_values(rank_col).head(top_n)
    else:  # Higher is better
        plot_df = feature_ranking.sort_values(rank_col, ascending=False).head(top_n)
    
    # Create figure
    plt.figure(figsize=(12, max(6, 0.3 * len(plot_df))))
    
    # Create bar plot
    bars = plt.barh(plot_df['feature'], plot_df[rank_col])
    
    # Color bars by early warning if requested and column exists
    if highlight_early_warning and early_warning_col is not None:
        for i, (_, row) in enumerate(plot_df.iterrows()):
            if row[early_warning_col]:
                bars[i].set_color('red')
                bars[i].set_alpha(0.8)
    
    # Add value labels
    for i, v in enumerate(plot_df[rank_col]):
        plt.text(v + 0.01 * plt.xlim()[1], i, f"{v:.3f}", va='center')
    
    # Add legend if early warning features are highlighted
    if highlight_early_warning and early_warning_col is not None and any(plot_df[early_warning_col]):
        legend_elements = [
            Patch(facecolor='C0', label='Regular Feature'),
            Patch(facecolor='red', alpha=0.8, label='Early Warning Feature')
        ]
        plt.legend(handles=legend_elements, loc='lower right')
    
    plt.title("Feature Ranking")
    plt.xlabel(rank_col.replace('_', ' ').title())
    plt.ylabel("Feature")
    plt.grid(True, axis='x')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_temporal_patterns(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    dates: Optional[List[str]] = None,
    save_dir: Optional[str] = None
):
    """
    Visualize temporal patterns in time series data.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        dates: List of dates corresponding to time steps (optional)
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Prepare x-axis
    if dates is not None:
        x = [pd.to_datetime(date) for date in dates]
        x_formatter = mdates.DateFormatter('%Y-%m-%d')
        x_label = "Date"
    else:
        x = np.arange(time_series.shape[1])
        x_formatter = None
        x_label = "Time Step"
    
    # Get bleached and healthy samples
    bleached_indices = np.where(labels == 1)[0]
    healthy_indices = np.where(labels == 0)[0]
    
    # Create a grid of plots for all features
    n_features = min(len(feature_names), time_series.shape[2])
    n_cols = 2
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig = plt.figure(figsize=(15, 5 * n_rows))
    gs = gridspec.GridSpec(n_rows, n_cols)
    
    for f in range(n_features):
        ax = plt.subplot(gs[f])
        feature_name = feature_names[f]
        
        # Calculate statistics for each class
        bleached_mean = np.mean(time_series[bleached_indices, :, f], axis=0)
        healthy_mean = np.mean(time_series[healthy_indices, :, f], axis=0)
        
        bleached_std = np.std(time_series[bleached_indices, :, f], axis=0)
        healthy_std = np.std(time_series[healthy_indices, :, f], axis=0)
        
        # Plot means
        ax.plot(x, bleached_mean, 'r-', linewidth=2, label='Bleached (Mean)')
        ax.plot(x, healthy_mean, 'g-', linewidth=2, label='Healthy (Mean)')
        
        # Plot confidence intervals
        ax.fill_between(
            x,
            bleached_mean - bleached_std,
            bleached_mean + bleached_std,
            color='red',
            alpha=0.2,
            label='Bleached (±1 SD)'
        )
        
        ax.fill_between(
            x,
            healthy_mean - healthy_std,
            healthy_mean + healthy_std,
            color='green',
            alpha=0.2,
            label='Healthy (±1 SD)'
        )
        
        # Format x-axis for dates
        if x_formatter:
            ax.xaxis.set_major_formatter(x_formatter)
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        ax.set_title(f"Temporal Pattern: {feature_name}")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "temporal_patterns.png"), dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create correlation heatmap between features over time
    plt.figure(figsize=(12, 10))
    
    # Reshape time series to [num_samples * time_steps, num_features]
    reshaped_ts = time_series.reshape(-1, time_series.shape[2])
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(reshaped_ts.T)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        xticklabels=feature_names,
        yticklabels=feature_names
    )
    
    plt.title("Feature Correlation Across Time")
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "feature_correlation_time.png"), dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_reef_image_with_predictions(
    image: np.ndarray,
    prediction: float,
    ground_truth: Optional[int] = None,
    feature_importance: Optional[np.ndarray] = None,
    save_path: Optional[str] = None
):
    """
    Visualize coral reef image with prediction and optional feature importance heatmap.
    
    Args:
        image: Image data of shape [height, width, channels]
        prediction: Prediction probability
        ground_truth: Ground truth label (0 for healthy, 1 for bleached) (optional)
        feature_importance: Feature importance heatmap of shape [height, width] (optional)
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create figure with subplots
    if feature_importance is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    # Display image
    ax1.imshow(image)
    ax1.set_title("Coral Reef Image")
    ax1.axis('off')
    
    # Add prediction text
    prediction_text = f"Prediction: {prediction:.3f} ({int(round(prediction)):d})"
    if ground_truth is not None:
        ground_truth_text = f"Ground Truth: {ground_truth:d}"
        text_color = 'green' if int(round(prediction)) == ground_truth else 'red'
        prediction_text = f"{prediction_text}\n{ground_truth_text}"
    else:
        text_color = 'black'
    
    ax1.text(
        10, 10, prediction_text,
        fontsize=12, color='white', backgroundcolor='black',
        bbox={'facecolor': 'black', 'alpha': 0.7, 'pad': 5}
    )
    
    # Display feature importance heatmap if provided
    if feature_importance is not None:
        im = ax2.imshow(feature_importance, cmap='hot', interpolation='bilinear')
        ax2.set_title("Feature Importance Heatmap")
        ax2.axis('off')
        
        # Add colorbar
        fig.colorbar(im, ax=ax2, label='Importance')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_environmental_correlations(
    env_data: pd.DataFrame,
    bleaching_events: List[Dict[str, Any]],
    feature_names: Optional[List[str]] = None,
    save_path: Optional[str] = None
):
    """
    Visualize correlations between environmental parameters and bleaching events.
    
    Args:
        env_data: DataFrame with environmental parameters
        bleaching_events: List of dictionaries with bleaching event information
        feature_names: List of feature names to include (optional)
        save_path: Path to save visualization (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Select features to visualize
    if feature_names is None:
        feature_names = env_data.columns.tolist()
    
    # Create figure
    n_features = len(feature_names)
    plt.figure(figsize=(15, 3 * n_features))
    
    # Plot each environmental parameter
    for i, feature in enumerate(feature_names):
        if feature in env_data.columns:
            ax = plt.subplot(n_features, 1, i+1)
            
            # Plot environmental parameter
            ax.plot(env_data.index, env_data[feature], 'b-', linewidth=1.5)
            
            # Mark bleaching events
            for event in bleaching_events:
                ax.axvspan(
                    event['start_date'], 
                    event['end_date'], 
                    alpha=0.3, 
                    color='red',
                    label=f"Bleaching ({event['severity']})" if 'severity' in event else "Bleaching"
                )
            
            ax.set_title(f"{feature}")
            ax.set_ylabel("Value")
            
            # Add legend on first plot only
            if i == 0:
                ax.legend()
            
            # Format x-axis dates
            if pd.api.types.is_datetime64_any_dtype(env_data.index):
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
            
            ax.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Create correlation matrix
    corr_df = pd.DataFrame()
    
    # Add binary bleaching event indicator
    bleaching = np.zeros(len(env_data))
    for event in bleaching_events:
        mask = (env_data.index >= event['start_date']) & (env_data.index <= event['end_date'])
        bleaching[mask] = 1
    
    corr_df['bleaching'] = bleaching
    
    # Add environmental parameters
    for feature in feature_names:
        if feature in env_data.columns:
            corr_df[feature] = env_data[feature].values
    
    # Calculate and visualize correlation matrix
    plt.figure(figsize=(10, 8))
    
    corr_matrix = corr_df.corr()
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        vmin=-1, vmax=1
    )
    
    plt.title("Correlation between Environmental Parameters and Bleaching")
    plt.tight_layout()
    
    if save_path:
        base, ext = os.path.splitext(save_path)
        corr_save_path = f"{base}_corr{ext}"
        plt.savefig(corr_save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_comprehensive_report(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    model_metrics: Dict[str, Dict[str, float]],
    top_features: List[str],
    early_warning_features: List[str],
    save_dir: Optional[str] = None
):
    """
    Create a comprehensive visualization report for coral bleaching prediction.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        model_metrics: Dictionary of model metrics
        top_features: List of top feature names
        early_warning_features: List of early warning feature names
        save_dir: Directory to save report (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for comprehensive report
    fig = plt.figure(figsize=(18, 24))
    gs = gridspec.GridSpec(4, 2, height_ratios=[1, 1, 1, 1])
    
    # 1. Model Comparison Plot
    ax1 = plt.subplot(gs[0, 0])
    
    # Create bar chart of model metrics
    metrics_df = pd.DataFrame(model_metrics).T
    metrics_df = metrics_df.reset_index().rename(columns={'index': 'model'})
    
    # Sort by accuracy
    if 'accuracy' in metrics_df.columns:
        metrics_df = metrics_df.sort_values('accuracy', ascending=False)
    
    # Plot accuracy, precision, recall for each model
    models = metrics_df['model'].tolist()
    x = np.arange(len(models))
    width = 0.25
    
    metrics_to_plot = ['accuracy', 'precision', 'recall']
    colors = ['blue', 'green', 'red']
    
    for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
        if metric in metrics_df.columns:
            ax1.bar(x + (i - 1) * width, metrics_df[metric], width, label=metric.capitalize(), color=color)
    
    ax1.set_title("Model Performance Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45)
    ax1.set_ylabel("Score")
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, axis='y')
    
    # 2. Feature Importance Plot
    ax2 = plt.subplot(gs[0, 1])
    
    # Create dummy dataframe for feature importance
    importance_df = pd.DataFrame({
        'feature': top_features,
        'importance': np.linspace(1, 0.1, len(top_features))
    })
    
    # Mark early warning features
    importance_df['early_warning'] = importance_df['feature'].isin(early_warning_features)
    
    # Sort by importance
    importance_df = importance_df.sort_values('importance', ascending=False)
    
    # Plot top 10 features
    bars = ax2.barh(importance_df['feature'].head(10), importance_df['importance'].head(10))
    
    # Color bars by early warning
    for i, row in importance_df.head(10).reset_index().iterrows():
        if row['early_warning']:
            bars[i].set_color('red')
    
    ax2.set_title("Top Feature Importance")
    ax2.set_xlabel("Importance")
    ax2.set_ylabel("Feature")
    ax2.invert_yaxis()  # Highest importance at the top
    
    # Add legend for early warning features
    if any(importance_df['early_warning']):
        legend_elements = [
            Patch(facecolor='C0', label='Regular Feature'),
            Patch(facecolor='red', label='Early Warning Feature')
        ]
        ax2.legend(handles=legend_elements, loc='lower right')
    
    ax2.grid(True, axis='x')
    
    # 3. Feature Change Over Time Plot
    ax3 = plt.subplot(gs[1, :])
    
    # Get feature indices for top features
    feature_indices = [feature_names.index(name) for name in top_features[:3] if name in feature_names]
    
    # Get bleached and healthy samples
    bleached_indices = np.where(labels == 1)[0]
    healthy_indices = np.where(labels == 0)[0]
    
    # Calculate average patterns
    bleached_avg = np.mean(time_series[bleached_indices], axis=0)
    healthy_avg = np.mean(time_series[healthy_indices], axis=0)
    
    # Time steps
    time_steps = np.arange(time_series.shape[1])
    
    # Plot each top feature
    for idx in feature_indices:
        feature_name = feature_names[idx]
        ax3.plot(time_steps, bleached_avg[:, idx], '-', linewidth=2, label=f"{feature_name} (Bleached)")
        ax3.plot(time_steps, healthy_avg[:, idx], '--', linewidth=2, label=f"{feature_name} (Healthy)")
    
    ax3.set_title("Feature Changes Over Time")
    ax3.set_xlabel("Time Step")
    ax3.set_ylabel("Value")
    ax3.legend()
    ax3.grid(True)
    
    # 4. Early Warning Signals Plot
    ax4 = plt.subplot(gs[2, 0])
    
    # Create dummy early warning signals
    ews = np.zeros((time_series.shape[1]))
    warning_threshold = 0.7
    
    # Simulate increasing signal
    ews[10:] = np.linspace(0.2, 0.9, time_series.shape[1] - 10)
    
    # Plot early warning signal
    ax4.plot(time_steps, ews, 'b-', linewidth=2, label='Early Warning Signal')
    
    # Highlight warning threshold
    ax4.axhline(y=warning_threshold, color='r', linestyle='--', label='Warning Threshold')
    
    # Highlight warning period
    warning_start = np.where(ews >= warning_threshold)[0][0]
    ax4.axvspan(warning_start, time_series.shape[1] - 1, alpha=0.2, color='red')
    
    ax4.set_title("Early Warning Signal")
    ax4.set_xlabel("Time Step")
    ax4.set_ylabel("Signal Strength")
    ax4.legend()
    ax4.grid(True)
    
    # 5. Feature Correlation Matrix
    ax5 = plt.subplot(gs[2, 1])
    
    # Get feature indices for top features
    feature_indices = [feature_names.index(name) for name in top_features[:5] if name in feature_names]
    selected_features = [feature_names[i] for i in feature_indices]
    
    # Extract selected features
    feature_data = np.zeros((time_series.shape[0], len(feature_indices)))
    for i, idx in enumerate(feature_indices):
        feature_data[:, i] = np.mean(time_series[:, :, idx], axis=1)
    
    # Calculate correlation matrix
    corr_matrix = np.corrcoef(feature_data.T)
    
    # Create heatmap
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap='coolwarm',
        square=True,
        xticklabels=selected_features,
        yticklabels=selected_features,
        ax=ax5
    )
    
    ax5.set_title("Feature Correlation Matrix")
    
    # 6. Class Distribution Plot
    ax6 = plt.subplot(gs[3, 0])
    
    # Count samples in each class
    class_counts = np.bincount(labels.astype(int))
    class_names = ['Healthy', 'Bleached']
    
    # Create pie chart
    ax6.pie(
        class_counts, 
        labels=class_names, 
        autopct='%1.1f%%',
        colors=['green', 'red'],
        explode=(0, 0.1),
        shadow=True,
        startangle=90
    )
    
    ax6.set_title("Class Distribution")
    
    # 7. Time Series Pattern Plot
    ax7 = plt.subplot(gs[3, 1])
    
    # Select a representative feature
    if len(feature_indices) > 0:
        feature_idx = feature_indices[0]
        feature_name = feature_names[feature_idx]
        
        # Plot average pattern with confidence interval
        ax7.plot(time_steps, bleached_avg[:, feature_idx], 'r-', linewidth=2, label='Bleached')
        ax7.plot(time_steps, healthy_avg[:, feature_idx], 'g-', linewidth=2, label='Healthy')
        
        # Calculate standard deviation
        bleached_std = np.std(time_series[bleached_indices, :, feature_idx], axis=0)
        healthy_std = np.std(time_series[healthy_indices, :, feature_idx], axis=0)
        
        # Plot confidence intervals
        ax7.fill_between(
            time_steps,
            bleached_avg[:, feature_idx] - bleached_std,
            bleached_avg[:, feature_idx] + bleached_std,
            color='red',
            alpha=0.2
        )
        
        ax7.fill_between(
            time_steps,
            healthy_avg[:, feature_idx] - healthy_std,
            healthy_avg[:, feature_idx] + healthy_std,
            color='green',
            alpha=0.2
        )
        
        ax7.set_title(f"Pattern Comparison: {feature_name}")
        ax7.set_xlabel("Time Step")
        ax7.set_ylabel("Value")
        ax7.legend()
        ax7.grid(True)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, "comprehensive_report.png"), dpi=300, bbox_inches='tight')
    
    plt.show()


def visualize_model_attention(
    attention_weights: Dict[str, np.ndarray],
    time_series: np.ndarray,
    feature_names: List[str],
    save_dir: Optional[str] = None
):
    """
    Visualize attention weights from attention-based models.
    
    Args:
        attention_weights: Dictionary of attention weights for different models
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        feature_names: List of feature names
        save_dir: Directory to save visualizations (optional)
    """
    # Set plotting style
    set_plotting_style()
    
    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figure for each model's attention weights
    for model_name, weights in attention_weights.items():
        # Average attention weights across samples
        if weights.ndim > 2:
            avg_weights = np.mean(weights, axis=0)
        else:
            avg_weights = weights
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(
            avg_weights,
            cmap='YlOrRd',
            xticklabels=5,  # Show every 5th time step
            yticklabels=5
        )
        
        plt.title(f"{model_name} Attention Weights")
        plt.xlabel("Time Step")
        plt.ylabel("Time Step")
        
        if save_dir:
            plt.savefig(os.path.join(save_dir, f"{model_name}_attention.png"), dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Visualize attention for each feature
        if len(feature_names) > 0:
            # Create temporal attention plot
            plt.figure(figsize=(12, 6))
            
            # Calculate temporal attention as row-wise average
            temporal_attention = np.mean(avg_weights, axis=0)
            
            # Plot temporal attention
            plt.plot(temporal_attention, 'b-', linewidth=2)
            
            # Highlight peaks
            peak_indices = signal.find_peaks(temporal_attention)[0]
            plt.plot(peak_indices, temporal_attention[peak_indices], 'ro', markersize=8)
            
            plt.title(f"{model_name} Temporal Attention")
            plt.xlabel("Time Step")
            plt.ylabel("Attention Weight")
            plt.grid(True)
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{model_name}_temporal_attention.png"), dpi=300, bbox_inches='tight')
            
            plt.show()
            
            # Visualize feature importance based on attention
            feature_importance = np.zeros(len(feature_names))
            
            # Calculate importance by correlating each feature with attention
            for i, feature_name in enumerate(feature_names):
                # Average feature values across samples
                avg_feature = np.mean(time_series[:, :, i], axis=0)
                
                # Calculate correlation with attention
                corr = np.corrcoef(avg_feature, temporal_attention)[0, 1]
                
                # Use absolute correlation as importance
                feature_importance[i] = abs(corr)
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            
            # Sort features by importance
            sorted_indices = np.argsort(feature_importance)[::-1]
            sorted_features = [feature_names[i] for i in sorted_indices[:10]]  # Top 10 features
            sorted_importance = feature_importance[sorted_indices[:10]]
            
            plt.barh(sorted_features, sorted_importance)
            
            plt.title(f"{model_name} Feature Importance from Attention")
            plt.xlabel("Importance")
            plt.ylabel("Feature")
            plt.grid(True, axis='x')
            
            if save_dir:
                plt.savefig(os.path.join(save_dir, f"{model_name}_feature_importance.png"), dpi=300, bbox_inches='tight')
            
            plt.show()
