"""
Comprehensive analysis module for coral bleaching prediction.

This module combines feature comparison, model comparison, early warning detection,
and visualization functionalities for the Duke Coral Health project.
"""

import os
import numpy as np
import pandas as pd
import polars as plrs # Optional, will try to use if available
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from matplotlib.colors import ListedColormap
from scipy import signal, stats
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    confusion_matrix, roc_curve, precision_recall_curve, auc,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xgboost as xgb # Optional, will try to use if available
from xgboost import DMatrix
import pytorch_lightning as pl
from tqdm import tqdm
import cv2
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import matplotlib.dates as mdates
from matplotlib.ticker import MaxNLocator

# Attempt to import models from a 'models' directory or the current directory if structured differently
try:
    # Assuming models.py is in the same directory or src.models
    from models import (
        CoralLightningModel as CNNLSTMModel,
        CoralTransformerLightning as TransformerModel,
        TCNLightningModel as TCNModel,
        XGBoostCoralModel,
        EnsembleLightningModel as EnsembleModel,
        XGB_FeatureExtractor
    )
except ImportError as e:
    print(f"Warning: Could not import one or more model definitions: {e}. Ensure models.py is accessible.")
    # Define dummy classes if import fails, to allow script to run without full model functionality
    class DummyModel(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.device = 'cpu'
        def forward(self, *args, **kwargs): return torch.randn(1)
        def parameters(self): return iter([torch.nn.Parameter(torch.randn(1))]) # For device check
    CNNLSTMModel = TCNModel = TransformerModel = EnsembleModel = DummyModel
    XGBoostCoralModel = object # Non-PyTorch, use object as placeholder
    class XGB_FeatureExtractor:
        def __init__(self, *args, **kwargs): pass
        def extract_all_features(self, *args, **kwargs): 
            return np.random.randn(50, 100), [f'feat_{i}' for i in range(100)] # Dummy features

    
# --- Content from visualization.py (with adaptations) ---

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
    set_plotting_style()
    
    if sample_indices is None:
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]
        sample_indices = []
        if len(class_0_indices) > 0: sample_indices.extend(class_0_indices[:min(3, len(class_0_indices))])
        if len(class_1_indices) > 0: sample_indices.extend(class_1_indices[:min(3, len(class_1_indices))])
    
    if save_dir: os.makedirs(save_dir, exist_ok=True)
    
    for f in range(min(len(feature_names), time_series.shape[2])):
        feature_name = feature_names[f]
        plt.figure(figsize=(12, 6))
        for idx in sample_indices:
            if idx >= len(labels): continue # Safety check
            label_text = 'Healthy' if labels[idx] == 0 else 'Bleached'
            color = 'green' if labels[idx] == 0 else 'red'
            plt.plot(time_series[idx, :, f], color=color, alpha=0.7, label=f"Sample {idx} ({label_text})")
        plt.title(f"Time Series: {feature_name}"); plt.xlabel("Time Step"); plt.ylabel("Value")
        plt.grid(True); plt.legend()
        if save_dir: plt.savefig(os.path.join(save_dir, f"timeseries_{feature_name}.png"), dpi=300, bbox_inches='tight')
        plt.close() # Close figure after saving/showing
    
    if np.any(labels==0) and np.any(labels==1): # Ensure both classes exist for mean calculation
        class_0_mean = np.mean(time_series[labels == 0], axis=0)
        class_1_mean = np.mean(time_series[labels == 1], axis=0)
        
        for f in range(min(len(feature_names), time_series.shape[2])):
            feature_name = feature_names[f]
            plt.figure(figsize=(12, 6))
            plt.plot(class_0_mean[:, f], 'g-', linewidth=2, label='Healthy (Mean)')
            plt.plot(class_1_mean[:, f], 'r-', linewidth=2, label='Bleached (Mean)')
            
            if np.sum(labels == 0) > 1:
                class_0_std = np.std(time_series[labels == 0, :, f], axis=0)
                plt.fill_between(range(len(class_0_mean[:, f])), class_0_mean[:, f] - class_0_std, class_0_mean[:, f] + class_0_std, color='green', alpha=0.2, label='Healthy (±1 SD)')
            if np.sum(labels == 1) > 1:
                class_1_std = np.std(time_series[labels == 1, :, f], axis=0)
                plt.fill_between(range(len(class_1_mean[:, f])), class_1_mean[:, f] - class_1_std, class_1_mean[:, f] + class_1_std, color='red', alpha=0.2, label='Bleached (±1 SD)')
            
            plt.title(f"Average Patterns: {feature_name}"); plt.xlabel("Time Step"); plt.ylabel("Value")
            plt.grid(True); plt.legend()
            if save_dir: plt.savefig(os.path.join(save_dir, f"avg_pattern_{feature_name}.png"), dpi=300, bbox_inches='tight')
            plt.close()


def visualize_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 20,
    save_path: Optional[str] = None
):
    set_plotting_style()
    importance_col = 'importance' 
    if 'feature' not in importance_df.columns: raise ValueError("DataFrame must have 'feature' column.")
    if 'importance' not in importance_df.columns: 
        potential_cols = [col for col in importance_df.columns if 'importance' in col.lower() or 'score' in col.lower()]
        if not potential_cols: raise ValueError("DataFrame must have an importance-like column.")
        importance_col = potential_cols[0]

    plot_df = importance_df.sort_values(importance_col, ascending=False).head(top_n)
    plt.figure(figsize=(12, max(6, 0.3 * len(plot_df))))
    ax = sns.barplot(data=plot_df, y='feature', x=importance_col, orient='h')
    for i, v in enumerate(plot_df[importance_col]):
        ax.text(v + 0.01 * ax.get_xlim()[1], i, f"{v:.3f}", va='center')
    plt.title("Feature Importance"); plt.xlabel("Importance Score"); plt.ylabel("Feature")
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_learning_curves(
    history_dict: Dict[str, Dict[str, Any]], 
    metrics: List[str] = ['loss', 'accuracy'],
    save_path: Optional[str] = None
):
    set_plotting_style()
    n_plots = len(metrics)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 5 * n_plots), squeeze=False)
    axes = axes.flatten()

    model_names = list(history_dict.keys())
    if not model_names: 
        print("No model history to plot.")
        return
        
    colors = plt.cm.get_cmap('tab10', len(model_names)) # Use get_cmap for newer matplotlib

    for i, metric_name in enumerate(metrics):
        ax = axes[i]
        for j, model_name in enumerate(model_names):
            history = history_dict.get(model_name, {})
            epochs = range(1, len(history.get('train_loss', [])) + 1)
            
            # Handling different ways metric might be stored
            train_metric_key = f'train_{metric_name}'
            val_metric_key = f'val_{metric_name}'
            
            if metric_name == 'loss': # Standard keys for loss
                train_metric_key = 'train_loss'
                val_metric_key = 'val_loss'
            
            if train_metric_key in history:
                 ax.plot(epochs, history[train_metric_key], color=colors(j), linestyle='-', label=f"{model_name} Train {metric_name.capitalize()}")
            
            if 'val_metrics' in history and val_metric_key in history['val_metrics']: # Nested val_metrics
                 ax.plot(epochs, history['val_metrics'][val_metric_key], color=colors(j), linestyle='--', label=f"{model_name} Val {metric_name.capitalize()}")
            elif val_metric_key in history: # Flat val_metrics
                 ax.plot(epochs, history[val_metric_key], color=colors(j), linestyle='--', label=f"{model_name} Val {metric_name.capitalize()}")


        ax.set_title(f"Learning Curves: {metric_name.capitalize()}")
        ax.set_xlabel("Epoch"); ax.set_ylabel(metric_name.capitalize())
        ax.legend(loc='best'); ax.grid(True)
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_model_comparison(
    metrics_summary: pd.DataFrame, 
    metrics_to_plot: List[str] = ['accuracy', 'precision', 'recall', 'f1', 'auc'],
    save_path: Optional[str] = None
):
    set_plotting_style()
    
    if metrics_summary.index.name != 'model' and 'model' not in metrics_summary.columns:
        metrics_summary = metrics_summary.reset_index().rename(columns={'index': 'model'})
    
    # Filter for metrics present in the dataframe
    actual_metrics_to_plot = [m for m in metrics_to_plot if m in metrics_summary.columns]
    if not actual_metrics_to_plot:
        print("None of the specified metrics to plot are in the summary DataFrame.")
        return

    df_melted = metrics_summary.melt(id_vars='model', value_vars=actual_metrics_to_plot,
                                     var_name='metric', value_name='score')

    plt.figure(figsize=(max(12, len(metrics_summary['model']) * 1.5), 7))
    sns.barplot(x='metric', y='score', hue='model', data=df_melted, palette='viridis')
    
    plt.title("Model Performance Comparison")
    plt.ylabel("Score"); plt.xlabel("Metric")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=45, ha="right")
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, axis='y')
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# --- Content from early_warning_detection.py (with adaptations) ---

class EarlyWarningDetector:
    """
    Detector for early warning signals of coral bleaching events.
    """
    def __init__(self, results_dir: str = './results/early_warning', window_size: int = 5, detrend: bool = True, smoothing: bool = True):
        self.results_dir = results_dir; os.makedirs(results_dir, exist_ok=True)
        self.window_size = window_size; self.detrend = detrend; self.smoothing = smoothing
        self.top_features = self._load_top_features()

    def _load_top_features(self) -> List[str]:
        feature_path = os.path.join(os.path.dirname(self.results_dir), 'feature_analysis/early_warning_features.csv') 
        if os.path.exists(feature_path):
            try: return pd.read_csv(feature_path)['feature'].tolist()
            except: return []
        return []
    
    def preprocess_timeseries(self, ts: np.ndarray) -> np.ndarray:
        if self.detrend: ts_detrended = signal.detrend(ts)
        else: ts_detrended = ts.copy() # Use a copy to avoid modifying original
        
        if self.smoothing and len(ts_detrended) >= self.window_size :
            # Ensure window_size is odd for 'same' padding to align correctly, or handle padding manually
            current_window_size = self.window_size if self.window_size % 2 != 0 else self.window_size -1
            if current_window_size < 1: current_window_size = 1 # min window size
            
            if len(ts_detrended) >= current_window_size:
                ts_smoothed = np.convolve(ts_detrended, np.ones(current_window_size)/current_window_size, mode='same')
                ts_processed = ts_smoothed
            else: # Not enough data points for convolution with 'same' padding effectively
                ts_processed = ts_detrended

        else:
            ts_processed = ts_detrended
        return ts_processed

    def calculate_rolling_statistics(self, ts: np.ndarray, window_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        ws = window_size if window_size is not None else self.window_size
        ws = max(2, min(ws, len(ts) -1 if len(ts) > 1 else 1)) # Ensure ws is at least 2 and less than ts length
        
        if len(ts) <= ws: # Not enough data points for rolling window
            default_val = np.array([0.0]) # Return a single zero value
            return {k: default_val for k in ['ar1', 'variance', 'skewness', 'kurtosis']}

        n_windows = len(ts) - ws + 1
        stats_dict = {k: np.zeros(n_windows) for k in ['ar1', 'variance', 'skewness', 'kurtosis']}

        for i in range(n_windows):
            window = ts[i:i+ws]
            if len(window) < 2: continue # Should not happen with ws check, but safety
            
            # Lag-1 Autocorrelation
            if len(window) > 1:
                # np.corrcoef returns a 2x2 matrix, we need the off-diagonal element
                # It can also return nan if std is zero.
                c = np.corrcoef(window[:-1], window[1:])
                stats_dict['ar1'][i] = c[0,1] if not np.isnan(c[0,1]) else 0.0
            else:
                stats_dict['ar1'][i] = 0.0
                
            stats_dict['variance'][i] = np.var(window)
            stats_dict['skewness'][i] = stats.skew(window) if len(window) > 0 else 0
            stats_dict['kurtosis'][i] = stats.kurtosis(window) if len(window) > 0 else 0 # Fisher's definition (normal ==> 0.0)
        return stats_dict
    
    def analyze_feature_for_ews(self, feature_data: np.ndarray, feature_name: str, labels: np.ndarray):
        set_plotting_style()
        bleached_indices = np.where(labels == 1)[0]
        healthy_indices = np.where(labels == 0)[0]

        if len(bleached_indices) == 0 or len(healthy_indices) == 0:
            print(f"Not enough samples for both classes for feature {feature_name}.")
            return {}

        avg_bleached_ts = np.mean([self.preprocess_timeseries(feature_data[idx]) for idx in bleached_indices], axis=0)
        avg_healthy_ts = np.mean([self.preprocess_timeseries(feature_data[idx]) for idx in healthy_indices], axis=0)

        rolling_stats_bleached = self.calculate_rolling_statistics(avg_bleached_ts)
        rolling_stats_healthy = self.calculate_rolling_statistics(avg_healthy_ts)

        # Check if any stats were successfully computed
        if not any(len(v)>0 for v in rolling_stats_bleached.values()):
            print(f"Could not compute rolling stats for bleached samples of {feature_name}")
            return {}
        if not any(len(v)>0 for v in rolling_stats_healthy.values()):
            print(f"Could not compute rolling stats for healthy samples of {feature_name}")
            return {}


        plt.figure(figsize=(15, 10))
        plot_idx = 1
        for stat_name in rolling_stats_bleached.keys():
            if plot_idx > 4: break # Limit to 4 subplots for this figure
            plt.subplot(2, 2, plot_idx)
            if len(rolling_stats_healthy.get(stat_name,[])) > 0:
                plt.plot(rolling_stats_healthy[stat_name], label=f'Healthy - {stat_name}', color='green')
            if len(rolling_stats_bleached.get(stat_name,[])) > 0:
                plt.plot(rolling_stats_bleached[stat_name], label=f'Bleached - {stat_name}', color='red')
            plt.title(f'{feature_name} - Rolling {stat_name}')
            plt.xlabel('Time Window Index'); plt.ylabel(stat_name.capitalize())
            plt.legend(); plt.grid(True)
            plot_idx +=1
        plt.tight_layout()
        if self.results_dir: plt.savefig(os.path.join(self.results_dir, f"ews_stats_{feature_name}.png"))
        plt.close()
        
        return {"rolling_stats_bleached": rolling_stats_bleached, "rolling_stats_healthy": rolling_stats_healthy}

    def analyze_all_features_for_ews(self, time_series_data: np.ndarray, feature_names: List[str], labels: np.ndarray):
        all_results = {}
        features_to_analyze = self.top_features if self.top_features else feature_names
        for f_name in tqdm(features_to_analyze, desc="Analyzing EWS for features"):
            if f_name in feature_names:
                f_idx = feature_names.index(f_name)
                # Ensure feature_data is 1D for EWS analysis if time_series_data is [samples, timesteps, features]
                # The analyze_feature_for_ews expects [samples, timesteps] for one feature
                single_feature_ts_data = time_series_data[:, :, f_idx]
                results = self.analyze_feature_for_ews(single_feature_ts_data, f_name, labels)
                all_results[f_name] = results
        return all_results


# --- Content from feature_comparison.py (now FeatureAnalyzer in this file) ---

class FeatureAnalyzer:
    """
    Analyzes features important for coral bleaching prediction.
    Identifies and visualizes key features across different models.
    """
    def __init__(self, models_dir: str = './results/model_comparison', results_dir: str = './results/feature_analysis', feature_names_all: Optional[List[str]] = None):
        self.models_dir = models_dir; self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.feature_names_all = feature_names_all 
        self.feature_importance_cache = self._load_all_feature_importances() 

    def _load_all_feature_importances(self) -> Dict[str, pd.DataFrame]:
        importances = {}
        xgb_imp_path = os.path.join(self.models_dir, 'xgboost_feature_importances.csv') 
        if os.path.exists(xgb_imp_path):
            try: importances['xgboost_native'] = pd.read_csv(xgb_imp_path)
            except Exception as e: print(f"Could not load {xgb_imp_path}: {e}")
        return importances

    def get_xgboost_importance(self, xgb_model: Optional[XGBoostCoralModel] = None, top_n:int = 20) -> Optional[pd.DataFrame]:
        df = None
        if 'xgboost_native' in self.feature_importance_cache:
            df = self.feature_importance_cache['xgboost_native']
        elif xgb_model and hasattr(xgb_model, 'get_feature_importances'):
            imp_dict = xgb_model.get_feature_importances()
            if imp_dict: # Check if not None and not empty
                df = pd.DataFrame(list(imp_dict.items()), columns=['feature', 'importance']).sort_values('importance', ascending=False)
                self.feature_importance_cache['xgboost_native'] = df 
        
        if df is None or df.empty:
            print("XGBoost model or its importance data not available/empty.")
            return None
        
        visualize_feature_importance(df, top_n=top_n, save_path=os.path.join(self.results_dir, 'xgboost_native_importance.png'))
        return df.head(top_n) # Return only top_n for consistency with method name

    def calculate_permutation_importance(
        self, model_name: str, model_instance: Any, 
        X: np.ndarray, y: np.ndarray, 
        current_feature_names: List[str], 
        n_repeats: int = 10,
        # For PyTorch models that take multiple inputs (images, time_series)
        # X_images: Optional[np.ndarray] = None, 
        # X_time_series: Optional[np.ndarray] = None
    ) -> Optional[pd.DataFrame]:
        set_plotting_style()
        
        # This function needs to be adapted based on how X is structured for different models
        # If X is a single feature matrix (e.g., for XGBoost or post-feature-extraction for DL models)
        # If DL models are passed directly, X would need to be split into images, ts, etc.

        # Assuming X is the final feature matrix for the model
        # For PyTorch models, model_instance is a LightningModule.
        # We need to call its forward pass.
        
        # Scorer for permutation importance
        def roc_auc_scorer(estimator, X_test, y_test):
            if hasattr(estimator, 'predict_proba'):
                y_pred_proba = estimator.predict_proba(X_test)[:, 1]
            elif isinstance(estimator, pl.LightningModule): # PyTorch Lightning model
                estimator.eval()
                X_tensor = torch.from_numpy(X_test).float().to(estimator.device)
                # This assumes X_test is a single tensor. If it's (images, ts), this needs adaptation.
                # For simplicity, assume X_test is the concatenated feature vector if used this way.
                # Or, this permutation importance needs to be done at a higher level
                # where original image/ts data is available for PyTorch models.
                
                # Simplified: if model is PL, assume it's already wrapped or we error out
                # For now, let's focus on XGBoost working. DL permutation importance is more complex with multi-input.
                raise NotImplementedError("Permutation importance for PyTorch Lightning models needs specific data handling.")
            else:
                raise ValueError("Model type not supported for roc_auc_scorer in permutation_importance.")
            return roc_auc_score(y_test, y_pred_proba)

        perm_importance_result = None
        if model_name == 'xgboost' and isinstance(model_instance, XGBoostCoralModel):
            # XGBoostCoralModel has a predict method that takes images and ts.
            # Sklearn permutation_importance needs X, y.
            # We need to use the already extracted features X that were passed.
            # The model_instance should be the Booster object here.
            booster = model_instance.model # Get the actual XGBoost booster
            if booster is None: 
                print("XGBoost model not trained.")
                return None

            # Wrap booster for sklearn compatibility
            class XGBBoosterWrapper:
                def __init__(self, booster_model, feature_names_list):
                    self.booster_model = booster_model
                    self.feature_names_list = feature_names_list
                def predict_proba(self, X_data):
                    # XGBoost predict returns probs of positive class for binary:logistic
                    dmatrix = DMatrix(X_data, feature_names=self.feature_names_list)
                    preds = self.booster_model.predict(dmatrix)
                    # Sklearn scorers often expect (N, n_classes) or (N,) for positive class
                    return np.vstack((1-preds, preds)).T # Return as (N, 2) for roc_auc scoring
            
            wrapped_booster = XGBBoosterWrapper(booster, current_feature_names)
            perm_importance_result = permutation_importance(wrapped_booster, X, y, scoring=roc_auc_scorer, n_repeats=n_repeats, random_state=42)
        
        elif isinstance(model_instance, pl.LightningModule):
            print(f"Permutation importance for PyTorch model {model_name} requires specific data handling (images, ts separately). Skipping for now.")
            return None # Or implement specific logic
        else:
            print(f"Model type {model_name} not directly supported for permutation importance in this generic function. Skipping.")
            return None

        if perm_importance_result is None: return None

        df = pd.DataFrame({
            'feature': current_feature_names,
            'importance_mean': perm_importance_result.importances_mean,
            'importance_std': perm_importance_result.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        self.feature_importance_cache[f'{model_name}_permutation'] = df
        save_path = os.path.join(self.results_dir, f'{model_name}_permutation_importance.png')
        visualize_feature_importance(df, top_n=20, save_path=save_path) # visualize_feature_importance expects 'importance' col
        return df

# --- Content from compare_models.py (now ModelEvaluator in this file) ---
class ModelEvaluator:
    """
    Evaluates and compares multiple models for coral bleaching prediction.
    """
    def __init__(self, models_config: Dict[str, Dict[str, Any]], 
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu', 
                 results_dir: str = './results/model_comparison'):
        self.models_config = models_config
        self.device = device
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.models = {}
        self.history = {}  # Store training history
        self._initialize_models()

    def _initialize_models(self):
        for name, config in self.models_config.items():
            print(f"Initializing model: {name}")
            model_params = config.get('model_params', {})
            learning_rate = config.get('learning_rate', 1e-3)  # Default learning rate

            if name == 'cnn_lstm':
                self.models[name] = CNNLSTMModel(model_params=model_params, learning_rate=learning_rate).to(self.device)
            elif name == 'tcn':
                self.models[name] = TCNModel(model_params=model_params, learning_rate=learning_rate).to(self.device)
            elif name == 'transformer':
                # Ensure binary classification settings
                model_params['num_classes'] = 2  # For binary classification with CrossEntropy
                self.models[name] = TransformerModel(model_config=model_params, learning_rate=learning_rate).to(self.device)
            else:
                print(f"Warning: Model type {name} not recognized for initialization in ModelEvaluator.")
            
            self.history[name] = {'train_loss': [], 'val_loss': [], 'val_metrics': {}}

    def train_all_models(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 10):
        for name, model_instance in self.models.items():
            if isinstance(model_instance, pl.LightningModule):
                print(f"\nTraining {name} using PyTorch Lightning Trainer...")
                # Configure trainer based on device
                trainer_kwargs = {
                    'max_epochs': num_epochs,
                    'callbacks': [pl.callbacks.EarlyStopping(monitor='val_loss', patience=3)],
                    'logger': pl.loggers.CSVLogger(save_dir=self.results_dir, name=name)
                }
                
                if self.device == 'cuda':
                    trainer_kwargs.update({
                        'accelerator': 'gpu',
                        'devices': 1
                    })
                else:
                    trainer_kwargs.update({
                        'accelerator': 'cpu',
                        'devices': 1  # Use 1 CPU core
                    })
                
                trainer = pl.Trainer(**trainer_kwargs)
                trainer.fit(model_instance, train_loader, val_loader)
            else:
                print(f"Skipping training for {name} (not a LightningModule).")

    def test_all_models(self, test_loader: DataLoader) -> pd.DataFrame:
        all_results = {}
        for name, model_instance in self.models.items():
            if isinstance(model_instance, pl.LightningModule):
                print(f"\nTesting {name} using PyTorch Lightning Trainer...")
                # Configure trainer based on device
                trainer_kwargs = {
                    'logger': False
                }
                
                if self.device == 'cuda':
                    trainer_kwargs.update({
                        'accelerator': 'gpu',
                        'devices': 1
                    })
                else:
                    trainer_kwargs.update({
                        'accelerator': 'cpu',
                        'devices': 1  # Use 1 CPU core
                    })
                
                trainer = pl.Trainer(**trainer_kwargs)
                test_metrics = trainer.test(model_instance, test_loader, verbose=False)
                all_results[name] = test_metrics[0] if test_metrics else {}
            else:
                print(f"Skipping testing for {name}.")

        results_df = pd.DataFrame.from_dict(all_results, orient='index')
        results_df.to_csv(os.path.join(self.results_dir, "model_comparison_test_results.csv"))
        print("\nModel Test Results:")
        print(results_df)
        return results_df

    def generate_comparison_report(self, test_loader: DataLoader):
        print("\nGenerating Model Comparison Report...")
        test_results_df = self.test_all_models(test_loader)
        
        if not test_results_df.empty:
            visualize_model_comparison(
                test_results_df,
                metrics_to_plot=[col for col in test_results_df.columns if 'loss' not in col.lower()],
                save_path=os.path.join(self.results_dir, "model_test_metrics_comparison.png")
            )
        
        print("\nReport generation completed.")

# --- Main analysis execution block (example) ---
if __name__ == "__main__":
    print("Coral Health Analysis Module")
    set_plotting_style()

    # --- Dummy Data Generation ---
    NUM_SAMPLES = 50; SEQ_LEN = 24; NUM_TS_FEATURES = 8; IMG_SIZE = 64
    dummy_train_images_pt = torch.randn(NUM_SAMPLES, 3, IMG_SIZE, IMG_SIZE)
    dummy_train_ts_pt = torch.randn(NUM_SAMPLES, SEQ_LEN, NUM_TS_FEATURES)
    # Generate both float and long labels for different models
    dummy_train_labels_float = torch.randint(0, 2, (NUM_SAMPLES,)).float()  # For BCEWithLogitsLoss
    dummy_train_labels_long = torch.randint(0, 2, (NUM_SAMPLES,)).long()    # For CrossEntropyLoss
    dummy_val_images_pt = torch.randn(NUM_SAMPLES//2, 3, IMG_SIZE, IMG_SIZE)
    dummy_val_ts_pt = torch.randn(NUM_SAMPLES//2, SEQ_LEN, NUM_TS_FEATURES)
    dummy_val_labels_float = torch.randint(0, 2, (NUM_SAMPLES//2,)).float()
    dummy_val_labels_long = torch.randint(0, 2, (NUM_SAMPLES//2,)).long()
    dummy_test_images_pt = torch.randn(NUM_SAMPLES//2, 3, IMG_SIZE, IMG_SIZE)
    dummy_test_ts_pt = torch.randn(NUM_SAMPLES//2, SEQ_LEN, NUM_TS_FEATURES)
    dummy_test_labels_float = torch.randint(0, 2, (NUM_SAMPLES//2,)).float()
    dummy_test_labels_long = torch.randint(0, 2, (NUM_SAMPLES//2,)).long()

    class DummyDataset(Dataset):
        def __init__(self, images, time_series, labels):
            self.images = images
            self.time_series = time_series
            self.labels = labels
            self.labels_float = labels.float()  # For BCE loss
            self.labels_long = labels.long()    # For CrossEntropy loss
        
        def __len__(self):
            return len(self.images)
        
        def __getitem__(self, idx):
            return {
                'image': self.images[idx],
                'time_series': self.time_series[idx],
                'label': self.labels[idx],
                'label_float': self.labels_float[idx],
                'label_long': self.labels_long[idx]
            }

    # Create dummy datasets
    dummy_train_dataset = DummyDataset(
        dummy_train_images_pt,
        dummy_train_ts_pt,
        dummy_train_labels_long
    )
    dummy_val_dataset = DummyDataset(
        dummy_val_images_pt,
        dummy_val_ts_pt,
        dummy_val_labels_long
    )
    dummy_test_dataset = DummyDataset(
        dummy_test_images_pt,
        dummy_test_ts_pt,
        dummy_test_labels_long
    )
    
    # Add num_workers to improve performance
    train_loader = DataLoader(dummy_train_dataset, batch_size=16, num_workers=4)
    val_loader = DataLoader(dummy_val_dataset, batch_size=16, num_workers=4)
    test_loader = DataLoader(dummy_test_dataset, batch_size=16, num_workers=4)

    dummy_train_images_np = np.random.randint(0, 255, (NUM_SAMPLES, IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    dummy_train_ts_np = np.random.randn(NUM_SAMPLES, SEQ_LEN, NUM_TS_FEATURES)
    dummy_train_labels_np = np.random.randint(0, 2, NUM_SAMPLES)
    
    # --- Feature Analyzer Example ---
    print("\n--- Feature Analyzer Example ---")
    models_comp_dir_main = './results_analysis_test/model_comparison'
    feat_analysis_dir_main = './results_analysis_test/feature_analysis'
    os.makedirs(models_comp_dir_main, exist_ok=True)
    # Simulate XGBoost feature extraction and importance file generation
    if xgb:
        xgb_feat_extractor = XGB_FeatureExtractor(wavelet_level=1, pretrained_backbone='resnet18') # Use smaller backbone for speed
        temp_xgb_features, temp_xgb_feat_names = xgb_feat_extractor.extract_all_features(dummy_train_images_np, dummy_train_ts_np)
        dummy_xgb_imp_df = pd.DataFrame({'feature': temp_xgb_feat_names, 'importance': np.random.rand(len(temp_xgb_feat_names))})
        dummy_xgb_imp_df.to_csv(os.path.join(models_comp_dir_main, 'xgboost_feature_importances.csv'), index=False)
        feature_analyzer = FeatureAnalyzer(models_dir=models_comp_dir_main, results_dir=feat_analysis_dir_main, feature_names_all=temp_xgb_feat_names)
        xgb_imp = feature_analyzer.get_xgboost_importance(top_n=10)
        if xgb_imp is not None: print("Top XGBoost features (from dummy file):\n", xgb_imp.head())
    
    # --- Early Warning Detector Example ---
    print("\n--- Early Warning Detector Example ---")
    ts_feature_names = [f'env_feat_{i}' for i in range(NUM_TS_FEATURES)]
    ews_detector = EarlyWarningDetector(results_dir=os.path.join(feat_analysis_dir_main, "ews_specific")) 
    if NUM_TS_FEATURES > 0:
        ews_detector.analyze_all_features_for_ews(dummy_train_ts_np, ts_feature_names, dummy_train_labels_np)
    
    # --- Model Evaluator Example ---
    print("\n--- Model Evaluator Example ---")
    cnn_lstm_params_eval = {'input_channels': 3, 'img_size': IMG_SIZE, 'time_input_dim': NUM_TS_FEATURES, 'seq_len': SEQ_LEN,
                       'cnn_backbone': 'resnet18', 'cnn_output_dim': 32, 'lstm_hidden_dim': 16, # Smaller
                       'lstm_layers': 1, 'lstm_output_dim': 16, 'wavelet_output_dim': 8, 'wavelet_level': 1,
                       'fusion_output_dim': 16, 'dropout': 0.1}
    tcn_params_eval = {'input_channels': 3, 'img_size': IMG_SIZE, 'time_input_dim': NUM_TS_FEATURES, 'sequence_length': SEQ_LEN,
                  'img_feature_dim': 32, 'time_feature_dim': 16, 'hidden_dims': [8, 8], # Smaller
                  'fusion_hidden_dim': 16, 'fusion_output_dim': 8, 'cnn_backbone': 'resnet18',
                  'kernel_size': 3, 'dropout': 0.1, 'use_attention': False}
    vit_params_eval = {'img_size': IMG_SIZE, 'patch_size': 16 if IMG_SIZE>=16 else 8, 'in_channels': 3, 
                  'time_input_dim': NUM_TS_FEATURES, 'max_len': SEQ_LEN,
                  'img_embed_dim': 24, 'time_embed_dim': 16, 'fusion_dim': 16, # Smaller, divisible
                  'img_num_heads': 3, 'time_num_heads': 2, 'fusion_num_heads': 2,
                  'img_ff_dim': 24*2, 'time_ff_dim': 16*2, 'img_num_layers': 1, 'time_num_layers': 1,
                  'num_classes': 2, 'dropout': 0.1, 'fusion_type': 'concat'}

    models_config_eval = {
        'cnn_lstm': {'model_params': cnn_lstm_params_eval, 'learning_rate': 1e-4},
        'tcn': {'model_params': tcn_params_eval, 'learning_rate': 1e-4},
        'transformer': {'model_params': vit_params_eval, 'learning_rate': 1e-4},
    }

    model_evaluator = ModelEvaluator(models_config=models_config_eval, results_dir=models_comp_dir_main)
    model_evaluator.train_all_models(train_loader, val_loader, num_epochs=1) 
    test_results = model_evaluator.test_all_models(test_loader)
    if not test_results.empty:
        model_evaluator.generate_comparison_report(test_loader)

    print("\nAnalysis script completed.")