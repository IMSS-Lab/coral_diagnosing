"""
Feature comparison module for coral bleaching prediction.

This module analyzes and compares features important for coral bleaching prediction.
It includes functions for:
- Identifying the most important features across different models
- Visualizing feature importance and relationships
- Analyzing feature changes over time before bleaching events
- Detecting subtle early warning signals in environmental parameters
- Ranking features by their predictive power for early warning
"""

import os
import numpy as np
import pandas as pd
import polars as plrs
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.inspection import permutation_importance
from scipy import stats
from scipy.signal import find_peaks
import torch
from torch.utils.data import DataLoader
import xgboost as xgb
from tqdm import tqdm

# Import models
from src.models.cnn_lstm import CoralLightningModel as CNNLSTMModel
from src.models.vit import CoralTransformerLightning as TransformerModel
from src.models.tcn import TCNLightningModel as TCNModel
from src.models.xgb import XGBoostCoralModel


class FeatureAnalyzer:
    """
    Analyzes features important for coral bleaching prediction.
    
    Identifies and visualizes key features across different models.
    """
    
    def __init__(
        self,
        models_dir: str = './results/model_comparison',
        results_dir: str = './results/feature_comparison',
        feature_names: Optional[List[str]] = None
    ):
        """
        Initialize feature analyzer.
        
        Args:
            models_dir: Directory containing trained models
            results_dir: Directory to save results
            feature_names: List of feature names (optional)
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        self.feature_names = feature_names
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load feature importance results if available
        self.feature_importance = self._load_feature_importance()
        
        # Early warning features
        self.early_warning_features = []
    
    def _load_feature_importance(self) -> Dict[str, pd.DataFrame]:
        """
        Load feature importance from files.
        
        Returns:
            Dictionary of feature importance dataframes
        """
        importance_dict = {}
        
        # Check if feature importance file exists for XGBoost
        xgb_importance_path = os.path.join(self.models_dir, 'feature_importance.csv')
        if os.path.exists(xgb_importance_path):
            importance_dict['xgboost'] = pd.read_csv(xgb_importance_path)
        
        return importance_dict
    
    def analyze_xgboost_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyze XGBoost feature importance.
        
        Args:
            top_n: Number of top features to retain
            
        Returns:
            DataFrame of feature importance
        """
        if 'xgboost' not in self.feature_importance:
            print("XGBoost feature importance not found!")
            return pd.DataFrame()
        
        importance_df = self.feature_importance['xgboost']
        
        # Sort by importance and take top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df, x='importance', y='feature')
        plt.title('Top Feature Importance (XGBoost)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'xgboost_feature_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def calculate_permutation_importance(
        self,
        model_name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        n_repeats: int = 10,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        Calculate permutation importance for a model.
        
        Args:
            model_name: Name of the model
            model: Model instance
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            n_repeats: Number of permutation repeats
            random_state: Random seed
            
        Returns:
            DataFrame of feature importance
        """
        print(f"Calculating permutation importance for {model_name}...")
        
        # Define prediction function
        if model_name == 'xgboost':
            def predict_fn(X_subset):
                dmatrix = xgb.DMatrix(X_subset)
                return model.predict(dmatrix)
        else:
            # PyTorch models
            def predict_fn(X_subset):
                X_tensor = torch.tensor(X_subset, dtype=torch.float32).to(next(model.parameters()).device)
                with torch.no_grad():
                    outputs = model(X_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    return torch.sigmoid(outputs).cpu().numpy()
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            predict_fn, X, y, 
            n_repeats=n_repeats,
            random_state=random_state
        )
        
        # Create dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance_mean', ascending=False)
        
        # Save to CSV
        importance_df.to_csv(os.path.join(self.results_dir, f'{model_name}_permutation_importance.csv'), index=False)
        
        # Visualize top 20
        plt.figure(figsize=(12, 8))
        sns.barplot(data=importance_df.head(20), x='importance_mean', y='feature')
        plt.title(f'Top 20 Feature Importance ({model_name} - Permutation)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'{model_name}_permutation_importance.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return importance_df
    
    def find_consistent_features(self, importance_dfs: Dict[str, pd.DataFrame], top_n: int = 10) -> pd.DataFrame:
        """
        Find features that are consistently important across models.
        
        Args:
            importance_dfs: Dictionary of feature importance dataframes
            top_n: Number of top features to consider from each model
            
        Returns:
            DataFrame of consistent features
        """
        print("Finding consistently important features...")
        
        # Check if we have enough models
        if len(importance_dfs) < 2:
            print("Need at least two models for comparison!")
            return pd.DataFrame()
        
        # Get top N features from each model
        top_features = {}
        for model_name, df in importance_dfs.items():
            top_features[model_name] = set(df.head(top_n)['feature'].tolist())
        
        # Find features common to all models
        common_features = set.intersection(*top_features.values())
        
        # Find features common to at least half of the models
        half_count = len(importance_dfs) // 2 + 1
        feature_counts = {}
        
        for feature_set in top_features.values():
            for feature in feature_set:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
        
        common_half = {feature for feature, count in feature_counts.items() if count >= half_count}
        
        # Create result dataframe
        result = {
            'feature': list(common_features.union(common_half)),
            'in_all_models': [feature in common_features for feature in common_features.union(common_half)],
            'model_count': [feature_counts[feature] for feature in common_features.union(common_half)]
        }
        
        # Add importance rank from each model
        for model_name, df in importance_dfs.items():
            # Create rank dictionary
            ranks = {feature: rank for rank, feature in enumerate(df['feature'].tolist(), 1)}
            
            # Add rank to result
            result[f'{model_name}_rank'] = [ranks.get(feature, float('nan')) for feature in result['feature']]
        
        # Create dataframe
        result_df = pd.DataFrame(result)
        
        # Sort by number of models and average rank
        result_df['avg_rank'] = result_df[[col for col in result_df.columns if col.endswith('_rank')]].mean(axis=1)
        result_df = result_df.sort_values(['model_count', 'avg_rank'], ascending=[False, True])
        
        # Save to CSV
        result_df.to_csv(os.path.join(self.results_dir, 'consistent_features.csv'), index=False)
        
        # Visualize
        plt.figure(figsize=(12, 8))
        sns.barplot(data=result_df.head(min(20, len(result_df))), x='model_count', y='feature')
        plt.title('Consistent Features Across Models')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'consistent_features.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return result_df
    
    def analyze_feature_correlations(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Analyze feature correlations with the target.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            DataFrame of feature correlations
        """
        print("Analyzing feature correlations...")
        
        # Calculate correlation with target
        correlations = []
        
        for i in range(X.shape[1]):
            corr = stats.pointbiserialr(X[:, i], y)[0]  # Point-biserial correlation for binary target
            correlations.append(abs(corr))  # Use absolute correlation
        
        # Create dataframe
        corr_df = pd.DataFrame({
            'feature': feature_names,
            'correlation': correlations
        })
        
        # Sort by correlation
        corr_df = corr_df.sort_values('correlation', ascending=False)
        
        # Save to CSV
        corr_df.to_csv(os.path.join(self.results_dir, 'feature_correlations.csv'), index=False)
        
        # Visualize top N
        plt.figure(figsize=(12, 8))
        sns.barplot(data=corr_df.head(top_n), x='correlation', y='feature')
        plt.title('Top Feature Correlations with Target')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return corr_df
    
    def analyze_mutual_information(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Analyze mutual information between features and target.
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            top_n: Number of top features to show
            
        Returns:
            DataFrame of mutual information
        """
        print("Analyzing mutual information...")
        
        # Calculate mutual information
        mi = mutual_info_classif(X, y, random_state=42)
        
        # Create dataframe
        mi_df = pd.DataFrame({
            'feature': feature_names,
            'mutual_info': mi
        })
        
        # Sort by mutual information
        mi_df = mi_df.sort_values('mutual_info', ascending=False)
        
        # Save to CSV
        mi_df.to_csv(os.path.join(self.results_dir, 'mutual_information.csv'), index=False)
        
        # Visualize top N
        plt.figure(figsize=(12, 8))
        sns.barplot(data=mi_df.head(top_n), x='mutual_info', y='feature')
        plt.title('Top Features by Mutual Information')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'mutual_information.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return mi_df
    
    def analyze_feature_changes(
        self,
        time_series: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        top_features: List[str],
        window_size: int = 5
    ) -> Dict[str, np.ndarray]:
        """
        Analyze how features change over time before bleaching events.
        
        Args:
            time_series: Time series data of shape [num_samples, time_steps, num_features]
            labels: Binary labels (0 for healthy, 1 for bleached)
            feature_names: List of feature names
            top_features: List of top feature names to analyze
            window_size: Window size for change detection
            
        Returns:
            Dictionary of feature change patterns
        """
        print("Analyzing feature changes before bleaching events...")
        
        # Get indices of bleached samples
        bleached_indices = np.where(labels == 1)[0]
        healthy_indices = np.where(labels == 0)[0]
        
        # Get feature indices
        feature_indices = {name: idx for idx, name in enumerate(feature_names)}
        
        # Extract top features
        top_indices = [feature_indices[name] for name in top_features if name in feature_indices]
        
        # Initialize result dictionary
        result = {
            'feature_names': [feature_names[idx] for idx in top_indices],
            'bleached_patterns': [],
            'healthy_patterns': []
        }
        
        # Analyze each top feature
        for feature_idx in top_indices:
            # Get feature data for bleached samples
            bleached_data = time_series[bleached_indices, :, feature_idx]
            healthy_data = time_series[healthy_indices, :, feature_idx]
            
            # Calculate average pattern
            bleached_pattern = np.mean(bleached_data, axis=0)
            healthy_pattern = np.mean(healthy_data, axis=0)
            
            # Store patterns
            result['bleached_patterns'].append(bleached_pattern)
            result['healthy_patterns'].append(healthy_pattern)
            
            # Plot pattern
            plt.figure(figsize=(10, 6))
            plt.plot(bleached_pattern, 'r-', label='Bleached')
            plt.plot(healthy_pattern, 'g-', label='Healthy')
            plt.title(f'Feature Pattern: {feature_names[feature_idx]}')
            plt.xlabel('Time Step')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(self.results_dir, f'feature_pattern_{feature_names[feature_idx]}.png'), dpi=300, bbox_inches='tight')
            plt.show()
        
        # Convert to numpy arrays
        result['bleached_patterns'] = np.array(result['bleached_patterns'])
        result['healthy_patterns'] = np.array(result['healthy_patterns'])
        
        # Save patterns to CSV
        patterns_df = pd.DataFrame()
        
        for i, feature_name in enumerate(result['feature_names']):
            patterns_df[f'{feature_name}_bleached'] = result['bleached_patterns'][i]
            patterns_df[f'{feature_name}_healthy'] = result['healthy_patterns'][i]
        
        patterns_df.to_csv(os.path.join(self.results_dir, 'feature_patterns.csv'), index=False)
        
        # Detect early warning signals
        self._detect_early_warning_signals(result, window_size)
        
        return result
    
    def _detect_early_warning_signals(self, patterns: Dict[str, Any], window_size: int = 5) -> None:
        """
        Detect early warning signals in feature patterns.
        
        Args:
            patterns: Dictionary of feature patterns
            window_size: Window size for detection
        """
        print("Detecting early warning signals...")
        
        early_warning_features = []
        
        for i, feature_name in enumerate(patterns['feature_names']):
            bleached_pattern = patterns['bleached_patterns'][i]
            healthy_pattern = patterns['healthy_patterns'][i]
            
            # Calculate difference pattern
            diff_pattern = bleached_pattern - healthy_pattern
            
            # Apply smoothing
            smoothed_diff = np.convolve(diff_pattern, np.ones(window_size)/window_size, mode='valid')
            
            # Find peaks and troughs
            peaks, _ = find_peaks(smoothed_diff)
            troughs, _ = find_peaks(-smoothed_diff)
            
            # Check for significant changes
            if len(peaks) > 0 or len(troughs) > 0:
                # Calculate rate of change
                rate_of_change = np.diff(smoothed_diff)
                
                # Check if early changes are present
                early_change = False
                early_idx = len(smoothed_diff) // 3  # First third of the time series
                
                if np.any(abs(rate_of_change[:early_idx]) > np.std(rate_of_change) * 1.5):
                    early_change = True
                    early_warning_features.append(feature_name)
                
                # Plot difference pattern with peaks and troughs
                plt.figure(figsize=(10, 6))
                plt.plot(smoothed_diff, 'b-', label='Difference (Bleached - Healthy)')
                plt.plot(peaks, smoothed_diff[peaks], 'ro', label='Peaks')
                plt.plot(troughs, smoothed_diff[troughs], 'go', label='Troughs')
                
                plt.title(f'Difference Pattern: {feature_name} (Early Warning: {early_change})')
                plt.xlabel('Time Step')
                plt.ylabel('Difference Value')
                plt.legend()
                plt.grid(True)
                plt.savefig(os.path.join(self.results_dir, f'difference_pattern_{feature_name}.png'), dpi=300, bbox_inches='tight')
                plt.show()
        
        # Save early warning features
        self.early_warning_features = early_warning_features
        
        if early_warning_features:
            print(f"Early warning features: {', '.join(early_warning_features)}")
            
            # Save to CSV
            pd.DataFrame({'feature': early_warning_features}).to_csv(
                os.path.join(self.results_dir, 'early_warning_features.csv'), 
                index=False
            )
    
    def rank_features(
        self,
        importance_dfs: Dict[str, pd.DataFrame],
        correlation_df: pd.DataFrame,
        mutual_info_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank features by combining multiple importance metrics.
        
        Args:
            importance_dfs: Dictionary of feature importance dataframes
            correlation_df: Feature correlation dataframe
            mutual_info_df: Mutual information dataframe
            
        Returns:
            DataFrame of ranked features
        """
        print("Ranking features...")
        
        # Get all unique features
        all_features = set()
        
        # Add features from importance dataframes
        for df in importance_dfs.values():
            all_features.update(df['feature'].tolist())
        
        # Add features from correlation and mutual info
        all_features.update(correlation_df['feature'].tolist())
        all_features.update(mutual_info_df['feature'].tolist())
        
        # Create feature ranking dictionary
        feature_ranks = {feature: {'count': 0, 'avg_rank': 0, 'ranks': []} for feature in all_features}
        
        # Add importance ranks
        for model_name, df in importance_dfs.items():
            for rank, feature in enumerate(df['feature'].tolist(), 1):
                feature_ranks[feature]['count'] += 1
                feature_ranks[feature]['ranks'].append(rank)
        
        # Add correlation rank
        for rank, feature in enumerate(correlation_df['feature'].tolist(), 1):
            feature_ranks[feature]['count'] += 1
            feature_ranks[feature]['ranks'].append(rank)
        
        # Add mutual info rank
        for rank, feature in enumerate(mutual_info_df['feature'].tolist(), 1):
            feature_ranks[feature]['count'] += 1
            feature_ranks[feature]['ranks'].append(rank)
        
        # Calculate average rank
        for feature, data in feature_ranks.items():
            if data['ranks']:
                data['avg_rank'] = sum(data['ranks']) / len(data['ranks'])
            else:
                data['avg_rank'] = float('inf')
        
        # Create dataframe
        rank_df = pd.DataFrame({
            'feature': list(feature_ranks.keys()),
            'appearance_count': [data['count'] for data in feature_ranks.values()],
            'avg_rank': [data['avg_rank'] for data in feature_ranks.values()]
        })
        
        # Add early warning flag
        rank_df['early_warning'] = rank_df['feature'].isin(self.early_warning_features)
        
        # Sort by appearance count and average rank
        rank_df = rank_df.sort_values(['appearance_count', 'avg_rank'], ascending=[False, True])
        
        # Save to CSV
        rank_df.to_csv(os.path.join(self.results_dir, 'features_ranked.csv'), index=False)
        
        # Visualize top 20
        plt.figure(figsize=(12, 8))
        sns.scatterplot(
            data=rank_df.head(min(50, len(rank_df))),
            x='avg_rank',
            y='appearance_count',
            hue='early_warning',
            size='early_warning',
            sizes=[50, 200],
            alpha=0.7
        )
        
        # Add feature labels
        for _, row in rank_df.head(min(20, len(rank_df))).iterrows():
            plt.text(
                row['avg_rank'] + 0.1,
                row['appearance_count'],
                row['feature'],
                fontsize=8
            )
        
        plt.title('Feature Ranking')
        plt.xlabel('Average Rank (lower is better)')
        plt.ylabel('Number of Appearances')
        plt.grid(True)
        plt.savefig(os.path.join(self.results_dir, 'feature_ranking.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return rank_df


def compare_features(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    models_dir: str = './results/model_comparison',
    results_dir: str = './results/feature_comparison'
) -> pd.DataFrame:
    """
    Compare features important for coral bleaching prediction.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        models_dir: Directory containing trained models
        results_dir: Directory to save results
        
    Returns:
        DataFrame of ranked features
    """
    # Initialize feature analyzer
    analyzer = FeatureAnalyzer(models_dir, results_dir, feature_names)
    
    # Analyze XGBoost feature importance
    xgb_importance = analyzer.analyze_xgboost_importance()
    
    # Extract feature matrix from time series (for simplicity, flatten time series)
    X = time_series.reshape(time_series.shape[0], -1)
    feature_names_full = [f"{name}_{t}" for name in feature_names for t in range(time_series.shape[1])]
    
    # Analyze feature correlations
    correlation_df = analyzer.analyze_feature_correlations(X, labels, feature_names_full)
    
    # Analyze mutual information
    mutual_info_df = analyzer.analyze_mutual_information(X, labels, feature_names_full)
    
    # Combine importance dataframes
    importance_dfs = {'xgboost': xgb_importance}
    
    # Find consistent features
    consistent_features = analyzer.find_consistent_features(importance_dfs)
    
    # Analyze feature changes
    if not consistent_features.empty:
        top_features = consistent_features['feature'].head(10).tolist()
    else:
        # Use top features from XGBoost if no consistent features
        top_features = xgb_importance['feature'].head(10).tolist() if not xgb_importance.empty else []
    
    # Filter to only include features in the original feature_names
    top_features = [f for f in top_features if f in feature_names]
    
    if top_features:
        analyzer.analyze_feature_changes(time_series, labels, feature_names, top_features)
    
    # Rank features
    rank_df = analyzer.rank_features(importance_dfs, correlation_df, mutual_info_df)
    
    return rank_df


if __name__ == "__main__":
    import sys
    sys.path.append('..')  # Add parent directory to path
    import preprocessing
    
    # Load data
    data_dir = "../data/processed"
    train_data, val_data, test_data, _ = preprocessing.load_processed_data(data_dir)
    
    # Define feature names (example)
    feature_names = [
        'temperature', 'salinity', 'ph', 'dissolved_oxygen',
        'turbidity', 'chlorophyll', 'nitrate', 'phosphate'
    ]
    
    # Compare features
    rank_df = compare_features(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names
    )
    
    print("Top 10 features:")
    print(rank_df.head(10))