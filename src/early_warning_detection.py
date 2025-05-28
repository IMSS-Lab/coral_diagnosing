"""
Early warning detection module for coral bleaching prediction.

This module focuses on detecting subtle changes in environmental parameters
that indicate an impending coral bleaching event. It implements various
early warning signal (EWS) methods from the scientific literature on critical transitions.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import signal, stats
from sklearn.ensemble import IsolationForest
from tqdm import tqdm
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.nonparametric.smoothers_lowess import lowess


class EarlyWarningDetector:
    """
    Detector for early warning signals of coral bleaching events.
    
    Implements various methods for detecting critical transitions.
    """
    
    def __init__(
        self,
        results_dir: str = './results/feature_comparison',
        window_size: int = 5,
        detrend: bool = True,
        smoothing: bool = True
    ):
        """
        Initialize early warning detector.
        
        Args:
            results_dir: Directory to save results
            window_size: Window size for rolling statistics
            detrend: Whether to detrend time series before analysis
            smoothing: Whether to apply smoothing
        """
        self.results_dir = results_dir
        self.window_size = window_size
        self.detrend = detrend
        self.smoothing = smoothing
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
        
        # Load top features if available
        self.top_features = self._load_top_features()
    
    def _load_top_features(self) -> List[str]:
        """
        Load top features from file.
        
        Returns:
            List of top feature names
        """
        feature_path = os.path.join(self.results_dir, 'early_warning_features.csv')
        if os.path.exists(feature_path):
            df = pd.read_csv(feature_path)
            return df['feature'].tolist()
        else:
            return []
    
    def preprocess_timeseries(self, ts: np.ndarray) -> np.ndarray:
        """
        Preprocess time series data.
        
        Args:
            ts: Time series data
            
        Returns:
            Preprocessed time series
        """
        # Apply detrending if enabled
        if self.detrend:
            ts_detrended = signal.detrend(ts)
        else:
            ts_detrended = ts
        
        # Apply smoothing if enabled
        if self.smoothing:
            ts_smoothed = np.convolve(ts_detrended, np.ones(self.window_size)/self.window_size, mode='valid')
            
            # Pad ends to maintain original length
            pad_size = len(ts) - len(ts_smoothed)
            pad_left = pad_size // 2
            pad_right = pad_size - pad_left
            ts_processed = np.pad(ts_smoothed, (pad_left, pad_right), mode='edge')
        else:
            ts_processed = ts_detrended
        
        return ts_processed
    
    def calculate_ar1(self, ts: np.ndarray) -> float:
        """
        Calculate lag-1 autocorrelation.
        
        Args:
            ts: Time series data
            
        Returns:
            Lag-1 autocorrelation coefficient
        """
        if len(ts) < 2:
            return 0
        
        # Calculate autocorrelation at lag 1
        return np.corrcoef(ts[:-1], ts[1:])[0, 1]
    
    def calculate_variance(self, ts: np.ndarray) -> float:
        """
        Calculate variance.
        
        Args:
            ts: Time series data
            
        Returns:
            Variance
        """
        return np.var(ts)
    
    def calculate_skewness(self, ts: np.ndarray) -> float:
        """
        Calculate skewness.
        
        Args:
            ts: Time series data
            
        Returns:
            Skewness
        """
        if len(ts) < 2:
            return 0
        
        return stats.skew(ts)
    
    def calculate_kurtosis(self, ts: np.ndarray) -> float:
        """
        Calculate kurtosis.
        
        Args:
            ts: Time series data
            
        Returns:
            Kurtosis
        """
        if len(ts) < 2:
            return 0
        
        return stats.kurtosis(ts)
    
    def calculate_cv(self, ts: np.ndarray) -> float:
        """
        Calculate coefficient of variation.
        
        Args:
            ts: Time series data
            
        Returns:
            Coefficient of variation
        """
        if np.mean(ts) == 0:
            return 0
        
        return np.std(ts) / np.mean(ts)
    
    def calculate_return_rate(self, ts: np.ndarray) -> float:
        """
        Calculate return rate.
        
        Args:
            ts: Time series data
            
        Returns:
            Return rate
        """
        if len(ts) < 2:
            return 0
        
        diffs = np.diff(ts)
        return -np.corrcoef(ts[:-1], diffs)[0, 1]
    
    def calculate_density_ratio(self, ts: np.ndarray) -> float:
        """
        Calculate spectral density ratio.
        
        Args:
            ts: Time series data
            
        Returns:
            Spectral density ratio
        """
        if len(ts) < 4:
            return 0
        
        # Calculate periodogram
        freqs, psd = signal.periodogram(ts)
        
        if len(freqs) < 4:
            return 0
        
        # Calculate low and high frequency power
        low_idx = len(freqs) // 4
        low_power = np.sum(psd[:low_idx])
        high_power = np.sum(psd[low_idx:])
        
        if high_power == 0:
            return 0
        
        return low_power / high_power
    
    def calculate_rolling_statistics(
        self,
        ts: np.ndarray,
        window_size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate rolling statistics.
        
        Args:
            ts: Time series data
            window_size: Window size for rolling statistics (optional)
            
        Returns:
            Dictionary of rolling statistics
        """
        if window_size is None:
            window_size = self.window_size
        
        # Ensure minimum window size
        window_size = max(2, min(window_size, len(ts) - 1))
        
        # Initialize arrays
        n_windows = len(ts) - window_size + 1
        ar1 = np.zeros(n_windows)
        variance = np.zeros(n_windows)
        skewness = np.zeros(n_windows)
        kurtosis = np.zeros(n_windows)
        cv = np.zeros(n_windows)
        return_rate = np.zeros(n_windows)
        density_ratio = np.zeros(n_windows)
        
        # Calculate rolling statistics
        for i in range(n_windows):
            window = ts[i:i+window_size]
            window_processed = self.preprocess_timeseries(window)
            
            ar1[i] = self.calculate_ar1(window_processed)
            variance[i] = self.calculate_variance(window_processed)
            skewness[i] = self.calculate_skewness(window_processed)
            kurtosis[i] = self.calculate_kurtosis(window_processed)
            cv[i] = self.calculate_cv(window_processed)
            return_rate[i] = self.calculate_return_rate(window_processed)
            density_ratio[i] = self.calculate_density_ratio(window_processed)
        
        return {
            'ar1': ar1,
            'variance': variance,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'cv': cv,
            'return_rate': return_rate,
            'density_ratio': density_ratio
        }
    
    def detect_critical_transition(
        self,
        ts: np.ndarray,
        threshold: float = 2.0
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect critical transition in time series.
        
        Args:
            ts: Time series data
            threshold: Z-score threshold for detection
            
        Returns:
            Tuple of (detected flag, results dictionary)
        """
        # Preprocess time series
        ts_processed = self.preprocess_timeseries(ts)
        
        # Calculate rolling statistics
        rolling_stats = self.calculate_rolling_statistics(ts_processed)
        
        # Calculate z-scores for last window compared to early windows
        early_end = len(ts) // 2
        results = {}
        
        for stat_name, stat_values in rolling_stats.items():
            if len(stat_values) > early_end:
                early_values = stat_values[:early_end]
                late_values = stat_values[early_end:]
                
                # Calculate mean and std of early values
                early_mean = np.mean(early_values)
                early_std = np.std(early_values)
                
                # Avoid division by zero
                if early_std == 0:
                    early_std = 1e-10
                
                # Calculate z-scores for late values
                z_scores = (late_values - early_mean) / early_std
                
                # Store results
                results[stat_name] = {
                    'early_mean': early_mean,
                    'early_std': early_std,
                    'z_scores': z_scores,
                    'max_z': np.max(np.abs(z_scores)),
                    'exceeds_threshold': np.any(np.abs(z_scores) > threshold)
                }
        
        # Detect critical transition if any statistic exceeds threshold
        detected = any(result['exceeds_threshold'] for result in results.values())
        
        return detected, results
    
    def detect_anomalies(
        self,
        ts: np.ndarray,
        contamination: float = 0.1
    ) -> np.ndarray:
        """
        Detect anomalies in time series using Isolation Forest.
        
        Args:
            ts: Time series data
            contamination: Expected proportion of anomalies
            
        Returns:
            Binary array indicating anomalies
        """
        if len(ts) < 10:
            return np.zeros(len(ts), dtype=bool)
        
        # Preprocess time series
        ts_processed = self.preprocess_timeseries(ts)
        
        # Create features
        X = np.zeros((len(ts_processed) - 1, 2))
        X[:, 0] = ts_processed[:-1]  # Current value
        X[:, 1] = ts_processed[1:]   # Next value
        
        # Detect anomalies
        clf = IsolationForest(contamination=contamination, random_state=42)
        y_pred = clf.fit_predict(X)
        
        # Convert to binary array (1 for anomalies, 0 for normal)
        anomalies = np.zeros(len(ts), dtype=bool)
        anomalies[1:] = (y_pred == -1)
        
        return anomalies
    
    def analyze_feature(
        self,
        feature_data: np.ndarray,
        feature_name: str,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Analyze a feature for early warning signals.
        
        Args:
            feature_data: Feature data of shape [num_samples, time_steps]
            feature_name: Feature name
            labels: Binary labels (0 for healthy, 1 for bleached)
            
        Returns:
            Dictionary of analysis results
        """
        print(f"Analyzing feature: {feature_name}")
        
        # Get indices of bleached and healthy samples
        bleached_indices = np.where(labels == 1)[0]
        healthy_indices = np.where(labels == 0)[0]
        
        # Calculate average patterns
        bleached_avg = np.mean(feature_data[bleached_indices], axis=0)
        healthy_avg = np.mean(feature_data[healthy_indices], axis=0)
        
        # Calculate difference pattern
        diff_pattern = bleached_avg - healthy_avg
        
        # Detect critical transition
        transition_detected, transition_results = self.detect_critical_transition(bleached_avg)
        
        # Detect anomalies
        bleached_anomalies = self.detect_anomalies(bleached_avg)
        healthy_anomalies = self.detect_anomalies(healthy_avg)
        
        # Calculate early warning metrics for individual samples
        bleached_metrics = []
        healthy_metrics = []
        
        for idx in tqdm(bleached_indices, desc=f"Processing {feature_name} (bleached)"):
            ts = feature_data[idx]
            _, results = self.detect_critical_transition(ts)
            bleached_metrics.append(results)
        
        for idx in tqdm(healthy_indices, desc=f"Processing {feature_name} (healthy)"):
            ts = feature_data[idx]
            _, results = self.detect_critical_transition(ts)
            healthy_metrics.append(results)
        
        # Aggregate metrics
        bleached_agg = self._aggregate_metrics(bleached_metrics)
        healthy_agg = self._aggregate_metrics(healthy_metrics)
        
        # Visualize results
        self._visualize_feature_analysis(
            feature_name=feature_name,
            bleached_avg=bleached_avg,
            healthy_avg=healthy_avg,
            diff_pattern=diff_pattern,
            transition_results=transition_results,
            bleached_anomalies=bleached_anomalies,
            healthy_anomalies=healthy_anomalies,
            bleached_agg=bleached_agg,
            healthy_agg=healthy_agg
        )
        
        # Compile results
        results = {
            'feature_name': feature_name,
            'transition_detected': transition_detected,
            'transition_results': transition_results,
            'bleached_anomalies': bleached_anomalies,
            'healthy_anomalies': healthy_anomalies,
            'bleached_metrics': bleached_metrics,
            'healthy_metrics': healthy_metrics,
            'bleached_agg': bleached_agg,
            'healthy_agg': healthy_agg
        }
        
        return results
    
    def _aggregate_metrics(self, metrics_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """
        Aggregate metrics across samples.
        
        Args:
            metrics_list: List of metrics dictionaries
            
        Returns:
            Dictionary of aggregated metrics
        """
        if not metrics_list:
            return {}
        
        # Initialize aggregated metrics
        aggregated = {}
        
        # Get all metric names
        metric_names = metrics_list[0].keys()
        
        for metric_name in metric_names:
            # Extract values for this metric
            values = [m[metric_name]['max_z'] for m in metrics_list if metric_name in m]
            
            # Calculate statistics
            aggregated[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values),
                'exceeds_threshold': np.mean([m[metric_name]['exceeds_threshold'] for m in metrics_list if metric_name in m])
            }
        
        return aggregated
    
    def _visualize_feature_analysis(
        self,
        feature_name: str,
        bleached_avg: np.ndarray,
        healthy_avg: np.ndarray,
        diff_pattern: np.ndarray,
        transition_results: Dict[str, Any],
        bleached_anomalies: np.ndarray,
        healthy_anomalies: np.ndarray,
        bleached_agg: Dict[str, Dict[str, float]],
        healthy_agg: Dict[str, Dict[str, float]]
    ) -> None:
        """
        Visualize feature analysis results.
        
        Args:
            feature_name: Feature name
            bleached_avg: Average pattern for bleached samples
            healthy_avg: Average pattern for healthy samples
            diff_pattern: Difference pattern
            transition_results: Critical transition results
            bleached_anomalies: Anomalies in bleached samples
            healthy_anomalies: Anomalies in healthy samples
            bleached_agg: Aggregated metrics for bleached samples
            healthy_agg: Aggregated metrics for healthy samples
        """
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 16))
        
        # Plot average patterns
        ax1 = fig.add_subplot(3, 2, 1)
        ax1.plot(bleached_avg, 'r-', label='Bleached')
        ax1.plot(healthy_avg, 'g-', label='Healthy')
        
        # Highlight anomalies
        ax1.scatter(
            np.where(bleached_anomalies)[0],
            bleached_avg[bleached_anomalies],
            color='red',
            marker='o',
            s=100,
            label='Bleached Anomalies'
        )
        ax1.scatter(
            np.where(healthy_anomalies)[0],
            healthy_avg[healthy_anomalies],
            color='green',
            marker='o',
            s=100,
            label='Healthy Anomalies'
        )
        
        ax1.set_title(f'Average Patterns: {feature_name}')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot difference pattern
        ax2 = fig.add_subplot(3, 2, 2)
        ax2.plot(diff_pattern, 'b-')
        ax2.axhline(y=0, color='k', linestyle='--')
        ax2.set_title(f'Difference Pattern: {feature_name}')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Difference Value')
        ax2.grid(True)
        
        # Plot rolling statistics
        ax3 = fig.add_subplot(3, 2, 3)
        
        for stat_name, result in transition_results.items():
            if stat_name in ['ar1', 'variance', 'return_rate']:
                ax3.plot(result['z_scores'], label=f"{stat_name} (max z={result['max_z']:.2f})")
        
        ax3.axhline(y=2.0, color='r', linestyle='--', label='Threshold')
        ax3.axhline(y=-2.0, color='r', linestyle='--')
        ax3.set_title(f'Z-Scores of Key EWS Metrics: {feature_name}')
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Z-Score')
        ax3.legend()
        ax3.grid(True)
        
        # Plot aggregated metrics comparison
        ax4 = fig.add_subplot(3, 2, 4)
        
        metric_names = list(bleached_agg.keys())
        x = np.arange(len(metric_names))
        width = 0.35
        
        ax4.bar(
            x - width/2,
            [bleached_agg[m]['mean'] for m in metric_names],
            width,
            label='Bleached',
            color='red',
            alpha=0.7
        )
        ax4.bar(
            x + width/2,
            [healthy_agg[m]['mean'] for m in metric_names],
            width,
            label='Healthy',
            color='green',
            alpha=0.7
        )
        
        ax4.set_xticks(x)
        ax4.set_xticklabels(metric_names, rotation=45)
        ax4.set_title(f'Comparison of EWS Metrics: {feature_name}')
        ax4.set_ylabel('Mean Max Z-Score')
        ax4.legend()
        ax4.grid(True)
        
        # Plot autocorrelation
        ax5 = fig.add_subplot(3, 2, 5)
        
        # Calculate autocorrelation
        bleached_acf = acf(bleached_avg, nlags=min(40, len(bleached_avg) // 2))
        healthy_acf = acf(healthy_avg, nlags=min(40, len(healthy_avg) // 2))
        
        ax5.plot(bleached_acf, 'r-', label='Bleached')
        ax5.plot(healthy_acf, 'g-', label='Healthy')
        ax5.set_title(f'Autocorrelation: {feature_name}')
        ax5.set_xlabel('Lag')
        ax5.set_ylabel('Autocorrelation')
        ax5.legend()
        ax5.grid(True)
        
        # Plot power spectrum
        ax6 = fig.add_subplot(3, 2, 6)
        
        # Calculate power spectrum
        bleached_freq, bleached_psd = signal.periodogram(bleached_avg)
        healthy_freq, healthy_psd = signal.periodogram(healthy_avg)
        
        ax6.semilogy(bleached_freq, bleached_psd, 'r-', label='Bleached')
        ax6.semilogy(healthy_freq, healthy_psd, 'g-', label='Healthy')
        ax6.set_title(f'Power Spectrum: {feature_name}')
        ax6.set_xlabel('Frequency')
        ax6.set_ylabel('Power Spectral Density')
        ax6.legend()
        ax6.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f'ews_analysis_{feature_name}.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_all_features(
        self,
        feature_data: np.ndarray,
        feature_names: List[str],
        labels: np.ndarray,
        top_n: Optional[int] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Analyze all features for early warning signals.
        
        Args:
            feature_data: Feature data of shape [num_samples, time_steps, num_features]
            feature_names: List of feature names
            labels: Binary labels (0 for healthy, 1 for bleached)
            top_n: Number of top features to analyze (optional)
            
        Returns:
            Dictionary of analysis results for each feature
        """
        # Determine features to analyze
        if self.top_features and top_n is None:
            # Use pre-loaded top features
            analyze_features = self.top_features
            feature_indices = [feature_names.index(name) for name in analyze_features if name in feature_names]
        elif top_n is not None:
            # Use top N features
            feature_indices = list(range(min(top_n, len(feature_names))))
            analyze_features = [feature_names[i] for i in feature_indices]
        else:
            # Analyze all features
            feature_indices = list(range(len(feature_names)))
            analyze_features = feature_names
        
        print(f"Analyzing {len(analyze_features)} features: {analyze_features}")
        
        # Analyze each feature
        results = {}
        
        for idx, name in zip(feature_indices, analyze_features):
            feature_series = feature_data[:, :, idx]
            results[name] = self.analyze_feature(feature_series, name, labels)
        
        # Summarize results
        self._summarize_results(results)
        
        return results
    
    def _summarize_results(self, results: Dict[str, Dict[str, Any]]) -> None:
        """
        Summarize analysis results.
        
        Args:
            results: Dictionary of analysis results for each feature
        """
        # Create summary dataframe
        summary_data = []
        
        for feature_name, result in results.items():
            # Get key metrics
            transition_detected = result['transition_detected']
            
            # Get max z-scores
            max_z_scores = {
                metric: result['transition_results'][metric]['max_z']
                for metric in result['transition_results']
            }
            
            # Get exceedance rates
            exceedance_rates = {
                metric: result['bleached_agg'][metric]['exceeds_threshold'] if metric in result['bleached_agg'] else 0
                for metric in result['transition_results']
            }
            
            # Add to summary data
            summary_data.append({
                'feature': feature_name,
                'transition_detected': transition_detected,
                **{f"{metric}_max_z": score for metric, score in max_z_scores.items()},
                **{f"{metric}_exceedance": rate for metric, rate in exceedance_rates.items()}
            })
        
        # Create dataframe
        summary_df = pd.DataFrame(summary_data)
        
        # Sort by whether transition was detected, then by max z-score of ar1
        if 'ar1_max_z' in summary_df.columns:
            summary_df = summary_df.sort_values(['transition_detected', 'ar1_max_z'], ascending=[False, False])
        else:
            summary_df = summary_df.sort_values('transition_detected', ascending=False)
        
        # Save to CSV
        summary_df.to_csv(os.path.join(self.results_dir, 'ews_summary.csv'), index=False)
        
        # Visualize summary
        self._visualize_summary(summary_df)
    
    def _visualize_summary(self, summary_df: pd.DataFrame) -> None:
        """
        Visualize summary results.
        
        Args:
            summary_df: Summary dataframe
        """
        # Get columns for max z-scores
        z_score_cols = [col for col in summary_df.columns if col.endswith('_max_z')]
        
        if not z_score_cols:
            return
        
        # Create heatmap data
        heatmap_data = summary_df.sort_values('transition_detected', ascending=False)[['feature'] + z_score_cols]
        heatmap_data = heatmap_data.set_index('feature')
        
        # Rename columns for better display
        heatmap_data.columns = [col.replace('_max_z', '') for col in heatmap_data.columns]
        
        # Create heatmap
        plt.figure(figsize=(12, max(8, len(summary_df) * 0.4)))
        sns.heatmap(
            heatmap_data,
            cmap='YlOrRd',
            annot=True,
            fmt='.2f',
            linewidths=.5
        )
        plt.title('Max Z-Scores of Early Warning Signals by Feature')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'ews_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Create bar chart for overall ranking
        if len(summary_df) > 0:
            # Calculate overall score (mean of z-scores)
            summary_df['overall_score'] = summary_df[z_score_cols].mean(axis=1)
            
            # Sort by overall score
            plot_df = summary_df.sort_values('overall_score', ascending=False)
            
            # Plot bar chart
            plt.figure(figsize=(12, max(8, len(summary_df) * 0.4)))
            bars = plt.barh(plot_df['feature'], plot_df['overall_score'])
            
            # Color bars by whether transition was detected
            for i, detected in enumerate(plot_df['transition_detected']):
                bars[i].set_color('red' if detected else 'blue')
            
            plt.xlabel('Overall EWS Score')
            plt.title('Features Ranked by Early Warning Signal Strength')
            plt.grid(True, axis='x')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'ews_ranking.png'), dpi=300, bbox_inches='tight')
            plt.show()


def detect_early_warnings(
    time_series: np.ndarray,
    labels: np.ndarray,
    feature_names: List[str],
    results_dir: str = './results/feature_comparison',
    window_size: int = 5,
    detrend: bool = True,
    smoothing: bool = True,
    top_n: Optional[int] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Detect early warning signals of coral bleaching.
    
    Args:
        time_series: Time series data of shape [num_samples, time_steps, num_features]
        labels: Binary labels (0 for healthy, 1 for bleached)
        feature_names: List of feature names
        results_dir: Directory to save results
        window_size: Window size for rolling statistics
        detrend: Whether to detrend time series before analysis
        smoothing: Whether to apply smoothing
        top_n: Number of top features to analyze (optional)
        
    Returns:
        Dictionary of analysis results for each feature
    """
    # Initialize early warning detector
    detector = EarlyWarningDetector(
        results_dir=results_dir,
        window_size=window_size,
        detrend=detrend,
        smoothing=smoothing
    )
    
    # Analyze all features
    results = detector.analyze_all_features(time_series, feature_names, labels, top_n)
    
    return results


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
    
    # Detect early warning signals
    results = detect_early_warnings(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        top_n=None  # Analyze all features
    )