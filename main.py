"""
Main execution script for Duke Coral Health project.

This script orchestrates the entire pipeline for coral bleaching prediction:
1. Data preprocessing
2. Feature engineering
3. Model training and comparison
4. Feature analysis
5. Early warning signal detection
6. Visualization and reporting

The pipeline can be executed end-to-end or specific components can be run
by passing appropriate command-line arguments.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import yaml
import json
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union

# Import project modules
from analysis.preprocessing import (
    preprocess_pipeline, 
    load_processed_data, 
    create_data_loaders
)
from analysis.feature_engineering import (
    extract_all_features,
    select_best_features,
    visualize_feature_importance,
    save_features
)
from analysis.compare_models import (
    compare_models
)
from analysis.compare_features import (
    compare_features
)
from analysis.early_warning_detection import (
    detect_early_warnings
)
from analysis.visualization import (
    visualize_time_series,
    visualize_feature_importance,
    visualize_feature_distribution,
    visualize_correlation_matrix,
    visualize_learning_curves,
    visualize_model_comparison,
    visualize_confusion_matrices,
    visualize_roc_curves,
    visualize_pr_curves,
    visualize_feature_changes_over_time,
    visualize_early_warning_signals,
    create_comprehensive_report
)

# Import model classes
from models.cnn_lstm_attention import CoralNet as CNNLSTMModel
from models.vit import DualTransformerModel 
from models.tcn import TCNCoralModel
from models.xgboost_model import XGBoostCoralModel
from models.ensemble import EnsembleModel


# Load configuration
def load_config(config_path: str = './config/config.yaml') -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    print(f"Loading configuration from {config_path}")
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        print("Configuration loaded successfully")
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        # Return default configuration
        return {
            'data': {
                'data_dir': './data',
                'processed_dir': './data/processed',
                'results_dir': './results',
                'use_cached_data': True,
                'test_size': 0.2,
                'val_size': 0.2,
                'random_seed': 42
            },
            'features': {
                'feature_selection': True,
                'num_features': 100,
                'feature_selection_method': 'mutual_info'
            },
            'models': {
                'batch_size': 32,
                'num_epochs': 20,
                'early_stopping': True,
                'use_wavelet': True,
                'learning_rate': 0.001,
                'weight_decay': 1e-4,
                'model_configs': {
                    'cnn_lstm': {
                        'enabled': True,
                        'backbone': 'efficientnet_b0',
                        'hidden_dim': 64,
                        'dropout': 0.3
                    },
                    'transformer': {
                        'enabled': True,
                        'embed_dim': 384,
                        'vision_depth': 6,
                        'temporal_depth': 4,
                        'num_heads': 8
                    },
                    'tcn': {
                        'enabled': True,
                        'hidden_dims': [64, 128, 256],
                        'kernel_size': 3
                    },
                    'xgboost': {
                        'enabled': True,
                        'n_estimators': 100,
                        'learning_rate': 0.1,
                        'max_depth': 5
                    },
                    'ensemble': {
                        'enabled': True,
                        'weights': {
                            'cnn_lstm': 0.3,
                            'transformer': 0.3,
                            'tcn': 0.2,
                            'xgboost': 0.2
                        }
                    }
                }
            },
            'early_warning': {
                'window_size': 5,
                'detrend': True,
                'smoothing': True
            },
            'visualization': {
                'save_visualizations': True
            }
        }


# Data preprocessing step
def run_preprocessing(
    config: Dict[str, Any]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, Any]]:
    """
    Run data preprocessing pipeline.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Tuple of (train_data, val_data, test_data, normalization_params)
    """
    print("Starting data preprocessing step")
    
    data_config = config['data']
    data_dir = data_config['data_dir']
    processed_dir = data_config['processed_dir']
    use_cached = data_config['use_cached_data']
    test_size = data_config['test_size']
    val_size = data_config['val_size']
    random_seed = data_config['random_seed']
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)
    
    # Check if processed data exists
    if use_cached and os.path.exists(processed_dir):
        print("Loading preprocessed data from cache")
        train_data, val_data, test_data, norm_params = load_processed_data(processed_dir)
    else:
        print("Processing raw data")
        # Define feature names - example, replace with actual feature names from your data
        required_features = [
            'temperature', 'salinity', 'ph', 'dissolved_oxygen',
            'turbidity', 'chlorophyll', 'nitrate', 'phosphate'
        ]
        
        # Run preprocessing pipeline
        train_data, val_data, test_data, norm_params = preprocess_pipeline(
            data_dir=data_dir,
            save_dir=processed_dir,
            required_features=required_features,
            test_size=test_size,
            val_size=val_size
        )
    
    print(f"Preprocessing complete. Dataset sizes - Train: {len(train_data['labels'])}, "
          f"Val: {len(val_data['labels'])}, Test: {len(test_data['labels'])}")
    
    return train_data, val_data, test_data, norm_params


# Feature engineering step
def run_feature_engineering(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> Tuple[np.ndarray, List[str]]:
    """
    Run feature engineering pipeline.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        config: Configuration dictionary
        
    Returns:
        Tuple of (selected_features, selected_feature_names)
    """
    print("Starting feature engineering step")
    
    feature_config = config['features']
    results_dir = config['data']['results_dir']
    
    # Create results directory if it doesn't exist
    features_dir = os.path.join(results_dir, 'feature_engineering')
    os.makedirs(features_dir, exist_ok=True)
    
    # Extract features from training data
    print("Extracting features from all data")
    train_features, feature_names = extract_all_features(
        images=train_data['images'],
        timeseries=train_data['timeseries']
    )
    
    # Save all extracted features
    save_features(
        features=train_features,
        feature_names=feature_names,
        labels=train_data['labels'],
        save_path=os.path.join(features_dir, 'train_features_all.csv')
    )
    
    # Perform feature selection if enabled
    if feature_config['feature_selection']:
        print("Performing feature selection")
        selected_features, selected_names = select_best_features(
            features=train_features,
            labels=train_data['labels'],
            feature_names=feature_names,
            k=feature_config['num_features'],
            method=feature_config['feature_selection_method']
        )
        
        # Save selected features
        save_features(
            features=selected_features,
            feature_names=selected_names,
            labels=train_data['labels'],
            save_path=os.path.join(features_dir, 'train_features_selected.csv')
        )
        
        # Visualize feature importance
        visualize_feature_importance(
            importance_df=pd.DataFrame({
                'feature': selected_names,
                'importance': np.random.rand(len(selected_names))  # Placeholder, will be updated by model
            }),
            save_path=os.path.join(features_dir, 'feature_importance_initial.png')
        )
        
        return selected_features, selected_names
    else:
        return train_features, feature_names


# Model training and comparison step
def run_model_comparison(
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run model training and comparison.
    
    Args:
        train_data: Training data
        val_data: Validation data
        test_data: Test data
        config: Configuration dictionary
        
    Returns:
        Dictionary of model comparison results
    """
    print("Starting model training and comparison")
    
    models_config = config['models']
    results_dir = config['data']['results_dir']
    batch_size = models_config['batch_size']
    num_epochs = models_config['num_epochs']
    use_wavelet = models_config['use_wavelet']
    
    # Create data loaders
    print("Creating data loaders")
    train_loader, val_loader, test_loader = create_data_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=batch_size
    )
    
    # Prepare model configurations
    model_configs = {}
    
    # CNN-LSTM configuration
    if models_config['model_configs']['cnn_lstm']['enabled']:
        cnn_lstm_config = models_config['model_configs']['cnn_lstm']
        model_configs['cnn_lstm'] = {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'num_features': train_data['timeseries'].shape[2],
                'wavelet_dim': 128,  # This should match the actual wavelet dimension
                'cnn_backbone': cnn_lstm_config['backbone'],
                'lstm_hidden_dim': cnn_lstm_config['hidden_dim'],
                'dropout_rate': cnn_lstm_config['dropout']
            },
            'optimizer_params': {
                'lr': models_config['learning_rate'],
                'weight_decay': models_config['weight_decay']
            }
        }
    
    # Transformer configuration
    if models_config['model_configs']['transformer']['enabled']:
        transformer_config = models_config['model_configs']['transformer']
        model_configs['transformer'] = {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'time_features': train_data['timeseries'].shape[2],
                'embed_dim': transformer_config['embed_dim'],
                'vision_depth': transformer_config['vision_depth'],
                'temporal_depth': transformer_config['temporal_depth'],
                'num_heads': transformer_config['num_heads']
            },
            'optimizer_params': {
                'lr': models_config['learning_rate'],
                'weight_decay': models_config['weight_decay']
            }
        }
    
    # TCN configuration
    if models_config['model_configs']['tcn']['enabled']:
        tcn_config = models_config['model_configs']['tcn']
        model_configs['tcn'] = {
            'model_params': {
                'time_steps': train_data['timeseries'].shape[1],
                'num_features': train_data['timeseries'].shape[2],
                'tcn_hidden_dims': tcn_config['hidden_dims'],
                'tcn_kernel_size': tcn_config['kernel_size']
            },
            'optimizer_params': {
                'lr': models_config['learning_rate'],
                'weight_decay': models_config['weight_decay']
            }
        }
    
    # XGBoost configuration
    if models_config['model_configs']['xgboost']['enabled']:
        xgb_config = models_config['model_configs']['xgboost']
        model_configs['xgboost'] = {
            'model_params': {
                'n_estimators': xgb_config['n_estimators'],
                'learning_rate': xgb_config['learning_rate'],
                'max_depth': xgb_config['max_depth']
            }
        }
    
    # Create results directory
    model_comparison_dir = os.path.join(results_dir, 'model_comparison')
    os.makedirs(model_comparison_dir, exist_ok=True)
    
    # Run model comparison
    print(f"Comparing {len(model_configs)} models: {', '.join(model_configs.keys())}")
    compare_models(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        models_config=model_configs,
        num_epochs=num_epochs,
        wavelet_features=use_wavelet,
        results_dir=model_comparison_dir
    )
    
    # Load and return comparison results
    results_path = os.path.join(model_comparison_dir, 'model_comparison_results.csv')
    if os.path.exists(results_path):
        model_results = pd.read_csv(results_path)
        print("Model comparison results:")
        print(model_results)
        return model_results.to_dict()
    else:
        print("Model comparison results file not found")
        return {}


# Feature comparison step
def run_feature_comparison(
    train_data: Dict[str, np.ndarray],
    feature_names: List[str],
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Run feature comparison analysis.
    
    Args:
        train_data: Training data
        feature_names: List of feature names
        config: Configuration dictionary
        
    Returns:
        DataFrame with ranked features
    """
    print("Starting feature comparison analysis")
    
    results_dir = config['data']['results_dir']
    
    # Create results directory
    feature_comparison_dir = os.path.join(results_dir, 'feature_comparison')
    os.makedirs(feature_comparison_dir, exist_ok=True)
    
    # Run feature comparison
    rank_df = compare_features(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        models_dir=os.path.join(results_dir, 'model_comparison'),
        results_dir=feature_comparison_dir
    )
    
    print(f"Feature comparison complete. Top features: {', '.join(rank_df['feature'].head(5).tolist())}")
    
    return rank_df


# Early warning detection step
def run_early_warning_detection(
    train_data: Dict[str, np.ndarray],
    feature_names: List[str],
    ranked_features: pd.DataFrame,
    config: Dict[str, Any]
) -> Dict[str, np.ndarray]:
    """
    Run early warning signal detection.
    
    Args:
        train_data: Training data
        feature_names: List of feature names
        ranked_features: DataFrame with ranked features
        config: Configuration dictionary
        
    Returns:
        Dictionary of early warning signal results
    """
    print("Starting early warning signal detection")
    
    results_dir = config['data']['results_dir']
    early_warning_config = config['early_warning']
    
    # Create results directory
    early_warning_dir = os.path.join(results_dir, 'early_warning')
    os.makedirs(early_warning_dir, exist_ok=True)
    
    # Get early warning features from ranked features
    if 'early_warning' in ranked_features.columns:
        early_warning_features = ranked_features[ranked_features['early_warning']]['feature'].tolist()
    else:
        # If early warning column doesn't exist, use top features
        early_warning_features = ranked_features['feature'].head(5).tolist()
    
    # Run early warning detection
    ews_results = detect_early_warnings(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        results_dir=early_warning_dir,
        window_size=early_warning_config['window_size'],
        detrend=early_warning_config['detrend'],
        smoothing=early_warning_config['smoothing']
    )
    
    print(f"Early warning detection complete. Detected features: {', '.join(early_warning_features)}")
    
    return ews_results


# Visualization and reporting step
def run_visualization(
    train_data: Dict[str, np.ndarray],
    feature_names: List[str],
    ranked_features: pd.DataFrame,
    model_results: Dict[str, Any],
    ews_results: Dict[str, np.ndarray],
    config: Dict[str, Any]
) -> None:
    """
    Run visualization and reporting.
    
    Args:
        train_data: Training data
        feature_names: List of feature names
        ranked_features: DataFrame with ranked features
        model_results: Dictionary of model results
        ews_results: Dictionary of early warning signal results
        config: Configuration dictionary
    """
    print("Starting visualization and reporting")
    
    results_dir = config['data']['results_dir']
    vis_config = config['visualization']
    
    # Create visualization directory
    vis_dir = os.path.join(results_dir, 'visualization')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize time series data
    print("Visualizing time series data")
    visualize_time_series(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        save_dir=vis_dir if vis_config['save_visualizations'] else None
    )
    
    # Visualize feature importance
    print("Visualizing feature importance")
    importance_df = pd.DataFrame({
        'feature': ranked_features['feature'],
        'importance': ranked_features['avg_rank'] if 'avg_rank' in ranked_features.columns else ranked_features.iloc[:, 1]
    })
    
    visualize_feature_importance(
        importance_df=importance_df,
        save_path=os.path.join(vis_dir, 'feature_importance.png') if vis_config['save_visualizations'] else None
    )
    
    # Extract feature matrix from time series
    feature_matrix = train_data['timeseries'].reshape(train_data['timeseries'].shape[0], -1)
    
    # Visualize feature distributions
    print("Visualizing feature distributions")
    visualize_feature_distribution(
        features=feature_matrix,
        labels=train_data['labels'],
        feature_names=[f"{name}_{t}" for name in feature_names for t in range(train_data['timeseries'].shape[1])],
        top_features=ranked_features['feature'].head(10).tolist(),
        save_dir=vis_dir if vis_config['save_visualizations'] else None
    )
    
    # Visualize feature correlations
    print("Visualizing feature correlations")
    visualize_correlation_matrix(
        features=feature_matrix,
        feature_names=[f"{name}_{t}" for name in feature_names for t in range(train_data['timeseries'].shape[1])],
        top_features=ranked_features['feature'].head(20).tolist(),
        save_path=os.path.join(vis_dir, 'correlation_matrix.png') if vis_config['save_visualizations'] else None
    )
    
    # Load learning curves if available
    history_path = os.path.join(results_dir, 'model_comparison', 'change_over_time.csv')
    if os.path.exists(history_path):
        history_df = pd.read_csv(history_path)
        
        print("Visualizing learning curves")
        visualize_learning_curves(
            history_df=history_df,
            metrics=['loss', 'accuracy'] if 'loss' in history_df.columns[0] else ['accuracy'],
            save_path=os.path.join(vis_dir, 'learning_curves.png') if vis_config['save_visualizations'] else None
        )
    
    # Load and visualize model comparison results
    model_results_path = os.path.join(results_dir, 'model_comparison', 'model_comparison_results.csv')
    if os.path.exists(model_results_path):
        model_metrics_df = pd.read_csv(model_results_path)
        
        print("Visualizing model comparison")
        visualize_model_comparison(
            metrics_df=model_metrics_df,
            save_path=os.path.join(vis_dir, 'model_comparison.png') if vis_config['save_visualizations'] else None
        )
    
    # Visualize feature changes over time
    print("Visualizing feature changes over time")
    visualize_feature_changes_over_time(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        top_features=ranked_features['feature'].head(5).tolist(),
        save_dir=vis_dir if vis_config['save_visualizations'] else None
    )
    
    # Visualize early warning signals
    print("Visualizing early warning signals")
    visualize_early_warning_signals(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        ews_results=ews_results,
        save_dir=vis_dir if vis_config['save_visualizations'] else None
    )
    
    # Create comprehensive report
    print("Creating comprehensive report")
    
    # Get early warning features
    if 'early_warning' in ranked_features.columns:
        early_warning_features = ranked_features[ranked_features['early_warning']]['feature'].tolist()
    else:
        early_warning_features = []
    
    # Convert model results to expected format
    model_metrics = {}
    if isinstance(model_results, dict):
        if 'model' in model_results:
            # Results from DataFrame dict
            for i, model_name in enumerate(model_results['model']):
                model_metrics[model_name] = {
                    metric: model_results[metric][i] 
                    for metric in model_results.keys() 
                    if metric != 'model'
                }
        else:
            # Direct dict from model comparison
            model_metrics = model_results
    
    create_comprehensive_report(
        time_series=train_data['timeseries'],
        labels=train_data['labels'],
        feature_names=feature_names,
        model_metrics=model_metrics,
        top_features=ranked_features['feature'].head(10).tolist(),
        early_warning_features=early_warning_features,
        save_dir=vis_dir if vis_config['save_visualizations'] else None
    )
    
    print("Visualization and reporting complete")


# Main function to run the pipeline
def main():
    """Run the full pipeline or specific components."""
    parser = argparse.ArgumentParser(description='Coral Health Project Pipeline')
    parser.add_argument('--config', type=str, default='./config/config.yaml', help='Path to configuration file')
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing step')
    parser.add_argument('--engineer', action='store_true', help='Run feature engineering step')
    parser.add_argument('--compare_models', action='store_true', help='Run model comparison step')
    parser.add_argument('--compare_features', action='store_true', help='Run feature comparison step')
    parser.add_argument('--early_warning', action='store_true', help='Run early warning detection step')
    parser.add_argument('--visualize', action='store_true', help='Run visualization step')
    parser.add_argument('--all', action='store_true', help='Run all steps')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Determine which steps to run
    run_all = args.all or not any([
        args.preprocess, args.engineer, args.compare_models, 
        args.compare_features, args.early_warning, args.visualize
    ])
    
    # Create results directory if it doesn't exist
    os.makedirs(config['data']['results_dir'], exist_ok=True)
    
    # Save configuration
    config_out_path = os.path.join(config['data']['results_dir'], 'config_used.yaml')
    with open(config_out_path, 'w') as f:
        yaml.dump(config, f)
    
    # Initialize variables to store results from each step
    train_data, val_data, test_data = None, None, None
    norm_params = None
    selected_features, feature_names = None, None
    model_results = {}
    ranked_features = None
    ews_results = {}
    
    # Run preprocessing step
    if run_all or args.preprocess:
        train_data, val_data, test_data, norm_params = run_preprocessing(config)
    else:
        # Load preprocessed data if not running preprocessing
        print("Loading preprocessed data")
        train_data, val_data, test_data, norm_params = load_processed_data(config['data']['processed_dir'])
    
    # Define default feature names if not available
    if feature_names is None:
        feature_names = [
            'temperature', 'salinity', 'ph', 'dissolved_oxygen',
            'turbidity', 'chlorophyll', 'nitrate', 'phosphate'
        ]
    
    # Run feature engineering step
    if run_all or args.engineer:
        selected_features, feature_names = run_feature_engineering(train_data, val_data, test_data, config)
    
    # Run model comparison step
    if run_all or args.compare_models:
        model_results = run_model_comparison(train_data, val_data, test_data, config)
    
    # Run feature comparison step
    if run_all or args.compare_features:
        ranked_features = run_feature_comparison(train_data, feature_names, config)
    else:
        # Create dummy ranked features if not running feature comparison
        ranked_features = pd.DataFrame({
            'feature': feature_names,
            'importance': np.linspace(1, 0.1, len(feature_names))
        })
    
    # Run early warning detection step
    if run_all or args.early_warning:
        ews_results = run_early_warning_detection(train_data, feature_names, ranked_features, config)
    
    # Run visualization step
    if run_all or args.visualize:
        run_visualization(train_data, feature_names, ranked_features, model_results, ews_results, config)
    
    print("Pipeline execution complete!")


if __name__ == "__main__":
    # Start timing
    start_time = datetime.datetime.now()
    print(f"Starting pipeline at {start_time}")
    
    # Run main function
    main()
    
    # End timing
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Pipeline completed at {end_time}")
    print(f"Total execution time: {elapsed_time}")