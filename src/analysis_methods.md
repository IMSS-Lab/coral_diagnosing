# Analytical Methods for Coral Bleaching Prediction Study

This section details the analytical methods employed for data preprocessing, feature engineering, model development, evaluation, and interpretation in the study of coral bleaching prediction.

## 1. Data Preprocessing and Preparation

### 1.1. Data Acquisition and Cleaning
Raw data, comprising satellite imagery of coral reefs and corresponding environmental time series (e.g., sea surface temperature, salinity, pH), along with coral health labels (bleached/healthy), were collected.
*   **Image Cleaning**: Images were standardized by resizing to a uniform dimension (e.g., 224x224 pixels). Color channels were consistently ordered (e.g., BGR to RGB). Pixel values were normalized, typically to the range $[0, 1]$ or by standardizing with mean and standard deviation computed across the training dataset.
*   **Time Series Cleaning**: Environmental time series data were processed to handle missing values, which could involve imputation (e.g., mean, median, or interpolation) or marking missing entries. Time series were resampled or interpolated to ensure consistent sequence lengths across samples. Numerical features were often standardized (z-score normalization) or scaled to a specific range (min-max scaling).
*   **Data Alignment**: Imagery, time series, and labels were meticulously aligned based on unique sample identifiers and temporal stamps to ensure correct correspondence for multi-modal model training.

### 1.2. Data Splitting
The aligned dataset was partitioned into training, validation, and testing sets. Stratified splitting was employed based on the coral health labels to maintain proportional class representation in each subset, crucial for handling potential class imbalances.

### 1.3. Data Augmentation and Loading
For training deep learning models, image data augmentation techniques were applied to the training set to increase data variability and prevent overfitting. Common augmentations included random horizontal/vertical flips, rotations, and color jittering (adjustments to brightness, contrast, saturation). PyTorch `Dataset` and `DataLoader` classes were utilized for efficient batching, shuffling, and parallel data loading during model training.

## 2. Feature Engineering

To enhance model performance, particularly for traditional machine learning models like XGBoost, and to provide interpretable inputs, various features were engineered.

### 2.1. Image-based Features
*   **Deep Features**: Pre-trained Convolutional Neural Networks (CNNs), such as ResNet or EfficientNet, were used as feature extractors. The activations from intermediate or final convolutional layers (before the classification head) were taken as high-level image representations.
*   **Handcrafted Spatial Features**:
    *   *Color Statistics*: Mean, standard deviation, and potentially higher-order moments were calculated for each color channel (e.g., R, G, B, HSV). Color ratios were also computed.
    *   *Texture Features*: Metrics describing image texture were extracted. This included statistics from image gradients (e.g., mean and standard deviation of Sobel filter responses) and potentially features derived from Gray-Level Co-occurrence Matrices (GLCM), such as contrast, dissimilarity, homogeneity, energy, and correlation.

### 2.2. Time Series-based Features
*   **Statistical Features**: For each environmental variable within a relevant time window, a suite of statistical descriptors was computed:
    *   Central tendency: mean, median.
    *   Dispersion: standard deviation, variance, min, max, range, Interquartile Range (IQR).
    *   Shape: skewness, kurtosis.
*   **Temporal Dynamics Features**:
    *   *Trend*: The slope of a linear regression line fitted to the time series segment.
    *   *Autocorrelation*: Autocorrelation Function (ACF) values at specific lags (e.g., lag-1, lag-2) to capture temporal dependence.
    *   *Complexity/Variability*: Shannon entropy (for discretized values) or spectral entropy; sum of squared values (energy).
    *   *Spectral Features*: Dominant frequencies identified using Fast Fourier Transform (FFT) or power spectral density (PSD) via Welch's method, and power in specific frequency bands.
*   **Wavelet Features**:
    *   Discrete Wavelet Transform (DWT) using wavelets like Daubechies (e.g., 'db4') was applied to decompose time series into different frequency sub-bands (approximation and detail coefficients) across multiple decomposition levels.
    *   Statistical features (mean, standard deviation, energy, entropy) were calculated for the coefficients at each level and for each sub-band, providing a time-frequency representation.

## 3. Feature Analysis and Selection

### 3.1. Feature Importance Assessment
*   **Model-Specific Importance**: For tree-based models like XGBoost, intrinsic importance scores (e.g., "gain," which measures the average training loss reduction gained when a feature is used in a split) were utilized.
*   **Permutation Importance**: This model-agnostic technique was used to evaluate feature importance by measuring the decrease in model performance (e.g., AUC, F1-score) when the values of a single feature were randomly shuffled. This process was repeated multiple times for robustness.

### 3.2. Feature Correlation Analysis
*   **Feature-Target Correlation**: For binary classification, point-biserial correlation coefficients were calculated between continuous features and the binary target variable.
*   **Inter-Feature Correlation**: Pearson correlation matrices were computed to understand multicollinearity among features, which can inform feature selection and model interpretation.

### 3.3. Mutual Information
Mutual information was calculated between each feature and the target variable. This measures the amount of information obtained about one variable through observing the other, capturing non-linear dependencies.

### 3.4. Dimensionality Reduction for Visualization
To visualize high-dimensional feature spaces, dimensionality reduction techniques were employed:
*   **Principal Component Analysis (PCA)**: For linear dimensionality reduction, retaining principal components that explain the most variance.
*   **t-distributed Stochastic Neighbor Embedding (t-SNE)** and **Uniform Manifold Approximation and Projection (UMAP)**: For non-linear dimensionality reduction, primarily used for visualizing clusters and relationships in 2D or 3D.

## 4. Model Training and Evaluation

### 4.1. Model Training
*   **Optimization**: Deep learning models were trained using optimizers such as AdamW (Adam with weight decay).
*   **Loss Functions**: For binary classification, Binary Cross-Entropy with Logits (BCEWithLogitsLoss) was standard. For multi-class tasks (if applicable), CrossEntropyLoss was used. Class weighting was often incorporated into the loss function to address class imbalance by assigning higher penalties to misclassifications of the minority class.
*   **Learning Rate Scheduling**: Learning rate schedulers like ReduceLROnPlateau (reducing LR when a metric stops improving) or CosineAnnealingLR (cosine decay of LR) were used to adjust the learning rate during training.
*   **Regularization**: Techniques such as dropout, L1/L2 weight decay, and batch normalization were employed to prevent overfitting.
*   **Early Stopping**: Training was often monitored on a validation set, and stopped if performance on this set ceased to improve for a predefined number of epochs, particularly for XGBoost and potentially for deep learning models.
*   **Cross-Validation**: For models like XGBoost, Stratified K-Fold cross-validation was used during training or hyperparameter tuning to obtain more robust performance estimates and reduce variance due to a particular train-validation split.

### 4.2. Model Evaluation
Model performance was assessed using a range of standard classification metrics:
*   **Accuracy**: Proportion of correctly classified samples.
*   **Precision**: Ability of the classifier not to label a negative sample as positive ($TP / (TP + FP)$).
*   **Recall (Sensitivity)**: Ability of the classifier to find all positive samples ($TP / (TP + FN)$).
*   **F1-Score**: Harmonic mean of precision and recall ($2 \cdot (Precision \cdot Recall) / (Precision + Recall)$).
*   **Area Under the Receiver Operating Characteristic Curve (AUC-ROC)**: Measures the model's ability to distinguish between classes across various thresholds.
*   **Area Under the Precision-Recall Curve (AUCPR or Average Precision)**: Summarizes the trade-off between precision and recall across thresholds, particularly useful for imbalanced datasets.
*   **Confusion Matrix**: A table showing true positives, true negatives, false positives, and false negatives, providing a detailed view of classification performance.

### 4.3. Performance Visualization
*   **Learning Curves**: Plots of training and validation loss/metrics against training epochs to diagnose overfitting, underfitting, or convergence issues.
*   **Metric Comparison Plots**: Bar charts comparing key evaluation metrics across different models or experimental setups.
*   **ROC and PR Curves**: Visualizations of the trade-offs for ROC (TPR vs. FPR) and PR (Precision vs. Recall) curves.
*   **Confusion Matrix Heatmaps**: Visual representations of confusion matrices.

## 5. Early Warning Signal (EWS) Detection

Methods were implemented to detect subtle changes in environmental parameters or model-derived indicators that might precede a coral bleaching event.

### 5.1. Time Series Pattern Analysis
*   The temporal evolution of key environmental features leading up to documented bleaching events was analyzed by comparing average patterns and variability (e.g., standard deviation bands) between healthy periods and pre-bleaching periods.
*   Difference patterns (e.g., bleached average - healthy average) were examined to highlight deviations.

### 5.2. Rolling Window Statistics for EWS Indicators
Generic EWS indicators, often associated with systems approaching critical transitions (tipping points), were calculated on time series data using a rolling window approach:
*   **Variance**: Increasing variance is a common EWS.
*   **Autocorrelation at lag-1 (AR1)**: Increasing AR1 (slower return to equilibrium after perturbation) is another key EWS.
*   **Skewness and Kurtosis**: Changes in the shape of the data distribution.
*   **Coefficient of Variation (CV)**.
*   **Return Rate**: Measures how quickly a system returns to its previous state after small perturbations.
*   **Spectral Density Ratio**: Ratio of low-frequency to high-frequency power in the power spectrum, indicating a shift towards slower dynamics.
Time series were often detrended (e.g., Gaussian kernel smoothing followed by subtraction of the smoothed series) and/or smoothed (e.g., moving average) before calculating EWS indicators to focus on fluctuations around a trend.

### 5.3. Trend Analysis and Thresholding
*   Trends in the EWS indicators (e.g., variance, AR1) were assessed, often using Kendall's tau correlation coefficient, to quantify monotonic increases.
*   Z-score based thresholding was applied to EWS indicators (comparing recent values to a baseline period) to identify statistically significant deviations that could signal an impending transition.

### 5.4. Anomaly Detection
Multivariate anomaly detection methods, such as Isolation Forest, were applied to time series data or derived EWS indicators to identify unusual patterns that might serve as early warnings.

### 5.5. Feature Ranking for Early Warning
Features were ranked based on their ability to discriminate between pre-bleaching and healthy states, or by the strength of EWS trends they exhibited, to identify the most informative parameters for early warning.