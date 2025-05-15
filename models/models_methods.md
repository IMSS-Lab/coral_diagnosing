# Predictive Models for Coral Bleaching

This document outlines the architectures and key components of various predictive models developed for forecasting coral bleaching events. These models leverage multi-modal data, including satellite imagery and environmental time series, to capture complex spatio-temporal dynamics.

## 1. CNN-LSTM with Attention and Wavelet Features

The CNN-LSTM model is designed to concurrently process visual information from coral reef images and temporal patterns from environmental data.

### Architecture:

The model comprises three main feature extraction branches followed by a fusion mechanism:

1.  **CNN Module (Image Features)**:
    *   Utilizes a pre-trained Convolutional Neural Network (CNN) backbone (e.g., ResNet18, ResNet50, EfficientNet-B0) to extract high-level visual features from input images.
    *   The backbone's classification head is replaced with an identity layer or a custom projection layer to produce a fixed-size feature vector $\mathbf{f}_{cnn} \in \mathbb{R}^{D_{cnn}}$.
    *   Dropout and Batch Normalization are used for regularization.

2.  **LSTM Module (Time Series Features)**:
    *   Processes sequential environmental data $\mathbf{X}_{ts} \in \mathbb{R}^{T \times F_{ts}}$ (where $T$ is sequence length, $F_{ts}$ is number of time series features) using a multi-layer Long Short-Term Memory (LSTM) network.
    *   The LSTM can be uni- or bi-directional.
    *   A **Self-Attention** mechanism is applied to the LSTM outputs $\mathbf{H}_{lstm} \in \mathbb{R}^{T \times D_{lstm\_hidden}}$ to weigh the importance of different time steps:
        *   Query ($Q$), Key ($K$), Value ($V$) matrices are projected from $\mathbf{H}_{lstm}$.
        *   Attention scores are computed as: $$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$ where $d_k$ is the dimension of the key vectors.
    *   The attended features are typically pooled (e.g., mean pooling) and projected to an output feature vector $\mathbf{f}_{lstm} \in \mathbb{R}^{D_{lstm}}$.

3.  **Wavelet Module (Time Series Features)**:
    *   Applies a Discrete Wavelet Transform (DWT) (e.g., 'db4') to each environmental time series feature up to a specified decomposition level $L$.
    *   From the approximation ($cA_L$) and detail ($cD_1, \dots, cD_L$) coefficients at each level, statistical features are extracted (e.g., mean, standard deviation, energy, entropy).
    *   These wavelet-derived features are concatenated and processed by a Multi-Layer Perceptron (MLP) to produce $\mathbf{f}_{wavelet} \in \mathbb{R}^{D_{wavelet}}$.

4.  **Feature Fusion Module**:
    *   The feature vectors from the three branches ($\mathbf{f}_{cnn}, \mathbf{f}_{lstm}, \mathbf{f}_{wavelet}$) are concatenated: $\mathbf{f}_{concat} = [\mathbf{f}_{cnn}; \mathbf{f}_{lstm}; \mathbf{f}_{wavelet}]$.
    *   $\mathbf{f}_{concat}$ is passed through one or more fully connected layers with activation functions (e.g., ReLU) and normalization to produce a fused representation $\mathbf{f}_{fused}$.

5.  **Output Layer**:
    *   A final linear layer maps $\mathbf{f}_{fused}$ to a single logit for binary classification (healthy vs. bleached).
    *   A sigmoid function converts the logit to a probability.
    *   The model is typically trained using a Binary Cross-Entropy with Logits loss (BCEWithLogitsLoss), potentially with positive class weighting to handle class imbalance:
        $$L = -w \cdot [y \cdot \log(\sigma(x)) + (1-y) \cdot \log(1-\sigma(x))]$$
        where $x$ is the logit, $y$ is the true label, $\sigma$ is the sigmoid function, and $w$ is the positive class weight.

## 2. Temporal Convolutional Network (TCN) based Model

This model also combines image and time series data but uses a Temporal Convolutional Network (TCN) for processing sequential information.

### Architecture:

1.  **CNN Module (Image Features)**:
    *   Identical or similar to the CNN module in the CNN-LSTM model, producing $\mathbf{f}_{cnn} \in \mathbb{R}^{D_{cnn}}$.

2.  **TCN Module (Time Series Features)**:
    *   The core of this module is a stack of **Temporal Blocks**. Each block consists of:
        *   Two layers of 1D dilated causal convolutions. Causal convolutions are ensured by padding the input sequence appropriately on the left and then "chomping" an equivalent amount from the output's right end, ensuring no future information is used.
        *   Weight normalization and dropout are applied after each convolution.
        *   ReLU activation functions.
        *   A residual connection sums the input to the block with the output of the two convolutional layers (a 1x1 convolution is used on the input if channel numbers differ).
    *   Dilations are typically increased exponentially with network depth (e.g., $1, 2, 4, 8, \dots$), allowing the TCN to have a large receptive field with relatively few layers.
    *   The TCN processes input time series $\mathbf{X}_{ts} \in \mathbb{R}^{T \times F_{ts}}$ (often transposed to $\mathbb{R}^{F_{ts} \times T}$ for Conv1D) and outputs $\mathbf{H}_{tcn} \in \mathbb{R}^{D_{tcn\_hidden} \times T}$.
    *   An optional **Sequence Attention** mechanism (similar to self-attention) can be applied to $\mathbf{H}_{tcn}$ (after transposing to $\mathbb{R}^{T \times D_{tcn\_hidden}}$).
    *   The output is typically pooled (e.g., mean pooling over the time dimension or using the last time step's output) and projected to $\mathbf{f}_{tcn} \in \mathbb{R}^{D_{tcn}}$.

3.  **Feature Fusion Module**:
    *   Concatenates image features $\mathbf{f}_{cnn}$ and TCN-derived time series features $\mathbf{f}_{tcn}$: $\mathbf{f}_{concat} = [\mathbf{f}_{cnn}; \mathbf{f}_{tcn}]$.
    *   May employ a more sophisticated fusion strategy, such as context-aware gating:
        *   Image and time features are projected: $\mathbf{p}_{img} = W_{img}\mathbf{f}_{cnn}$, $\mathbf{p}_{time} = W_{time}\mathbf{f}_{tcn}$.
        *   A context vector $\mathbf{c} = [\mathbf{p}_{img}; \mathbf{p}_{time}; \mathbf{p}_{img} \odot \mathbf{p}_{time}; \mathbf{p}_{img} - \mathbf{p}_{time}]$ (or similar).
        *   Gates are computed: $\mathbf{g}_{img} = \sigma(W_{g,img}\mathbf{c})$, $\mathbf{g}_{time} = \sigma(W_{g,time}\mathbf{c})$.
        *   Gated features: $\mathbf{f}'_{img} = \mathbf{g}_{img} \odot \mathbf{p}_{img}$, $\mathbf{f}'_{time} = \mathbf{g}_{time} \odot \mathbf{p}_{time}$.
        *   These gated features are then further processed.
    *   The fused representation $\mathbf{f}_{fused}$ is obtained.

4.  **Output Layer**:
    *   Similar to the CNN-LSTM model, a linear layer maps $\mathbf{f}_{fused}$ to a classification logit, trained with BCEWithLogitsLoss.

## 3. Vision Transformer (ViT) based Dual Transformer Model

This architecture employs transformers for both image and time series modalities, followed by a cross-modal fusion mechanism.

### Architecture:

1.  **Image Transformer (ViT)**:
    *   **Patch Embedding**: The input image $\mathbf{I} \in \mathbb{R}^{H \times W \times C}$ is divided into a sequence of non-overlapping patches $\mathbf{x}_p \in \mathbb{R}^{N_p \times (P^2 \cdot C)}$, where $P$ is patch size and $N_p = HW/P^2$ is number of patches. Each patch is linearly projected into an embedding of dimension $D_{img}$.
        $$\mathbf{E}_{patches} = [\mathbf{x}_{p_1}W_E; \mathbf{x}_{p_2}W_E; \dots ; \mathbf{x}_{p_{N_p}}W_E]$$
    *   **CLS Token**: A learnable classification token $\mathbf{x}_{cls} \in \mathbb{R}^{1 \times D_{img}}$ is prepended to the sequence of patch embeddings.
    *   **Positional Embedding**: Learnable positional embeddings $\mathbf{E}_{pos} \in \mathbb{R}^{(N_p+1) \times D_{img}}$ are added to the patch embeddings to retain spatial information.
        $$\mathbf{Z}_0 = [\mathbf{x}_{cls}; \mathbf{E}_{patches}] + \mathbf{E}_{pos}$$
    *   **Transformer Encoder**: $\mathbf{Z}_0$ is passed through a stack of $L_{img}$ transformer encoder layers. Each layer consists of:
        *   Multi-Head Self-Attention (MHSA): $MHSA(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W^O$, where $head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$.
        *   Layer Normalization (LN).
        *   Position-wise Feed-Forward Network (FFN).
        $$\mathbf{Z}'_l = MHSA(LN(\mathbf{Z}_{l-1})) + \mathbf{Z}_{l-1}$$
        $$\mathbf{Z}_l = FFN(LN(\mathbf{Z}'_l)) + \mathbf{Z}'_l$$
    *   The output features $\mathbf{f}_{img\_transformer} \in \mathbb{R}^{(N_p+1) \times D_{img}}$ are the output of the last transformer layer. The state of the CLS token $\mathbf{f}_{img\_cls} = \mathbf{Z}_{L_{img}}[0]$ can be used as the image representation.

2.  **Time Series Transformer**:
    *   **Feature Embedding**: Each time step's feature vector $\mathbf{x}_{ts,t} \in \mathbb{R}^{F_{ts}}$ from the input $\mathbf{X}_{ts} \in \mathbb{R}^{T \times F_{ts}}$ is linearly projected to an embedding of dimension $D_{time}$.
    *   **Positional Encoding**: Sinusoidal positional encodings are added to the feature embeddings:
        $$PE_{(pos, 2i)} = \sin(pos/10000^{2i/D_{time}})$$
        $$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/D_{time}})$$
    *   **CLS Token**: A learnable classification token $\mathbf{x}_{ts\_cls} \in \mathbb{R}^{1 \times D_{time}}$ is prepended.
    *   **Transformer Encoder**: Similar to the image transformer, the sequence is processed by $L_{time}$ transformer encoder layers, yielding $\mathbf{f}_{ts\_transformer} \in \mathbb{R}^{(T+1) \times D_{time}}$. The CLS token state $\mathbf{f}_{ts\_cls}$ can represent the time series.

3.  **Fusion Module (Cross-Modal Attention)**:
    *   The full sequence outputs from both transformers (or just their CLS tokens, projected to a common fusion dimension $D_{fusion}$) are used.
    *   Let $\mathbf{F}_{img} \in \mathbb{R}^{N_{img} \times D_{fusion}}$ and $\mathbf{F}_{ts} \in \mathbb{R}^{N_{ts} \times D_{fusion}}$ be the (projected) outputs.
    *   **Cross-Attention**:
        $$\mathbf{A}_{img \to ts} = MHSA(LN(\mathbf{F}_{img}), LN(\mathbf{F}_{ts}), LN(\mathbf{F}_{ts}))$$
        $$\mathbf{A}_{ts \to img} = MHSA(LN(\mathbf{F}_{ts}), LN(\mathbf{F}_{img}), LN(\mathbf{F}_{img}))$$
    *   Updated features are obtained via residual connections and FFNs:
        $$\mathbf{F}'_{img} = \mathbf{F}_{img} + \text{Dropout}(\mathbf{A}_{img \to ts})$$
        $$\mathbf{F}'_{ts} = \mathbf{F}_{ts} + \text{Dropout}(\mathbf{A}_{ts \to img})$$
        $$\mathbf{F}''_{img} = \mathbf{F}'_{img} + \text{Dropout}(FFN(LN(\mathbf{F}'_{img})))$$
        $$\mathbf{F}''_{ts} = \mathbf{F}'_{ts} + \text{Dropout}(FFN(LN(\mathbf{F}'_{ts})))$$
    *   The CLS tokens from $\mathbf{F}''_{img}$ and $\mathbf{F}''_{ts}$ are extracted.
    *   **Fusion Strategy**:
        *   If `fusion_type` is 'concat' or 'both': Concatenate the CLS tokens $[\mathbf{F}''_{img}[0]; \mathbf{F}''_{ts}[0]]$ and project to $D_{fusion}$.
        *   If `fusion_type` is 'attention' only: Average the CLS tokens and project.
    *   This yields the final fused feature vector $\mathbf{f}_{fused}$.

4.  **Output Layer**:
    *   A linear layer maps $\mathbf{f}_{fused}$ to $N_{classes}$ logits (e.g., 2 for binary classification with CrossEntropyLoss, or 1 for BCEWithLogitsLoss).
    *   The model can also output individual predictions from $\mathbf{f}_{img\_cls}$ and $\mathbf{f}_{ts\_cls}$ for auxiliary losses or multi-task learning.
        $$Loss_{total} = w_{fus}L_{fus} + w_{img}L_{img} + w_{ts}L_{ts}$$

## 4. XGBoost Model

The XGBoost (Extreme Gradient Boosting) model provides a tree-based approach, often excelling with tabular data derived from feature engineering.

### Feature Engineering:

A crucial step for XGBoost is comprehensive feature extraction from both image and time series modalities.

1.  **Image Features**:
    *   **Deep Features**: Output from a pre-trained CNN backbone (e.g., ResNet, EfficientNet) after removing the final classification layer. This results in a high-dimensional vector per image.
    *   **Handcrafted Spatial Features**:
        *   Color Statistics: Mean, standard deviation for each color channel (R, G, B), color ratios (e.g., R/G, R/B).
        *   Texture Features: Statistics derived from image gradients (e.g., Sobel filter output means, stds), or potentially Gray-Level Co-occurrence Matrix (GLCM) features like contrast, dissimilarity, homogeneity, energy, correlation.

2.  **Time Series Features**:
    *   **Statistical Features**: For each environmental variable over its time window: mean, std, min, max, median, 25th percentile (q25), 75th percentile (q75), Interquartile Range (IQR), skewness, kurtosis.
    *   **Temporal Dynamics**:
        *   Trend: Slope of a linear regression fitted to the time series.
        *   Autocorrelation: Lag-1 and Lag-2 autocorrelation coefficients.
        *   Entropy: Shannon entropy of the (discretized) time series values.
        *   Energy: Sum of squared values.
        *   Peak Frequency: Dominant frequency from Fast Fourier Transform (FFT).
    *   **Wavelet Features**:
        *   DWT applied to each time series (e.g., 'db4' wavelet, multiple levels).
        *   From approximation and detail coefficients at each level: mean, std, energy, entropy.

### XGBoost Model:

*   **Algorithm**: XGBoost is an ensemble learning method based on gradient boosted decision trees. It builds trees sequentially, where each new tree corrects errors made by previously trained trees.
*   **Objective Function**: For binary classification, the `binary:logistic` objective is commonly used. The objective function $Obj(\Theta)$ to be minimized at each iteration $t$ for adding a new tree $f_t$ is:
    $$Obj^{(t)} \approx \sum_{i=1}^{n} [g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i)] + \Omega(f_t)$$
    where $g_i = \partial_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ and $h_i = \partial^2_{\hat{y}^{(t-1)}} l(y_i, \hat{y}^{(t-1)})$ are the first and second order gradients of the loss function $l$ with respect to the prediction $\hat{y}^{(t-1)}$ at step $t-1$. $\Omega(f_t) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$ is the regularization term, penalizing tree complexity ($T$ is number of leaves, $w_j$ are leaf weights).
*   **Training**:
    *   Input features are typically scaled (e.g., StandardScaler).
    *   The model can be trained with early stopping based on a validation set's performance (e.g., AUC).
    *   Cross-validation (e.g., Stratified K-Fold) can be used for robust hyperparameter tuning and performance estimation.
*   **Feature Importance**: XGBoost provides built-in feature importance scores (e.g., 'gain', 'weight', 'cover'), which can be used for feature selection or interpretation. SHAP (SHapley Additive exPlanations) values can also be computed for more detailed local and global explanations.
*   **Hyperparameters**: Key parameters include `n_estimators` (number of trees), `learning_rate` (eta), `max_depth`, `subsample`, `colsample_bytree`, `gamma` (min_split_loss), `lambda` (L2 reg), `alpha` (L1 reg).

## 5. Ensemble Model

The ensemble model aims to improve predictive performance and robustness by combining the strengths of the individual base models.

### Architecture:

1.  **Base Models**:
    *   CNN-LSTM Model
    *   TCN-based Model
    *   Dual Transformer Model
    *   XGBoost Model
    Each base model is trained (or loaded if pre-trained). For PyTorch models, they are set to evaluation mode (`model.eval()`).

2.  **Prediction Aggregation**:
    *   For a given input sample (image, time series), predictions (probabilities) are obtained from each trained base model $P_m(\mathbf{x})$, where $m$ indexes the model.
    *   **Weighted Averaging**: The final ensemble prediction $P_{ens}(\mathbf{x})$ is a weighted average of the individual model predictions:
        $$P_{ens}(\mathbf{x}) = \sum_{m=1}^{M} w_m \cdot P_m(\mathbf{x})$$
        where $w_m$ is the weight assigned to model $m$, and $\sum w_m = 1$.
    *   Weights $w_m$ can be:
        *   Pre-defined (e.g., based on individual model validation performance).
        *   Learned as parameters, optimized on a validation set to minimize an ensemble loss function (e.g., by applying softmax to learnable logits to ensure they sum to 1).

### Uncertainty Quantification:

*   For deep learning base models (CNN-LSTM, TCN, Transformer), uncertainty can be estimated using **Monte Carlo (MC) Dropout**.
    *   Dropout layers are kept active during inference.
    *   Multiple forward passes ($N_{samples}$) are performed for the same input.
    *   The mean of these $N_{samples}$ predictions is taken as the final prediction.
    *   The variance of these $N_{samples}$ predictions serves as a measure of model uncertainty.
*   The ensemble variance can be estimated by combining individual model variances, potentially weighted by their ensemble weights:
    $$Var_{ens} \approx \sum_{m=1}^{M} w_m^2 \cdot Var_m$$ (assuming independence, a simplification).

### Feature Importance and Early Warning:

*   The ensemble can leverage feature importance from its constituents (especially XGBoost).
*   Early warning signal capabilities can be integrated by analyzing trends in predictions or uncertainty from the ensemble or individual models as they process sequential data leading up to a potential event.

The ensemble model provides a framework to potentially achieve superior performance by mitigating individual model weaknesses and leveraging their diverse ways of processing information.