# Clinical/Metadata-Only Baseline Report (MLP)

## 1. Environment & Hardware
- **Python**: `3.10.10`
- **PyTorch**: `2.14.1+cpu`
- **Compute Device**: `CPU` (N/A (CPU Mode))
- **System RAM**: `Available`

## 2. Dataset Partitioning & Feature Schema
- **Training Samples**: 17578
- **Validation Samples**: 3812
- **Test Samples**: 3941
- **Predictive Clinical Features**: `age_approx` (Numerical), `sex` (Categorical), `anatom_site_general` (Categorical)
- **Preprocessed Input Feature Dimensions**: 13 (1 continuous + One-Hot Encoded Categories)

## 3. Preprocessing & Imputation Strategy
- **`age_approx`**: Missing count = 311 (1.73%). Strategy: Median imputation (55.0 years) + `StandardScaler`.
- **`sex`**: Missing count = 273 (1.52%). Strategy: Explicit `'Unknown'` category + `OneHotEncoder`.
- **`anatom_site_general`**: Missing count = 1828 (10.39%). Strategy: Explicit `'Unknown'` category + `OneHotEncoder`.
- **Leakage Prevention Guarantee**: Scikit-learn `ColumnTransformer` fitted **ONLY** on Training split (`preprocessor.joblib`).

## 4. Model Architecture & Training Settings
- **Architecture**: Multi-Layer Perceptron (MLP)
  - Input Layer: 13 units
  - Hidden Layer 1: 128 units (ReLU, Dropout 0.3)
  - Hidden Layer 2: 64 units (ReLU, Dropout 0.2)
  - Metadata Embedding Layer: 32 units (ReLU) -> Saved for Phase 5 Fusion
  - Output Classifier: 8 units (Logits for 8 diagnostic classes)
- **Optimizer**: `AdamW` (LR = 1e-3, Weight Decay = 1e-4)
- **Loss Function**: Class-Weighted CrossEntropyLoss (Fitted on Training Split only)
- **Best Epoch**: Epoch 3 (Best Validation Macro F1 = 0.2032)

## 5. Overall Test Performance Metrics
- **Test Accuracy**: **36.26%**
- **Macro F1 Score**: **0.1963**
- **Weighted F1 Score**: **0.3996**
- **Macro Precision**: **0.2362**
- **Macro Recall / Sensitivity**: **0.2631**
- **Macro Specificity**: **0.9074**
- **Macro ROC-AUC**: **0.6817**

## 6. Per-Class Performance Breakdown

| class | precision | recall_sensitivity | specificity | f1 | support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NV** | 0.7815 | 0.5881 | 0.8538 | 0.6712 | 1855 |
| **MEL** | 0.4245 | 0.0740 | 0.9746 | 0.1261 | 797 |
| **BCC** | 0.2455 | 0.2268 | 0.8898 | 0.2357 | 538 |
| **BKL** | 0.1760 | 0.1084 | 0.9417 | 0.1341 | 406 |
| **AK** | 0.1436 | 0.3841 | 0.9087 | 0.2090 | 151 |
| **SCC** | 0.0874 | 0.4141 | 0.8886 | 0.1444 | 99 |
| **VASC** | 0.0130 | 0.0200 | 0.9805 | 0.0157 | 50 |
| **DF** | 0.0183 | 0.2889 | 0.8214 | 0.0345 | 45 |

## 7. Inference Speed Performance
- **Evaluated Test Samples**: 3941
- **Total Test Inference Time**: 0.0060 seconds
- **Average Time Per Sample**: 0.0015 ms
- **Throughput**: 654125.53 samples/second

## 8. Artifact Locations
- **Best Model Checkpoint**: `models/metadata_only/best_model.pt`
- **Preprocessor Pipeline**: `models/metadata_only/preprocessor.joblib`
- **Class Weights Config**: `models/metadata_only/class_weights.json`
- **Exploration Figures**: `reports/figures/metadata_only/exploration/`
- **Confusion Matrix Plots**: `reports/figures/metadata_only/confusion_matrix_normalized.png`
- **ROC Curves**: `reports/figures/metadata_only/roc_curve.png`
- **Precision-Recall Curves**: `reports/figures/metadata_only/precision_recall_curve.png`
- **Classification Report CSV**: `reports/results/metadata_only_classification_report.csv`
- **Metrics JSON**: `reports/results/metadata_only_metrics.json`

## 9. Research Interpretation & Limitations
- **Standalone Clinical Utility**: Clinical metadata alone (`age_approx`, `sex`, `anatom_site_general`) provides moderate signal (e.g. NV nevus vs. BCC/MEL age/site correlations) but is **insufficient for standalone clinical diagnosis** compared to lesion imaging.
- **Multimodal Purpose**: The 32-dimensional learned metadata feature embedding will be fused with the 2048-dimensional ResNet50 image embedding in **Phase 5 (Multimodal Fusion)** to evaluate multi-source performance gains.
