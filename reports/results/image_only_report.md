# Image-Only Baseline Report (ResNet50)

## 1. Environment & Hardware
- **Python**: `3.10.10`
- **PyTorch**: `2.14.1+cpu`
- **Torchvision**: `0.29.1+cpu`
- **Compute Device**: `CPU` (N/A (CPU Mode))
- **System RAM**: `Available`

## 2. Dataset Partitioning
- **Training Images**: 17578
- **Validation Images**: 3812
- **Test Images**: 3941
- **Classes**: 8 (`NV`, `MEL`, `BCC`, `BKL`, `AK`, `SCC`, `VASC`, `DF`)

## 3. Training & Model Architecture
- **Backbone Architecture**: ResNet50 (Pretrained ImageNet-1k weights)
- **Input Resolution**: $224 \times 224$ RGB
- **Optimizer**: `AdamW` (Weight decay = 1e-4)
- **Loss Function**: Class-Weighted CrossEntropyLoss
- **Learning Rate Schedule**: `ReduceLROnPlateau` (Stage 1 LR: 1e-3, Stage 2 LR: 1e-4)
- **Batch Size**: 64
- **Best Validation Macro F1**: **0.6630**

## 4. Overall Test Performance Metrics
- **Test Accuracy**: **71.25%**
- **Macro F1 Score**: **0.2440**
- **Weighted F1 Score**: **0.7128**
- **Macro Precision**: **0.4605**
- **Macro Recall / Sensitivity**: **0.2332**
- **Macro Specificity**: **0.8958**
- **Macro ROC-AUC**: **0.6356**

## 5. Per-Class Performance Breakdown

| class | precision | recall_sensitivity | specificity | f1 | support |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NV** | 0.8421 | 0.8682 | 0.3226 | 0.8550 | 129 |
| **MEL** | 1.0000 | 0.0645 | 1.0000 | 0.1212 | 31 |
| **BCC** | 0.0000 | 0.0000 | 0.9250 | 0.0000 | 0 |
| **BKL** | 0.0000 | 0.0000 | 0.9187 | 0.0000 | 0 |
| **AK** | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |
| **SCC** | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |
| **VASC** | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |
| **DF** | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0 |

## 6. Inference Speed Performance
- **Evaluated Test Subset**: 160 images
- **Total Test Inference Time**: 34.80 seconds
- **Average Time Per Image**: 217.51 ms
- **Throughput**: 4.60 images/second

## 7. Artifact Locations
- **Best Model Checkpoint**: `models/image_only/best_model.pt`
- **Class Weights Config**: `models/image_only/class_weights.json`
- **Confusion Matrix Plots**: `reports/figures/image_only/confusion_matrix_normalized.png`
- **ROC Curves**: `reports/figures/image_only/roc_curve.png`
- **Precision-Recall Curves**: `reports/figures/image_only/precision_recall_curve.png`
- **Classification Report CSV**: `reports/results/image_only_classification_report.csv`
- **Metrics JSON**: `reports/results/image_only_metrics.json`
