# Multimodal Feature Fusion Report (Phase 5)

## 1. Research Objective & Modality Representations
The primary research objective is to test whether combining medical image representations (2048-dimensional ResNet50 embedding) and clinical metadata representations (32-dimensional MLP embedding) improves diagnostic classification over single-modality baselines.

- **Image Modality Branch**: ResNet50 Backbone (2048-dimensional feature vector)
- **Metadata Modality Branch**: Clinical MLP (32-dimensional feature vector derived from `age_approx`, `sex`, `anatom_site_general`)
- **Data Alignment Guarantee**: 100% of samples across splits are joined explicitly using `image_id` with 0 missing or mismatched labels.

## 2. Multimodal Architectures Evaluated
1. **Simple Concatenation Fusion (`ISICMultimodalConcatClassifier`)**:
   - 2048 + 32 = 2080-dimensional fused vector
   - Dense(2080 -> 512) -> ReLU -> Dropout(0.3) -> Dense(512 -> 128) -> ReLU -> Dropout(0.2) -> Classifier(8)
2. **Multi-Head Attention Fusion (`ISICMultimodalAttentionClassifier`)**:
   - Linear projections: Image (2048 -> 256), Metadata (32 -> 256) into `[IMAGE_TOKEN, METADATA_TOKEN]` (2 x 256).
   - 4-Head Multi-Head Self-Attention over modality tokens with residual connection and LayerNorm.
   - Classifier (512 -> 128 -> 8).

## 3. Master Model Comparison Table

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC | Sensitivity | Specificity | Inference Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image Only — ResNet50** | 0.6391 | 0.2271 | 0.1669 | 0.1487 | 0.6056 | 0.6734 | 0.1669 | 0.8959 | 92.8600 |
| **Metadata Only — MLP** | 0.3626 | 0.2362 | 0.2631 | 0.1963 | 0.3996 | 0.6817 | 0.2631 | 0.9074 | 0.0060 |
| **Multimodal — Concatenation** | 0.6863 | 0.1599 | 0.1628 | 0.1495 | 0.6154 | 0.5546 | 0.1628 | 0.8915 | 0.0468 |
| **Multimodal — Attention** | 0.6925 | 0.1396 | 0.1785 | 0.1538 | 0.6519 | 0.7309 | 0.1785 | 0.9210 | 0.1263 |


## 4. Ablation Analysis

| Model | Image | Metadata | Concatenation | Attention | Macro F1 | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Image Only** | ✓ | - | - | - | 0.1487 | 0.6734 |
| **Metadata Only** | - | ✓ | - | - | 0.1963 | 0.6817 |
| **Multimodal Concatenation** | ✓ | ✓ | ✓ | - | 0.1495 | 0.5546 |
| **Multimodal Attention** | ✓ | ✓ | - | ✓ | 0.1538 | 0.7309 |


## 5. Attention & Representation Interaction Analysis
- **Attention Weight Matrix**: Inter-modality self-attention matrix saved to `reports/figures/multimodal/attention_weights.png`.
- *Scientific Interpretation Note*: Attention weights demonstrate representation interaction across modality token spaces rather than direct causal feature importance.

## 6. Experimental Research Findings
- **Single-Modality Baseline Comparison**:
  - Image-Only ResNet50 Macro F1: **0.1487** (ROC-AUC: **0.6734**)
  - Metadata-Only MLP Macro F1: **0.1963** (ROC-AUC: **0.6817**)
- **Multimodal Fusion Performance**:
  - Multimodal Concatenation Macro F1: **0.1495** (ROC-AUC: **0.5546**)
  - Multimodal Attention Macro F1: **0.1538** (ROC-AUC: **0.7309**)
- **Best Overall Model**: **Metadata Only — MLP**

## 7. Artifact Locations
- **Model Checkpoints**: `models/multimodal/concat_model.pt`, `models/multimodal/attention_model.pt`
- **Master Comparison Table CSV**: `reports/results/model_comparison.csv`
- **ROC Comparison Figure**: `reports/figures/comparison/roc_comparison.png`
- **Embedding PCA Figure**: `reports/figures/embeddings/pca_embeddings_2d.png`
- **Attention Matrix Figure**: `reports/figures/multimodal/attention_weights.png`
- **Reusable Inference Module**: `src/models/multimodal_inference.py`
