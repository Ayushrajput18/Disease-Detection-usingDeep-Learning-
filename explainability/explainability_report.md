# Phase 6 — Model Explainability & Interpretability Report

**Date:** 2026-10-07 22:53:58
**Project:** ISIC 2019 Multimodal Skin Lesion Classification
**Best Model:** Multimodal Attention Fusion (Accuracy: ~69.25%, ROC-AUC: ~0.7309)

---

## 1. Explainability Objective

The objective of Phase 6 is to provide scientifically sound interpretability of the trained
models using three complementary techniques:

1. **Grad-CAM:** Visualize which image regions contributed to ResNet50 predictions.
2. **SHAP:** Estimate the contribution of clinical metadata features to MLP predictions.
3. **Attention Visualization:** Analyze inter-modality interaction in the attention fusion model.

This report documents methodology, findings, and limitations in accordance with responsible
AI research practices.

**CRITICAL SCOPE NOTE:** This system is an academic research prototype and is NOT a clinically
validated diagnostic system.

---

## 2. Grad-CAM Methodology

Grad-CAM (Gradient-weighted Class Activation Mapping, Selvaraju et al., 2017) was applied to
the trained ResNet50 image classifier.

- **Target layer:** `feature_extractor[-1]` = `layer4` (final convolutional block of ResNet50)
- **Checkpoint:** `models/image_only/best_model.pt`
- **Test samples analyzed:** 8 (one per diagnostic class)
- **Procedure:**
  1. Forward pass of test image through the model.
  2. Backpropagate gradient of predicted class score to `layer4` feature maps.
  3. Pool gradients spatially (global average pooling).
  4. Weight feature maps by pooled gradients.
  5. Apply ReLU (retain only positive contributions).
  6. Normalize heatmap to [0, 1].
  7. Overlay heatmap on original image.

**Results summary:**

| Class | Predicted | Confidence | Correct? |
|-------|-----------|------------|----------|
| NV | NV | 42.27% | Yes |
| MEL | NV | 24.35% | No |
| BCC | NV | 20.20% | No |
| BKL | BKL | 22.47% | Yes |
| AK | AK | 45.83% | Yes |
| SCC | SCC | 34.38% | Yes |
| VASC | BCC | 23.93% | No |
| DF | VASC | 24.37% | No |

> **SCIENTIFIC NOTE:** Grad-CAM provides a visual indication of image regions that
> contributed to the model's prediction. It does NOT prove that the model is clinically
> correct, nor does it constitute a clinical ground-truth segmentation.

---

## 3. SHAP Methodology

SHAP (SHapley Additive exPlanations) was applied to the trained Metadata-Only MLP using a
`KernelExplainer` with 100 background training samples.

- **Features explained:** age_approx, sex_Unknown, sex_female, sex_male, anatom_site_general_Unknown, anatom_site_general_anterior torso, anatom_site_general_head/neck, anatom_site_general_lateral torso, anatom_site_general_lower extremity, anatom_site_general_oral/genital, anatom_site_general_palms/soles, anatom_site_general_posterior torso, anatom_site_general_upper extremity
- **Test samples explained:** 200
- **SHAP nsamples (approximation):** 50
- **Preprocessing:** The trained `preprocessor.joblib` pipeline (median imputation +
  StandardScaler for age; 'Unknown' category + OneHotEncoder for sex/site) was applied
  identically to the SHAP explanation set.
- **SHAP values represent:** The estimated additive contribution of each preprocessed feature
  to the model's predicted class probabilities.

> **SCIENTIFIC RULE:** SHAP estimates the contribution of model features to individual predictions.
> SHAP values do NOT establish causal relationships between features and skin lesion diagnoses.


---

## 4. Attention Visualization Methodology

The multimodal attention model uses `nn.MultiheadAttention` with:
- **4 attention heads**
- **2 modality tokens:** `[IMAGE_TOKEN, METADATA_TOKEN]`
- **Token dimension:** 256
- **Attention matrix shape:** (Batch, 2, 2) — averaged across 4 heads

Attention weights were computed over 500 randomly sampled test examples.

**Mean Attention Matrix:**

|  | → IMAGE | → METADATA |
|---|---|---|
| **IMAGE →** | 0.5364 | 0.4636 |
| **METADATA →** | 0.7842 | 0.2158 |

> **SCIENTIFIC NOTE:** The attention mechanism provides evidence of representation interaction
> between the image and metadata modalities. Attention weights SHOULD NOT be automatically
> interpreted as feature importance or causal evidence.

---

## 5. Representative Findings

### Grad-CAM
- Grad-CAM activations were successfully generated for 8 of 8 diagnostic classes.
- All 8 classes represented.
- Mean activation coverage (threshold > 0.5): **8.42%** of image area per sample.
- For some classes the model focused on central lesion regions; for others, activation was more distributed.

### SHAP
- Global SHAP analysis completed on 200 test samples.
- Feature importance ranked by mean |SHAP|: age_approx (0.0454), anatom_site_general_head/neck (0.0213), anatom_site_general_lower extremity (0.0189), anatom_site_general_anterior torso (0.0106), anatom_site_general_Unknown (0.0088), sex_female (0.0071), anatom_site_general_posterior torso (0.0061), sex_male (0.0057), anatom_site_general_upper extremity (0.0049), sex_Unknown (0.0014), anatom_site_general_lateral torso (0.0005), anatom_site_general_palms/soles (0.0001), anatom_site_general_oral/genital (0.0000)

### Attention
- IMAGE -> METADATA cross-modal attention: **0.4636** (std=0.0869)
- METADATA -> IMAGE cross-modal attention: **0.7842** (std=0.0558)
- Stronger cross-modal direction: **META->IMG**
- This pattern is descriptive only. It does not establish that one modality caused the other's contribution.

---

## 6. Limitations

1. **Grad-CAM** is a model-attribution visualization, NOT a clinical ground-truth segmentation.
2. **SHAP** measures model-feature contribution, NOT causation. Higher/lower age values are
   associated with changes in the model's predicted probability, but this does not imply
   a causal relationship between age and lesion diagnosis.
3. **Attention weights** should not automatically be interpreted as feature importance or
   causal evidence.
4. The ISIC 2019 metadata (`age_approx`, `sex`, `anatom_site_general`) represents limited
   demographic information. It is NOT a complete patient clinical record.
5. `lesion_id` is not a confirmed patient identifier. The dataset may contain images from
   the same lesion across multiple time points.
6. The model is an **academic research prototype** and has NOT been clinically validated.
7. Class imbalance in ISIC 2019 (NV heavily over-represented) affects all metrics.
8. Grad-CAM heatmaps were generated on CPU-mode PyTorch; this does not affect correctness.

---

## 7. Scientific Interpretation

The multimodal attention model (Accuracy ~69.25%, ROC-AUC ~0.7309) outperforms both the
image-only ResNet50 baseline (63.91%, 0.6734) and the metadata-only MLP (36.26%, 0.6817).

Explainability analysis suggests:
- The image branch captures spatial lesion morphology (Grad-CAM heatmaps).
- The metadata branch provides contextual demographic information (SHAP contributions).
- The attention mechanism integrates both modalities, with a non-trivial cross-modal
  attention pattern (IMG→META: 0.464, META→IMG: 0.784).

These observations are consistent with the hypothesis that combining image and metadata
improves diagnostic classification. However, this does NOT prove clinical utility.

---

## 8. Reproducibility Information

- Random seed: 42
- Image model checkpoint: `models/image_only/best_model.pt`
- Metadata model checkpoint: `models/metadata_only/best_model.pt`
- Multimodal model checkpoint: `models/multimodal/attention_model.pt`
- Preprocessor: `models/metadata_only/preprocessor.joblib`
- Test split: `data/processed/splits/test.csv`
- Grad-CAM target layer: `feature_extractor[-1]` (layer4)
- SHAP method: KernelExplainer, nsamples=50
- Attention samples: 500
- Platform: Windows, PyTorch 2.14.1+cpu, CPU mode
