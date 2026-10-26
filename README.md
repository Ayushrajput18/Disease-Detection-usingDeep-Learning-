# Disease Detection using Deep Learning
## Multimodal Skin Lesion Diagnostic Classification & Clinical Research Platform (DermaFusion)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![ISIC 2019](https://img.shields.io/badge/Dataset-ISIC%202019-teal.svg)](https://challenge.isic-archive.com/landing/2019/)
[![Architecture](https://img.shields.io/badge/Architecture-ResNet50%20%2B%20Cross--Attention-purple.svg)]()
[![Web App](https://img.shields.io/badge/Interface-DermaFusion%20Web%20App-0284c7.svg)]()

An end-to-end deep learning system for automated skin cancer diagnosis and explainability built on the **ISIC 2019 benchmark dataset** (25,331 clinical dermoscopic images across 8 diagnostic classes).

The system integrates **Dual-Stream Cross-Attention Fusion** combining visual representations from a convolutional neural network (ResNet50) and patient clinical metadata (Age, Sex, Anatomical Site, Lesion Diameter, Elevation, Evolutionary History) into a unified diagnostic decision support system.

---

## 🌟 Key Highlights

- **8 Diagnostic Classes**: Melanoma (MEL), Melanocytic Nevus (NV), Basal Cell Carcinoma (BCC), Benign Keratosis (BKL), Actinic Keratosis (AK), Squamous Cell Carcinoma (SCC), Vascular Lesion (VASC), Dermatofibroma (DF).
- **Data Leakage Prevention**: Strict StratifiedGroupKFold split grouped on lesion identifier (lesion_group_id) to ensure images of the same lesion never cross between train, validation, and test partitions.
- **Dual-Stream Cross-Attention Architecture**: Projects 2048-dimensional visual embeddings and 32-dimensional clinical embeddings into a 256-dimensional token space with 4-head bidirectional cross-attention.
- **Explainable AI (XAI)**:
  - **Grad-CAM**: Visualizes ResNet50 layer4 convolutional feature map saliency with directional axes and hotspot foci.
  - **SHAP (KernelExplainer)**: Computes additive feature importance and patient-specific risk delta.
  - **Cross-Attention Token Matrix**: Quantifies bidirectional relevance between image patches and clinical variables.
  - **Monte Carlo Dropout**: Quantifies Epistemic Uncertainty ($\sigma$) to detect ambiguous, borderline lesions.
- **DermaFusion Clinical Protocol Web App**: High-fidelity clinical research interface featuring optical microscope viewports, pre-loaded demo cohorts, and side-by-side modality ablation comparisons.

---

## 📊 Benchmark Model Performance

Evaluated on **3,941 held-out test lesions** with zero data leakage:

| Model Architecture | Modalities Evaluated | Test Accuracy | Macro ROC-AUC | Weighted F1 | Epistemic Uncertainty ($\sigma$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Multimodal Attention Fusion (Best)** | **Dermoscopy + Clinical Metadata** | **69.25%** | **0.7309** | **0.6519** | **0.041 (Low)** |
| Multimodal Concatenation | Dermoscopy + Clinical Metadata | 68.63% | 0.5546 | 0.6154 | 0.052 |
| Image-Only ResNet50 Baseline | Dermoscopy Image Only | 63.91% | 0.6734 | 0.6056 | 0.082 |
| Metadata-Only MLP Baseline | Patient Demographics Only | 36.26% | 0.6817 | 0.3996 | 0.115 (High) |

---

## 📁 Repository Structure

`
├── configs/                      # Experiment hyperparameters (YAML)
│   ├── image_only.yaml
│   ├── metadata_only.yaml
│   ├── multimodal_attention.yaml
│   └── multimodal_concat.yaml
├── data/
│   └── processed/
│       ├── embeddings/           # Precomputed 2048-d and 32-d modality vectors
│       └── splits/               # Leakage-free train, val, test splits (CSV)
├── explainability/               # Generated XAI artifacts
│   ├── attention/                # Cross-attention matrices and head breakdowns
│   ├── gradcam/                  # 8-class Grad-CAM heatmaps and grids
│   └── shap/                     # SHAP summary plots and feature ranking CSV
├── models/                       # Trained PyTorch model checkpoints & preprocessor
│   ├── image_only/best_model.pt
│   ├── metadata_only/best_model.pt
│   ├── metadata_only/preprocessor.joblib
│   └── multimodal/attention_model.pt
├── reports/                      # Detailed research reports and ablation figures
├── src/                          # Model definitions & dataset loaders
│   ├── dataset.py
│   └── models/
│       ├── image_model.py        # ISICImageClassifier (ResNet50)
│       ├── clinical_model.py     # ISICMetadataClassifier (MLP)
│       └── multimodal_model.py   # Cross-Attention & Concat Fusion
├── web_app/                      # DermaFusion interactive web platform
│   ├── app.py                    # Flask inference API & XAI engine
│   ├── static/                   # CSS, JS, and pre-loaded sample cases
│   └── templates/index.html      # High-density clinical research interface
├── run_audit.py                  # Phase 1: ISIC 2019 dataset audit
├── prepare_dataset.py            # Phase 2: Deduplication & group splitting
├── train_image_model.py          # Phase 3: ResNet50 image-only baseline
├── train_clinical_model.py       # Phase 4: Metadata MLP baseline
├── train_multimodal.py           # Phase 5: Multimodal fusion training
└── run_phase6_explainability.py  # Phase 6: Full XAI pipeline (Grad-CAM + SHAP)
`

---

## 🚀 Quickstart Guide

### 1. Installation
Clone the repository and install dependencies:
`ash
git clone https://github.com/Ayushrajput18/Disease-Detection-usingDeep-Learning-.git
cd Disease-Detection-usingDeep-Learning-
pip install torch torchvision numpy pandas scikit-learn matplotlib shap joblib pyyaml flask Pillow
`

### 2. Launch the DermaFusion Web Application
`ash
python web_app/app.py
`
Open **http://localhost:5055** in your browser.

- Select any of the **8 pre-loaded authentic ISIC demo cohorts** (Melanoma, Nevus, Basal Cell Carcinoma, etc.).
- Inspect the **calibrated probability distribution**, **ICD-O-3 classification**, and **sub-model ablation analysis**.
- Navigate to the **Explainability Lab** to examine **Grad-CAM activation foci**, **SHAP feature waterfall attributions**, and **cross-attention matrices**.

---

## 🔬 Scientific Disclaimer
This project is an **academic research prototype** developed for educational and research purposes. It is **NOT** a certified medical diagnostic device and must **NEVER** replace professional clinical examination, dermoscopy, or biopsy histopathology.
