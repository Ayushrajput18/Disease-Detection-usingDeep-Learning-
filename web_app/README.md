# ISIC 2019 Multimodal Skin Lesion Web Application

## Overview
An interactive web application demonstrating multimodal deep learning for skin lesion classification using dermoscopic images and clinical metadata (Age, Sex, Anatomical Site).

---

## 1. Status of Model Training
**Training is 100% COMPLETE.** No further training is required.

All 4 models have been trained, validated, and saved with complete reproducibility:
1. **Multimodal Attention Fusion (Best)**: `models/multimodal/attention_model.pt`
   - Test Accuracy: **~69.25%** | ROC-AUC: **~0.7309**
2. **Multimodal Concat Fusion**: `models/multimodal/concat_model.pt`
   - Test Accuracy: **~68.75%** | ROC-AUC: **~0.7251**
3. **Image-Only ResNet50**: `models/image_only/best_model.pt`
   - Test Accuracy: **~63.91%** | ROC-AUC: **~0.6734**
4. **Metadata-Only MLP**: `models/metadata_only/best_model.pt` + `preprocessor.joblib`
   - Test Accuracy: **~36.26%** | ROC-AUC: **~0.6817**

---

## 2. Web Application Features
- **Live Local Server**: Hosted at `http://localhost:5055`
- **1-Click Pre-loaded Samples**: 8 authentic test cases from the ISIC 2019 test split covering all diagnostic categories:
  - Melanoma (`MEL`)
  - Melanocytic Nevus (`NV`)
  - Basal Cell Carcinoma (`BCC`)
  - Benign Keratosis (`BKL`)
  - Actinic Keratosis (`AK`)
  - Squamous Cell Carcinoma (`SCC`)
  - Vascular Lesion (`VASC`)
  - Dermatofibroma (`DF`)
- **Custom Image Upload**: Drag-and-drop or file picker for custom dermoscopy images.
- **Clinical Metadata Inputs**: Age, Sex, and Anatomical Site.
- **Model Selection & Comparison**: Run individual models or compare all 4 architectures side-by-side.
- **Explainability**:
  - **Live Grad-CAM**: Visualizes ResNet50 `layer4` attention regions with an interactive *Overlay* vs *Original* toggle.
  - **Cross-Modal Attention Matrix**: Displays inter-modality token weights between Image and Metadata.
- **Probability Distribution**: Full 8-class softmax breakdown with color-coded severity badges (Benign, Pre-Malignant, Malignant).

---

## 3. How to Run / Restart the Web App
Run the following command from the project root:
```bash
python web_app/app.py
```
Then navigate to:
```
http://localhost:5055
```
