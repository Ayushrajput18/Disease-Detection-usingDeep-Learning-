# Dataset Preparation & Leakage Prevention Report

## 1. Executive Summary & Source
- **Dataset Source**: ISIC 2019 (Training Subset)
- **Total Original Images**: 25331
- **Ground Truth Match Rate**: 100% (25,331 / 25,331)
- **Metadata Match Rate**: 100% (25,331 / 25,331)
- **Random Seed**: `42`

## 2. Duplicate Handling & Canonical Mapping
- **Exact Duplicate Pairs Detected**: 50 (100 total images)
- **Duplicate Mapping File**: `data/processed/duplicate_mapping.csv`
- **Canonical Subset Size**: 25281 images
- **Deduplication Rule**: All exact duplicate images were grouped into `duplicate_group_id`s. For primary experiments, 50 canonical images are retained, while 50 duplicate instances are tracked and isolated to ensure zero leakage across splits.

## 3. Diagnosis Target Definition
- **Target Variable**: `diagnosis` (mapped to `diagnosis_index`)
- **Class Mapping**:
```json
{
  "NV": 0,
  "MEL": 1,
  "BCC": 2,
  "BKL": 3,
  "AK": 4,
  "SCC": 5,
  "VASC": 6,
  "DF": 7
}
```
- **Validation**: 100% of samples contain exactly 1 positive diagnostic label. Zero sample anomalies detected.

## 4. Clinical Features & Non-Predictive Identifiers
- **Legitimate Predictive Clinical Features**:
  1. `age_approx` (Numerical, continuous)
  2. `sex` (Categorical: male, female)
  3. `anatom_site_general` (Categorical: 8 sites)
- **Strictly Non-Predictive Identifiers** (Excluded from model feature sets):
  - `image_id`, `lesion_id`, `lesion_group_id`, `combined_leakage_group_id`, `image_path`

## 5. Clinical Preprocessing Design
- **`age_approx`**: Missing count = 437 (1.73%). Imputation strategy: Training-set median imputation (55.0 years).
- **`sex`**: Missing count = 384 (1.52%). Imputation strategy: Explicit `'Unknown'` category.
- **`anatom_site_general`**: Missing count = 2,631 (10.39%). Imputation strategy: Explicit `'Unknown'` category.
- **Data Leakage Guarantee**: Preprocessing pipelines (e.g. `ColumnTransformer` / `StandardScaler` / `SimpleImputer`) are fitted **ONLY** on the training split.

## 6. Image Preprocessing Plan
- **Model Input Size**: $224 \times 224$ RGB pixels (designed for ResNet50 baseline).
- **Resizing Strategy**: Maintain aspect ratio with center crop / pad.
- **Training Augmentation**: Horizontal flip, mild random rotations ($\pm 15^\circ$), subtle brightness/contrast shifts ($\pm 10\%$).
- **Validation/Test Preprocessing**: Deterministic resizing and normalization ONLY. No augmentation applied.

## 7. Grouping Strategy & Leakage Prevention
- **Primary Grouping Identifier**: `combined_leakage_group_id`
- **Construction**: Merges `lesion_id` annotations with `duplicate_group_id` via connected components analysis.
- **Leakage Prevention Verification**:
  - Unique Lesion/Leakage Groups: 13927
  - Train-Validation Group Overlap: **0**
  - Train-Test Group Overlap: **0**
  - Validation-Test Group Overlap: **0**
  - Exact Duplicate Group Overlap Across Splits: **0**

## 8. Train / Validation / Test Splits & Class Distributions

| Partition | Total Images | Unique Lesion/Leakage Groups | Share (%) |
| :--- | :--- | :--- | :--- |
| **Training** | 17578 | 9745 | 69.39% |
| **Validation** | 3812 | 2091 | 15.05% |
| **Test** | 3941 | 2091 | 15.56% |

### Diagnostic Class Distribution Across Splits

| Class | Entire Dataset (%) | Train Count (%) | Validation Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- | :--- |
| **NV** | 12875 (50.83%) | 9017 (51.3%) | 2003 (52.54%) | 1855 (47.07%) |
| **MEL** | 4522 (17.85%) | 3078 (17.51%) | 647 (16.97%) | 797 (20.22%) |
| **BCC** | 3323 (13.12%) | 2299 (13.08%) | 486 (12.75%) | 538 (13.65%) |
| **BKL** | 2624 (10.36%) | 1818 (10.34%) | 400 (10.49%) | 406 (10.3%) |
| **AK** | 867 (3.42%) | 602 (3.42%) | 114 (2.99%) | 151 (3.83%) |
| **SCC** | 628 (2.48%) | 432 (2.46%) | 97 (2.54%) | 99 (2.51%) |
| **VASC** | 253 (1.0%) | 173 (0.98%) | 30 (0.79%) | 50 (1.27%) |
| **DF** | 239 (0.94%) | 159 (0.9%) | 35 (0.92%) | 45 (1.14%) |

## 9. Class Imbalance Handling Strategy
Inverse class weights calculated for the training split:
```json
{
  "NV": 0.2437,
  "MEL": 0.7139,
  "BCC": 0.9557,
  "BKL": 1.2086,
  "AK": 3.6499,
  "SCC": 5.0862,
  "VASC": 12.7009,
  "DF": 13.8192
}
```
*Note: Validation and test sets remain unweighted to represent real-world clinical distributions.*

## 10. Reproducibility & Saved Artifacts
- **Splits Directory**: `data/processed/splits/` (`train.csv`, `validation.csv`, `test.csv`)
- **Dataset Manifest**: `data/processed/dataset_manifest.json`
- **Class Mapping**: `reports/dataset/class_mapping.json`
- **Duplicate Mapping**: `data/processed/duplicate_mapping.csv`
- **Sample Figure**: `reports/figures/sample_images.png`

## 11. Known Limitations & Patient ID Clarification
- **Patient ID Availability**: ISIC 2019 metadata does **NOT** provide verified Patient IDs. All grouping is executed strictly at the `lesion_id` level. Patient-level grouping cannot be guaranteed beyond available lesion tags.
- **Class Imbalance**: High imbalance remains (NV represents ~50.8% of samples while DF represents ~0.94%). Loss weighting or weighted sampling will be required during model training.
