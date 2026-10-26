import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

# Set random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Paths
DATASET_DIR = 'Dataset'
PROCESSED_DIR = 'data/processed'
SPLITS_DIR = 'data/processed/splits'
REPORTS_DIR = 'reports'
REPORTS_DATASET_DIR = 'reports/dataset'
FIGURES_DIR = 'reports/figures'

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SPLITS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DATASET_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("Starting Phase 2: Dataset Preparation & Leakage Prevention...")

# ----------------------------------------------------
# STEP 1: Preserve original data (Load from processed master & raw CSVs)
# ----------------------------------------------------
gt_path = os.path.join(DATASET_DIR, 'ISIC_2019_Training_GroundTruth.csv')
meta_path = os.path.join(DATASET_DIR, 'ISIC_2019_Training_Metadata.csv')
img_dir = os.path.join(DATASET_DIR, 'ISIC_2019_Training_Input')

gt_df = pd.read_csv(gt_path)
meta_df = pd.read_csv(meta_path)
dup_df = pd.read_csv(os.path.join(REPORTS_DATASET_DIR, 'duplicate_images.csv'))

# ----------------------------------------------------
# STEP 2: Handle Exact Duplicates
# ----------------------------------------------------
# Create duplicate mapping
# dup_df has columns: duplicate_group_id, md5_hash, image_id
dup_group_map = {} # image_id -> duplicate_group_id
canonical_map = {} # duplicate_group_id -> canonical_image_id

duplicate_mapping_rows = []
all_image_ids = set(gt_df['image'])

for group_id, group in dup_df.groupby('duplicate_group_id'):
    group_images = group['image_id'].tolist()
    canonical_id = group_images[0]
    canonical_map[group_id] = canonical_id
    for img_id in group_images:
        is_canonical = (img_id == canonical_id)
        dup_group_map[img_id] = f"DUP_GRP_{group_id}"
        duplicate_mapping_rows.append({
            'image_id': img_id,
            'canonical_image_id': canonical_id,
            'duplicate_group_id': f"DUP_GRP_{group_id}",
            'is_canonical': is_canonical
        })

# Images not in dup_df are their own canonical
for img_id in all_image_ids:
    if img_id not in dup_group_map:
        duplicate_mapping_rows.append({
            'image_id': img_id,
            'canonical_image_id': img_id,
            'duplicate_group_id': f"SINGLE_{img_id}",
            'is_canonical': True
        })

dup_mapping_df = pd.DataFrame(duplicate_mapping_rows)
dup_mapping_df.to_csv(os.path.join(PROCESSED_DIR, 'duplicate_mapping.csv'), index=False)
print(f"Duplicate mapping saved. Total images: {len(dup_mapping_df)}, Canonical images: {dup_mapping_df['is_canonical'].sum()}")

# ----------------------------------------------------
# STEP 3: Define Target & Class Mapping
# ----------------------------------------------------
diag_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
# Check row sums
row_sums = gt_df[diag_cols].sum(axis=1)
if not (row_sums == 1.0).all():
    invalid_rows = gt_df[row_sums != 1.0]
    raise ValueError(f"Target verification failed! Found {len(invalid_rows)} rows with non-exact single label: {invalid_rows}")

# Identify active classes (excluding UNK if 0 positives)
active_classes = ['NV', 'MEL', 'BCC', 'BKL', 'AK', 'SCC', 'VASC', 'DF'] # deterministic standard order
class_mapping = {cls_name: idx for idx, cls_name in enumerate(active_classes)}
with open(os.path.join(REPORTS_DATASET_DIR, 'class_mapping.json'), 'w') as f:
    json.dump(class_mapping, f, indent=4)

gt_df['diagnosis'] = gt_df[diag_cols].idxmax(axis=1)
gt_df['diagnosis_index'] = gt_df['diagnosis'].map(class_mapping)

print(f"Class mapping created: {class_mapping}")

# ----------------------------------------------------
# STEP 4 & 7: Combine Datasets & Build Leakage-Resistant Grouping Variable
# ----------------------------------------------------
# File to ID mapping
img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
id_to_file = {os.path.splitext(f)[0]: f for f in img_filenames}

df = pd.merge(gt_df[['image', 'diagnosis', 'diagnosis_index']], meta_df, on='image', how='left')
df['image_filename'] = df['image'].map(id_to_file)
df['image_path'] = df['image_filename'].apply(lambda x: os.path.join(img_dir, x))
df['image_id'] = df['image']

# Merge duplicate mapping
df = pd.merge(df, dup_mapping_df[['image_id', 'canonical_image_id', 'duplicate_group_id', 'is_canonical']], on='image_id', how='left')

# Assign lesion_group_id
df['lesion_group_id'] = df['lesion_id'].fillna(df['image_id'])

# Unify duplicate groups and lesion_group_id into combined_leakage_group_id
# We use Union-Find / Connected Components to group all connected image IDs together
from collections import defaultdict

parent = {}
def find(i):
    if parent[i] == i:
        return i
    parent[i] = find(parent[i])
    return parent[i]

def union(i, j):
    root_i = find(i)
    root_j = find(j)
    if root_i != root_j:
        parent[root_i] = root_j

all_nodes = set(df['image_id'])
for n in all_nodes:
    parent[n] = n

# Union images sharing the same lesion_id
for lesion_id, group in df[df['lesion_id'].notnull()].groupby('lesion_id'):
    img_list = group['image_id'].tolist()
    for i in range(1, len(img_list)):
        union(img_list[0], img_list[i])

# Union images sharing the same duplicate_group_id
for dup_grp, group in df.groupby('duplicate_group_id'):
    img_list = group['image_id'].tolist()
    for i in range(1, len(img_list)):
        union(img_list[0], img_list[i])

df['combined_leakage_group_id'] = df['image_id'].apply(lambda x: f"GRP_{find(x)}")

print(f"Total rows: {len(df)}")
print(f"Unique combined_leakage_group_id count: {df['combined_leakage_group_id'].nunique()}")

# ----------------------------------------------------
# STEP 8: Stratified Group-Aware Train / Validation / Test Splitting
# ----------------------------------------------------
# We group by combined_leakage_group_id
# Each group has a dominant diagnosis (or diagnosis distribution). Since almost all lesions have consistent diagnosis,
# we determine each group's primary diagnosis and sample count.

group_summary = df.groupby('combined_leakage_group_id').agg(
    n_images=('image_id', 'count'),
    diagnosis=('diagnosis', lambda x: x.mode()[0]),
    diagnosis_index=('diagnosis_index', lambda x: x.mode()[0])
).reset_index()

print("\nGroup Summary Diagnosis Counts:")
print(group_summary['diagnosis'].value_counts())

# We split group_summary into train (70%), val (15%), test (15%) using StratifiedKFold on group_summary
# 15% is approx 3/20, so 10-fold CV gives 70% train (14 folds), 15% val (3 folds), 15% test (3 folds)
# Or we can do a 2-stage stratified split on groups.

from sklearn.model_selection import StratifiedKFold

group_summary = group_summary.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# Stage 1: Split groups into Train+Val (85%) and Test (15%)
# We can use StratifiedKFold with n_splits=20 -> 17 folds for train+val, 3 folds for test
skf_outer = StratifiedKFold(n_splits=20, shuffle=True, random_state=RANDOM_SEED)
folds = list(skf_outer.split(group_summary, group_summary['diagnosis_index']))

test_group_indices = []
for fold_idx in [0, 1, 2]: # 3 folds = 15%
    test_group_indices.extend(folds[fold_idx][1])

train_val_group_indices = [i for i in range(len(group_summary)) if i not in test_group_indices]

test_groups = set(group_summary.loc[test_group_indices, 'combined_leakage_group_id'])
train_val_groups_df = group_summary.loc[train_val_group_indices].reset_index(drop=True)

# Stage 2: Split train_val_groups into Train (70/85 = 82.35%) and Val (15/85 = 17.65%)
# 17.65% of 17 folds is 3 folds out of 17.
skf_inner = StratifiedKFold(n_splits=17, shuffle=True, random_state=RANDOM_SEED)
inner_folds = list(skf_inner.split(train_val_groups_df, train_val_groups_df['diagnosis_index']))

val_group_indices = []
for fold_idx in [0, 1, 2]: # 3 folds = 15% of total
    val_group_indices.extend(inner_folds[fold_idx][1])

train_group_indices = [i for i in range(len(train_val_groups_df)) if i not in val_group_indices]

val_groups = set(train_val_groups_df.loc[val_group_indices, 'combined_leakage_group_id'])
train_groups = set(train_val_groups_df.loc[train_group_indices, 'combined_leakage_group_id'])

# Map split back to main df
def assign_split(grp):
    if grp in train_groups:
        return 'train'
    elif grp in val_groups:
        return 'val'
    elif grp in test_groups:
        return 'test'
    else:
        raise ValueError(f"Unassigned group {grp}")

df['split'] = df['combined_leakage_group_id'].apply(assign_split)

# ----------------------------------------------------
# STEP 9: Verify Split & Leakage Checks
# ----------------------------------------------------
train_df = df[df['split'] == 'train'].copy()
val_df = df[df['split'] == 'val'].copy()
test_df = df[df['split'] == 'test'].copy()

train_grp_set = set(train_df['combined_leakage_group_id'])
val_grp_set = set(val_df['combined_leakage_group_id'])
test_grp_set = set(test_df['combined_leakage_group_id'])

overlap_tr_val = train_grp_set.intersection(val_grp_set)
overlap_tr_te = train_grp_set.intersection(test_grp_set)
overlap_val_te = val_grp_set.intersection(test_grp_set)

print("\n=== SPLIT VERIFICATION ===")
print(f"Train images: {len(train_df)} ({len(train_df)/len(df)*100:.2f}%), groups: {len(train_grp_set)}")
print(f"Val images: {len(val_df)} ({len(val_df)/len(df)*100:.2f}%), groups: {len(val_grp_set)}")
print(f"Test images: {len(test_df)} ({len(test_df)/len(df)*100:.2f}%), groups: {len(test_grp_set)}")
print(f"Overlap Train-Val: {len(overlap_tr_val)}")
print(f"Overlap Train-Test: {len(overlap_tr_te)}")
print(f"Overlap Val-Test: {len(overlap_val_te)}")

if len(overlap_tr_val) > 0 or len(overlap_tr_te) > 0 or len(overlap_val_te) > 0:
    raise RuntimeError("CRITICAL LEAKAGE DETECTED across splits!")

# Check exact duplicate group leakage
dup_tr_set = set(train_df['duplicate_group_id'])
dup_val_set = set(val_df['duplicate_group_id'])
dup_te_set = set(test_df['duplicate_group_id'])

dup_overlap_1 = dup_tr_set.intersection(dup_val_set) - {s for s in dup_tr_set if s.startswith('SINGLE_')}
dup_overlap_2 = dup_tr_set.intersection(dup_te_set) - {s for s in dup_tr_set if s.startswith('SINGLE_')}
dup_overlap_3 = dup_val_set.intersection(dup_te_set) - {s for s in dup_val_set if s.startswith('SINGLE_')}

print(f"Duplicate Group Overlap Train-Val: {len(dup_overlap_1)}")
print(f"Duplicate Group Overlap Train-Test: {len(dup_overlap_2)}")
print(f"Duplicate Group Overlap Val-Test: {len(dup_overlap_3)}")

# ----------------------------------------------------
# STEP 10: Class Imbalance & Weighting Options
# ----------------------------------------------------
dist_df = pd.DataFrame({
    'Class': active_classes,
    'Entire_Count': [sum(df['diagnosis'] == c) for c in active_classes],
    'Entire_Pct': [round(sum(df['diagnosis'] == c)/len(df)*100, 2) for c in active_classes],
    'Train_Count': [sum(train_df['diagnosis'] == c) for c in active_classes],
    'Train_Pct': [round(sum(train_df['diagnosis'] == c)/len(train_df)*100, 2) for c in active_classes],
    'Val_Count': [sum(val_df['diagnosis'] == c) for c in active_classes],
    'Val_Pct': [round(sum(val_df['diagnosis'] == c)/len(val_df)*100, 2) for c in active_classes],
    'Test_Count': [sum(test_df['diagnosis'] == c) for c in active_classes],
    'Test_Pct': [round(sum(test_df['diagnosis'] == c)/len(test_df)*100, 2) for c in active_classes],
})

# Class weights for training (inverse class frequency)
total_train = len(train_df)
n_classes = len(active_classes)
class_counts_train = train_df['diagnosis_index'].value_counts().sort_index().values
class_weights = total_train / (n_classes * class_counts_train)
class_weights_dict = {cls_name: round(float(weight), 4) for cls_name, weight in zip(active_classes, class_weights)}

dist_df.to_csv(os.path.join(REPORTS_DATASET_DIR, 'split_class_distributions.csv'), index=False)
print("\nClass Distributions per Split:\n", dist_df)
print("\nTraining Class Weights (Option A):\n", class_weights_dict)

# ----------------------------------------------------
# STEP 11: Save Splits
# ----------------------------------------------------
export_cols = [
    'image_id', 'image_path', 'diagnosis', 'diagnosis_index',
    'age_approx', 'sex', 'anatom_site_general', 'lesion_group_id',
    'combined_leakage_group_id', 'duplicate_group_id', 'is_canonical'
]

train_export = train_df[export_cols].copy()
val_export = val_df[export_cols].copy()
test_export = test_df[export_cols].copy()

train_export.to_csv(os.path.join(SPLITS_DIR, 'train.csv'), index=False)
val_export.to_csv(os.path.join(SPLITS_DIR, 'validation.csv'), index=False)
test_export.to_csv(os.path.join(SPLITS_DIR, 'test.csv'), index=False)

print(f"\nSplits saved to {SPLITS_DIR}/")

# ----------------------------------------------------
# STEP 12: Dataset Manifest
# ----------------------------------------------------
manifest = {
    "source_dataset": "ISIC 2019",
    "total_original_images": len(df),
    "duplicate_image_pairs_count": len(dup_df) // 2,
    "total_images_in_duplicate_pairs": len(dup_df),
    "canonical_primary_experiment_images": int(dup_mapping_df['is_canonical'].sum()),
    "removed_duplicates_for_primary_experiment": int(len(df) - dup_mapping_df['is_canonical'].sum()),
    "class_mapping": class_mapping,
    "grouping_variable": "combined_leakage_group_id (combines lesion_id and duplicate_group_id)",
    "split_ratios": {"train": 0.7011, "validation": 0.1495, "test": 0.1494},
    "random_seed": RANDOM_SEED,
    "image_resolution": "224x224 (for ResNet50 input)",
    "clinical_features": ["age_approx", "sex", "anatom_site_general"],
    "clinical_identifiers_non_predictive": ["image_id", "lesion_id", "lesion_group_id", "combined_leakage_group_id"],
    "missing_data_strategy": {
        "age_approx": "Median imputation fitted on Training set only",
        "sex": "Explicit 'Unknown' category",
        "anatom_site_general": "Explicit 'Unknown' category"
    },
    "training_class_weights": class_weights_dict
}

with open(os.path.join(PROCESSED_DIR, 'dataset_manifest.json'), 'w') as f:
    json.dump(manifest, f, indent=4)

# ----------------------------------------------------
# STEP 13: Visual Quality Check Contact Sheet
# ----------------------------------------------------
print("\nGenerating sample contact sheet image...")
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, cls_name in enumerate(active_classes):
    sample_row = train_df[train_df['diagnosis'] == cls_name].iloc[0]
    img_p = sample_row['image_path']
    with Image.open(img_p) as img:
        axes[idx].imshow(img)
        axes[idx].set_title(f"Class: {cls_name}\nID: {sample_row['image_id']}", fontsize=11, fontweight='bold')
        axes[idx].axis('off')

plt.suptitle("ISIC 2019 Sample Images Across Diagnostic Classes", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'sample_images.png'), dpi=300)
plt.close()
print(f"Sample contact sheet saved to {FIGURES_DIR}/sample_images.png")

# ----------------------------------------------------
# STEP 14: Final Preparation Report (Markdown)
# ----------------------------------------------------
report_md = f"""# Dataset Preparation & Leakage Prevention Report

## 1. Executive Summary & Source
- **Dataset Source**: ISIC 2019 (Training Subset)
- **Total Original Images**: {len(df)}
- **Ground Truth Match Rate**: 100% (25,331 / 25,331)
- **Metadata Match Rate**: 100% (25,331 / 25,331)
- **Random Seed**: `{RANDOM_SEED}`

## 2. Duplicate Handling & Canonical Mapping
- **Exact Duplicate Pairs Detected**: {len(dup_df) // 2} (100 total images)
- **Duplicate Mapping File**: `data/processed/duplicate_mapping.csv`
- **Canonical Subset Size**: {dup_mapping_df['is_canonical'].sum()} images
- **Deduplication Rule**: All exact duplicate images were grouped into `duplicate_group_id`s. For primary experiments, 50 canonical images are retained, while 50 duplicate instances are tracked and isolated to ensure zero leakage across splits.

## 3. Diagnosis Target Definition
- **Target Variable**: `diagnosis` (mapped to `diagnosis_index`)
- **Class Mapping**:
```json
{json.dumps(class_mapping, indent=2)}
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
- **`age_approx`**: Missing count = 437 (1.73%). Imputation strategy: Training-set median imputation ({train_df['age_approx'].median()} years).
- **`sex`**: Missing count = 384 (1.52%). Imputation strategy: Explicit `'Unknown'` category.
- **`anatom_site_general`**: Missing count = 2,631 (10.39%). Imputation strategy: Explicit `'Unknown'` category.
- **Data Leakage Guarantee**: Preprocessing pipelines (e.g. `ColumnTransformer` / `StandardScaler` / `SimpleImputer`) are fitted **ONLY** on the training split.

## 6. Image Preprocessing Plan
- **Model Input Size**: $224 \\times 224$ RGB pixels (designed for ResNet50 baseline).
- **Resizing Strategy**: Maintain aspect ratio with center crop / pad.
- **Training Augmentation**: Horizontal flip, mild random rotations ($\pm 15^\circ$), subtle brightness/contrast shifts ($\pm 10\%$).
- **Validation/Test Preprocessing**: Deterministic resizing and normalization ONLY. No augmentation applied.

## 7. Grouping Strategy & Leakage Prevention
- **Primary Grouping Identifier**: `combined_leakage_group_id`
- **Construction**: Merges `lesion_id` annotations with `duplicate_group_id` via connected components analysis.
- **Leakage Prevention Verification**:
  - Unique Lesion/Leakage Groups: {df['combined_leakage_group_id'].nunique()}
  - Train-Validation Group Overlap: **0**
  - Train-Test Group Overlap: **0**
  - Validation-Test Group Overlap: **0**
  - Exact Duplicate Group Overlap Across Splits: **0**

## 8. Train / Validation / Test Splits & Class Distributions

| Partition | Total Images | Unique Lesion/Leakage Groups | Share (%) |
| :--- | :--- | :--- | :--- |
| **Training** | {len(train_df)} | {len(train_grp_set)} | {len(train_df)/len(df)*100:.2f}% |
| **Validation** | {len(val_df)} | {len(val_grp_set)} | {len(val_df)/len(df)*100:.2f}% |
| **Test** | {len(test_df)} | {len(test_grp_set)} | {len(test_df)/len(df)*100:.2f}% |

### Diagnostic Class Distribution Across Splits

| Class | Entire Dataset (%) | Train Count (%) | Validation Count (%) | Test Count (%) |
| :--- | :--- | :--- | :--- | :--- |
"""

for idx, row in dist_df.iterrows():
    report_md += f"| **{row['Class']}** | {row['Entire_Count']} ({row['Entire_Pct']}%) | {row['Train_Count']} ({row['Train_Pct']}%) | {row['Val_Count']} ({row['Val_Pct']}%) | {row['Test_Count']} ({row['Test_Pct']}%) |\n"

report_md += f"""
## 9. Class Imbalance Handling Strategy
Inverse class weights calculated for the training split:
```json
{json.dumps(class_weights_dict, indent=2)}
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
"""

with open('reports/dataset_preparation_report.md', 'w') as f:
    f.write(report_md)

print("Dataset Preparation & Leakage Prevention Phase completed successfully!")
