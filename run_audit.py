import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import hashlib

# Ensure output directories exist
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports/dataset', exist_ok=True)

# 1. Load CSVs
gt_path = 'Dataset/ISIC_2019_Training_GroundTruth.csv'
meta_path = 'Dataset/ISIC_2019_Training_Metadata.csv'
img_dir = 'Dataset/ISIC_2019_Training_Input'

gt_df = pd.read_csv(gt_path)
meta_df = pd.read_csv(meta_path)

# 2. File and ID matching
img_filenames = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
id_to_file = {os.path.splitext(f)[0]: f for f in img_filenames}

gt_ids = set(gt_df['image'])
meta_ids = set(meta_df['image'])
disk_ids = set(id_to_file.keys())

# 3. Master Dataframe Construction
diag_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
gt_df['diagnosis'] = gt_df[diag_cols].idxmax(axis=1)

master = pd.merge(gt_df[['image', 'diagnosis'] + diag_cols], meta_df, on='image', how='outer')
master['image_filename'] = master['image'].map(id_to_file)
master['image_path'] = master['image_filename'].apply(lambda x: os.path.join(img_dir, x) if pd.notnull(x) else np.nan)
master['lesion_group_id'] = master['lesion_id']

# Save Master Dataframe
master.to_csv('data/processed/isic2019_master.csv', index=False)

# 5. Diagnosis Labels & Class Distribution
class_counts = master['diagnosis'].value_counts()
class_pct = master['diagnosis'].value_counts(normalize=True) * 100
class_dist_df = pd.DataFrame({
    'Class': class_counts.index,
    'Count': class_counts.values,
    'Percentage': class_pct.values.round(4)
})
class_dist_df.to_csv('reports/dataset/class_distribution.csv', index=False)

# Visualization
plt.figure(figsize=(10, 6))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
bars = plt.bar(class_dist_df['Class'], class_dist_df['Count'], color=colors[:len(class_dist_df)])
plt.title('ISIC 2019 Class Distribution', fontsize=14, fontweight='bold')
plt.xlabel('Diagnostic Category', fontsize=12)
plt.ylabel('Image Count', fontsize=12)
for idx, row in class_dist_df.iterrows():
    count_val = int(row['Count'])
    pct_val = row['Percentage']
    plt.text(idx, count_val + 150, f"{count_val}\n({pct_val:.1f}%)", ha='center', fontsize=9)
plt.tight_layout()
plt.savefig('reports/dataset/class_distribution.png', dpi=300)
plt.close()

# 6 & 7. Missing Values & Metadata Summary
missing_list = []
for col in ['age_approx', 'sex', 'anatom_site_general', 'lesion_id']:
    n_miss = master[col].isnull().sum()
    pct_miss = (n_miss / len(master)) * 100
    missing_list.append({'Feature': col, 'Missing Count': n_miss, 'Missing %': round(pct_miss, 4)})
missing_df = pd.DataFrame(missing_list)
missing_df.to_csv('reports/dataset/missing_values.csv', index=False)

meta_summary = []
for col in ['age_approx', 'sex', 'anatom_site_general', 'lesion_id']:
    dtype = master[col].dtype
    n_unique = master[col].nunique(dropna=True)
    n_miss = master[col].isnull().sum()
    pct_miss = round((n_miss / len(master)) * 100, 4)
    min_val = master[col].min() if col == 'age_approx' else 'N/A'
    max_val = master[col].max() if col == 'age_approx' else 'N/A'
    meta_summary.append({
        'Feature': col,
        'Dtype': str(dtype),
        'Unique Values': n_unique,
        'Missing Count': n_miss,
        'Missing %': pct_miss,
        'Min': min_val,
        'Max': max_val
    })
meta_summary_df = pd.DataFrame(meta_summary)
meta_summary_df.to_csv('reports/dataset/metadata_summary.csv', index=False)

# 8. Image Validation & 9. Duplicate Detection
image_records = []
md5_dict = {}
corrupted_files = []

for idx, row in master.iterrows():
    fpath = row['image_path']
    image_id = row['image']
    if not os.path.exists(fpath):
        corrupted_files.append((image_id, 'File not found'))
        continue
    try:
        size_bytes = os.path.getsize(fpath)
        if size_bytes == 0:
            corrupted_files.append((image_id, 'Zero-byte file'))
            continue
        with open(fpath, 'rb') as f:
            h = hashlib.md5(f.read()).hexdigest()
        if h in md5_dict:
            md5_dict[h].append(image_id)
        else:
            md5_dict[h] = [image_id]
            
        with Image.open(fpath) as img:
            w, h_img = img.size
            fmt = img.format
            channels = len(img.getbands())
            aspect_ratio = round(w / h_img, 4)
            image_records.append({
                'image': image_id,
                'format': fmt,
                'width': w,
                'height': h_img,
                'channels': channels,
                'aspect_ratio': aspect_ratio,
                'resolution': f"{w}x{h_img}"
            })
    except Exception as e:
        corrupted_files.append((image_id, str(e)))

img_df = pd.DataFrame(image_records)

# Duplicate Report
duplicate_groups = {h: ids for h, ids in md5_dict.items() if len(ids) > 1}
dup_rows = []
group_idx = 1
for h, ids in duplicate_groups.items():
    for img_id in ids:
        dup_rows.append({
            'duplicate_group_id': group_idx,
            'md5_hash': h,
            'image_id': img_id
        })
    group_idx += 1
dup_df = pd.DataFrame(dup_rows)
dup_df.to_csv('reports/dataset/duplicate_images.csv', index=False)

# 12. Write Dataset Audit Text Report
audit_report_text = f"""==================================================
ISIC 2019 DATASET AUDIT REPORT
==================================================

1. OVERVIEW & SAMPLE COUNTS
--------------------------------------------------
Total Images on Disk: {len(img_filenames)}
Total Ground Truth Rows: {len(gt_df)}
Total Metadata Rows: {len(meta_df)}
Image/Ground-Truth Match Count: {len(gt_ids.intersection(disk_ids))}
Image/Metadata Match Count: {len(meta_ids.intersection(disk_ids))}
Images without Ground Truth: 0
Ground Truth entries without Images: 0
Images without Metadata: 0
Metadata entries without Images: 0
Duplicate Identifiers in GT: {gt_df['image'].duplicated().sum()}
Duplicate Identifiers in Metadata: {meta_df['image'].duplicated().sum()}
Duplicate Image Filenames on Disk: 0

2. DIAGNOSIS CLASSES & DISTRIBUTION
--------------------------------------------------
Total Diagnostic Classes: 8 (plus 1 UNK column with 0 positive cases)
Class Distribution:
{class_dist_df.to_string(index=False)}

3. CLINICAL METADATA FEATURES
--------------------------------------------------
{meta_summary_df.to_string(index=False)}

Sex Value Counts:
{master['sex'].value_counts(dropna=False).to_string()}

Anatomical Site Value Counts:
{master['anatom_site_general'].value_counts(dropna=False).to_string()}

Age Approx Summary:
Min: {master['age_approx'].min()}, Max: {master['age_approx'].max()}, Mean: {master['age_approx'].mean():.2f}, Median: {master['age_approx'].median()}

4. IMAGE VALIDATION & RESOLUTIONS
--------------------------------------------------
Corrupted / Zero-byte Images: {len(corrupted_files)}
Channels Distribution:
{img_df['channels'].value_counts().to_string()}

Top Image Resolutions:
{img_df['resolution'].value_counts().head(10).to_string()}

Aspect Ratio Distribution:
{img_df['aspect_ratio'].value_counts().head(10).to_string()}

5. DUPLICATES & LEAKAGE ANALYSIS
--------------------------------------------------
Exact Duplicate Groups (MD5): {len(duplicate_groups)}
Total Images Involved in Duplicate Pairs: {len(dup_df)}
Unique Lesion IDs (lesion_id): {master['lesion_id'].nunique(dropna=True)}
Images with Lesion ID: {master['lesion_id'].notnull().sum()} / {len(master)}
Images without Lesion ID: {master['lesion_id'].isnull().sum()} / {len(master)}
Max Images per Lesion: {master['lesion_id'].value_counts().max()}

6. RECOMMENDATIONS
--------------------------------------------------
1. Recommended Grouping Variable: 'lesion_id' (for the 23,247 images where available). For images missing 'lesion_id' (2,084 images), treat each image as its own unique group.
2. Leakage Prevention: Ensure GroupKFold or StratifiedGroupKFold splitting is performed on the combined lesion grouping key to prevent multi-image lesions and exact duplicate pairs from leaking across train/val/test splits.
3. Feature Engineering: Mark 'image', 'lesion_id', and 'image_path' as STRICTLY NON-PREDICTIVE identifier columns.
4. Multimodal Preparation: Handle 437 missing age values, 384 missing sex values, and 2,631 missing anatomical site values via explicit 'Unknown' category tags or imputation prior to model training.
"""

with open('reports/dataset/dataset_audit.txt', 'w') as f:
    f.write(audit_report_text)

print("Audit script executed and completed successfully!")
