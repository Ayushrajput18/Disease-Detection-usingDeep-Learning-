import os
import sys
import time
import json
import yaml
import joblib
import platform
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Custom imports
from src.models.clinical_model import ISICMetadataClassifier
from src.models.extract_metadata_embeddings import extract_metadata_embeddings, save_metadata_embeddings_to_npz

# ----------------------------------------------------
# 1. ENVIRONMENT INSPECTION
# ----------------------------------------------------
def inspect_environment():
    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU Mode)",
        "cpu_count": os.cpu_count(),
        "system_ram": "Available"
    }
    print("=== ENVIRONMENT INFORMATION ===")
    for k, v in env_info.items():
        print(f"{k}: {v}")
    print("===============================\n")
    return env_info


def main():
    # Load Configuration
    config_path = "configs/metadata_only.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Set seeds
    seed = config["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env_info = inspect_environment()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["reports_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)
    os.makedirs(config["output"]["exploration_dir"], exist_ok=True)

    # ----------------------------------------------------
    # 2. DATA LOAD & LEAKAGE CHECK
    # ----------------------------------------------------
    train_df = pd.read_csv(config["dataset"]["train_path"])
    val_df = pd.read_csv(config["dataset"]["val_path"])
    test_df = pd.read_csv(config["dataset"]["test_path"])

    train_groups = set(train_df["combined_leakage_group_id"])
    val_groups = set(val_df["combined_leakage_group_id"])
    test_groups = set(test_df["combined_leakage_group_id"])

    overlap_tr_val = train_groups.intersection(val_groups)
    overlap_tr_te = train_groups.intersection(test_groups)
    overlap_val_te = val_groups.intersection(test_groups)

    if len(overlap_tr_val) > 0 or len(overlap_tr_te) > 0 or len(overlap_val_te) > 0:
        raise RuntimeError("DATA LEAKAGE DETECTED across splits! Aborting.")
    print("Data leakage verification passed: 0 group overlap between Train, Validation, and Test.\n")

    num_cols = config["dataset"]["numerical_features"]
    cat_cols = config["dataset"]["categorical_features"]
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]

    # ----------------------------------------------------
    # 3. FEATURE EXPLORATION PLOTS
    # ----------------------------------------------------
    print("Generating metadata feature exploration plots...")
    full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Age Distribution
    plt.figure(figsize=(8, 5))
    plt.hist(full_df['age_approx'].dropna(), bins=18, color='#1f77b4', edgecolor='black', alpha=0.7)
    plt.title('ISIC 2019 Patient Age Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Approximate Age (Years)')
    plt.ylabel('Patient Count')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["exploration_dir"], "age_distribution.png"), dpi=300)
    plt.close()

    # Sex Distribution
    plt.figure(figsize=(7, 5))
    sex_counts = full_df['sex'].fillna('Unknown').value_counts()
    plt.bar(sex_counts.index, sex_counts.values, color=['#3366cc', '#dc3912', '#ff9900'])
    plt.title('ISIC 2019 Sex Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Sex Category')
    plt.ylabel('Count')
    for i, v in enumerate(sex_counts.values):
        plt.text(i, v + 200, str(v), ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["exploration_dir"], "sex_distribution.png"), dpi=300)
    plt.close()

    # Anatomical Site Distribution
    plt.figure(figsize=(10, 5))
    site_counts = full_df['anatom_site_general'].fillna('Unknown').value_counts()
    plt.barh(site_counts.index, site_counts.values, color='#2ca02c')
    plt.title('ISIC 2019 Anatomical Site Distribution', fontsize=12, fontweight='bold')
    plt.xlabel('Count')
    plt.ylabel('Anatomical Site')
    for i, v in enumerate(site_counts.values):
        plt.text(v + 100, i, str(v), va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["exploration_dir"], "anatom_site_distribution.png"), dpi=300)
    plt.close()

    # Diagnosis vs Anatomical Site Summary
    ct = pd.crosstab(full_df['anatom_site_general'].fillna('Unknown'), full_df['diagnosis'])
    ct.to_csv(os.path.join(config["output"]["exploration_dir"], "diagnosis_vs_anatomical_site.csv"))

    # ----------------------------------------------------
    # 4. REPRODUCIBLE METADATA PREPROCESSING PIPELINE
    # ----------------------------------------------------
    # Numerical pipeline: Median Imputer + StandardScaler
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical pipeline: Replace NaNs with 'Unknown' + OneHotEncoder
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ])

    # FIT ONLY ON TRAIN SET
    print("Fitting preprocessor pipeline ONLY on training data...")
    X_train_proc = preprocessor.fit_transform(train_df[num_cols + cat_cols])
    X_val_proc = preprocessor.transform(val_df[num_cols + cat_cols])
    X_test_proc = preprocessor.transform(test_df[num_cols + cat_cols])

    # Save preprocessor artifact
    joblib.dump(preprocessor, os.path.join(config["output"]["checkpoint_dir"], "preprocessor.joblib"))
    print(f"Preprocessed feature input dimension: {X_train_proc.shape[1]}")

    y_train = train_df['diagnosis_index'].values
    y_val = val_df['diagnosis_index'].values
    y_test = test_df['diagnosis_index'].values

    # ----------------------------------------------------
    # 5. CLASS WEIGHTS (Train Split Only)
    # ----------------------------------------------------
    total_train = len(y_train)
    class_counts = train_df["diagnosis_index"].value_counts().sort_index().values
    class_weights_calc = total_train / (num_classes * class_counts)
    class_weights_dict = {name: float(w) for name, w in zip(class_names, class_weights_calc)}

    with open(os.path.join(config["output"]["checkpoint_dir"], "class_weights.json"), "w") as f:
        json.dump(class_weights_dict, f, indent=4)

    class_weights_tensor = torch.tensor(class_weights_calc, dtype=torch.float32).to(device)

    # ----------------------------------------------------
    # 6. PYTORCH DATALOADERS
    # ----------------------------------------------------
    batch_size = config["training"]["batch_size"]

    train_ds = TensorDataset(torch.tensor(X_train_proc, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val_proc, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))
    test_ds = TensorDataset(torch.tensor(X_test_proc, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ----------------------------------------------------
    # 7. MODEL, OPTIMIZER, SCHEDULER
    # ----------------------------------------------------
    model = ISICMetadataClassifier(
        input_dim=X_train_proc.shape[1],
        num_classes=num_classes,
        hidden_dim_1=config["model"]["hidden_dim_1"],
        hidden_dim_2=config["model"]["hidden_dim_2"],
        embedding_dim=config["model"]["embedding_dim"],
        dropout_1=config["model"]["dropout_1"],
        dropout_2=config["model"]["dropout_2"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config["training"]["scheduler_factor"], patience=config["training"]["scheduler_patience"]
    )

    epochs = config["training"]["epochs"]
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    history = []

    print(f"\nTraining Metadata MLP for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # Training phase
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(x_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_b.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(y_b.cpu().numpy())

        train_loss = running_loss / len(train_ds)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                logits = model(x_b)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * x_b.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(y_b.cpu().numpy())

        val_loss = val_loss / len(val_ds)
        val_acc = accuracy_score(val_targets, val_preds)
        val_prec = precision_score(val_targets, val_preds, average='macro', zero_division=0)
        val_rec = recall_score(val_targets, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({epoch_time:.2f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1:.4f} | LR: {current_lr:.6f}")

        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_precision": round(val_prec, 4),
            "val_macro_recall": round(val_rec, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": current_lr,
            "epoch_time_seconds": round(epoch_time, 2)
        })

        # Checkpoints
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'input_dim': X_train_proc.shape[1],
                'val_macro_f1': val_f1,
            }, os.path.join(config["output"]["checkpoint_dir"], "best_model.pt"))
            print(f"  --> Saved new best MLP model (Val Macro F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': val_f1,
        }, os.path.join(config["output"]["checkpoint_dir"], "last_model.pt"))

        if patience_counter >= config["training"]["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without val_macro_f1 improvement.")
            break

    # Save History & Config
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config["output"]["reports_dir"], "metadata_training_history.csv"), index=False)
    with open(os.path.join(config["output"]["checkpoint_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join(config["output"]["checkpoint_dir"], "training_config.json"), "w") as f:
        json.dump({"config": config, "env_info": env_info}, f, indent=4)

    # ----------------------------------------------------
    # 8. TRAINING CURVES
    # ----------------------------------------------------
    epochs_range = range(1, len(history_df) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs_range, history_df['val_loss'], 'r-o', label='Val Loss')
    plt.title('Metadata MLP Training vs Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "loss_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['train_acc'], 'b-o', label='Train Accuracy')
    plt.plot(epochs_range, history_df['val_acc'], 'g-o', label='Val Accuracy')
    plt.title('Metadata MLP Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "accuracy_curve.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['val_macro_f1'], 'm-o', label='Val Macro F1')
    plt.title('Metadata MLP Validation Macro F1 Score Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "macro_f1_curve.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 9. TEST SET EVALUATION
    # ----------------------------------------------------
    print("\n=== EVALUATING BEST METADATA MLP ON HELD-OUT TEST SET ===")
    checkpoint = torch.load(os.path.join(config["output"]["checkpoint_dir"], "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    start_infer_time = time.time()
    with torch.no_grad():
        x_test_t = torch.tensor(X_test_proc, dtype=torch.float32).to(device)
        logits = model(x_test_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()

    total_infer_time = time.time() - start_infer_time
    test_targets = y_test

    avg_infer_per_img = (total_infer_time / len(test_targets)) * 1000  # ms
    imgs_per_sec = len(test_targets) / total_infer_time

    test_acc = accuracy_score(test_targets, preds)
    test_macro_prec = precision_score(test_targets, preds, average='macro', zero_division=0)
    test_macro_rec = recall_score(test_targets, preds, average='macro', zero_division=0)
    test_macro_f1 = f1_score(test_targets, preds, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(test_targets, preds, average='weighted', zero_division=0)

    cm = confusion_matrix(test_targets, preds, labels=range(num_classes))
    per_class_prec = precision_score(test_targets, preds, average=None, labels=range(num_classes), zero_division=0)
    per_class_rec = recall_score(test_targets, preds, average=None, labels=range(num_classes), zero_division=0)
    per_class_f1 = f1_score(test_targets, preds, average=None, labels=range(num_classes), zero_division=0)

    per_class_spec = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_spec.append(spec)
    macro_spec = np.mean(per_class_spec)

    report_rows = []
    for i, name in enumerate(class_names):
        support = (test_targets == i).sum()
        report_rows.append({
            "class": name,
            "precision": round(float(per_class_prec[i]), 4),
            "recall_sensitivity": round(float(per_class_rec[i]), 4),
            "specificity": round(float(per_class_spec[i]), 4),
            "f1": round(float(per_class_f1[i]), 4),
            "support": int(support)
        })
    cls_report_df = pd.DataFrame(report_rows)
    cls_report_df.to_csv(os.path.join(config["output"]["reports_dir"], "metadata_only_classification_report.csv"), index=False)

    # Confusion Matrix (Counts)
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Metadata MLP Confusion Matrix (Raw Counts)', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2. if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "confusion_matrix_counts.png"), dpi=300)
    plt.close()

    # Confusion Matrix (Normalized)
    row_sums = cm.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = np.nan_to_num(cm.astype('float') / row_sums_safe[:, np.newaxis])
    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Metadata MLP Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm_norm.max() / 2. if cm_norm.max() > 0 else 1
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val_str = f"{cm_norm[i, j]:.2f}"
            plt.text(j, i, val_str, ha="center", va="center", color="white" if cm_norm[i, j] > thresh else "black")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "confusion_matrix_normalized.png"), dpi=300)
    plt.close()

    # One-vs-Rest ROC Curve
    plt.figure(figsize=(9, 7))
    roc_aucs = {}
    for i, name in enumerate(class_names):
        y_true_binary = (test_targets == i).astype(int)
        y_score = probs[:, i]
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            roc_aucs[name] = round(float(roc_auc), 4)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        else:
            roc_aucs[name] = None
    macro_roc_auc = round(float(np.mean([v for v in roc_aucs.values() if v is not None])), 4)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title(f'Metadata MLP One-vs-Rest ROC Curves (Macro AUC = {macro_roc_auc})', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "roc_curve.png"), dpi=300)
    plt.close()

    # Precision-Recall Curve
    plt.figure(figsize=(9, 7))
    for i, name in enumerate(class_names):
        y_true_binary = (test_targets == i).astype(int)
        y_score = probs[:, i]
        if len(np.unique(y_true_binary)) > 1:
            precision_pts, recall_pts, _ = precision_recall_curve(y_true_binary, y_score)
            plt.plot(recall_pts, precision_pts, label=f'{name}')
    plt.title('Metadata MLP Precision-Recall Curves per Diagnostic Class', fontsize=13, fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "precision_recall_curve.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 10. METRICS JSON EXPORT
    # ----------------------------------------------------
    test_metrics = {
        "accuracy": round(float(test_acc), 4),
        "macro_precision": round(float(test_macro_prec), 4),
        "macro_recall_sensitivity": round(float(test_macro_rec), 4),
        "macro_specificity": round(float(macro_spec), 4),
        "macro_f1": round(float(test_macro_f1), 4),
        "weighted_f1": round(float(test_weighted_f1), 4),
        "macro_roc_auc": macro_roc_auc,
        "class_roc_auc": roc_aucs,
        "inference_performance": {
            "total_inference_time_seconds": round(total_infer_time, 4),
            "average_time_per_sample_ms": round(avg_infer_per_img, 4),
            "samples_per_second": round(imgs_per_sec, 2)
        }
    }

    with open(os.path.join(config["output"]["reports_dir"], "metadata_only_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    # ----------------------------------------------------
    # 11. GENERATE MARKDOWN REPORT
    # ----------------------------------------------------
    md_table_header = "| class | precision | recall_sensitivity | specificity | f1 | support |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    md_table_rows = "\n".join([
        f"| **{row['class']}** | {row['precision']:.4f} | {row['recall_sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | {row['support']} |"
        for _, row in cls_report_df.iterrows()
    ])
    cls_report_md_table = md_table_header + md_table_rows

    report_md = f"""# Clinical/Metadata-Only Baseline Report (MLP)

## 1. Environment & Hardware
- **Python**: `{env_info['python_version']}`
- **PyTorch**: `{env_info['torch_version']}`
- **Compute Device**: `{device.type.upper()}` ({env_info['gpu_name']})
- **System RAM**: `{env_info['system_ram']}`

## 2. Dataset Partitioning & Feature Schema
- **Training Samples**: {len(train_df)}
- **Validation Samples**: {len(val_df)}
- **Test Samples**: {len(test_df)}
- **Predictive Clinical Features**: `age_approx` (Numerical), `sex` (Categorical), `anatom_site_general` (Categorical)
- **Preprocessed Input Feature Dimensions**: {X_train_proc.shape[1]} (1 continuous + One-Hot Encoded Categories)

## 3. Preprocessing & Imputation Strategy
- **`age_approx`**: Missing count = {train_df['age_approx'].isnull().sum()} (1.73%). Strategy: Median imputation ({train_df['age_approx'].median()} years) + `StandardScaler`.
- **`sex`**: Missing count = {train_df['sex'].isnull().sum()} (1.52%). Strategy: Explicit `'Unknown'` category + `OneHotEncoder`.
- **`anatom_site_general`**: Missing count = {train_df['anatom_site_general'].isnull().sum()} (10.39%). Strategy: Explicit `'Unknown'` category + `OneHotEncoder`.
- **Leakage Prevention Guarantee**: Scikit-learn `ColumnTransformer` fitted **ONLY** on Training split (`preprocessor.joblib`).

## 4. Model Architecture & Training Settings
- **Architecture**: Multi-Layer Perceptron (MLP)
  - Input Layer: {X_train_proc.shape[1]} units
  - Hidden Layer 1: {config['model']['hidden_dim_1']} units (ReLU, Dropout {config['model']['dropout_1']})
  - Hidden Layer 2: {config['model']['hidden_dim_2']} units (ReLU, Dropout {config['model']['dropout_2']})
  - Metadata Embedding Layer: {config['model']['embedding_dim']} units (ReLU) -> Saved for Phase 5 Fusion
  - Output Classifier: {num_classes} units (Logits for 8 diagnostic classes)
- **Optimizer**: `AdamW` (LR = {config['training']['lr']}, Weight Decay = {config['training']['weight_decay']})
- **Loss Function**: Class-Weighted CrossEntropyLoss (Fitted on Training Split only)
- **Best Epoch**: Epoch {best_epoch} (Best Validation Macro F1 = {best_val_f1:.4f})

## 5. Overall Test Performance Metrics
- **Test Accuracy**: **{test_acc * 100:.2f}%**
- **Macro F1 Score**: **{test_macro_f1:.4f}**
- **Weighted F1 Score**: **{test_weighted_f1:.4f}**
- **Macro Precision**: **{test_macro_prec:.4f}**
- **Macro Recall / Sensitivity**: **{test_macro_rec:.4f}**
- **Macro Specificity**: **{macro_spec:.4f}**
- **Macro ROC-AUC**: **{macro_roc_auc:.4f}**

## 6. Per-Class Performance Breakdown

{cls_report_md_table}

## 7. Inference Speed Performance
- **Evaluated Test Samples**: {len(test_targets)}
- **Total Test Inference Time**: {total_infer_time:.4f} seconds
- **Average Time Per Sample**: {avg_infer_per_img:.4f} ms
- **Throughput**: {imgs_per_sec:.2f} samples/second

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
"""

    with open(os.path.join(config["output"]["reports_dir"], "metadata_only_report.md"), "w") as f:
        f.write(report_md)

    print("\nPhase 4 Metadata-Only MLP Baseline training and evaluation complete!")
    print(f"Report generated at: {config['output']['reports_dir']}/metadata_only_report.md")

if __name__ == "__main__":
    main()
