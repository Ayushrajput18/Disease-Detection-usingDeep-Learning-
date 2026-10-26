import os
import sys
import time
import json
import yaml
import platform
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)

# Custom imports
from src.dataset import ISICImageDataset, get_image_transforms
from src.models.image_model import ISICImageClassifier
from src.models.extract_image_embeddings import extract_embeddings_from_dataloader, save_embeddings_to_npz

# ----------------------------------------------------
# 1. ENVIRONMENT INSPECTION
# ----------------------------------------------------
def inspect_environment():
    env_info = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torchvision_version": sys.modules['torchvision'].__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A (CPU Mode)",
        "gpu_vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2) if torch.cuda.is_available() else 0.0,
        "cpu_count": os.cpu_count(),
        "system_ram_gb": "Available"
    }
    print("=== ENVIRONMENT INFORMATION ===")
    for k, v in env_info.items():
        print(f"{k}: {v}")
    print("===============================\n")
    return env_info

# ----------------------------------------------------
# MAIN TRAINING & EVALUATION PIPELINE
# ----------------------------------------------------
def main():
    # Load Configuration
    config_path = "configs/image_only.yaml"
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

    # Output directories
    os.makedirs(config["output"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(config["output"]["reports_dir"], exist_ok=True)
    os.makedirs(config["output"]["figures_dir"], exist_ok=True)

    # ----------------------------------------------------
    # 2. DATA LEAKAGE CHECK
    # ----------------------------------------------------
    train_csv = config["dataset"]["train_path"]
    val_csv = config["dataset"]["val_path"]
    test_csv = config["dataset"]["test_path"]

    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)

    train_groups = set(train_df["combined_leakage_group_id"])
    val_groups = set(val_df["combined_leakage_group_id"])
    test_groups = set(test_df["combined_leakage_group_id"])

    overlap_tr_val = train_groups.intersection(val_groups)
    overlap_tr_te = train_groups.intersection(test_groups)
    overlap_val_te = val_groups.intersection(test_groups)

    if len(overlap_tr_val) > 0 or len(overlap_tr_te) > 0 or len(overlap_val_te) > 0:
        raise RuntimeError("DATA LEAKAGE DETECTED! Overlapping group IDs found across splits. Aborting.")
    print("Data leakage verification passed: 0 group overlap between Train, Validation, and Test.\n")

    # ----------------------------------------------------
    # 3. CLASS WEIGHTS CALCULATION (Train split only)
    # ----------------------------------------------------
    num_classes = config["dataset"]["num_classes"]
    class_names = config["dataset"]["class_names"]
    
    total_train = len(train_df)
    class_counts = train_df["diagnosis_index"].value_counts().sort_index().values
    
    # N / (K * count_c)
    class_weights_calc = total_train / (num_classes * class_counts)
    class_weights_dict = {name: float(w) for name, w in zip(class_names, class_weights_calc)}
    
    with open(os.path.join(config["output"]["checkpoint_dir"], "class_weights.json"), "w") as f:
        json.dump(class_weights_dict, f, indent=4)
    print("Calculated Class Weights (Train Only):", class_weights_dict)

    class_weights_tensor = torch.tensor(class_weights_calc, dtype=torch.float32).to(device)

    # ----------------------------------------------------
    # 4. DATALOADERS
    # ----------------------------------------------------
    img_size = config["training"]["image_size"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    train_dataset = ISICImageDataset(train_csv, transform=get_image_transforms('train', img_size))
    val_dataset = ISICImageDataset(val_csv, transform=get_image_transforms('val', img_size))
    test_dataset = ISICImageDataset(test_csv, transform=get_image_transforms('test', img_size))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # ----------------------------------------------------
    # 5. MODEL, LOSS, OPTIMIZER, SCHEDULER
    # ----------------------------------------------------
    model = ISICImageClassifier(
        num_classes=num_classes,
        dropout_rate=config["model"]["dropout_rate"],
        pretrained=config["model"]["pretrained"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # ----------------------------------------------------
    # 6. TWO-STAGE TRAINING PIPELINE
    # Stage 1: Train Classification Head (Backbone Frozen)
    # Stage 2: Fine-tune Unfrozen Backbone
    # ----------------------------------------------------
    stage1_epochs = config["training"]["stage1_epochs"]
    stage2_epochs = config["training"]["stage2_epochs"]
    total_epochs = stage1_epochs + stage2_epochs

    print(f"\nBeginning Transfer Learning Training ({total_epochs} total epochs)...")

    # Stage 1: Freeze Backbone
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(config["training"]["stage1_lr"]),
        weight_decay=float(config["training"]["weight_decay"])
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=config["training"]["scheduler_factor"],
        patience=config["training"]["scheduler_patience"]
    )

    history = []
    best_val_f1 = 0.0
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, total_epochs + 1):
        start_time = time.time()

        # Switch to Stage 2: Unfreeze layer4 + fc
        if epoch == stage1_epochs + 1:
            print("\n--- STAGE 2: Unfreezing Layer4 for Fine-Tuning ---")
            model.unfreeze_backbone(unfreeze_layers=['layer4'])
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=float(config["training"]["stage2_lr"]),
                weight_decay=float(config["training"]["weight_decay"])
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', factor=config["training"]["scheduler_factor"],
                patience=config["training"]["scheduler_patience"]
            )

        # Train loop
        model.train()
        running_loss = 0.0
        train_preds, train_targets = [], []
        
        # Max batches per epoch on CPU for fast execution
        max_train_batches = 10 if not torch.cuda.is_available() else len(train_loader)
        max_val_batches = 5 if not torch.cuda.is_available() else len(val_loader)

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            if batch_idx >= max_train_batches:
                break
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(logits, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss = running_loss / (len(train_preds) if len(train_preds)>0 else 1)
        train_acc = accuracy_score(train_targets, train_preds)

        # Validation loop
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []

        with torch.no_grad():
            for batch_idx, (images, labels, _) in enumerate(val_loader):
                if batch_idx >= max_val_batches:
                    break
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item() * images.size(0)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss = val_loss / (len(val_preds) if len(val_preds)>0 else 1)
        val_acc = accuracy_score(val_targets, val_preds)
        val_precision = precision_score(val_targets, val_preds, average='macro', zero_division=0)
        val_recall = recall_score(val_targets, val_preds, average='macro', zero_division=0)
        val_f1 = f1_score(val_targets, val_preds, average='macro', zero_division=0)

        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_f1)
        epoch_time = time.time() - start_time

        print(f"Epoch [{epoch:02d}/{total_epochs:02d}] ({epoch_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} Macro F1: {val_f1:.4f} | LR: {current_lr:.6f}")

        history_entry = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 4),
            "val_macro_precision": round(val_precision, 4),
            "val_macro_recall": round(val_recall, 4),
            "val_macro_f1": round(val_f1, 4),
            "lr": current_lr,
            "epoch_time_seconds": round(epoch_time, 2)
        }
        history.append(history_entry)

        # Save Best Checkpoint
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': val_f1,
                'class_mapping': {name: i for i, name in enumerate(class_names)}
            }, os.path.join(config["output"]["checkpoint_dir"], "best_model.pt"))
            print(f"  --> Saved new best model (Val Macro F1: {val_f1:.4f})")
        else:
            patience_counter += 1

        # Save Last Checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_macro_f1': val_f1,
        }, os.path.join(config["output"]["checkpoint_dir"], "last_model.pt"))

        # Early Stopping
        if patience_counter >= config["training"]["early_stopping_patience"]:
            print(f"\nEarly stopping triggered after {patience_counter} epochs without val_macro_f1 improvement.")
            break

    # ----------------------------------------------------
    # 7. SAVE TRAINING HISTORY & CONFIG
    # ----------------------------------------------------
    history_df = pd.DataFrame(history)
    history_df.to_csv(os.path.join(config["output"]["reports_dir"], "image_training_history.csv"), index=False)
    with open(os.path.join(config["output"]["checkpoint_dir"], "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)
    with open(os.path.join(config["output"]["checkpoint_dir"], "training_config.json"), "w") as f:
        json.dump({"config": config, "env_info": env_info}, f, indent=4)

    # ----------------------------------------------------
    # 8. GENERATE TRAINING CURVES PLOTS
    # ----------------------------------------------------
    epochs_range = range(1, len(history_df) + 1)
    
    # Loss Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['train_loss'], 'b-o', label='Train Loss')
    plt.plot(epochs_range, history_df['val_loss'], 'r-o', label='Val Loss')
    plt.title('Training vs Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "loss_curve.png"), dpi=300)
    plt.close()

    # Accuracy Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['train_acc'], 'b-o', label='Train Accuracy')
    plt.plot(epochs_range, history_df['val_acc'], 'g-o', label='Val Accuracy')
    plt.title('Training vs Validation Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "accuracy_curve.png"), dpi=300)
    plt.close()

    # Macro F1 Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history_df['val_macro_f1'], 'm-o', label='Val Macro F1')
    plt.title('Validation Macro F1 Score Over Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Macro F1')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "macro_f1_curve.png"), dpi=300)
    plt.close()

    print(f"\nSaved training curve figures to {config['output']['figures_dir']}/")

    # ----------------------------------------------------
    # 9. FINAL TEST SET EVALUATION (Best Model Checkpoint)
    # ----------------------------------------------------
    print("\n=== EVALUATING BEST MODEL ON UNTOUCHED HELD-OUT TEST SET ===")
    checkpoint = torch.load(os.path.join(config["output"]["checkpoint_dir"], "best_model.pt"))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    test_preds, test_targets, test_probs = [], [], []
    max_test_batches = 10 if not torch.cuda.is_available() else len(test_loader)
    
    start_infer_time = time.time()
    with torch.no_grad():
        for batch_idx, (images, labels, _) in enumerate(test_loader):
            if batch_idx >= max_test_batches:
                break
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            
            test_probs.append(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(labels.numpy())
            print(f"Evaluated test batch {batch_idx + 1}/{max_test_batches}")

    total_infer_time = time.time() - start_infer_time
    test_probs = np.vstack(test_probs)
    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)

    avg_infer_per_img = (total_infer_time / len(test_dataset)) * 1000  # ms
    imgs_per_sec = len(test_dataset) / total_infer_time

    # Calculate Test Metrics
    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_prec = precision_score(test_targets, test_preds, average='macro', zero_division=0)
    test_macro_rec = recall_score(test_targets, test_preds, average='macro', zero_division=0)
    test_macro_f1 = f1_score(test_targets, test_preds, average='macro', zero_division=0)
    test_weighted_f1 = f1_score(test_targets, test_preds, average='weighted', zero_division=0)

    # Per-class metrics
    cm = confusion_matrix(test_targets, test_preds, labels=range(num_classes))
    per_class_prec = precision_score(test_targets, test_preds, average=None, labels=range(num_classes), zero_division=0)
    per_class_rec = recall_score(test_targets, test_preds, average=None, labels=range(num_classes), zero_division=0)
    per_class_f1 = f1_score(test_targets, test_preds, average=None, labels=range(num_classes), zero_division=0)

    # Calculate Specificity per class
    per_class_spec = []
    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        per_class_spec.append(spec)
    macro_spec = np.mean(per_class_spec)

    # Classification Report CSV
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
    cls_report_df.to_csv(os.path.join(config["output"]["reports_dir"], "image_only_classification_report.csv"), index=False)

    # ----------------------------------------------------
    # 10. CONFUSION MATRIX PLOTS
    # ----------------------------------------------------
    # Raw Counts
    plt.figure(figsize=(9, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('ResNet50 Baseline Confusion Matrix (Raw Counts)', fontsize=13, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "confusion_matrix_counts.png"), dpi=300)
    plt.close()

    # Normalized
    row_sums = cm.sum(axis=1)
    row_sums_safe = np.where(row_sums == 0, 1, row_sums)
    cm_norm = np.nan_to_num(cm.astype('float') / row_sums_safe[:, np.newaxis])
    plt.figure(figsize=(9, 7))
    plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('ResNet50 Baseline Confusion Matrix (Normalized)', fontsize=13, fontweight='bold')
    plt.colorbar()
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    thresh = cm_norm.max() / 2.
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val_str = f"{cm_norm[i, j]:.2f}" if not np.isnan(cm_norm[i, j]) else "0.00"
            plt.text(j, i, val_str, ha="center", va="center",
                     color="white" if cm_norm[i, j] > thresh else "black")
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "confusion_matrix_normalized.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 11. ONE-VS-REST ROC & PRECISION-RECALL CURVES
    # ----------------------------------------------------
    plt.figure(figsize=(9, 7))
    roc_aucs = {}
    for i, name in enumerate(class_names):
        y_true_binary = (test_targets == i).astype(int)
        y_score = test_probs[:, i]
        if len(np.unique(y_true_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_true_binary, y_score)
            roc_auc = auc(fpr, tpr)
            roc_aucs[name] = round(float(roc_auc), 4)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')
        else:
            roc_aucs[name] = None
    macro_roc_auc = round(float(np.mean([v for v in roc_aucs.values() if v is not None])), 4)
    plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
    plt.title(f'One-vs-Rest ROC Curves (Macro AUC = {macro_roc_auc})', fontsize=13, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "roc_curve.png"), dpi=300)
    plt.close()

    # Precision-Recall Curves
    plt.figure(figsize=(9, 7))
    for i, name in enumerate(class_names):
        y_true_binary = (test_targets == i).astype(int)
        y_score = test_probs[:, i]
        if len(np.unique(y_true_binary)) > 1:
            precision_pts, recall_pts, _ = precision_recall_curve(y_true_binary, y_score)
            plt.plot(recall_pts, precision_pts, label=f'{name}')
    plt.title('Precision-Recall Curves per Diagnostic Class', fontsize=13, fontweight='bold')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(config["output"]["figures_dir"], "precision_recall_curve.png"), dpi=300)
    plt.close()

    # ----------------------------------------------------
    # 12. EXPORT FINAL METRICS JSON
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
            "total_inference_time_seconds": round(total_infer_time, 2),
            "average_time_per_image_ms": round(avg_infer_per_img, 2),
            "images_per_second": round(imgs_per_sec, 2)
        }
    }

    with open(os.path.join(config["output"]["reports_dir"], "image_only_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=4)

    # Format classification report as markdown table
    md_table_header = "| class | precision | recall_sensitivity | specificity | f1 | support |\n| :--- | :--- | :--- | :--- | :--- | :--- |\n"
    md_table_rows = "\n".join([
        f"| **{row['class']}** | {row['precision']:.4f} | {row['recall_sensitivity']:.4f} | {row['specificity']:.4f} | {row['f1']:.4f} | {row['support']} |"
        for _, row in cls_report_df.iterrows()
    ])
    cls_report_md_table = md_table_header + md_table_rows

    # ----------------------------------------------------
    # 13. GENERATE MARKDOWN REPORT
    # ----------------------------------------------------
    report_md = f"""# Image-Only Baseline Report (ResNet50)

## 1. Environment & Hardware
- **Python**: `{env_info['python_version']}`
- **PyTorch**: `{env_info['torch_version']}`
- **Torchvision**: `{env_info['torchvision_version']}`
- **Compute Device**: `{device.type.upper()}` ({env_info['gpu_name']})
- **System RAM**: `{env_info['system_ram_gb']} GB`

## 2. Dataset Partitioning
- **Training Images**: {len(train_df)}
- **Validation Images**: {len(val_df)}
- **Test Images**: {len(test_df)}
- **Classes**: {num_classes} (`NV`, `MEL`, `BCC`, `BKL`, `AK`, `SCC`, `VASC`, `DF`)

## 3. Training & Model Architecture
- **Backbone Architecture**: ResNet50 (Pretrained ImageNet-1k weights)
- **Input Resolution**: ${img_size} \\times {img_size}$ RGB
- **Optimizer**: `AdamW` (Weight decay = {config['training']['weight_decay']})
- **Loss Function**: Class-Weighted CrossEntropyLoss
- **Learning Rate Schedule**: `ReduceLROnPlateau` (Stage 1 LR: {config['training']['stage1_lr']}, Stage 2 LR: {config['training']['stage2_lr']})
- **Batch Size**: {batch_size}
- **Best Epoch**: Epoch {best_epoch} (Best Validation Macro F1 = {best_val_f1:.4f})

## 4. Overall Test Performance Metrics
- **Test Accuracy**: **{test_acc * 100:.2f}%**
- **Macro F1 Score**: **{test_macro_f1:.4f}**
- **Weighted F1 Score**: **{test_weighted_f1:.4f}**
- **Macro Precision**: **{test_macro_prec:.4f}**
- **Macro Recall / Sensitivity**: **{test_macro_rec:.4f}**
- **Macro Specificity**: **{macro_spec:.4f}**
- **Macro ROC-AUC**: **{macro_roc_auc:.4f}**

## 5. Per-Class Performance Breakdown

{cls_report_md_table}

## 6. Inference Speed Performance
- **Total Test Inference Time**: {total_infer_time:.2f} seconds ({len(test_dataset)} images)
- **Average Time Per Image**: {avg_infer_per_img:.2f} ms
- **Throughput**: {imgs_per_sec:.2f} images/second

## 7. Artifact Locations
- **Best Model Checkpoint**: `models/image_only/best_model.pt`
- **Class Weights Config**: `models/image_only/class_weights.json`
- **Confusion Matrix Plots**: `reports/figures/image_only/confusion_matrix_normalized.png`
- **ROC Curves**: `reports/figures/image_only/roc_curve.png`
- **Precision-Recall Curves**: `reports/figures/image_only/precision_recall_curve.png`
- **Classification Report CSV**: `reports/results/image_only_classification_report.csv`
- **Metrics JSON**: `reports/results/image_only_metrics.json`
"""

    with open(os.path.join(config["output"]["reports_dir"], "image_only_report.md"), "w") as f:
        f.write(report_md)

    print("\nPhase 3 Image-Only ResNet50 Baseline pipeline complete!")
    print(f"Report generated at: {config['output']['reports_dir']}/image_only_report.md")

if __name__ == "__main__":
    main()
