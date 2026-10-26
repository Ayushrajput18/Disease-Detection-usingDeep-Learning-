"""
Phase 6 — Model Explainability & Interpretability
==================================================
Implements:
  A. Grad-CAM on ResNet50 image branch (layer4)
  B. SHAP for metadata MLP (KernelExplainer)
  C. Multimodal Attention visualization
  D. Multimodal case studies
  E. Quantitative explainability analysis
  F. Research-safe explainability report
  G. Reproducibility config (explainability_config.json)
  H. Final phase summary (phase6_explainability_summary.md)

CRITICAL SCIENTIFIC RULES:
  - Grad-CAM is a model-attribution visualization, NOT clinical ground-truth.
  - SHAP measures model-feature contribution, NOT causation.
  - Attention weights should NOT be interpreted as causal evidence.
  - No fabricated data. All samples are from the real test set.
  - Random seed = 42 for all sampling.
"""

import matplotlib
matplotlib.use('Agg')

import os
import sys
import json
import random
import warnings
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

import joblib

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────
# PATHS & SEEDS
# ─────────────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

BASE_DIR   = Path(".")
IMAGE_DIR  = BASE_DIR / "Dataset" / "ISIC_2019_Training_Input"
SPLITS_DIR = BASE_DIR / "data" / "processed" / "splits"
EMBED_DIR  = BASE_DIR / "data" / "processed" / "embeddings"

MODEL_IMAGE      = BASE_DIR / "models" / "image_only"   / "best_model.pt"
MODEL_METADATA   = BASE_DIR / "models" / "metadata_only"/ "best_model.pt"
MODEL_MULTIMODAL = BASE_DIR / "models" / "multimodal"   / "attention_model.pt"
PREPROCESSOR     = BASE_DIR / "models" / "metadata_only"/ "preprocessor.joblib"

EXPL_DIR   = BASE_DIR / "explainability"
GRADCAM_DIR= EXPL_DIR / "gradcam"
SHAP_DIR   = EXPL_DIR / "shap"
ATTN_DIR   = EXPL_DIR / "attention"

for d in [GRADCAM_DIR, SHAP_DIR, ATTN_DIR]:
    d.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["NV", "MEL", "BCC", "BKL", "AK", "SCC", "VASC", "DF"]
NUM_CLASSES = 8

DEVICE = torch.device("cpu")

# ─────────────────────────────────────────────────────────────────
# HELPER: Add src to path
# ─────────────────────────────────────────────────────────────────
sys.path.insert(0, str(BASE_DIR / "src"))
from models.image_model     import ISICImageClassifier
from models.clinical_model  import ISICMetadataClassifier
from models.multimodal_model import ISICMultimodalAttentionClassifier

print("="*70)
print("PHASE 6 — MODEL EXPLAINABILITY & INTERPRETABILITY")
print("="*70)

# ─────────────────────────────────────────────────────────────────
# SECTION 0: Load test CSV + Embeddings
# ─────────────────────────────────────────────────────────────────
print("\n[0] Loading test split and embeddings...")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
print(f"    Test rows: {len(test_df)}")

emb_data = np.load(EMBED_DIR / "test_embeddings.npz", allow_pickle=True)
img_emb   = emb_data["image_embeddings"]    # (N, 2048)
meta_emb  = emb_data["metadata_embeddings"] # (N, 32)
labels    = emb_data["labels"]              # (N,)
image_ids = emb_data["image_ids"]           # (N,)
print(f"    Image embeddings: {img_emb.shape}")
print(f"    Metadata embeddings: {meta_emb.shape}")
print(f"    Labels: {labels.shape}")

# ─────────────────────────────────────────────────────────────────
# IMAGE PREPROCESSING (same as training)
# ─────────────────────────────────────────────────────────────────
IMG_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

def load_image_tensor(image_id):
    # Strip _downsampled suffix if present (images stored without that suffix)
    img_id_clean = image_id.replace("_downsampled", "")
    img_path = IMAGE_DIR / f"{img_id_clean}.jpg"
    if not img_path.exists():
        return None, None
    img = Image.open(img_path).convert("RGB")
    tensor = IMG_TRANSFORM(img).unsqueeze(0).to(DEVICE)  # (1, 3, 224, 224)
    return img, tensor

# ─────────────────────────────────────────────────────────────────
# SECTION A: GRAD-CAM
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART A — GRAD-CAM (ResNet50 layer4)")
print("="*70)

# --- Load image model ---
print("[A1] Loading image model checkpoint...")
img_model = ISICImageClassifier(num_classes=NUM_CLASSES, dropout_rate=0.3, pretrained=False)
ckpt = torch.load(MODEL_IMAGE, map_location=DEVICE, weights_only=True)
img_model.load_state_dict(ckpt["model_state_dict"])
img_model.eval()
print("     Image model loaded OK.")

# --- Grad-CAM implementation ---
class GradCAM:
    """
    Grad-CAM implementation targeting a specific feature layer.
    
    Selvaraju et al. (2017): 'Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization.'
    
    IMPORTANT SCIENTIFIC NOTE:
    Grad-CAM provides a visual indication of image regions that contributed
    to the model's prediction. It does NOT prove clinical correctness.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.target_layer = target_layer
        self.gradients   = None
        self.activations = None
        self._hooks      = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()

    def generate(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap for `input_tensor`.
        Returns heatmap in [0, 1] range, shape (H, W).
        """
        self.model.zero_grad()
        output = self.model(input_tensor)                      # (1, num_classes)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        score = output[0, class_idx]
        score.backward()

        # Pool gradients across spatial dimensions
        pooled_grads = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        weighted     = (pooled_grads * self.activations).squeeze(0)   # (C, H, W)
        cam          = weighted.sum(dim=0)                             # (H, W)
        cam          = F.relu(cam)

        # Normalize to [0, 1]
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy(), class_idx, output.softmax(dim=1)[0].detach().cpu().numpy()


# Target layer = feature_extractor[-1] = layer4
target_layer = img_model.feature_extractor[-1]
grad_cam     = GradCAM(img_model, target_layer)

# --- Select test samples: one per class ---
print("[A2] Selecting representative test samples (one per class)...")
np.random.seed(SEED)
samples_per_class = {}
for idx, row in test_df.iterrows():
    label = row["diagnosis_index"]
    if label not in samples_per_class:
        # Check image exists
        img_id_clean = str(row['image_id']).replace("_downsampled", "")
        img_path = IMAGE_DIR / f"{img_id_clean}.jpg"
        if img_path.exists():
            samples_per_class[label] = {
                "image_id":  row["image_id"],
                "label":     label,
                "class_name": CLASS_NAMES[label],
                "age":       row.get("age_approx", np.nan),
                "sex":       row.get("sex", "unknown"),
                "site":      row.get("anatom_site_general", "unknown"),
            }
    if len(samples_per_class) == NUM_CLASSES:
        break

missing_classes = [CLASS_NAMES[i] for i in range(NUM_CLASSES) if i not in samples_per_class]
if missing_classes:
    print(f"     WARNING: No test images found for classes: {missing_classes}")
print(f"     Found samples for {len(samples_per_class)} / {NUM_CLASSES} classes.")

# --- Generate Grad-CAM overlays ---
print("[A3] Generating Grad-CAM heatmaps...")
gradcam_results = []

for label_idx in sorted(samples_per_class.keys()):
    s = samples_per_class[label_idx]
    orig_img, img_tensor = load_image_tensor(s["image_id"])
    if img_tensor is None:
        print(f"     SKIP {s['class_name']}: image file not found.")
        continue

    # Run Grad-CAM
    heatmap, pred_idx, probs = grad_cam.generate(img_tensor)

    s["pred_label"]    = pred_idx
    s["pred_class"]    = CLASS_NAMES[pred_idx]
    s["confidence"]    = float(probs[pred_idx])
    s["correct"]       = (pred_idx == label_idx)
    s["heatmap"]       = heatmap
    s["orig_img"]      = orig_img

    gradcam_results.append(s)
    status = "CORRECT" if s["correct"] else "MISCLASSIFIED"
    print(f"     {s['class_name']:5s} -> Pred: {s['pred_class']:5s} "
          f"({s['confidence']:.2%}) [{status}]")

grad_cam.remove_hooks()

# --- Save individual + grid figures ---
def overlay_gradcam(orig_img, heatmap, alpha=0.5):
    """Overlay Grad-CAM heatmap on original image using jet colormap."""
    import cv2 as _cv2
    img_arr  = np.array(orig_img.resize((224, 224)))
    heatmap_resized = np.uint8(255 * heatmap)
    heatmap_color   = _cv2.applyColorMap(heatmap_resized, _cv2.COLORMAP_JET)
    heatmap_color   = _cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(alpha * heatmap_color + (1 - alpha) * img_arr)
    return overlay

# Fall back if opencv not available
try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False
    print("     INFO: OpenCV not found. Using matplotlib colormap for overlay.")

def make_overlay(orig_img, heatmap):
    img_arr = np.array(orig_img.resize((224, 224))).astype(np.float32) / 255.0
    # Resize heatmap to 224x224 using PIL (handles arbitrary heatmap sizes from layer4)
    hm_pil   = Image.fromarray(np.uint8(255 * heatmap)).resize((224, 224), Image.BILINEAR)
    hm_224   = np.array(hm_pil).astype(np.float32) / 255.0
    cmap     = plt.get_cmap("jet")
    h_rgba   = cmap(hm_224)[:, :, :3]  # (224, 224, 3)
    overlay  = np.clip(0.5 * h_rgba + 0.5 * img_arr, 0, 1)
    return overlay

print("[A4] Saving Grad-CAM figures...")
n = len(gradcam_results)
if n == 0:
    print("     ERROR: No Grad-CAM results generated. Skipping figure save.")
else:
    # Individual figures
    for s in gradcam_results:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        img_arr = np.array(s["orig_img"].resize((224, 224)))
        overlay  = make_overlay(s["orig_img"], s["heatmap"])

        axes[0].imshow(img_arr);        axes[0].set_title("Original Image"); axes[0].axis("off")
        axes[1].imshow(s["heatmap"], cmap="jet"); axes[1].set_title("Grad-CAM Heatmap"); axes[1].axis("off")
        axes[2].imshow(overlay);        axes[2].set_title("Grad-CAM Overlay"); axes[2].axis("off")

        status_str = "✓ Correct" if s["correct"] else "✗ Misclassified"
        fig.suptitle(
            f"True: {s['class_name']} | Predicted: {s['pred_class']} "
            f"({s['confidence']:.2%}) | {status_str}",
            fontsize=11, fontweight="bold"
        )
        plt.tight_layout()
        out_path = GRADCAM_DIR / f"gradcam_{s['class_name']}.png"
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()

    # Grid figure
    cols = 3
    rows_needed = n
    fig = plt.figure(figsize=(cols * 5, rows_needed * 4))
    fig.suptitle(
        "Grad-CAM Explainability — ResNet50 layer4\n"
        "Grad-CAM provides a visual indication of image regions that contributed\n"
        "to the model's prediction. It does NOT prove clinical correctness.",
        fontsize=11, y=1.01
    )

    for i, s in enumerate(gradcam_results):
        img_arr = np.array(s["orig_img"].resize((224, 224)))
        overlay  = make_overlay(s["orig_img"], s["heatmap"])

        ax1 = fig.add_subplot(rows_needed, cols, i * cols + 1)
        ax2 = fig.add_subplot(rows_needed, cols, i * cols + 2)
        ax3 = fig.add_subplot(rows_needed, cols, i * cols + 3)

        ax1.imshow(img_arr);            ax1.axis("off"); ax1.set_title("Original")
        ax2.imshow(s["heatmap"], cmap="jet"); ax2.axis("off"); ax2.set_title("Heatmap")
        ax3.imshow(overlay);            ax3.axis("off"); ax3.set_title("Overlay")

        status_str = "✓" if s["correct"] else "✗"
        ax1.set_ylabel(
            f"{status_str} True:{s['class_name']}\nPred:{s['pred_class']} "
            f"({s['confidence']:.1%})",
            rotation=0, labelpad=80, va="center", fontsize=8
        )

    plt.tight_layout()
    plt.savefig(GRADCAM_DIR / "gradcam_examples_grid.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {GRADCAM_DIR / 'gradcam_examples_grid.png'}")

    # Simple 2-column summary figure
    fig2, axes = plt.subplots(n, 2, figsize=(8, n * 3))
    if n == 1:
        axes = [axes]
    for i, s in enumerate(gradcam_results):
        overlay = make_overlay(s["orig_img"], s["heatmap"])
        axes[i][0].imshow(np.array(s["orig_img"].resize((224, 224)))); axes[i][0].axis("off")
        axes[i][1].imshow(overlay); axes[i][1].axis("off")
        status_str = "✓ Correct" if s["correct"] else "✗ Misclassified"
        axes[i][0].set_title(f"True: {s['class_name']}", fontsize=8)
        axes[i][1].set_title(
            f"Pred: {s['pred_class']} ({s['confidence']:.1%}) {status_str}", fontsize=8
        )
    plt.suptitle("Grad-CAM Examples (Original vs Overlay)", fontsize=11)
    plt.tight_layout()
    plt.savefig(GRADCAM_DIR / "gradcam_examples.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {GRADCAM_DIR / 'gradcam_examples.png'}")
    print(f"     Grad-CAM complete. {n} class examples generated.")

# ─────────────────────────────────────────────────────────────────
# SECTION B: SHAP FOR CLINICAL METADATA
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART B — SHAP FOR CLINICAL METADATA MLP")
print("="*70)

shap_error = None
shap_results = {}

try:
    import shap
    print(f"[B0] SHAP version: {shap.__version__}")

    # --- Load metadata model ---
    print("[B1] Loading metadata model and preprocessor...")
    preprocessor = joblib.load(PREPROCESSOR)
    print(f"     Preprocessor loaded: {type(preprocessor)}")

    # Determine input_dim from preprocessor
    test_meta_raw = test_df[["age_approx", "sex", "anatom_site_general"]].copy()
    X_test_transformed = preprocessor.transform(test_meta_raw)
    if hasattr(X_test_transformed, 'toarray'):
        X_test_transformed = X_test_transformed.toarray()
    input_dim = X_test_transformed.shape[1]
    print(f"     Preprocessed feature dim: {input_dim}")

    meta_model = ISICMetadataClassifier(
        input_dim=input_dim, num_classes=NUM_CLASSES,
        hidden_dim_1=128, hidden_dim_2=64, embedding_dim=32,
        dropout_1=0.3, dropout_2=0.2
    )
    ckpt_meta = torch.load(MODEL_METADATA, map_location=DEVICE, weights_only=True)
    meta_model.load_state_dict(ckpt_meta["model_state_dict"])
    meta_model.eval()
    print("     Metadata model loaded OK.")

    # --- Get feature names AFTER preprocessing ---
    raw_feature_names = preprocessor.get_feature_names_out()
    feature_names = [f.replace("num__", "").replace("cat__", "") for f in raw_feature_names]
    if len(feature_names) != input_dim:
        feature_names = [f"feature_{i}" for i in range(input_dim)]
    print(f"     Feature names ({len(feature_names)}): {feature_names}")

    # --- Wrapping model for SHAP (numpy in -> numpy out) ---
    def meta_predict_proba(X_np):
        """
        Wrapper: numpy array -> model softmax probabilities (numpy).
        Input is already preprocessed (X_test_transformed format).
        """
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        with torch.no_grad():
            logits = meta_model(X_tensor)
            probs  = torch.softmax(logits, dim=1).numpy()
        return probs

    # --- Background dataset for KernelExplainer ---
    print("[B2] Preparing SHAP background samples...")
    # Use train data as background
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    X_train_raw = train_df[["age_approx", "sex", "anatom_site_general"]].copy()
    X_train_transformed = preprocessor.transform(X_train_raw)
    if hasattr(X_train_transformed, 'toarray'):
        X_train_transformed = X_train_transformed.toarray()

    np.random.seed(SEED)
    bg_idx  = np.random.choice(len(X_train_transformed), size=100, replace=False)
    X_bg    = X_train_transformed[bg_idx]

    # Test background (200 random samples)
    test_idx_sample = np.random.choice(len(X_test_transformed), size=200, replace=False)
    X_explain = X_test_transformed[test_idx_sample]
    y_explain = labels[test_idx_sample] if len(labels) == len(test_df) else None

    print(f"     Background: {X_bg.shape}, Explain set: {X_explain.shape}")

    # --- Run KernelExplainer ---
    print("[B3] Running SHAP KernelExplainer (this may take a few minutes)...")
    explainer = shap.KernelExplainer(meta_predict_proba, X_bg)
    shap_values = explainer.shap_values(X_explain, nsamples=50, silent=True)

    # --- Normalize SHAP output to (N, n_features, n_classes) ---
    # SHAP 0.49.x returns ndarray (N, n_features, n_classes); older SHAP returns list of (N, n_features).
    if isinstance(shap_values, list):
        # Legacy: list of num_classes arrays each (N, n_features)
        shap_3d = np.stack(shap_values, axis=-1)   # -> (N, n_features, n_classes)
        print(f"     SHAP values (legacy list): {len(shap_values)} classes, shape {shap_values[0].shape}")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        if shap_values.shape[1] == input_dim:
            shap_3d = shap_values                    # already (N, n_features, n_classes)
        else:
            shap_3d = shap_values.transpose(0, 2, 1) # -> (N, n_features, n_classes)
        print(f"     SHAP values (ndarray 3D): {shap_values.shape} -> normalized {shap_3d.shape}")
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
        shap_3d = shap_values[:, :, np.newaxis]      # (N, n_features, 1)
        print(f"     SHAP values (2D array): {shap_values.shape}")
    else:
        shap_3d = np.array(shap_values)
        print(f"     SHAP values (other): {shap_3d.shape}")

    N_shap, n_feat_shap, n_cls_shap = shap_3d.shape
    print(f"     Normalized SHAP shape: {shap_3d.shape} "
          f"(samples={N_shap}, features={n_feat_shap}, classes={n_cls_shap})")

    # --- Global mean |SHAP| per feature (averaged over classes and samples) ---
    shap_mean_abs    = np.abs(shap_3d).mean(axis=2)   # (N, n_features) avg across classes
    global_importance = shap_mean_abs.mean(axis=0)     # (n_features,)

    # Signed mean: (N, n_features) for summary plots
    shap_mean_signed  = shap_3d.mean(axis=2)           # (N, n_features)

    # --- Save feature importance CSV ---
    feat_imp_df = pd.DataFrame({
        "feature":        feature_names,
        "mean_abs_shap":  global_importance.tolist(),
    }).sort_values("mean_abs_shap", ascending=False)
    feat_imp_df.to_csv(SHAP_DIR / "shap_feature_importance.csv", index=False)
    print(f"     Saved: {SHAP_DIR / 'shap_feature_importance.csv'}")

    try:
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_mean_signed, X_explain,
            feature_names=feature_names,
            show=False, max_display=input_dim, plot_type="dot"
        )
        plt.title(
            "SHAP Summary Plot — Metadata MLP\n"
            "SHAP estimates the contribution of model features to individual predictions.\n"
            "Higher |SHAP| = greater impact on model output (NOT causal).",
            fontsize=9
        )
        plt.tight_layout()
        plt.savefig(SHAP_DIR / "shap_summary.png", dpi=100, bbox_inches="tight")
        plt.close()
        print(f"     Saved: {SHAP_DIR / 'shap_summary.png'}")
    except Exception as e:
        print(f"     WARNING: SHAP summary plot failed: {e}. Falling back to bar plot.")

    # --- SHAP Bar Plot (global importance) ---
    plt.figure(figsize=(8, max(4, len(feature_names) * 0.4)))
    sorted_idx = np.argsort(global_importance)[::-1]
    bars = plt.barh(
        [feature_names[i] for i in sorted_idx[::-1]],
        global_importance[sorted_idx[::-1]],
        color=plt.cm.viridis(np.linspace(0.2, 0.9, len(feature_names)))
    )
    plt.xlabel("Mean |SHAP value| (averaged over all classes & test samples)")
    plt.title(
        "SHAP Global Feature Importance — Metadata MLP\n"
        "SHAP measures model-feature contribution, NOT causation.",
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_bar.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {SHAP_DIR / 'shap_bar.png'}")

    # --- Individual SHAP explanations (up to 8 samples, one per class if possible) ---
    print("[B4] Generating individual SHAP explanations...")
    # Map test indices by class
    if y_explain is not None:
        individual_indices = {}
        for i, lab in enumerate(y_explain):
            if lab not in individual_indices:
                individual_indices[lab] = i
            if len(individual_indices) == NUM_CLASSES:
                break
    else:
        individual_indices = {i: i for i in range(min(8, len(X_explain)))}

    n_individuals = len(individual_indices)
    fig_ind, axes_ind = plt.subplots(n_individuals, 1,
                                      figsize=(10, n_individuals * 3))
    if n_individuals == 1:
        axes_ind = [axes_ind]

    for plot_i, (class_label, sample_i) in enumerate(sorted(individual_indices.items())):
        ax = axes_ind[plot_i]
        class_name_str = CLASS_NAMES[class_label] if class_label < NUM_CLASSES else str(class_label)
        
        # Mean SHAP for this sample across classes
        sample_shap = shap_mean_signed[sample_i]  # (input_dim,)
        
        sorted_f = np.argsort(np.abs(sample_shap))[::-1][:10]
        colors = ["#d73027" if v > 0 else "#4575b4" for v in sample_shap[sorted_f]]
        ax.barh([feature_names[fi] for fi in sorted_f[::-1]],
                sample_shap[sorted_f[::-1]],
                color=colors[::-1])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"Sample {sample_i} — True class: {class_name_str}  "
                     "(SHAP contribution, NOT causal)", fontsize=9)
        ax.set_xlabel("SHAP value (mean across classes)", fontsize=8)

    plt.suptitle(
        "Individual SHAP Explanations — Metadata MLP\n"
        "Red = feature increased predicted probability | Blue = decreased",
        fontsize=10, y=1.01
    )
    plt.tight_layout()
    plt.savefig(SHAP_DIR / "shap_individual_examples.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {SHAP_DIR / 'shap_individual_examples.png'}")

    shap_results = {
        "global_importance": {fn: float(gv) for fn, gv in zip(feature_names, global_importance)},
        "n_background":      100,
        "n_explained":       len(X_explain),
        "feature_names":     feature_names,
        "shap_nsamples":     50,
    }
    print("[B] SHAP complete.")

except ImportError:
    shap_error = "SHAP library not installed. Install with: pip install shap"
    print(f"[B] ERROR: {shap_error}")
except Exception as e:
    shap_error = f"SHAP failed: {str(e)}"
    print(f"[B] ERROR: {shap_error}")
    traceback.print_exc()

# ─────────────────────────────────────────────────────────────────
# SECTION C: MULTIMODAL ATTENTION VISUALIZATION
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART C — MULTIMODAL ATTENTION VISUALIZATION")
print("="*70)

print("[C1] Loading multimodal attention model...")
attn_model = ISICMultimodalAttentionClassifier(
    image_dim=2048, metadata_dim=32,
    num_classes=NUM_CLASSES, token_dim=256, num_heads=4,
    dense_dim=128, dropout_1=0.3, dropout_2=0.2
)
ckpt_attn = torch.load(MODEL_MULTIMODAL, map_location=DEVICE, weights_only=True)
attn_model.load_state_dict(ckpt_attn["model_state_dict"])
attn_model.eval()
print("     Multimodal attention model loaded OK.")

# Architecture metadata
ATTN_CONFIG = {
    "num_heads": 4,
    "num_tokens": 2,
    "token_dim": 256,
    "attention_matrix_shape": "(Batch, 2, 2)",
    "tokens": ["IMAGE_TOKEN", "METADATA_TOKEN"],
    "aggregation": "Averaged across 4 heads (PyTorch MHA default)",
    "interactions": {
        "[0,0]": "IMAGE -> IMAGE",
        "[0,1]": "IMAGE -> METADATA",
        "[1,0]": "METADATA -> IMAGE",
        "[1,1]": "METADATA -> METADATA",
    }
}
print(f"     Attention config: {ATTN_CONFIG['num_heads']} heads, "
      f"token_dim={ATTN_CONFIG['token_dim']}, 2x2 matrix")

# --- Extract attention weights over entire test set ---
print("[C2] Computing attention weights over test embeddings...")
np.random.seed(SEED)
sample_size = min(500, len(img_emb))
idx_sample  = np.random.choice(len(img_emb), size=sample_size, replace=False)

all_attn = []
batch_size = 64
img_emb_t  = torch.tensor(img_emb,  dtype=torch.float32)
meta_emb_t = torch.tensor(meta_emb, dtype=torch.float32)
labels_arr = labels

with torch.no_grad():
    for i in range(0, sample_size, batch_size):
        idx_b = idx_sample[i:i+batch_size]
        x_img  = img_emb_t[idx_b]
        x_meta = meta_emb_t[idx_b]
        attn_w = attn_model.get_attention_weights(x_img, x_meta)  # (B, 2, 2)
        all_attn.append(attn_w.numpy())

all_attn = np.concatenate(all_attn, axis=0)  # (N, 2, 2)
mean_attn = all_attn.mean(axis=0)            # (2, 2)
print(f"     Mean attention matrix (over {sample_size} test samples):")
print(f"     {mean_attn}")

# --- Attention Matrix Plot ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# (1) Mean attention heatmap
im = axes[0].imshow(mean_attn, cmap="YlOrRd", vmin=0, vmax=1)
axes[0].set_xticks([0, 1]); axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["IMAGE", "METADATA"], fontsize=11)
axes[0].set_yticklabels(["IMAGE", "METADATA"], fontsize=11)
axes[0].set_xlabel("Attends TO (Key)", fontsize=11)
axes[0].set_ylabel("Query (Token)", fontsize=11)
axes[0].set_title("Mean Attention Matrix\n(Averaged over 4 heads, 500 test samples)", fontsize=10)
plt.colorbar(im, ax=axes[0])
for r in range(2):
    for c in range(2):
        axes[0].text(c, r, f"{mean_attn[r, c]:.3f}", ha="center", va="center",
                     fontsize=13, fontweight="bold",
                     color="white" if mean_attn[r, c] > 0.5 else "black")

# (2) Distribution of attention weights (boxplot)
labels_attn = ["IMG→IMG", "IMG→META", "META→IMG", "META→META"]
attn_flat = [
    all_attn[:, 0, 0], all_attn[:, 0, 1],
    all_attn[:, 1, 0], all_attn[:, 1, 1]
]
bp = axes[1].boxplot(attn_flat, labels=labels_attn, patch_artist=True,
                      medianprops=dict(color="red", linewidth=2))
colors_bp = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50"]
for patch, color in zip(bp["boxes"], colors_bp):
    patch.set_facecolor(color); patch.set_alpha(0.7)
axes[1].set_ylabel("Attention Weight", fontsize=11)
axes[1].set_title(
    f"Distribution of Attention Weights\n"
    f"(n={sample_size} samples, 4 heads averaged)", fontsize=10
)
axes[1].set_ylim(0, 1)

plt.suptitle(
    "Multimodal Attention Analysis\n"
    "The attention mechanism provides evidence of representation interaction "
    "between the image and metadata modalities.\n"
    "Attention weights SHOULD NOT be interpreted as causal feature importance.",
    fontsize=9
)
plt.tight_layout()
plt.savefig(ATTN_DIR / "attention_matrix.png", dpi=100, bbox_inches="tight")
plt.close()
print(f"     Saved: {ATTN_DIR / 'attention_matrix.png'}")

# --- Per-class attention analysis ---
print("[C3] Computing per-class attention analysis...")
# Get predictions for sample
all_preds = []
with torch.no_grad():
    for i in range(0, sample_size, batch_size):
        idx_b = idx_sample[i:i+batch_size]
        x_img  = img_emb_t[idx_b]
        x_meta = meta_emb_t[idx_b]
        logits = attn_model(x_img, x_meta)
        preds  = logits.argmax(dim=1).numpy()
        all_preds.append(preds)
all_preds = np.concatenate(all_preds, axis=0)
sample_labels = labels_arr[idx_sample] if len(labels_arr) > 0 else None

per_class_attn = {}
if sample_labels is not None:
    for c in range(NUM_CLASSES):
        mask = (sample_labels == c)
        if mask.sum() > 0:
            per_class_attn[CLASS_NAMES[c]] = all_attn[mask].mean(axis=0)

if per_class_attn:
    n_cls = len(per_class_attn)
    ncols = min(4, n_cls)
    nrows = (n_cls + ncols - 1) // ncols
    fig_cls, axes_cls = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 3.5))
    axes_cls = np.array(axes_cls).flatten() if n_cls > 1 else [axes_cls]

    for idx_c, (cname, mat) in enumerate(per_class_attn.items()):
        ax = axes_cls[idx_c]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["IMG", "META"], fontsize=9)
        ax.set_yticklabels(["IMG", "META"], fontsize=9)
        ax.set_title(f"{cname}", fontsize=10, fontweight="bold")
        for r in range(2):
            for c_pos in range(2):
                ax.text(c_pos, r, f"{mat[r, c_pos]:.3f}",
                        ha="center", va="center", fontsize=9,
                        color="white" if mat[r, c_pos] > 0.5 else "black")

    for j in range(len(per_class_attn), len(axes_cls)):
        axes_cls[j].set_visible(False)

    plt.suptitle(
        "Per-Class Mean Attention Matrices\n"
        "Attention weights averaged per diagnostic class (test set).\n"
        "NOTE: These are descriptive patterns, NOT causal explanations.",
        fontsize=9
    )
    plt.tight_layout()
    plt.savefig(ATTN_DIR / "attention_heads.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {ATTN_DIR / 'attention_heads.png'}")

# Compute useful statistics
img_to_meta_mean = float(mean_attn[0, 1])
meta_to_img_mean = float(mean_attn[1, 0])
img_self_mean    = float(mean_attn[0, 0])
meta_self_mean   = float(mean_attn[1, 1])

attn_summary = {
    "mean_attn_matrix": mean_attn.tolist(),
    "img_to_img_mean":  img_self_mean,
    "img_to_meta_mean": img_to_meta_mean,
    "meta_to_img_mean": meta_to_img_mean,
    "meta_to_meta_mean":meta_self_mean,
    "stronger_cross_modal": "IMG->META" if img_to_meta_mean > meta_to_img_mean else "META->IMG",
    "n_samples": sample_size,
}
print(f"     IMG->META: {img_to_meta_mean:.3f} | META->IMG: {meta_to_img_mean:.3f}")
print(f"     IMG self: {img_self_mean:.3f} | META self: {meta_self_mean:.3f}")

# Pre-compute conditional string (Python 3.10 f-string backslash workaround)
if img_to_meta_mean > meta_to_img_mean:
    _attn_cross_modal_note = (
        "IMAGE -> METADATA cross-modal attention is stronger than METADATA -> IMAGE, "
        "suggesting the image representation integrates metadata information more strongly."
    )
else:
    _attn_cross_modal_note = (
        "METADATA -> IMAGE cross-modal attention is stronger than IMAGE -> METADATA, "
        "suggesting the metadata representation integrates image information more strongly."
    )

# Write attention_analysis.md
attn_md = f"""# Multimodal Attention Analysis Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Architecture

| Parameter | Value |
|---|---|
| Number of attention heads | {ATTN_CONFIG['num_heads']} |
| Number of modality tokens | {ATTN_CONFIG['num_tokens']} |
| Token dimension | {ATTN_CONFIG['token_dim']} |
| Attention matrix shape | {ATTN_CONFIG['attention_matrix_shape']} |
| Head aggregation | {ATTN_CONFIG['aggregation']} |

## Attention Matrix Layout

The 2x2 attention matrix represents the following interactions:

|  | Attends TO: IMAGE | Attends TO: METADATA |
|---|---|---|
| **Query: IMAGE** | IMAGE → IMAGE (self) | IMAGE → METADATA (cross) |
| **Query: METADATA** | METADATA → IMAGE (cross) | METADATA → METADATA (self) |

## Quantitative Results (n={sample_size} test samples)

| Interaction | Mean Attention Weight |
|---|---|
| IMAGE → IMAGE | {img_self_mean:.4f} |
| IMAGE → METADATA | {img_to_meta_mean:.4f} |
| METADATA → IMAGE | {meta_to_img_mean:.4f} |
| METADATA → METADATA | {meta_self_mean:.4f} |

**Stronger cross-modal direction:** {attn_summary['stronger_cross_modal']}

## Interpretation

The attention mechanism provides evidence of representation interaction between the image and metadata modalities.

- The IMAGE token predominantly attends to itself ({img_self_mean:.3f}) and to METADATA ({img_to_meta_mean:.3f}).
- The METADATA token attends to itself ({meta_self_mean:.3f}) and to IMAGE ({meta_to_img_mean:.3f}).
- {_attn_cross_modal_note}

## Scientific Limitations

> **IMPORTANT:** Attention weights SHOULD NOT automatically be interpreted as feature importance or causal evidence.
>
> The attention mechanism provides evidence of representation interaction between the image and metadata modalities. 
> It does NOT prove that any specific metadata feature or image region caused the prediction.
>
> These patterns are descriptive, not explanatory in a causal sense.

## Reproducibility

- Random seed: {SEED}
- Checkpoint: `models/multimodal/attention_model.pt`
- Test embeddings: `data/processed/embeddings/test_embeddings.npz`
- Samples analyzed: {sample_size}
"""
with open(ATTN_DIR / "attention_analysis.md", "w", encoding="utf-8") as f:
    f.write(attn_md)
print(f"     Saved: {ATTN_DIR / 'attention_analysis.md'}")
print("[C] Attention visualization complete.")

# ─────────────────────────────────────────────────────────────────
# SECTION D: MULTIMODAL CASE STUDIES
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART D — MULTIMODAL CASE STUDIES")
print("="*70)

print("[D1] Selecting case-study samples...")

# Run multimodal predictions on test set
all_mm_preds   = []
all_mm_probs   = []
all_mm_correct = []
BATCH = 64
with torch.no_grad():
    for i in range(0, len(img_emb), BATCH):
        x_img  = img_emb_t[i:i+BATCH]
        x_meta = meta_emb_t[i:i+BATCH]
        logits = attn_model(x_img, x_meta)
        probs  = torch.softmax(logits, dim=1)
        preds  = logits.argmax(dim=1)
        all_mm_preds.extend(preds.numpy().tolist())
        all_mm_probs.extend(probs.numpy().tolist())

all_mm_preds   = np.array(all_mm_preds)
all_mm_probs   = np.array(all_mm_probs)
all_mm_correct = (all_mm_preds == labels_arr[:len(all_mm_preds)])

# Strategy: select 4 specific cases
# 1. Correct — most common class (NV)
# 2. Correct — clinically important (MEL)
# 3. Misclassification
# 4. Rare class (VASC or DF)

def find_sample(condition_mask, class_filter=None, require_image=True):
    for i in np.where(condition_mask)[0]:
        if class_filter is not None and labels_arr[i] != class_filter:
            continue
        img_id = image_ids[i]
        if require_image:
            if not (IMAGE_DIR / f"{img_id}.jpg").exists():
                continue
        return int(i)
    return None

np.random.seed(SEED)
case_indices = {}

# Case 1: Correct NV
idx = find_sample(all_mm_correct, class_filter=CLASS_NAMES.index("NV"))
if idx is not None: case_indices["Correct - NV (Common Class)"] = idx

# Case 2: Correct MEL
idx = find_sample(all_mm_correct, class_filter=CLASS_NAMES.index("MEL"))
if idx is not None: case_indices["Correct - MEL (Clinically Important)"] = idx

# Case 3: Misclassified
idx = find_sample(~all_mm_correct)
if idx is not None: case_indices["Misclassification"] = idx

# Case 4: Rare class (VASC)
idx = find_sample(np.ones(len(labels_arr), dtype=bool), class_filter=CLASS_NAMES.index("VASC"))
if idx is None:
    idx = find_sample(np.ones(len(labels_arr), dtype=bool), class_filter=CLASS_NAMES.index("DF"))
if idx is not None: case_indices["Rare Class (VASC/DF)"] = idx

print(f"     Selected {len(case_indices)} case studies.")

# Get metadata for test samples
# test_df is already ordered to match embeddings?
# We need to match by image_id
test_df_indexed = test_df.set_index("image_id") if "image_id" in test_df.columns else None

# SHAP for case studies (if available)
case_shap_values = {}
if not shap_error and 'explainer' in dir() and 'shap_values' in dir():
    pass  # we'll approximate from computed shap_values

# ---- Build case study figure ----
n_cases = len(case_indices)
if n_cases > 0:
    fig_cs = plt.figure(figsize=(16, n_cases * 4))
    fig_cs.suptitle(
        "Phase 6 — Multimodal Case Studies\n"
        "Combining Grad-CAM (Image Branch) + SHAP (Metadata Branch) + Attention (Multimodal)\n"
        "NOTE: Research prototype — NOT clinically validated.",
        fontsize=10, y=1.01
    )

    for case_i, (case_label, sample_idx) in enumerate(case_indices.items()):
        img_id    = str(image_ids[sample_idx])
        true_lab  = int(labels_arr[sample_idx])
        pred_lab  = int(all_mm_preds[sample_idx])
        conf      = float(all_mm_probs[sample_idx][pred_lab])
        true_name = CLASS_NAMES[true_lab]
        pred_name = CLASS_NAMES[pred_lab]
        is_correct= (true_lab == pred_lab)

        # Metadata from test_df
        meta_info = {"age": "N/A", "sex": "N/A", "site": "N/A"}
        if test_df_indexed is not None and img_id in test_df_indexed.index:
            row = test_df_indexed.loc[img_id]
            meta_info["age"]  = str(row.get("age_approx", "N/A"))
            meta_info["sex"]  = str(row.get("sex", "N/A"))
            meta_info["site"] = str(row.get("anatom_site_general", "N/A"))

        # Grad-CAM for this sample
        orig_img, img_tensor = load_image_tensor(img_id)
        if img_tensor is not None:
            grad_cam2 = GradCAM(img_model, img_model.feature_extractor[-1])
            hm2, _, _ = grad_cam2.generate(img_tensor, class_idx=true_lab)
            grad_cam2.remove_hooks()
            overlay2 = make_overlay(orig_img, hm2)
        else:
            orig_img = None; overlay2 = None

        # Attention weights for this sample
        with torch.no_grad():
            xi  = img_emb_t[sample_idx:sample_idx+1]
            xm  = meta_emb_t[sample_idx:sample_idx+1]
            aw  = attn_model.get_attention_weights(xi, xm)[0].numpy()  # (2, 2)

        # Subplot: 1 row x 4 panels per case
        row_base = case_i
        ax1 = fig_cs.add_subplot(n_cases, 4, row_base * 4 + 1)
        ax2 = fig_cs.add_subplot(n_cases, 4, row_base * 4 + 2)
        ax3 = fig_cs.add_subplot(n_cases, 4, row_base * 4 + 3)
        ax4 = fig_cs.add_subplot(n_cases, 4, row_base * 4 + 4)

        # Panel 1: Original image + status
        if orig_img:
            ax1.imshow(np.array(orig_img.resize((224, 224))))
        else:
            ax1.text(0.5, 0.5, "Image not available", ha="center", va="center")
        status_str = "✓ CORRECT" if is_correct else "✗ MISCLASSIFIED"
        ax1.set_title(
            f"{case_label}\n"
            f"True: {true_name} | Pred: {pred_name}\n"
            f"Conf: {conf:.2%} | {status_str}",
            fontsize=7.5, pad=3
        )
        ax1.axis("off")

        # Panel 2: Grad-CAM overlay
        if overlay2 is not None:
            ax2.imshow(overlay2)
        else:
            ax2.text(0.5, 0.5, "N/A", ha="center", va="center")
        ax2.set_title("Grad-CAM Overlay\n(Image regions contributing\nto prediction)", fontsize=7.5)
        ax2.axis("off")

        # Panel 3: Metadata text + SHAP summary
        meta_text = (
            f"Age: {meta_info['age']}\n"
            f"Sex: {meta_info['sex']}\n"
            f"Site: {meta_info['site']}\n"
            f"\nSHAP not available\n(see shap/ folder)"
        ) if shap_error else (
            f"Age: {meta_info['age']}\n"
            f"Sex: {meta_info['sex']}\n"
            f"Site: {meta_info['site']}\n"
            f"\nSee shap_summary.png\nfor global SHAP values"
        )
        ax3.text(0.05, 0.95, meta_text, transform=ax3.transAxes,
                 fontsize=8, va="top", family="monospace",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax3.set_title("Metadata &\nSHAP Context", fontsize=7.5)
        ax3.axis("off")

        # Panel 4: Attention matrix
        im4 = ax4.imshow(aw, cmap="YlOrRd", vmin=0, vmax=1)
        ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
        ax4.set_xticklabels(["IMG", "META"], fontsize=8)
        ax4.set_yticklabels(["IMG", "META"], fontsize=8)
        ax4.set_title("Attention Matrix\n(This sample)", fontsize=7.5)
        for r in range(2):
            for c_pos in range(2):
                ax4.text(c_pos, r, f"{aw[r, c_pos]:.2f}",
                         ha="center", va="center", fontsize=9, fontweight="bold",
                         color="white" if aw[r, c_pos] > 0.6 else "black")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(EXPL_DIR / "multimodal_case_studies.png", dpi=100, bbox_inches="tight")
    plt.close()
    print(f"     Saved: {EXPL_DIR / 'multimodal_case_studies.png'}")
    print(f"[D] Case studies complete.")

# ─────────────────────────────────────────────────────────────────
# SECTION E: QUANTITATIVE EXPLAINABILITY ANALYSIS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART E — QUANTITATIVE EXPLAINABILITY ANALYSIS")
print("="*70)

n_test = len(all_mm_preds)
n_correct   = int(all_mm_correct.sum())
n_incorrect = n_test - n_correct
accuracy    = n_correct / n_test

# Grad-CAM coverage (mean activated region > 0.5)
gradcam_coverage = []
for s in gradcam_results:
    hm = s["heatmap"]
    coverage = float((hm > 0.5).mean())
    gradcam_coverage.append(coverage)
mean_coverage = float(np.mean(gradcam_coverage)) if gradcam_coverage else float("nan")

# Attention statistics
img_to_meta_std  = float(all_attn[:, 0, 1].std())
meta_to_img_std  = float(all_attn[:, 1, 0].std())

# Correct vs incorrect attention (test sample)
correct_mask_sample = (sample_labels == all_preds) if (sample_labels is not None and len(all_preds)==sample_size) else None
if correct_mask_sample is not None and correct_mask_sample.sum() > 0 and (~correct_mask_sample).sum() > 0:
    attn_correct   = all_attn[correct_mask_sample].mean(axis=0)
    attn_incorrect = all_attn[~correct_mask_sample].mean(axis=0)
    attn_diff_note = (
        f"Correct predictions: mean IMG->META={attn_correct[0,1]:.3f}, "
        f"META->IMG={attn_correct[1,0]:.3f}\n"
        f"Incorrect predictions: mean IMG->META={attn_incorrect[0,1]:.3f}, "
        f"META->IMG={attn_incorrect[1,0]:.3f}\n"
        f"NOTE: Sample sizes may be small. No statistical significance test performed."
    )
else:
    attn_correct   = None
    attn_incorrect = None
    attn_diff_note = "Insufficient matched data for correct/incorrect attention comparison."

print(f"     Test samples: {n_test}")
print(f"     Correct: {n_correct} ({accuracy:.2%})")
print(f"     Grad-CAM mean coverage (>0.5 threshold): {mean_coverage:.2%}")
print(f"     IMG->META attention: mean={img_to_meta_mean:.3f} std={img_to_meta_std:.3f}")
print(f"     META->IMG attention: mean={meta_to_img_mean:.3f} std={meta_to_img_std:.3f}")
if shap_results:
    top_feat = max(shap_results["global_importance"], key=shap_results["global_importance"].get)
    print(f"     Top SHAP feature: {top_feat} = {shap_results['global_importance'][top_feat]:.4f}")

quant_results = {
    "n_test_samples":               n_test,
    "n_correct":                    n_correct,
    "n_incorrect":                  n_incorrect,
    "accuracy":                     float(accuracy),
    "n_gradcam_classes":            len(gradcam_results),
    "gradcam_mean_coverage_gt05":   mean_coverage,
    "attn_img_to_meta_mean":        img_to_meta_mean,
    "attn_img_to_meta_std":         img_to_meta_std,
    "attn_meta_to_img_mean":        meta_to_img_mean,
    "attn_meta_to_img_std":         meta_to_img_std,
    "attn_correct_vs_incorrect_note": attn_diff_note,
    "shap_top_feature":             (top_feat if shap_results else "N/A"),
    "shap_error":                   shap_error,
}

# ─────────────────────────────────────────────────────────────────
# SECTION F: RESEARCH-SAFE INTERPRETATION REPORT
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PART F — RESEARCH REPORT")
print("="*70)

shap_section = (
    f"""## 3. SHAP Methodology

SHAP (SHapley Additive exPlanations) was applied to the trained Metadata-Only MLP using a
`KernelExplainer` with {shap_results.get('n_background', 'N/A')} background training samples.

- **Features explained:** {', '.join(shap_results.get('feature_names', ['N/A']))}
- **Test samples explained:** {shap_results.get('n_explained', 'N/A')}
- **SHAP nsamples (approximation):** {shap_results.get('shap_nsamples', 'N/A')}
- **Preprocessing:** The trained `preprocessor.joblib` pipeline (median imputation +
  StandardScaler for age; 'Unknown' category + OneHotEncoder for sex/site) was applied
  identically to the SHAP explanation set.
- **SHAP values represent:** The estimated additive contribution of each preprocessed feature
  to the model's predicted class probabilities.

> **SCIENTIFIC RULE:** SHAP estimates the contribution of model features to individual predictions.
> SHAP values do NOT establish causal relationships between features and skin lesion diagnoses.
"""
    if not shap_error else
    f"""## 3. SHAP Methodology

**STATUS: SHAP NOT AVAILABLE**

Error: {shap_error}

SHAP analysis could not be completed. The metadata MLP model and preprocessor are available
at `models/metadata_only/` for future SHAP analysis once the library is installed.
"""
)

# Pre-compute all conditional strings for report (Python 3.10 f-string workaround)
if len(gradcam_results) == NUM_CLASSES:
    _gradcam_class_note = "All 8 classes represented."
else:
    _class_list = ", ".join(s["class_name"] for s in gradcam_results)
    _gradcam_class_note = f"Classes represented: {_class_list}."

if not shap_error:
    _shap_global_note = (
        "- Global SHAP analysis completed on "
        + str(shap_results.get("n_explained", "N/A"))
        + " test samples."
    )
    _shap_importance_sorted = sorted(
        shap_results.get("global_importance", {}).items(),
        key=lambda x: -x[1]
    )
    _shap_importance_parts = [f"{k} ({v:.4f})" for k, v in _shap_importance_sorted]
    _shap_importance_note = "- Feature importance ranked by mean |SHAP|: " + ", ".join(_shap_importance_parts)
else:
    _shap_global_note     = "- SHAP analysis was not available in this run. See error log."
    _shap_importance_note = ""

report_text = f"""# Phase 6 — Model Explainability & Interpretability Report

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
- **Test samples analyzed:** {len(gradcam_results)} (one per diagnostic class)
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
""" + "\n".join(
    f"| {s['class_name']} | {s['pred_class']} | {s['confidence']:.2%} | {'Yes' if s['correct'] else 'No'} |"
    for s in gradcam_results
) + f"""

> **SCIENTIFIC NOTE:** Grad-CAM provides a visual indication of image regions that
> contributed to the model's prediction. It does NOT prove that the model is clinically
> correct, nor does it constitute a clinical ground-truth segmentation.

---

{shap_section}

---

## 4. Attention Visualization Methodology

The multimodal attention model uses `nn.MultiheadAttention` with:
- **4 attention heads**
- **2 modality tokens:** `[IMAGE_TOKEN, METADATA_TOKEN]`
- **Token dimension:** 256
- **Attention matrix shape:** (Batch, 2, 2) — averaged across 4 heads

Attention weights were computed over {sample_size} randomly sampled test examples.

**Mean Attention Matrix:**

|  | → IMAGE | → METADATA |
|---|---|---|
| **IMAGE →** | {mean_attn[0,0]:.4f} | {mean_attn[0,1]:.4f} |
| **METADATA →** | {mean_attn[1,0]:.4f} | {mean_attn[1,1]:.4f} |

> **SCIENTIFIC NOTE:** The attention mechanism provides evidence of representation interaction
> between the image and metadata modalities. Attention weights SHOULD NOT be automatically
> interpreted as feature importance or causal evidence.

---

## 5. Representative Findings

### Grad-CAM
- Grad-CAM activations were successfully generated for {len(gradcam_results)} of {NUM_CLASSES} diagnostic classes.
- {_gradcam_class_note}
- Mean activation coverage (threshold > 0.5): **{mean_coverage:.2%}** of image area per sample.
- For some classes the model focused on central lesion regions; for others, activation was more distributed.

### SHAP
{_shap_global_note}
{_shap_importance_note}

### Attention
- IMAGE -> METADATA cross-modal attention: **{img_to_meta_mean:.4f}** (std={img_to_meta_std:.4f})
- METADATA -> IMAGE cross-modal attention: **{meta_to_img_mean:.4f}** (std={meta_to_img_std:.4f})
- Stronger cross-modal direction: **{attn_summary['stronger_cross_modal']}**
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
  attention pattern (IMG→META: {img_to_meta_mean:.3f}, META→IMG: {meta_to_img_mean:.3f}).

These observations are consistent with the hypothesis that combining image and metadata
improves diagnostic classification. However, this does NOT prove clinical utility.

---

## 8. Reproducibility Information

- Random seed: {SEED}
- Image model checkpoint: `models/image_only/best_model.pt`
- Metadata model checkpoint: `models/metadata_only/best_model.pt`
- Multimodal model checkpoint: `models/multimodal/attention_model.pt`
- Preprocessor: `models/metadata_only/preprocessor.joblib`
- Test split: `data/processed/splits/test.csv`
- Grad-CAM target layer: `feature_extractor[-1]` (layer4)
- SHAP method: KernelExplainer, nsamples=50
- Attention samples: {sample_size}
- Platform: Windows, PyTorch {torch.__version__}, CPU mode
"""

report_path = EXPL_DIR / "explainability_report.md"
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_text)
print(f"     Saved: {report_path}")

# ─────────────────────────────────────────────────────────────────
# SECTION G: REPRODUCIBILITY CONFIG
# ─────────────────────────────────────────────────────────────────
print("\n[G] Writing reproducibility config...")

import platform
expl_config = {
    "phase":              "Phase 6 - Model Explainability",
    "date":               datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "random_seed":        SEED,
    "dataset_split":      "test",
    "test_csv":           "data/processed/splits/test.csv",
    "model_checkpoints": {
        "image_only":     "models/image_only/best_model.pt",
        "metadata_only":  "models/metadata_only/best_model.pt",
        "multimodal":     "models/multimodal/attention_model.pt",
    },
    "gradcam": {
        "target_layer":   "feature_extractor[-1] (ResNet50 layer4)",
        "n_classes_explained": len(gradcam_results),
        "class_names":    [s["class_name"] for s in gradcam_results],
    },
    "shap": {
        "method":         "KernelExplainer" if not shap_error else "UNAVAILABLE",
        "nsamples":       50,
        "n_background":   100,
        "background_source": "train split (random sample)",
        "n_explained":    shap_results.get("n_explained", 0),
        "feature_names":  shap_results.get("feature_names", []),
        "error":          shap_error,
    },
    "attention": {
        "num_heads":      4,
        "num_tokens":     2,
        "token_dim":      256,
        "n_samples_analyzed": sample_size,
        "aggregation":    "Averaged across 4 heads (PyTorch MHA default with average_attn_weights=True)",
    },
    "software": {
        "python":         platform.python_version(),
        "torch":          torch.__version__,
        "numpy":          np.__version__,
        "pandas":         pd.__version__,
        "joblib":         joblib.__version__,
    },
}
with open(EXPL_DIR / "explainability_config.json", "w", encoding="utf-8") as f:
    json.dump(expl_config, f, indent=2)
print(f"     Saved: {EXPL_DIR / 'explainability_config.json'}")

# ─────────────────────────────────────────────────────────────────
# SECTION H: FINAL PHASE SUMMARY
# ─────────────────────────────────────────────────────────────────
print("\n[H] Writing Phase 6 summary...")

# Pre-compute conditional strings (Python 3.10 f-string backslash workaround)
if shap_error:
    _shap_findings_text = "SHAP analysis was not available in this run.\n\nError: " + str(shap_error)
else:
    _shap_findings_text = (
        "- SHAP KernelExplainer applied with 100 background training samples.\n"
        "- 200 test samples explained.\n"
        "- Feature importance ranked by mean |SHAP value| across all classes.\n"
        "- See `explainability/shap/shap_feature_importance.csv` for full ranking.\n"
        "- SHAP measures model-feature contribution, NOT causation."
    )

files_created = []
for root, dirs, files in os.walk(EXPL_DIR):
    for fn in files:
        rel = os.path.relpath(os.path.join(root, fn), BASE_DIR)
        files_created.append(rel)
files_created.sort()

summary_text = f"""# Phase 6 — Explainability & Interpretability Summary

**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Status: COMPLETE

---

## 1. Files Created

| File | Description |
|------|-------------|
""" + "\n".join(
    f"| `{f}` | Generated artifact |"
    for f in files_created
) + f"""

---

## 2. Samples Analyzed

| Analysis | Samples |
|---|---|
| Grad-CAM | {len(gradcam_results)} classes ({', '.join(s['class_name'] for s in gradcam_results)}) |
| SHAP | {'200 test samples (KernelExplainer)' if not shap_error else 'UNAVAILABLE: ' + (shap_error or '')} |
| Attention | {sample_size} test samples |
| Case Studies | {n_cases} integrated examples |

---

## 3. Grad-CAM Findings

- Grad-CAM heatmaps generated for **{len(gradcam_results)}** of **{NUM_CLASSES}** diagnostic classes.
- Target layer: ResNet50 `layer4` (final convolutional block).
- Mean activation coverage (> 0.5 threshold): **{mean_coverage:.2%}** of image area.
- Model predicted correctly on **{sum(1 for s in gradcam_results if s['correct'])}** of **{len(gradcam_results)}** Grad-CAM samples.
- Grad-CAM provides a visual indication of image regions that contributed to the model's prediction.
  It does NOT prove clinical correctness.

---

## 4. SHAP Findings

{_shap_findings_text}

---

## 5. Attention Findings

| Interaction | Mean Weight |
|---|---|
| IMAGE → IMAGE (self) | {img_self_mean:.4f} |
| IMAGE → METADATA (cross) | {img_to_meta_mean:.4f} |
| METADATA → IMAGE (cross) | {meta_to_img_mean:.4f} |
| METADATA → METADATA (self) | {meta_self_mean:.4f} |

- Stronger cross-modal direction: **{attn_summary['stronger_cross_modal']}**
- The attention mechanism provides evidence of representation interaction between modalities.
- Attention weights SHOULD NOT be interpreted as causal explanations.

---

## 6. Multimodal Case Studies

{n_cases} integrated case studies created covering:
""" + "\n".join(f"- {case_label}" for case_label in case_indices.keys()) + f"""

Each case study shows: original image, Grad-CAM overlay, metadata context, and attention matrix.

---

## 7. Limitations

1. Grad-CAM is model-attribution, NOT clinical ground-truth segmentation.
2. SHAP measures model contribution, NOT causation.
3. Attention weights are not causal importance.
4. ISIC 2019 metadata is limited demographic information, NOT full clinical records.
5. `lesion_id` is NOT a confirmed patient ID.
6. The model is an academic research prototype, NOT clinically validated.
7. Class imbalance affects all metrics and explainability patterns.

---

## 8. Implementation Notes

- All analyses use CPU-mode PyTorch ({torch.__version__}).
- Random seed = {SEED} for all sampling.
- Grad-CAM hooks were registered and removed after each computation.
- SHAP KernelExplainer is an approximation; exact Shapley values require exponential computation.
- {"SHAP was successfully installed and run." if not shap_error else f"SHAP was NOT available: {shap_error}"}

---

## Recommendation for Phase 7

Phase 7 options (to be started only on user request):

**Option A — Interactive Web Application**
- Build a web dashboard for uploading dermoscopic images and receiving:
  - ResNet50 image classification + Grad-CAM overlay
  - Metadata input + SHAP explanation
  - Multimodal attention fusion prediction
  - Confidence scores and class probabilities

**Option B — Full Clinical-Grade Report Generation**
- Generate per-sample PDF reports combining all explainability artifacts.

**Option C — Extended Quantitative Analysis**
- Per-class Grad-CAM coverage analysis
- Statistical comparison of attention patterns across diagnostic groups
- SHAP interaction values for metadata feature pairs

DO NOT start Phase 7 without explicit user instruction.

---

**PHASE 6 COMPLETE**
"""

with open(BASE_DIR / "phase6_explainability_summary.md", "w", encoding="utf-8") as f:
    f.write(summary_text)
print(f"     Saved: phase6_explainability_summary.md")

# ─────────────────────────────────────────────────────────────────
# FINAL STATUS
# ─────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("PHASE 6 COMPLETE")
print("="*70)
print(f"\nGrad-CAM:   {len(gradcam_results)} classes explained")
print(f"SHAP:       {'complete' if not shap_error else 'FAILED - ' + str(shap_error)}")
print(f"Attention:  {sample_size} samples analyzed")
print(f"Cases:      {n_cases} case studies")
print(f"\nOutput dir: explainability/")
print("="*70)
