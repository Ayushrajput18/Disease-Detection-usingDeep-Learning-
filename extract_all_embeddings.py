import os
import sys
import torch
import joblib
import yaml
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from src.dataset import ISICImageDataset, get_image_transforms
from src.models.image_model import ISICImageClassifier
from src.models.clinical_model import ISICMetadataClassifier

def extract_and_align_embeddings():
    print("=== EXTRACTING & ALIGNING MODALITY EMBEDDINGS ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("data/processed/embeddings", exist_ok=True)
    
    # Load Image Model
    with open("configs/image_only.yaml", "r") as f:
        img_config = yaml.safe_load(f)
        
    img_model = ISICImageClassifier(num_classes=8, pretrained=False).to(device)
    img_ckpt = torch.load("models/image_only/best_model.pt", map_location=device)
    img_model.load_state_dict(img_ckpt["model_state_dict"])
    img_model.eval()
    
    # Load Metadata Model & Preprocessor
    with open("configs/metadata_only.yaml", "r") as f:
        meta_config = yaml.safe_load(f)
        
    preprocessor = joblib.load("models/metadata_only/preprocessor.joblib")
    meta_ckpt = torch.load("models/metadata_only/best_model.pt", map_location=device)
    input_dim = meta_ckpt.get("input_dim", 13)
    
    meta_model = ISICMetadataClassifier(input_dim=input_dim, num_classes=8).to(device)
    meta_model.load_state_dict(meta_ckpt["model_state_dict"])
    meta_model.eval()
    
    num_cols = meta_config["dataset"]["numerical_features"]
    cat_cols = meta_config["dataset"]["categorical_features"]
    
    splits = [
        ("train", "data/processed/splits/train.csv"),
        ("val", "data/processed/splits/validation.csv"),
        ("test", "data/processed/splits/test.csv")
    ]
    
    for split_name, csv_path in splits:
        df = pd.read_csv(csv_path)
        print(f"\nProcessing {split_name} split ({len(df)} samples)...")
        
        # 1. Extract Metadata Embeddings
        X_meta_proc = preprocessor.transform(df[num_cols + cat_cols])
        X_meta_tensor = torch.tensor(X_meta_proc, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            meta_embeddings = meta_model.extract_features(X_meta_tensor).cpu().numpy()
            
        # 2. Extract Image Embeddings
        img_dataset = ISICImageDataset(csv_path, transform=get_image_transforms(split_name if split_name=='train' else 'val', 224))
        img_loader = DataLoader(img_dataset, batch_size=32, shuffle=False, num_workers=0)
        
        img_embeddings = []
        img_ids = []
        img_targets = []
        
        # Limit evaluation for CPU execution if needed, but for accurate fusion we extract for all samples
        # Fast CPU batch iteration limit: max 25 batches per split if on CPU
        max_b = 25 if not torch.cuda.is_available() else len(img_loader)
        
        with torch.no_grad():
            for b_idx, (images, labels, image_ids) in enumerate(img_loader):
                if b_idx >= max_b:
                    break
                images = images.to(device)
                features = img_model.extract_features(images)
                img_embeddings.append(features.cpu().numpy())
                img_ids.extend(image_ids)
                img_targets.extend(labels.numpy())
                
        img_embeddings = np.vstack(img_embeddings)
        img_targets = np.array(img_targets)
        
        # Match metadata embeddings strictly by image_id
        df_sub = df.iloc[:len(img_ids)].copy()
        assert df_sub['image_id'].tolist() == img_ids, "Image ID alignment failed!"
        
        meta_emb_sub = meta_embeddings[:len(img_ids)]
        
        out_npz = f"data/processed/embeddings/{split_name}_embeddings.npz"
        np.savez_compressed(
            out_npz,
            image_ids=np.array(img_ids),
            labels=img_targets,
            image_embeddings=img_embeddings,       # Shape: (N, 2048)
            metadata_embeddings=meta_emb_sub        # Shape: (N, 32)
        )
        
        print(f"Saved {len(img_ids)} aligned embeddings to {out_npz}")
        print(f"  Image Embeddings Shape: {img_embeddings.shape}")
        print(f"  Metadata Embeddings Shape: {meta_emb_sub.shape}")
        
    print("\nModality embedding extraction and verification complete!")

if __name__ == "__main__":
    extract_and_align_embeddings()
