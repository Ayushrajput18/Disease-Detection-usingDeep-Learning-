import os
import torch
import numpy as np
import pandas as pd

def extract_metadata_embeddings(model, preprocessor, df, device):
    """
    Extracts 32-dimensional learned metadata feature vectors for a given DataFrame.
    
    Args:
        model: Trained ISICMetadataClassifier PyTorch model
        preprocessor: Fitted scikit-learn ColumnTransformer pipeline
        df: DataFrame containing clinical features ('age_approx', 'sex', 'anatom_site_general', 'image_id', 'diagnosis_index')
        device: torch.device (CPU or CUDA)
        
    Returns:
        image_ids: List of image_id strings
        labels: NumPy array of class indices
        embeddings: NumPy array of shape (N, 32)
    """
    model.eval()
    model.to(device)
    
    # Preprocess features
    X_processed = preprocessor.transform(df)
    X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        embeddings = model.extract_features(X_tensor).cpu().numpy()
        
    image_ids = df['image_id'].tolist()
    labels = df['diagnosis_index'].values
    
    return image_ids, labels, embeddings


def save_metadata_embeddings_to_npz(image_ids, labels, embeddings, output_path):
    """
    Saves extracted 32-dim metadata embeddings to a compressed .npz archive.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=np.array(image_ids),
        labels=labels,
        embeddings=embeddings
    )
    print(f"Saved {len(image_ids)} metadata embeddings to {output_path}. Shape: {embeddings.shape}")
