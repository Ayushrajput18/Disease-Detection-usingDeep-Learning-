import os
import torch
import pandas as pd
import numpy as np

def extract_embeddings_from_dataloader(model, dataloader, device):
    """
    Extracts 2048-dimensional feature vectors for all samples in a DataLoader.
    
    Returns:
        image_ids: List of image_id strings
        labels: List of integer class labels
        embeddings: NumPy array of shape (N, 2048)
    """
    model.eval()
    model.to(device)
    
    all_image_ids = []
    all_labels = []
    all_embeddings = []
    
    with torch.no_grad():
        for images, labels, image_ids in dataloader:
            images = images.to(device)
            features = model.extract_features(images)
            
            all_image_ids.extend(image_ids)
            all_labels.extend(labels.cpu().numpy())
            all_embeddings.append(features.cpu().numpy())
            
    all_embeddings = np.vstack(all_embeddings)
    return all_image_ids, np.array(all_labels), all_embeddings


def save_embeddings_to_npz(image_ids, labels, embeddings, output_path):
    """
    Saves extracted image embeddings and corresponding metadata to a compressed .npz archive.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(
        output_path,
        image_ids=np.array(image_ids),
        labels=labels,
        embeddings=embeddings
    )
    print(f"Saved {len(image_ids)} embeddings to {output_path}. Shape: {embeddings.shape}")
