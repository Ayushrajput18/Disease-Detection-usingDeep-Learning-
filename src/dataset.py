import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ISICImageDataset(Dataset):
    """
    PyTorch Dataset for ISIC 2019 Skin Lesion Image Classification.
    Reads split CSVs (train.csv, validation.csv, test.csv) and loads RGB images.
    """
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        image_id = row['image_id']
        label = int(row['diagnosis_index'])
        
        # Load image safely
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            raise IOError(f"Failed to read image at path: {img_path}. Error: {str(e)}")
            
        if self.transform:
            image_tensor = self.transform(image)
        else:
            image_tensor = transforms.ToTensor()(image)
            
        return image_tensor, torch.tensor(label, dtype=torch.long), image_id


def get_image_transforms(split='train', image_size=224):
    """
    Returns image transformation pipelines for training, validation, and testing.
    ImageNet normalization is applied for ResNet50 compatibility.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
    else:
        # Validation and Test: Deterministic preprocessing
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
