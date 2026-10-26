import os
import json
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.models.image_model import ISICImageClassifier
from src.models.clinical_model import ISICMetadataClassifier
from src.models.multimodal_model import ISICMultimodalConcatClassifier, ISICMultimodalAttentionClassifier

class MultimodalPredictor:
    """
    Reusable Multimodal Inference Engine for ISIC 2019 Skin Lesion Classification.
    Fuses medical image and clinical metadata for diagnostic prediction.
    """
    def __init__(self, concat_ckpt="models/multimodal/concat_model.pt", attention_ckpt="models/multimodal/attention_model.pt", device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Class Mapping
        with open("reports/dataset/class_mapping.json", "r") as f:
            self.class_mapping = json.load(f)
        self.index_to_class = {v: k for k, v in self.class_mapping.items()}
        self.class_names = [self.index_to_class[i] for i in range(len(self.index_to_class))]
        
        # Load Preprocessor
        self.preprocessor = joblib.load("models/metadata_only/preprocessor.joblib")
        
        # Image Transform
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std)
        ])
        
        # Load Single-Modality Backbone Models
        self.img_model = ISICImageClassifier(num_classes=8, pretrained=False).to(self.device)
        img_ckpt = torch.load("models/image_only/best_model.pt", map_location=self.device)
        self.img_model.load_state_dict(img_ckpt["model_state_dict"])
        self.img_model.eval()
        
        meta_ckpt = torch.load("models/metadata_only/best_model.pt", map_location=self.device)
        input_dim = meta_ckpt.get("input_dim", 13)
        self.meta_model = ISICMetadataClassifier(input_dim=input_dim, num_classes=8).to(self.device)
        self.meta_model.load_state_dict(meta_ckpt["model_state_dict"])
        self.meta_model.eval()
        
        # Load Multimodal Fusion Models if available
        self.concat_model = None
        if os.path.exists(concat_ckpt):
            self.concat_model = ISICMultimodalConcatClassifier().to(self.device)
            ckpt = torch.load(concat_ckpt, map_location=self.device)
            self.concat_model.load_state_dict(ckpt["model_state_dict"])
            self.concat_model.eval()
            
        self.attention_model = None
        if os.path.exists(attention_ckpt):
            self.attention_model = ISICMultimodalAttentionClassifier().to(self.device)
            ckpt = torch.load(attention_ckpt, map_location=self.device)
            self.attention_model.load_state_dict(ckpt["model_state_dict"])
            self.attention_model.eval()
            
    def preprocess_inputs(self, image_input, age, sex, anatom_site):
        # Image processing
        if isinstance(image_input, str):
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            img = image_input.convert('RGB')
        else:
            img = Image.fromarray(image_input).convert('RGB')
            
        img_tensor = self.img_transform(img).unsqueeze(0).to(self.device)
        
        # Metadata processing
        df_meta = pd.DataFrame([{
            'age_approx': float(age) if age is not None else np.nan,
            'sex': str(sex) if sex is not None else 'Unknown',
            'anatom_site_general': str(anatom_site) if anatom_site is not None else 'Unknown'
        }])
        meta_proc = self.preprocessor.transform(df_meta)
        meta_tensor = torch.tensor(meta_proc, dtype=torch.float32).to(self.device)
        
        return img_tensor, meta_tensor

    def predict_multimodal(self, image_input, age, sex, anatom_site, model_type="attention"):
        """
        Runs multimodal inference given medical image and clinical metadata inputs.
        
        Returns:
            predicted_class: Diagnostic category name string (e.g. 'MEL')
            class_probabilities: Dict mapping class names to probability floats
            confidence: Highest predicted class probability float
        """
        img_tensor, meta_tensor = self.preprocess_inputs(image_input, age, sex, anatom_site)
        
        with torch.no_grad():
            img_emb = self.img_model.extract_features(img_tensor)
            meta_emb = self.meta_model.extract_features(meta_tensor)
            
            if model_type == "attention" and self.attention_model is not None:
                logits = self.attention_model(img_emb, meta_emb)
            elif model_type == "concat" and self.concat_model is not None:
                logits = self.concat_model(img_emb, meta_emb)
            else:
                # Fallback to image model if multimodal weights unavailable
                logits = self.img_model(img_tensor)
                
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
        pred_idx = int(np.argmax(probs))
        predicted_class = self.index_to_class[pred_idx]
        confidence = float(probs[pred_idx])
        class_probabilities = {name: round(float(p), 4) for name, p in zip(self.class_names, probs)}
        
        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "class_probabilities": class_probabilities
        }
