import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ISICImageClassifier(nn.Module):
    """
    ResNet50-based Image Classifier for ISIC 2019 Skin Lesion Classification.
    
    Exposes 2048-dimensional feature embeddings for multimodal fusion
    and maintains access to final conv layer (layer4) for future Grad-CAM.
    """
    def __init__(self, num_classes=8, dropout_rate=0.3, pretrained=True):
        super(ISICImageClassifier, self).__init__()
        
        # Load torchvision ResNet50
        if pretrained:
            weights = ResNet50_Weights.DEFAULT
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet50(weights=None)
            
        # Extract features up to adaptive avgpool
        self.feature_extractor = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4  # Accessible for Grad-CAM targeting
        )
        
        self.avgpool = self.backbone.avgpool
        self.embedding_dim = self.backbone.fc.in_features  # 2048
        
        # Custom classification head
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc = nn.Linear(self.embedding_dim, num_classes)
        
    def extract_features(self, x):
        """
        Extracts 2048-dimensional normalized image embedding vector.
        """
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x
        
    def forward(self, x, return_embedding=False):
        """
        Forward pass returning classification logits (and optionally embedding).
        """
        embedding = self.extract_features(x)
        dropped = self.dropout(embedding)
        logits = self.fc(dropped)
        
        if return_embedding:
            return logits, embedding
        return logits

    def freeze_backbone(self):
        """
        Freezes feature extractor parameters for initial classification head training.
        """
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self, unfreeze_layers=['layer3', 'layer4']):
        """
        Unfreezes specified backbone layers for fine-tuning.
        If unfreeze_layers is 'all', unfreezes the entire backbone.
        """
        if unfreeze_layers == 'all':
            for param in self.feature_extractor.parameters():
                param.requires_grad = True
        else:
            for name, child in self.feature_extractor.named_children():
                if name in unfreeze_layers:
                    for param in child.parameters():
                        param.requires_grad = True
                else:
                    for param in child.parameters():
                        param.requires_grad = False
