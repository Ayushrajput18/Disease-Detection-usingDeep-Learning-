import torch
import torch.nn as nn

class ISICMetadataClassifier(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier for ISIC 2019 Clinical Metadata.
    
    Exposes a 32-dimensional metadata feature embedding for multimodal fusion (Phase 5).
    """
    def __init__(self, input_dim, num_classes=8, hidden_dim_1=128, hidden_dim_2=64, embedding_dim=32, dropout_1=0.3, dropout_2=0.2):
        super(ISICMetadataClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        
        # Layer 1: Dense -> ReLU -> Dropout
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_1)
        )
        
        # Layer 2: Dense -> ReLU -> Dropout
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_2)
        )
        
        # Metadata Embedding Layer (32-dim vector for multimodal fusion)
        self.embedding_layer = nn.Sequential(
            nn.Linear(hidden_dim_2, embedding_dim),
            nn.ReLU()
        )
        
        # Final Classification Head
        self.classifier = nn.Linear(embedding_dim, num_classes)
        
    def extract_features(self, x):
        """
        Extracts 32-dimensional learned metadata feature vector.
        """
        x = self.layer1(x)
        x = self.layer2(x)
        features = self.embedding_layer(x)
        return features
        
    def forward(self, x, return_embedding=False):
        """
        Forward pass returning classification logits (and optionally 32-dim embedding).
        """
        embedding = self.extract_features(x)
        logits = self.classifier(embedding)
        
        if return_embedding:
            return logits, embedding
        return logits
