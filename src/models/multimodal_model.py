import torch
import torch.nn as nn

class ISICMultimodalConcatClassifier(nn.Module):
    """
    Simple Concatenation Multimodal Fusion Model.
    Concatenates 2048-dim Image Embedding and 32-dim Metadata Embedding -> 2080-dim vector.
    """
    def __init__(self, image_dim=2048, metadata_dim=32, num_classes=8, dense_1=512, dense_2=128, dropout_1=0.3, dropout_2=0.2):
        super(ISICMultimodalConcatClassifier, self).__init__()
        
        fused_dim = image_dim + metadata_dim # 2080
        
        self.fusion_head = nn.Sequential(
            nn.Linear(fused_dim, dense_1),
            nn.ReLU(),
            nn.Dropout(p=dropout_1),
            nn.Linear(dense_1, dense_2),
            nn.ReLU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(dense_2, num_classes)
        )
        
    def forward(self, x_img, x_meta):
        # Concatenate along feature dimension
        x_fused = torch.cat([x_img, x_meta], dim=1)
        logits = self.fusion_head(x_fused)
        return logits


class ISICMultimodalAttentionClassifier(nn.Module):
    """
    Multi-Head Attention Multimodal Fusion Model.
    Projects Image (2048) and Metadata (32) embeddings into 256-dim token space,
    applies Multi-Head Self-Attention across modality tokens, and classifies into 8 diagnostic categories.
    """
    def __init__(self, image_dim=2048, metadata_dim=32, num_classes=8, token_dim=256, num_heads=4, dense_dim=128, dropout_1=0.3, dropout_2=0.2):
        super(ISICMultimodalAttentionClassifier, self).__init__()
        
        # Projection layers to common token space (256-d)
        self.img_projection = nn.Sequential(
            nn.Linear(image_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU()
        )
        
        self.meta_projection = nn.Sequential(
            nn.Linear(metadata_dim, token_dim),
            nn.LayerNorm(token_dim),
            nn.ReLU()
        )
        
        # Multi-Head Self-Attention over Modality Tokens [IMAGE_TOKEN, METADATA_TOKEN]
        self.multihead_attn = nn.MultiheadAttention(embed_dim=token_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(token_dim)
        self.dropout_attn = nn.Dropout(p=dropout_1)
        
        # Classification Head (2 tokens * 256 = 512 -> 128 -> 8)
        flatten_dim = 2 * token_dim
        self.classifier = nn.Sequential(
            nn.Linear(flatten_dim, dense_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_2),
            nn.Linear(dense_dim, num_classes)
        )
        
    def get_attention_weights(self, x_img, x_meta):
        """
        Extracts 2x2 attention matrix demonstrating inter-modality token interactions.
        """
        img_token = self.img_projection(x_img).unsqueeze(1)    # (B, 1, 256)
        meta_token = self.meta_projection(x_meta).unsqueeze(1)  # (B, 1, 256)
        tokens = torch.cat([img_token, meta_token], dim=1)      # (B, 2, 256)
        
        _, attn_weights = self.multihead_attn(tokens, tokens, tokens)
        return attn_weights # Shape: (B, 2, 2)

    def forward(self, x_img, x_meta):
        # 1. Project to common modality token space
        img_token = self.img_projection(x_img).unsqueeze(1)    # (B, 1, 256)
        meta_token = self.meta_projection(x_meta).unsqueeze(1)  # (B, 1, 256)
        tokens = torch.cat([img_token, meta_token], dim=1)      # (B, 2, 256)
        
        # 2. Multi-Head Self-Attention
        attn_out, _ = self.multihead_attn(tokens, tokens, tokens)
        tokens_attn = self.norm(tokens + self.dropout_attn(attn_out)) # Residual connection
        
        # 3. Flatten Modality Tokens & Classify
        flattened = torch.flatten(tokens_attn, start_dim=1)    # (B, 512)
        logits = self.classifier(flattened)
        return logits
