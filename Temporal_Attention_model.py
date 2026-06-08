import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class R2Plus1DTemporalAttention(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True, dropout_p=0.5):
        super().__init__()

        # Load the pre-trained backbone
        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)

        # 1. Extract the convolutional blocks (up to layer4)
        self.stem = backbone.stem
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        # 2. Spatial Average Pool: Crushes Height & Width, but PRESERVES Time
        self.spatial_pool = nn.AdaptiveAvgPool3d((None, 1, 1))

        # 3. Learned Temporal Attention Network
        # Input shape: (Batch, 512 channels, Time)
        self.attention_net = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(in_channels=128, out_channels=1, kernel_size=1)
        )

        # 4. Classification Head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(512, num_classes)
        )

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self):
        # Freeze all spatio-temporal convolutional layers
        for param in self.stem.parameters(): param.requires_grad = False
        for param in self.layer1.parameters(): param.requires_grad = False
        for param in self.layer2.parameters(): param.requires_grad = False
        for param in self.layer3.parameters(): param.requires_grad = False
        for param in self.layer4.parameters(): param.requires_grad = False

    def unfreeze_backbone(self):
        # Wake up the backbone for Stage 2
        for param in self.stem.parameters(): param.requires_grad = True
        for param in self.layer1.parameters(): param.requires_grad = True
        for param in self.layer2.parameters(): param.requires_grad = True
        for param in self.layer3.parameters(): param.requires_grad = True
        for param in self.layer4.parameters(): param.requires_grad = True

    def forward(self, x, return_attention=False):
        # Fix dimensions: (B, T, C, H, W) -> (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        # Forward pass through spatio-temporal convolutions
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # Shape out: (B, 512, T', H', W')

        # Collapse spatial dims: (B, 512, T', 1, 1) -> (B, 512, T')
        x = self.spatial_pool(x).squeeze(-1).squeeze(-1)

        # Calculate attention scores: (B, 1, T')
        attn_scores = self.attention_net(x)

        # Convert to probabilities (sum to 1 across the temporal dimension)
        attn_weights = F.softmax(attn_scores, dim=2)

        # Multiply features by their temporal attention weights
        weighted_x = x * attn_weights

        # Sum across the weighted temporal dimension to get a single global vector
        global_features = weighted_x.sum(dim=2)  # Shape: (B, 512)

        # Final classification
        logits = self.classifier(global_features)

        # Optional return for generating thesis visualizations later
        if return_attention:
            return logits, attn_weights

        return logits