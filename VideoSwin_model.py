import torch
import torch.nn as nn
from torchvision.models.video import swin3d_t, Swin3D_T_Weights


class VolleyballVideoSwinModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout_p=0.5):
        super(VolleyballVideoSwinModel, self).__init__()

        # Load Video Swin Transformer Tiny pretrained on Kinetics-400
        # Stochastic depth (drop_path_rate=0.1) is active by default
        backbone = swin3d_t(weights=Swin3D_T_Weights.KINETICS400_V1)

        # Replace the linear head (in_features=768 for Swin3D-T)
        in_features = backbone.head.in_features
        backbone.head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

        # Stage 1: freeze everything except the classification head
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'head' not in name:
                    param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # DataLoader delivers (Batch, Frames, C, H, W)
        # swin3d_t expects    (Batch, C, Frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


if __name__ == "__main__":
    model = VolleyballVideoSwinModel(num_classes=5)
    fake_video = torch.randn(2, 16, 3, 224, 224)
    output = model(fake_video)
    print(f"Output shape: {output.shape}")  # Expected: [2, 5]