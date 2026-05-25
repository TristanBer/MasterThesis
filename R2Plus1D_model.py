import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class VolleyballR2Plus1DModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout_p=0.5):
        super(VolleyballR2Plus1DModel, self).__init__()

        # Load R3D-18 pretrained on Kinetics-400
        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)

        # Replace the final classification layer with our own
        in_features = backbone.fc.in_features  # 512 for R3D-18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes)
        )
        self.model = backbone

        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:
                    param.requires_grad = False

    def unfreeze_backbone(self):
        #switch from Stage 1 to Stage 2 training
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x arrives as (Batch, Frames, C, H, W) from the DataLoader
        # R3D expects  (Batch, C, Frames, H, W) — so permutation is needed
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


if __name__ == "__main__":
    model = VolleyballIR2Plus1DModel(num_classes=5)
    fake_video = torch.randn(2, 16, 3, 112, 112)  # 2 clips, 16 frames
    output = model(fake_video)
    print(f"Output shape: {output.shape}")  # Expected: [2, 5]