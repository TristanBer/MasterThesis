import torch
import torch.nn as nn
from torchvision.models.video import x3d_m, X3D_M_Weights


class VolleyballX3DModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout_p=0.5):
        super(VolleyballX3DModel, self).__init__()

        # Load X3D-M pretrained on Kinetics-400
        backbone = x3d_m(weights=X3D_M_Weights.DEFAULT)

        # X3D-M's head is in blocks[-1]; update dropout and replace the
        # linear projection (in_features=2048) with a task-specific layer
        backbone.blocks[-1].dropout.p = dropout_p
        in_features = backbone.blocks[-1].proj.in_features  # 2048 for X3D-M
        backbone.blocks[-1].proj = nn.Linear(in_features, num_classes)

        self.model = backbone

        # Freeze everything except the last block (classification head)
        if freeze_backbone:
            last_block_key = f'blocks.{len(backbone.blocks) - 1}'
            for name, param in self.model.named_parameters():
                if last_block_key not in name:
                    param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # DataLoader delivers (Batch, Frames, C, H, W)
        # X3D expects       (Batch, C, Frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


if __name__ == "__main__":
    model = VolleyballX3DModel(num_classes=5)
    fake_video = torch.randn(2, 16, 3, 224, 224)
    output = model(fake_video)
    print(f"Output shape: {output.shape}")  # Expected: [2, 5]