import torch
import torch.nn as nn


class VolleyballX3DModel(nn.Module):
    def __init__(self, num_classes, freeze_backbone=True, dropout_p=0.5):
        super(VolleyballX3DModel, self).__init__()

        # Load X3D-M pretrained on Kinetics-400 via pytorchvideo
        self.model = torch.hub.load(
            'facebookresearch/pytorchvideo',
            'x3d_m',
            pretrained=True
        )

        # The classification head lives in blocks[-1]
        # Replace dropout probability, linear projection, and remove the
        # built-in Softmax — CrossEntropyLoss requires raw logits
        in_features = self.model.blocks[-1].proj.in_features  # 2048 for X3D-M
        self.model.blocks[-1].dropout  = nn.Dropout(p=dropout_p)
        self.model.blocks[-1].proj     = nn.Linear(in_features, num_classes)
        self.model.blocks[-1].activation = None  # Remove Softmax

        # Stage 1: freeze everything except the classification head
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'blocks.5' not in name:
                    param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, x):
        # DataLoader delivers (Batch, Frames, C, H, W)
        # X3D expects         (Batch, C, Frames, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        return self.model(x)


if __name__ == "__main__":
    model = VolleyballX3DModel(num_classes=5)
    fake_video = torch.randn(2, 16, 3, 224, 224)
    output = model(fake_video)
    print(f"Output shape: {output.shape}")  # Expected: [2, 5]