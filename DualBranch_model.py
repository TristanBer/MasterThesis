import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


class FineGrainedDualStreamModel(nn.Module):
    def __init__(self, num_classes=5, freeze_backbone=True, dropout_p=0.5):
        super().__init__()

        # --- BRANCH 1: APPEARANCE (Raw Frames) ---
        self.appearance_branch = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.appearance_branch.fc = nn.Identity()  # Remove the original Kinetics head

        # --- BRANCH 2: KINEMATICS (Frame Differences) ---
        self.motion_branch = r2plus1d_18(weights=R2Plus1D_18_Weights.DEFAULT)
        self.motion_branch.fc = nn.Identity()

        if freeze_backbone:
            self.freeze_backbones()

        # --- FUSION HEAD ---
        # R(2+1)D outputs 512 features. Two branches = 1024 features.
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(1024),  # <--- NEW: Stabilizes the concatenated features
            nn.Dropout(p=dropout_p),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes)
        )

    def freeze_backbones(self):
        for param in self.appearance_branch.parameters():
            param.requires_grad = False
        for param in self.motion_branch.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.appearance_branch.parameters():
            param.requires_grad = True
        for param in self.motion_branch.parameters():
            param.requires_grad = True

    def forward(self, x):
        # 0. Dimensionen korrigieren! 
        # Tauscht (B, T, C, H, W) zu (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)
        
        # 1. Appearance Features
        feat_app = self.appearance_branch(x)
        
        # 2. Dynamic Frame Differencing (Calculated on GPU instantly)
        # We subtract the previous frame from the current frame. 
        # To keep the temporal dimension the same length, we pad with the first frame.
        first_frame = x[:, :, 0:1, :, :]
        x_shifted = torch.cat([first_frame, x[:, :, :-1, :, :]], dim=2)
        x_diff = x - x_shifted  # This isolates the moving pixels!
        
        # 3. Motion Features
        feat_motion = self.motion_branch(x_diff)
        
        # 4. Fusion and Classification
        combined = torch.cat([feat_app, feat_motion], dim=1)
        return self.classifier(combined)