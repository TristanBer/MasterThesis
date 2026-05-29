"""
Dual-Head R(2+1)D-18 Architecture for Ablation Study A1
========================================================

This module implements a multi-task variant of the baseline
VolleyballR2Plus1DModel, in which the single 5-class classifier is decomposed
into three task-specific heads:

    Head 1 (Technique)  : {bump, overhead, N.A.}      -> 3 logits
    Head 2 (Direction)  : {backward, forward, N.A.}   -> 3 logits
    Head 3 (Others gate): {setting_action, other}     -> 2 logits

Rationale:
    The baseline 5-class label space (Others, bump_set_backward, bump_set_forward,
    overhead_set_backward, overhead_set_forward) is inherently structured as the
    Cartesian product of two semantically orthogonal sub-tasks (technique and
    direction), plus an exclusive "Others" category. By decoupling the
    classification heads, the shared spatio-temporal backbone (R(2+1)D-18,
    pre-trained on Kinetics-400; Tran et al., 2018) can learn representations
    that are simultaneously discriminative for both sub-tasks, while each head
    specialises in task-relevant features (Caruana, 1997; Ruder, 2017).

Mapping convention (must match dataset.VolleyballDataset.class_names ordering):
    class_idx 0 : Others                  -> technique=2 (N.A.), direction=2 (N.A.), is_other=1
    class_idx 1 : bump_set_backward       -> technique=0,         direction=0,         is_other=0
    class_idx 2 : bump_set_forward        -> technique=0,         direction=1,         is_other=0
    class_idx 3 : overhead_set_backward   -> technique=1,         direction=0,         is_other=0
    class_idx 4 : overhead_set_forward    -> technique=1,         direction=1,         is_other=0
"""

import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights


# ---------------------------------------------------------------------------
# Label decomposition mapping
# ---------------------------------------------------------------------------
# (technique_idx, direction_idx, is_other_idx)
# technique : 0=bump, 1=overhead, 2=N.A.
# direction : 0=backward, 1=forward, 2=N.A.
# is_other  : 0=setting_action, 1=other
LABEL_DECOMPOSITION = {
    0: (2, 2, 1),   # Others
    1: (0, 0, 0),   # bump_set_backward
    2: (0, 1, 0),   # bump_set_forward
    3: (1, 0, 0),   # overhead_set_backward
    4: (1, 1, 0),   # overhead_set_forward
}

TECHNIQUE_NAMES = ['bump', 'overhead', 'N.A.']
DIRECTION_NAMES = ['backward', 'forward', 'N.A.']
OTHER_NAMES = ['setting_action', 'other']


def decompose_labels(labels):
    """
    Decompose a tensor of 5-class labels into three task-specific tensors.

    Args:
        labels (torch.Tensor): shape (B,) of integer class indices in [0, 4].

    Returns:
        Tuple of three tensors (technique, direction, is_other), each of shape (B,).
    """
    technique = torch.empty_like(labels)
    direction = torch.empty_like(labels)
    is_other = torch.empty_like(labels)
    for i in range(labels.size(0)):
        t, d, o = LABEL_DECOMPOSITION[labels[i].item()]
        technique[i], direction[i], is_other[i] = t, d, o
    return technique, direction, is_other


def reconstruct_5class_prediction(technique_logits, direction_logits, other_logits):
    """
    Reconstruct the original 5-class prediction from the three head outputs.

    The reconstruction proceeds in two stages:
      (1) The "Others" gate is evaluated. If is_other is predicted, class 0 is returned.
      (2) Otherwise, the (technique, direction) pair determines the class index,
          restricted to the meaningful classes (bump/overhead, backward/forward).

    Args:
        technique_logits (torch.Tensor): shape (B, 3)
        direction_logits (torch.Tensor): shape (B, 3)
        other_logits     (torch.Tensor): shape (B, 2)

    Returns:
        torch.Tensor of shape (B,) with predicted class indices in [0, 4].
    """
    batch_size = technique_logits.size(0)
    preds = torch.zeros(batch_size, dtype=torch.long, device=technique_logits.device)

    other_pred = torch.argmax(other_logits, dim=1)                  # (B,)
    technique_pred = torch.argmax(technique_logits[:, :2], dim=1)   # restrict to bump/overhead
    direction_pred = torch.argmax(direction_logits[:, :2], dim=1)   # restrict to backward/forward

    for i in range(batch_size):
        if other_pred[i].item() == 1:
            preds[i] = 0  # Others
        else:
            t, d = technique_pred[i].item(), direction_pred[i].item()
            # Mapping: (bump,backward)=1, (bump,forward)=2,
            #          (overhead,backward)=3, (overhead,forward)=4
            preds[i] = 1 + 2 * t + d
    return preds


# ---------------------------------------------------------------------------
# Dual-Head R(2+1)D-18 model
# ---------------------------------------------------------------------------
class VolleyballR2Plus1DDualHeadModel(nn.Module):
    """
    Multi-task variant of VolleyballR2Plus1DModel.

    The shared R(2+1)D-18 backbone (pre-trained on Kinetics-400; Tran et al.,
    2018) is identical to the baseline to ensure that any observed performance
    differences can be attributed exclusively to the multi-head decomposition
    rather than to changes in the representational backbone.

    Architecture:
        Input (B, T, C, H, W)
            -> Permute to (B, C, T, H, W)
            -> R(2+1)D-18 spatio-temporal feature extraction
            -> Bottleneck feature vector (B, 512)
            -> Three parallel classification heads
    """

    def __init__(self, freeze_backbone=True, dropout_p=0.5):
        super(VolleyballR2Plus1DDualHeadModel, self).__init__()

        # --- Shared backbone (identical to baseline) ---
        backbone = r2plus1d_18(weights=R2Plus1D_18_Weights.KINETICS400_V1)
        in_features = backbone.fc.in_features  # 512 for R(2+1)D-18

        # Replace the final classification layer with an identity to expose the
        # 512-dim bottleneck features to the downstream multi-head module
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Stage 1: backbone frozen
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        # --- Three task-specific heads ---
        # Each head is intentionally lightweight (single linear layer with dropout)
        # to keep the parameter budget comparable to the single-head baseline;
        # the bulk of the model capacity remains in the shared backbone.
        self.technique_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 3),  # bump, overhead, N.A.
        )
        self.direction_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 3),  # backward, forward, N.A.
        )
        self.other_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, 2),  # setting_action, other
        )

    def unfreeze_backbone(self):
        """Stage 2: unfreeze the R(2+1)D backbone for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, C, H, W) as produced by the DataLoader.

        Returns:
            dict with keys 'technique', 'direction', 'other' mapping to logits.
        """
        # R(2+1)D expects (B, C, T, H, W) — permutation matches baseline
        x = x.permute(0, 2, 1, 3, 4)

        # Shared backbone produces a (B, 512) feature vector via the identity-replaced fc layer
        features = self.backbone(x)

        return {
            'technique': self.technique_head(features),
            'direction': self.direction_head(features),
            'other': self.other_head(features),
        }


# ---------------------------------------------------------------------------
# Multi-task loss
# ---------------------------------------------------------------------------
class MultiTaskLoss(nn.Module):
    """
    Weighted sum of three cross-entropy losses corresponding to the three
    classification heads.

    Mathematically:
        L_total = lambda_t * L_technique + lambda_d * L_direction + lambda_o * L_other

    The default weights (lambda_t = lambda_d = lambda_o = 1.0) impose an
    egalitarian prior over the three sub-tasks. The weights can be tuned via
    grid search on the validation fold. Kendall et al. (2018) propose learnable
    weighting via task-dependent uncertainty, which could be explored as a
    follow-up experiment.
    """

    def __init__(self,
                 weight_technique=1.0,
                 weight_direction=1.0,
                 weight_other=1.0,
                 technique_class_weights=None,
                 direction_class_weights=None,
                 other_class_weights=None,
                 label_smoothing=0.1):
        super().__init__()
        self.weight_technique = weight_technique
        self.weight_direction = weight_direction
        self.weight_other = weight_other

        self.loss_technique = nn.CrossEntropyLoss(
            weight=technique_class_weights, label_smoothing=label_smoothing)
        self.loss_direction = nn.CrossEntropyLoss(
            weight=direction_class_weights, label_smoothing=label_smoothing)
        self.loss_other = nn.CrossEntropyLoss(
            weight=other_class_weights, label_smoothing=label_smoothing)

    def forward(self, outputs, target_technique, target_direction, target_other):
        """
        Args:
            outputs (dict): output of VolleyballR2Plus1DDualHeadModel.forward()
            target_technique (torch.Tensor): shape (B,) in [0, 2]
            target_direction (torch.Tensor): shape (B,) in [0, 2]
            target_other     (torch.Tensor): shape (B,) in [0, 1]

        Returns:
            total_loss (torch.Tensor): scalar weighted sum
            components (dict): individual loss values for logging
        """
        l_t = self.loss_technique(outputs['technique'], target_technique)
        l_d = self.loss_direction(outputs['direction'], target_direction)
        l_o = self.loss_other(outputs['other'], target_other)

        total = (self.weight_technique * l_t
                 + self.weight_direction * l_d
                 + self.weight_other * l_o)

        components = {
            'technique': l_t.item(),
            'direction': l_d.item(),
            'other': l_o.item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = VolleyballR2Plus1DDualHeadModel(freeze_backbone=True)
    fake_video = torch.randn(2, 32, 3, 224, 224)  # 2 clips, 32 frames
    output = model(fake_video)
    print(f"Technique logits shape: {output['technique'].shape}")  # [2, 3]
    print(f"Direction logits shape: {output['direction'].shape}")  # [2, 3]
    print(f"Other logits shape:     {output['other'].shape}")      # [2, 2]

    # Verify which parameters are trainable in Stage 1
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\nStage 1 trainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # Test label decomposition
    fake_labels = torch.tensor([0, 1, 2, 3, 4])
    t, d, o = decompose_labels(fake_labels)
    print(f"\nLabel decomposition test:")
    print(f"  Original:  {fake_labels.tolist()}")
    print(f"  Technique: {t.tolist()}  (expected: [2, 0, 0, 1, 1])")
    print(f"  Direction: {d.tolist()}  (expected: [2, 0, 1, 0, 1])")
    print(f"  Other:     {o.tolist()}  (expected: [1, 0, 0, 0, 0])")

    # Test reconstruction
    fake_t_logits = torch.tensor([[0, 0, 5], [5, 0, 0], [5, 0, 0], [0, 5, 0], [0, 5, 0]], dtype=torch.float)
    fake_d_logits = torch.tensor([[0, 0, 5], [5, 0, 0], [0, 5, 0], [5, 0, 0], [0, 5, 0]], dtype=torch.float)
    fake_o_logits = torch.tensor([[0, 5], [5, 0], [5, 0], [5, 0], [5, 0]], dtype=torch.float)
    recon = reconstruct_5class_prediction(fake_t_logits, fake_d_logits, fake_o_logits)
    print(f"\nReconstruction test: {recon.tolist()}  (expected: [0, 1, 2, 3, 4])")

    # Test loss
    loss_fn = MultiTaskLoss()
    total, components = loss_fn(output, t[:2], d[:2], o[:2])
    print(f"\nLoss test: total={total.item():.4f}, components={components}")

    # Verify Stage 2 unfreezing
    model.unfreeze_backbone()
    trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nStage 2 trainable params: {trainable_s2:,} / {total:,} "
          f"({100 * trainable_s2 / total:.2f}%)")