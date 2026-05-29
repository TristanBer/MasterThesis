"""
Dual-Head CNN + BiLSTM Architecture for Ablation Study A1
==========================================================

This module implements a multi-task variant of the baseline VolleyballBaselineModel,
in which the single 5-class classifier is decomposed into three task-specific heads:

    Head 1 (Technique) : {bump, overhead, N.A.}   -> 3 logits
    Head 2 (Direction) : {backward, forward, N.A.} -> 3 logits
    Head 3 (Others gate): {is_other, is_setting_action} -> 2 logits

Rationale:
    The baseline 5-class label space (Others, bump_set_backward, bump_set_forward,
    overhead_set_backward, overhead_set_forward) is inherently structured as the
    Cartesian product of two semantically orthogonal sub-tasks (technique and
    direction), plus an exclusive "Others" category. By decoupling the
    classification heads, the shared spatio-temporal feature extractor (ResNet-18
    + BiLSTM) can learn representations that are simultaneously discriminative
    for both sub-tasks, while each head specialises in task-relevant features
    (Ruder, 2017; Caruana, 1997).

Mapping convention (must match dataset.VolleyballDataset.class_names ordering):
    class_idx 0 : Others                  -> technique=2 (N.A.), direction=2 (N.A.), is_other=1
    class_idx 1 : bump_set_backward       -> technique=0,         direction=0,         is_other=0
    class_idx 2 : bump_set_forward        -> technique=0,         direction=1,         is_other=0
    class_idx 3 : overhead_set_backward   -> technique=1,         direction=0,         is_other=0
    class_idx 4 : overhead_set_forward    -> technique=1,         direction=1,         is_other=0
"""

import torch
import torch.nn as nn
import torchvision.models as models


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
      (2) Otherwise, the (technique, direction) pair determines the class index.

    Args:
        technique_logits (torch.Tensor): shape (B, 3)
        direction_logits (torch.Tensor): shape (B, 3)
        other_logits     (torch.Tensor): shape (B, 2)

    Returns:
        torch.Tensor of shape (B,) with predicted class indices in [0, 4].
    """
    batch_size = technique_logits.size(0)
    preds = torch.zeros(batch_size, dtype=torch.long, device=technique_logits.device)

    other_pred = torch.argmax(other_logits, dim=1)        # (B,)
    technique_pred = torch.argmax(technique_logits[:, :2], dim=1)  # restrict to bump/overhead
    direction_pred = torch.argmax(direction_logits[:, :2], dim=1)  # restrict to backward/forward

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
# Dual-Head model
# ---------------------------------------------------------------------------
class VolleyballDualHeadModel(nn.Module):
    """
    Multi-task variant of VolleyballBaselineModel.

    The shared feature extraction trunk (ResNet-18 + BiLSTM) is identical to the
    baseline to ensure that any observed performance differences can be attributed
    exclusively to the multi-head decomposition rather than to changes in the
    representational backbone (Ruder, 2017).

    Architecture:
        Input (B, T, C, H, W)
            -> ResNet-18 per-frame feature extraction (B*T, 512, 1, 1)
            -> Reshape (B, T, 512)
            -> Bi-LSTM with hidden_dim*2 output features
            -> Last time-step feature vector (B, hidden_dim*2)
            -> Three parallel classification heads
    """

    def __init__(self, hidden_dim=256, num_layers=2, dropout_p=0.5):
        super(VolleyballDualHeadModel, self).__init__()

        # --- Shared backbone (identical to baseline) ---
        resnet = models.resnet18(weights='IMAGENET1K_V1')
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])

        # Stage 1: backbone frozen
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        feature_dim = hidden_dim * 2  # bidirectional concatenation

        # --- Three task-specific heads ---
        # Each head is intentionally lightweight to keep the parameter budget
        # comparable to the single-head baseline; the bulk of the capacity
        # remains in the shared trunk.
        self.technique_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, 3),  # bump, overhead, N.A.
        )
        self.direction_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, 3),  # backward, forward, N.A.
        )
        self.other_head = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(feature_dim, 2),  # setting_action, other
        )

    def unfreeze_backbone(self):
        """Stage 2: unfreeze the ResNet backbone for full fine-tuning."""
        for param in self.feature_extractor.parameters():
            param.requires_grad = True

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): shape (B, T, C, H, W)

        Returns:
            dict with keys 'technique', 'direction', 'other' mapping to logits.
        """
        batch_size, seq_len, c, h, w = x.shape

        # Flatten temporal dimension into batch for CNN processing
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)               # (B*T, 512, 1, 1)
        features = features.view(batch_size, seq_len, -1)  # (B, T, 512)

        # Temporal aggregation
        lstm_out, _ = self.lstm(features)
        last_step = lstm_out[:, -1, :]                     # (B, hidden_dim*2)

        # Multi-head output
        return {
            'technique': self.technique_head(last_step),
            'direction': self.direction_head(last_step),
            'other': self.other_head(last_step),
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

    The default weights (lambda_t = lambda_d = lambda_o = 1/3) impose an
    egalitarian prior over the three sub-tasks. The weights can be tuned via
    grid search on the validation fold (Kendall et al., 2018, suggest learnable
    weighting via task uncertainty, which can be a follow-up experiment).
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
            outputs (dict): output of VolleyballDualHeadModel.forward()
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
    model = VolleyballDualHeadModel()
    fake_video = torch.randn(2, 60, 3, 224, 224)
    output = model(fake_video)
    print(f"Technique logits shape: {output['technique'].shape}")  # [2, 3]
    print(f"Direction logits shape: {output['direction'].shape}")  # [2, 3]
    print(f"Other logits shape:     {output['other'].shape}")      # [2, 2]

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