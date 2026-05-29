"""
Training Script for R(2+1)D-18 Dual-Head Ablation Study (A1)
=============================================================

This script trains the multi-task variant of VolleyballR2Plus1DModel defined
in R2Plus1DDualHead_model.py. It mirrors the structure of R2Plus1D_train.py
(two-stage transfer learning, StratifiedGroupKFold partitioning, weighted
cross-entropy loss) to ensure that any observed performance differences are
attributable solely to the multi-task decomposition rather than to changes
in the training protocol.

Reporting strategy:
    The script produces both per-head metrics (technique, direction, other)
    AND the reconstructed 5-class metrics, enabling direct, fair comparison
    with the single-head R(2+1)D baseline.

References:
    Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41-75.
    Ruder, S. (2017). An Overview of Multi-Task Learning in Deep Neural
        Networks. arXiv:1706.05098.
    Tran, D., Wang, H., Torresani, L., Ray, J., LeCun, Y., & Paluri, M. (2018).
        A Closer Look at Spatiotemporal Convolutions for Action Recognition. CVPR.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
import os

from dataset import VolleyballDataset
from DualHead_R2plus1D_model import (VolleyballR2Plus1DDualHeadModel, MultiTaskLoss, decompose_labels, reconstruct_5class_prediction, TECHNIQUE_NAMES, DIRECTION_NAMES, OTHER_NAMES)

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = "/workspace/Master_Dataset_Extracted"

NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 8

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4

# Multi-task loss weights (lambda_t, lambda_d, lambda_o)
# Default to egalitarian weighting; can be tuned via grid search if necessary
LAMBDA_TECHNIQUE = 1.0
LAMBDA_DIRECTION = 1.0
LAMBDA_OTHER = 1.0

CHECKPOINT_PATH = "/workspace/R2Plus1DDualHead_best.pth"
LEARNING_CURVE_PATH = "/workspace/R2Plus1DDualHead_learning_curve.png"
CONFUSION_MATRIX_PATH_5CLASS = "/workspace/confusion_matrix_R2Plus1DDualHead_5class.png"
CONFUSION_MATRIX_PATH_TECHNIQUE = "/workspace/confusion_matrix_R2Plus1DDualHead_technique.png"
CONFUSION_MATRIX_PATH_DIRECTION = "/workspace/confusion_matrix_R2Plus1DDualHead_direction.png"
CONFUSION_MATRIX_PATH_OTHER = "/workspace/confusion_matrix_R2Plus1DDualHead_other.png"


# ---------------------------------------------------------------------------
# Class-weight computation per head
# ---------------------------------------------------------------------------
def compute_per_head_class_weights(dataset, train_indices, device):
    """
    Computes inverse-frequency class weights for each of the three classification
    heads, derived from the training subset only.

    For each 5-class training label, the corresponding (technique, direction,
    is_other) decomposition is computed, and inverse-frequency weights are
    derived per task independently. This mirrors the rationale of the baseline
    compute_class_weights() function (Pedregosa et al., 2011) but applied to
    the decomposed label spaces.

    Returns:
        Tuple of three tensors:
            (technique_weights[3], direction_weights[3], other_weights[2])
    """
    from DualHead_R2plus1D_model import LABEL_DECOMPOSITION

    train_labels = [dataset.labels[i] for i in train_indices]
    total = len(train_labels)

    # Decompose every training label
    technique_labels, direction_labels, other_labels = [], [], []
    for label in train_labels:
        t, d, o = LABEL_DECOMPOSITION[label]
        technique_labels.append(t)
        direction_labels.append(d)
        other_labels.append(o)

    def _inverse_freq(labels, num_classes):
        counts = Counter(labels)
        return [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]

    technique_weights = _inverse_freq(technique_labels, 3)
    direction_weights = _inverse_freq(direction_labels, 3)
    other_weights = _inverse_freq(other_labels, 2)

    print("\nPer-head class weights (inverse frequency):")
    print(f"  Technique  {TECHNIQUE_NAMES}: counts={dict(Counter(technique_labels))}, weights={[f'{w:.3f}' for w in technique_weights]}")
    print(f"  Direction  {DIRECTION_NAMES}: counts={dict(Counter(direction_labels))}, weights={[f'{w:.3f}' for w in direction_weights]}")
    print(f"  Other      {OTHER_NAMES}: counts={dict(Counter(other_labels))}, weights={[f'{w:.3f}' for w in other_weights]}")

    return (
        torch.tensor(technique_weights, dtype=torch.float32).to(device),
        torch.tensor(direction_weights, dtype=torch.float32).to(device),
        torch.tensor(other_weights, dtype=torch.float32).to(device),
    )


# ---------------------------------------------------------------------------
# Multi-task epoch runner
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, is_train, device, optimizer=None):
    """
    Runs a single epoch over the data loader.

    Returns:
        avg_loss            : float, total weighted loss averaged over batches
        component_losses    : dict with keys 'technique', 'direction', 'other'
        accuracies          : dict with keys 'technique', 'direction', 'other',
                              'reconstructed_5class'
        prediction_records  : dict with the same keys, each storing a list of
                              (all_preds, all_labels) for downstream reporting
    """
    model.train() if is_train else model.eval()

    total_loss = 0.0
    component_loss_sums = {'technique': 0.0, 'direction': 0.0, 'other': 0.0}
    correct = {'technique': 0, 'direction': 0, 'other': 0, 'reconstructed_5class': 0}
    total = 0

    predictions = {'technique': [], 'direction': [], 'other': [], 'reconstructed_5class': []}
    targets = {'technique': [], 'direction': [], 'other': [], 'reconstructed_5class': []}

    with torch.set_grad_enabled(is_train):
        for videos, labels_5class in loader:
            videos = videos.to(device)
            labels_5class = labels_5class.to(device)

            # Horizontal flip augmentation (training only) — identical to baseline
            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:
                        videos[i] = torch.flip(videos[i], dims=[3])  # Width dim

            # Decompose labels into three sub-task targets
            target_t, target_d, target_o = decompose_labels(labels_5class)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(videos)
                loss, components = criterion(outputs, target_t, target_d, target_o)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            for k in component_loss_sums:
                component_loss_sums[k] += components[k]

            # Per-head predictions
            pred_t = torch.argmax(outputs['technique'], dim=1)
            pred_d = torch.argmax(outputs['direction'], dim=1)
            pred_o = torch.argmax(outputs['other'], dim=1)

            # Reconstructed 5-class prediction (for fair comparison with baseline)
            pred_5class = reconstruct_5class_prediction(
                outputs['technique'], outputs['direction'], outputs['other']
            )

            batch_size = labels_5class.size(0)
            total += batch_size
            correct['technique'] += (pred_t == target_t).sum().item()
            correct['direction'] += (pred_d == target_d).sum().item()
            correct['other'] += (pred_o == target_o).sum().item()
            correct['reconstructed_5class'] += (pred_5class == labels_5class).sum().item()

            predictions['technique'].extend(pred_t.cpu().tolist())
            predictions['direction'].extend(pred_d.cpu().tolist())
            predictions['other'].extend(pred_o.cpu().tolist())
            predictions['reconstructed_5class'].extend(pred_5class.cpu().tolist())

            targets['technique'].extend(target_t.cpu().tolist())
            targets['direction'].extend(target_d.cpu().tolist())
            targets['other'].extend(target_o.cpu().tolist())
            targets['reconstructed_5class'].extend(labels_5class.cpu().tolist())

    n_batches = len(loader)
    avg_loss = total_loss / n_batches
    component_losses = {k: v / n_batches for k, v in component_loss_sums.items()}
    accuracies = {k: 100 * correct[k] / total for k in correct}

    return avg_loss, component_losses, accuracies, predictions, targets


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. DATA PREPARATION ---
    # Identical transforms to baseline R(2+1)D training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    full_dataset_train = VolleyballDataset(root_dir=ROOT_DIR, transform=train_transform, num_frames=NUM_FRAMES)
    full_dataset_val = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)

    # Identical match-grouping logic to baseline to ensure split-level reproducibility
    groups = []
    for path in full_dataset_train.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    sgkf = StratifiedGroupKFold(n_splits=5)
    train_indices, val_indices = next(sgkf.split(
        X=np.zeros(len(full_dataset_train)),
        y=full_dataset_train.labels,
        groups=groups
    ))

    train_matches = sorted(list(set([groups[i] for i in train_indices])))
    val_matches = sorted(list(set([groups[i] for i in val_indices])))
    print("\n================ DATA LEAKAGE VERIFICATION ================")
    print(f"Training Match Splits ({len(train_matches)} games): {train_matches}")
    print(f"Validation Match Splits ({len(val_matches)} games): {val_matches}")
    overlap = set(train_matches).intersection(set(val_matches))
    print(f"Intersection Overlap Check (Must be empty set): {overlap}")
    print("===========================================================\n")

    train_db = Subset(full_dataset_train, train_indices)
    val_db = Subset(full_dataset_val, val_indices)

    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    # --- 3. PER-HEAD CLASS WEIGHTS ---
    technique_weights, direction_weights, other_weights = compute_per_head_class_weights(
        full_dataset_train, train_indices, device
    )

    # --- 4. MODEL + LOSS ---
    model = VolleyballR2Plus1DDualHeadModel(freeze_backbone=True, dropout_p=0.5).to(device)

    criterion = MultiTaskLoss(
        weight_technique=LAMBDA_TECHNIQUE,
        weight_direction=LAMBDA_DIRECTION,
        weight_other=LAMBDA_OTHER,
        technique_class_weights=technique_weights,
        direction_class_weights=direction_weights,
        other_class_weights=other_weights,
        label_smoothing=0.1,
    )

    # Extended history for multi-task logging
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc_5class": [], "val_acc_5class": [],
        "train_acc_technique": [], "val_acc_technique": [],
        "train_acc_direction": [], "val_acc_direction": [],
        "train_acc_other": [], "val_acc_other": [],
    }
    best_val_acc = 0.0

    # --- 5. STAGE 1: TRAIN HEADS ONLY ---
    print(f"\n=== STAGE 1: Training heads only for {STAGE1_EPOCHS} epochs ===")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_S1)

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_comp, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, is_train=True, device=device, optimizer=optimizer
        )
        val_loss, val_comp, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, is_train=False, device=device
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc_5class"].append(train_acc['reconstructed_5class'])
        history["val_acc_5class"].append(val_acc['reconstructed_5class'])
        history["train_acc_technique"].append(train_acc['technique'])
        history["val_acc_technique"].append(val_acc['technique'])
        history["train_acc_direction"].append(train_acc['direction'])
        history["val_acc_direction"].append(val_acc['direction'])
        history["train_acc_other"].append(train_acc['other'])
        history["val_acc_other"].append(val_acc['other'])

        print(f"[S1 Epoch {epoch + 1}/{STAGE1_EPOCHS}] "
              f"Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"5-class Acc(T/V): {train_acc['reconstructed_5class']:.2f}%/{val_acc['reconstructed_5class']:.2f}% | "
              f"Tech(V): {val_acc['technique']:.2f}% | Dir(V): {val_acc['direction']:.2f}% | Other(V): {val_acc['other']:.2f}%")
        print(f"      Val component losses: technique={val_comp['technique']:.3f}, direction={val_comp['direction']:.3f}, other={val_comp['other']:.3f}")

        if val_acc['reconstructed_5class'] > best_val_acc:
            best_val_acc = val_acc['reconstructed_5class']
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (best 5-class val acc: {best_val_acc:.2f}%)")

    # --- 6. STAGE 2: FULL FINE-TUNING ---
    print(f"\n=== STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    model.unfreeze_backbone()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S2, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_comp, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, is_train=True, device=device, optimizer=optimizer
        )
        val_loss, val_comp, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, is_train=False, device=device
        )

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc_5class"].append(train_acc['reconstructed_5class'])
        history["val_acc_5class"].append(val_acc['reconstructed_5class'])
        history["train_acc_technique"].append(train_acc['technique'])
        history["val_acc_technique"].append(val_acc['technique'])
        history["train_acc_direction"].append(train_acc['direction'])
        history["val_acc_direction"].append(val_acc['direction'])
        history["train_acc_other"].append(train_acc['other'])
        history["val_acc_other"].append(val_acc['other'])

        print(f"[S2 Epoch {epoch + 1}/{STAGE2_EPOCHS}] "
              f"Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"5-class Acc(T/V): {train_acc['reconstructed_5class']:.2f}%/{val_acc['reconstructed_5class']:.2f}% | "
              f"Tech(V): {val_acc['technique']:.2f}% | Dir(V): {val_acc['direction']:.2f}% | Other(V): {val_acc['other']:.2f}%")
        print(f"      Val component losses: technique={val_comp['technique']:.3f}, direction={val_comp['direction']:.3f}, other={val_comp['other']:.3f}")

        if val_acc['reconstructed_5class'] > best_val_acc:
            best_val_acc = val_acc['reconstructed_5class']
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Checkpoint saved (best 5-class val acc: {best_val_acc:.2f}%)")

    # --- 7. FINAL EVALUATION ---
    print("\n=== FINAL EVALUATION (best checkpoint) ===")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True))
    _, _, final_acc, final_preds, final_targets = run_epoch(
        model, val_loader, criterion, is_train=False, device=device
    )

    print(f"\nFinal Val Accuracy:")
    print(f"  Reconstructed 5-class : {final_acc['reconstructed_5class']:.2f}%")
    print(f"  Technique head        : {final_acc['technique']:.2f}%")
    print(f"  Direction head        : {final_acc['direction']:.2f}%")
    print(f"  Other (gate) head     : {final_acc['other']:.2f}%")

    # 5-class report (primary comparison with baseline R(2+1)D)
    print("\nPer-class report (reconstructed 5-class):")
    present_classes = np.unique(final_targets['reconstructed_5class']).astype(int)
    present_names = [full_dataset_train.class_names[i] for i in present_classes]
    print(classification_report(
        final_targets['reconstructed_5class'],
        final_preds['reconstructed_5class'],
        target_names=present_names,
        labels=present_classes,
        zero_division=0,
    ))

    # Per-head reports
    print("\nPer-class report (Technique head):")
    print(classification_report(
        final_targets['technique'], final_preds['technique'],
        target_names=TECHNIQUE_NAMES, labels=[0, 1, 2], zero_division=0,
    ))
    print("\nPer-class report (Direction head):")
    print(classification_report(
        final_targets['direction'], final_preds['direction'],
        target_names=DIRECTION_NAMES, labels=[0, 1, 2], zero_division=0,
    ))
    print("\nPer-class report (Other gate):")
    print(classification_report(
        final_targets['other'], final_preds['other'],
        target_names=OTHER_NAMES, labels=[0, 1], zero_division=0,
    ))

    # --- 8. LEARNING CURVES ---
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss curve
    ax = axes[0, 0]
    ax.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    ax.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title("Total Multi-Task Loss")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Loss")
    ax.legend(); ax.grid(True)

    # 5-class reconstructed accuracy (primary metric)
    ax = axes[0, 1]
    ax.plot(epochs_range, history["train_acc_5class"], label="Train 5-class Acc", marker='o')
    ax.plot(epochs_range, history["val_acc_5class"], label="Val 5-class Acc", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title("Reconstructed 5-class Accuracy (primary metric)")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(); ax.grid(True)

    # Per-head accuracy (technique & direction)
    ax = axes[1, 0]
    ax.plot(epochs_range, history["val_acc_technique"], label="Technique (Val)", marker='o')
    ax.plot(epochs_range, history["val_acc_direction"], label="Direction (Val)", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title("Per-Head Validation Accuracy")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(); ax.grid(True)

    # Other-gate head accuracy
    ax = axes[1, 1]
    ax.plot(epochs_range, history["val_acc_other"], label="Other Gate (Val)", marker='o', color='purple')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title("Other-Gate Head Validation Accuracy")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
    ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(LEARNING_CURVE_PATH, dpi=300)
    print(f"\nLearning curves saved as '{LEARNING_CURVE_PATH}'")

    # --- 9. CONFUSION MATRICES ---
    # Reconstructed 5-class (primary, for direct comparison with baseline)
    print("\nGenerating confusion matrices...")
    cm_5class = confusion_matrix(
        final_targets['reconstructed_5class'],
        final_preds['reconstructed_5class'],
        labels=list(range(len(full_dataset_train.class_names))),
    )
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_5class, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset_train.class_names,
                yticklabels=full_dataset_train.class_names)
    plt.ylabel('Actual setting action'); plt.xlabel('Predicted setting action')
    plt.title('Confusion Matrix (Reconstructed 5-class) - Validation')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH_5CLASS, dpi=300)
    plt.close()

    # Per-head confusion matrices
    cm_technique = confusion_matrix(
        final_targets['technique'], final_preds['technique'], labels=[0, 1, 2]
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_technique, annot=True, fmt='d', cmap='Blues',
                xticklabels=TECHNIQUE_NAMES, yticklabels=TECHNIQUE_NAMES)
    plt.ylabel('Actual technique'); plt.xlabel('Predicted technique')
    plt.title('Confusion Matrix (Technique Head) - Validation')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH_TECHNIQUE, dpi=300)
    plt.close()

    cm_direction = confusion_matrix(
        final_targets['direction'], final_preds['direction'], labels=[0, 1, 2]
    )
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_direction, annot=True, fmt='d', cmap='Blues',
                xticklabels=DIRECTION_NAMES, yticklabels=DIRECTION_NAMES)
    plt.ylabel('Actual direction'); plt.xlabel('Predicted direction')
    plt.title('Confusion Matrix (Direction Head) - Validation')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH_DIRECTION, dpi=300)
    plt.close()

    cm_other = confusion_matrix(
        final_targets['other'], final_preds['other'], labels=[0, 1]
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_other, annot=True, fmt='d', cmap='Blues',
                xticklabels=OTHER_NAMES, yticklabels=OTHER_NAMES)
    plt.ylabel('Actual'); plt.xlabel('Predicted')
    plt.title('Confusion Matrix (Other Gate) - Validation')
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH_OTHER, dpi=300)
    plt.close()

    print(f"Confusion matrices saved:")
    print(f"  5-class:   {CONFUSION_MATRIX_PATH_5CLASS}")
    print(f"  Technique: {CONFUSION_MATRIX_PATH_TECHNIQUE}")
    print(f"  Direction: {CONFUSION_MATRIX_PATH_DIRECTION}")
    print(f"  Other:     {CONFUSION_MATRIX_PATH_OTHER}")
    print("\nTraining complete.")