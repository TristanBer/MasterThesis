"""
Training Script for R(2+1)D-18 Dual-Head Ablation Study (A1) — 5-Fold CV
========================================================================

This script trains the multi-task variant of VolleyballR2Plus1DModel defined
in DualHead_R2plus1D_model.py across all five StratifiedGroupKFold folds.
The structure mirrors R2Plus1D_train_5fold.py to guarantee that any observed
performance differences are attributable solely to the multi-task decomposition
rather than to changes in the training/validation protocol.

Output format:
    The reconstructed 5-class metrics are stored at the top level of
    SUMMARY_5fold.json, matching the single-head R(2+1)D baseline exactly,
    so the ablation comparison (Section 5.4.1) can be performed via identical
    JSON extraction paths. Per-head diagnostics (technique / direction / other)
    are nested under 'per_head_diagnostics' for supplementary analysis.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
import os
import json
import csv
import datetime
import random

from dataset import VolleyballDataset
from DualHead_R2plus1D_model import (
    VolleyballR2Plus1DDualHeadModel, MultiTaskLoss,
    decompose_labels, reconstruct_5class_prediction,
    TECHNIQUE_NAMES, DIRECTION_NAMES, OTHER_NAMES,
)

torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
torch.cuda.manual_seed_all(16)

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = "/workspace/Master_Dataset_Extracted"

NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 16

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 5e-5

# Multi-task loss weights (lambda_t, lambda_d, lambda_o)
LAMBDA_TECHNIQUE = 1.0
LAMBDA_DIRECTION = 1.0
LAMBDA_OTHER = 0.5

N_FOLDS = 5
MODEL_TAG = "R2Plus1DDualHead"
RESULTS_DIR = "/workspace/results_R2Plus1DDualHead_5fold"


# ---------------------------------------------------------------------------
# Class-weight computation per head (per fold)
# ---------------------------------------------------------------------------
def compute_per_head_class_weights(dataset, train_indices, device):
    """
    Inverse-frequency class weights for each of the three classification heads,
    derived from the training subset only.
    """
    from DualHead_R2plus1D_model import LABEL_DECOMPOSITION

    train_labels = [dataset.labels[i] for i in train_indices]
    total = len(train_labels)

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
    print(f"  Technique {TECHNIQUE_NAMES}: counts={dict(Counter(technique_labels))}, weights={[f'{w:.3f}' for w in technique_weights]}")
    print(f"  Direction {DIRECTION_NAMES}: counts={dict(Counter(direction_labels))}, weights={[f'{w:.3f}' for w in direction_weights]}")
    print(f"  Other     {OTHER_NAMES}: counts={dict(Counter(other_labels))}, weights={[f'{w:.3f}' for w in other_weights]}")

    weights_meta = {
        "technique": {TECHNIQUE_NAMES[c]: {"count": Counter(technique_labels).get(c, 0),
                                           "weight": technique_weights[c]} for c in range(3)},
        "direction": {DIRECTION_NAMES[c]: {"count": Counter(direction_labels).get(c, 0),
                                           "weight": direction_weights[c]} for c in range(3)},
        "other":     {OTHER_NAMES[c]:     {"count": Counter(other_labels).get(c, 0),
                                           "weight": other_weights[c]} for c in range(2)},
    }

    return (
        torch.tensor(technique_weights, dtype=torch.float32).to(device),
        torch.tensor(direction_weights, dtype=torch.float32).to(device),
        torch.tensor(other_weights, dtype=torch.float32).to(device),
        weights_meta,
    )


# ---------------------------------------------------------------------------
# Multi-task epoch runner — UNCHANGED from original
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, is_train, device, optimizer=None):
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

            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:
                        videos[i] = torch.flip(videos[i], dims=[3])

            target_t, target_d, target_o = decompose_labels(labels_5class)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(videos)
                loss, components = criterion(outputs, target_t, target_d, target_o)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            for k in component_loss_sums:
                component_loss_sums[k] += components[k]

            pred_t = torch.argmax(outputs['technique'], dim=1)
            pred_d = torch.argmax(outputs['direction'], dim=1)
            pred_o = torch.argmax(outputs['other'], dim=1)
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
# Per-fold training, evaluation, and persistence
# ---------------------------------------------------------------------------
def train_one_fold(fold_idx, train_indices, val_indices,
                   full_dataset_train, full_dataset_val, groups, device):
    """
    Trains and evaluates a single fold using the dual-head two-stage protocol.
    Returns the final-checkpoint 5-class metrics plus per-head metrics
    and full prediction/label arrays for OOF aggregation.
    """
    print(f"\n############## FOLD {fold_idx + 1}/{N_FOLDS} ##############")

    train_matches = sorted(list(set([groups[i] for i in train_indices])))
    val_matches = sorted(list(set([groups[i] for i in val_indices])))
    print("================ DATA LEAKAGE VERIFICATION ================")
    print(f"Training Match Splits ({len(train_matches)} games): {train_matches}")
    print(f"Validation Match Splits ({len(val_matches)} games): {val_matches}")
    overlap = set(train_matches).intersection(set(val_matches))
    print(f"Intersection Overlap Check (Must be empty set): {overlap}")
    print("===========================================================")

    train_db = Subset(full_dataset_train, train_indices)
    val_db = Subset(full_dataset_val, val_indices)
    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    # Per-fold class weights (per head, training subset only)
    technique_weights, direction_weights, other_weights, weights_meta = \
        compute_per_head_class_weights(full_dataset_train, train_indices, device)

    # Fresh model + fresh loss per fold (no weight leakage across folds)
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

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc_5class": [], "val_acc_5class": [],
        "train_acc_technique": [], "val_acc_technique": [],
        "train_acc_direction": [], "val_acc_direction": [],
        "train_acc_other": [], "val_acc_other": [],
    }
    best_val_acc = 0.0
    ckpt_path = f"/workspace/R2Plus1DDualHead_best_fold{fold_idx}.pth"

    # --- STAGE 1: heads only ---
    print(f"\n=== [Fold {fold_idx + 1}] STAGE 1: Training heads only for {STAGE1_EPOCHS} epochs ===")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_S1)

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_comp, train_acc, _, _ = run_epoch(
            model, train_loader, criterion, is_train=True, device=device, optimizer=optimizer
        )
        val_loss, val_comp, val_acc, _, _ = run_epoch(
            model, val_loader, criterion, is_train=False, device=device
        )

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc_5class"].append(train_acc['reconstructed_5class'])
        history["val_acc_5class"].append(val_acc['reconstructed_5class'])
        history["train_acc_technique"].append(train_acc['technique'])
        history["val_acc_technique"].append(val_acc['technique'])
        history["train_acc_direction"].append(train_acc['direction'])
        history["val_acc_direction"].append(val_acc['direction'])
        history["train_acc_other"].append(train_acc['other'])
        history["val_acc_other"].append(val_acc['other'])

        print(f"[Fold {fold_idx + 1} | S1 {epoch + 1}/{STAGE1_EPOCHS}] "
              f"Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"5-class Acc(T/V): {train_acc['reconstructed_5class']:.2f}%/{val_acc['reconstructed_5class']:.2f}% | "
              f"Tech(V): {val_acc['technique']:.2f}% | Dir(V): {val_acc['direction']:.2f}% | Other(V): {val_acc['other']:.2f}%")

        if val_acc['reconstructed_5class'] > best_val_acc:
            best_val_acc = val_acc['reconstructed_5class']
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved (best 5-class val acc: {best_val_acc:.2f}%)")

    # --- STAGE 2: full fine-tuning ---
    print(f"\n=== [Fold {fold_idx + 1}] STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
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

        history["train_loss"].append(train_loss); history["val_loss"].append(val_loss)
        history["train_acc_5class"].append(train_acc['reconstructed_5class'])
        history["val_acc_5class"].append(val_acc['reconstructed_5class'])
        history["train_acc_technique"].append(train_acc['technique'])
        history["val_acc_technique"].append(val_acc['technique'])
        history["train_acc_direction"].append(train_acc['direction'])
        history["val_acc_direction"].append(val_acc['direction'])
        history["train_acc_other"].append(train_acc['other'])
        history["val_acc_other"].append(val_acc['other'])

        print(f"[Fold {fold_idx + 1} | S2 {epoch + 1}/{STAGE2_EPOCHS}] "
              f"Loss(T/V): {train_loss:.3f}/{val_loss:.3f} | "
              f"5-class Acc(T/V): {train_acc['reconstructed_5class']:.2f}%/{val_acc['reconstructed_5class']:.2f}% | "
              f"Tech(V): {val_acc['technique']:.2f}% | Dir(V): {val_acc['direction']:.2f}% | Other(V): {val_acc['other']:.2f}%")

        if val_acc['reconstructed_5class'] > best_val_acc:
            best_val_acc = val_acc['reconstructed_5class']
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved (best 5-class val acc: {best_val_acc:.2f}%)")

    # --- FINAL EVALUATION (best checkpoint) ---
    print(f"\n=== [Fold {fold_idx + 1}] FINAL EVALUATION (best checkpoint) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    _, _, final_acc, final_preds, final_targets = run_epoch(
        model, val_loader, criterion, is_train=False, device=device
    )

    class_names_5 = full_dataset_train.class_names
    label_ids_5 = list(range(len(class_names_5)))

    # ---- 5-class reconstructed metrics (PRIMARY) ----
    final_acc_5class = final_acc['reconstructed_5class']
    macro_f1_5 = f1_score(final_targets['reconstructed_5class'], final_preds['reconstructed_5class'],
                          average='macro', labels=label_ids_5, zero_division=0) * 100
    weighted_f1_5 = f1_score(final_targets['reconstructed_5class'], final_preds['reconstructed_5class'],
                             average='weighted', labels=label_ids_5, zero_division=0) * 100
    report_5 = classification_report(final_targets['reconstructed_5class'], final_preds['reconstructed_5class'],
                                     target_names=class_names_5, labels=label_ids_5,
                                     zero_division=0, output_dict=True)
    cm_5 = confusion_matrix(final_targets['reconstructed_5class'], final_preds['reconstructed_5class'],
                            labels=label_ids_5)

    print(f"[Fold {fold_idx + 1}] 5-class reconstructed: Acc = {final_acc_5class:.2f}% | "
          f"Macro-F1 = {macro_f1_5:.2f}% | Weighted-F1 = {weighted_f1_5:.2f}%")
    print(f"  Technique head: Acc = {final_acc['technique']:.2f}%")
    print(f"  Direction head: Acc = {final_acc['direction']:.2f}%")
    print(f"  Other gate:     Acc = {final_acc['other']:.2f}%")

    # ---- Per-head metrics (DIAGNOSTIC) ----
    def head_metrics(preds, targs, names):
        ids = list(range(len(names)))
        acc = 100.0 * sum(int(p == t) for p, t in zip(preds, targs)) / len(targs)
        macro = f1_score(targs, preds, average='macro', labels=ids, zero_division=0) * 100
        weighted = f1_score(targs, preds, average='weighted', labels=ids, zero_division=0) * 100
        rep = classification_report(targs, preds, target_names=names, labels=ids,
                                    zero_division=0, output_dict=True)
        cm = confusion_matrix(targs, preds, labels=ids)
        return {"accuracy": acc, "macro_f1": macro, "weighted_f1": weighted,
                "classification_report": rep, "confusion_matrix": cm.tolist()}

    technique_metrics = head_metrics(final_preds['technique'], final_targets['technique'], TECHNIQUE_NAMES)
    direction_metrics = head_metrics(final_preds['direction'], final_targets['direction'], DIRECTION_NAMES)
    other_metrics = head_metrics(final_preds['other'], final_targets['other'], OTHER_NAMES)

    # ---- PERSIST PER-FOLD JSON ----
    fold_payload = {
        "model": MODEL_TAG,
        "fold": fold_idx,
        "hyperparameters": {
            "num_frames": NUM_FRAMES, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS,
            "lr_s1": LEARNING_RATE_S1, "lr_s2": LEARNING_RATE_S2,
            "weight_decay": 5e-4, "label_smoothing": 0.1, "dropout_p": 0.5,
            "lambda_technique": LAMBDA_TECHNIQUE,
            "lambda_direction": LAMBDA_DIRECTION,
            "lambda_other": LAMBDA_OTHER,
            "grad_clip_max_norm": 1.0,
        },
        "train_matches": train_matches,
        "val_matches": val_matches,
        "n_train_clips": int(len(train_indices)),
        "n_val_clips": int(len(val_indices)),
        "class_names": class_names_5,
        "per_head_class_weights": weights_meta,

        # 5-class reconstructed (primary, comparable to baseline R(2+1)D)
        "final_val_accuracy": final_acc_5class,
        "macro_f1": macro_f1_5,
        "weighted_f1": weighted_f1_5,
        "best_val_acc_during_training": best_val_acc,
        "classification_report": report_5,
        "confusion_matrix": cm_5.tolist(),
        "history": history,
        "val_predictions": final_preds['reconstructed_5class'],
        "val_labels": final_targets['reconstructed_5class'],

        # Per-head diagnostics
        "per_head": {
            "technique": {
                "class_names": list(TECHNIQUE_NAMES),
                **technique_metrics,
                "predictions": final_preds['technique'],
                "labels": final_targets['technique'],
            },
            "direction": {
                "class_names": list(DIRECTION_NAMES),
                **direction_metrics,
                "predictions": final_preds['direction'],
                "labels": final_targets['direction'],
            },
            "other": {
                "class_names": list(OTHER_NAMES),
                **other_metrics,
                "predictions": final_preds['other'],
                "labels": final_targets['other'],
            },
        },
    }
    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_results.json"), "w") as f:
        json.dump(fold_payload, f, indent=2)
    print(f"  -> Saved fold{fold_idx}_results.json")

    # ---- PER-FOLD CSVs (5-class confusion + per-class metrics) ----
    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_confusion_matrix_5class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual\\predicted"] + class_names_5)
        for r, name in enumerate(class_names_5):
            w.writerow([name] + cm_5[r].tolist())

    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_per_class_metrics_5class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names_5:
            row = report_5.get(name, {})
            w.writerow([name, row.get("precision", 0), row.get("recall", 0),
                        row.get("f1-score", 0), row.get("support", 0)])

    # ---- PER-FOLD LEARNING CURVES (4-panel multi-task) ----
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    ax.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title(f"Total Multi-Task Loss (Fold {fold_idx + 1})")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True)

    ax = axes[0, 1]
    ax.plot(epochs_range, history["train_acc_5class"], label="Train 5-class Acc", marker='o')
    ax.plot(epochs_range, history["val_acc_5class"], label="Val 5-class Acc", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title(f"Reconstructed 5-class Accuracy (Fold {fold_idx + 1})")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend(); ax.grid(True)

    ax = axes[1, 0]
    ax.plot(epochs_range, history["val_acc_technique"], label="Technique (Val)", marker='o')
    ax.plot(epochs_range, history["val_acc_direction"], label="Direction (Val)", marker='o')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title(f"Per-Head Validation Accuracy (Fold {fold_idx + 1})")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend(); ax.grid(True)

    ax = axes[1, 1]
    ax.plot(epochs_range, history["val_acc_other"], label="Other Gate (Val)", marker='o', color='purple')
    ax.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    ax.set_title(f"Other-Gate Head Validation Accuracy (Fold {fold_idx + 1})")
    ax.set_xlabel("Epochs"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100); ax.legend(); ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"R2Plus1DDualHead_learning_curve_fold{fold_idx}.png"), dpi=300)
    plt.close()

    # Per-fold 5-class confusion-matrix figure
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_5, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_5, yticklabels=class_names_5)
    plt.ylabel('Actual setting action'); plt.xlabel('Predicted setting action')
    plt.title(f'Confusion Matrix (Reconstructed 5-class) - Fold {fold_idx + 1}')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_R2Plus1DDualHead_fold{fold_idx}.png"), dpi=300)
    plt.close()

    return (
        final_acc_5class, macro_f1_5, weighted_f1_5,
        final_preds['reconstructed_5class'], final_targets['reconstructed_5class'],
        final_preds['technique'], final_targets['technique'],
        final_preds['direction'], final_targets['direction'],
        final_preds['other'], final_targets['other'],
        final_acc['technique'], final_acc['direction'], final_acc['other'],
    )


# ---------------------------------------------------------------------------
# Main: 5-fold loop + aggregation
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"All results will be written to: {RESULTS_DIR}")

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

    groups = []
    for path in full_dataset_train.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)

    # Cross-fold accumulators
    fold_acc_5class, fold_macro_f1_5, fold_weighted_f1_5 = [], [], []
    fold_acc_technique, fold_acc_direction, fold_acc_other = [], [], []

    oof_preds_5, oof_labels_5 = [], []
    oof_preds_t, oof_labels_t = [], []
    oof_preds_d, oof_labels_d = [], []
    oof_preds_o, oof_labels_o = [], []

    # ---- LOOP OVER ALL FIVE FOLDS ----
    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(
            X=np.zeros(len(full_dataset_train)),
            y=full_dataset_train.labels,
            groups=groups)):

        (acc5, mf1_5, wf1_5,
         p5, t5, pt, tt, pd, td, po, to_,
         acc_t, acc_d, acc_o) = train_one_fold(
            fold_idx, train_indices, val_indices,
            full_dataset_train, full_dataset_val, groups, device
        )

        fold_acc_5class.append(acc5)
        fold_macro_f1_5.append(mf1_5)
        fold_weighted_f1_5.append(wf1_5)
        fold_acc_technique.append(acc_t)
        fold_acc_direction.append(acc_d)
        fold_acc_other.append(acc_o)

        oof_preds_5.extend(p5); oof_labels_5.extend(t5)
        oof_preds_t.extend(pt); oof_labels_t.extend(tt)
        oof_preds_d.extend(pd); oof_labels_d.extend(td)
        oof_preds_o.extend(po); oof_labels_o.extend(to_)

    # ===== AGGREGATE ACROSS FOLDS =====
    fold_acc_5class = np.array(fold_acc_5class)
    fold_macro_f1_5 = np.array(fold_macro_f1_5)
    fold_weighted_f1_5 = np.array(fold_weighted_f1_5)
    fold_acc_technique = np.array(fold_acc_technique)
    fold_acc_direction = np.array(fold_acc_direction)
    fold_acc_other = np.array(fold_acc_other)

    class_names_5 = full_dataset_train.class_names
    label_ids_5 = list(range(len(class_names_5)))

    print("\n\n========= 5-FOLD CROSS-VALIDATION SUMMARY (DualHead) =========")
    for i in range(N_FOLDS):
        print(f"  Fold {i + 1}: 5-class Acc = {fold_acc_5class[i]:.2f}% | "
              f"Macro-F1 = {fold_macro_f1_5[i]:.2f}% | Weighted-F1 = {fold_weighted_f1_5[i]:.2f}% | "
              f"Tech = {fold_acc_technique[i]:.2f}% | Dir = {fold_acc_direction[i]:.2f}% | Other = {fold_acc_other[i]:.2f}%")
    print("---------------------------------------------------------------")
    print(f"  5-class Accuracy:    {fold_acc_5class.mean():.2f}% +/- {fold_acc_5class.std():.2f}")
    print(f"  5-class Macro-F1:    {fold_macro_f1_5.mean():.2f}% +/- {fold_macro_f1_5.std():.2f}")
    print(f"  5-class Weighted-F1: {fold_weighted_f1_5.mean():.2f}% +/- {fold_weighted_f1_5.std():.2f}")
    print(f"  Technique Accuracy:  {fold_acc_technique.mean():.2f}% +/- {fold_acc_technique.std():.2f}")
    print(f"  Direction Accuracy:  {fold_acc_direction.mean():.2f}% +/- {fold_acc_direction.std():.2f}")
    print(f"  Other Accuracy:      {fold_acc_other.mean():.2f}% +/- {fold_acc_other.std():.2f}")
    print("===============================================================")

    # ---- POOLED OOF (5-class reconstructed, PRIMARY) ----
    oof_report_text_5 = classification_report(oof_labels_5, oof_preds_5,
                                              target_names=class_names_5,
                                              labels=label_ids_5, zero_division=0)
    print("\n========= POOLED OUT-OF-FOLD CLASSIFICATION REPORT (5-class reconstructed) =========")
    print(oof_report_text_5)

    oof_report_dict_5 = classification_report(oof_labels_5, oof_preds_5,
                                              target_names=class_names_5, labels=label_ids_5,
                                              zero_division=0, output_dict=True)
    oof_macro_f1_5 = f1_score(oof_labels_5, oof_preds_5, average='macro',
                              labels=label_ids_5, zero_division=0) * 100
    oof_acc_5 = 100.0 * sum(int(p == t) for p, t in zip(oof_preds_5, oof_labels_5)) / len(oof_labels_5)
    oof_cm_5 = confusion_matrix(oof_labels_5, oof_preds_5, labels=label_ids_5)

    # ---- POOLED OOF per head (diagnostic) ----
    def pooled_head(preds, targs, names):
        ids = list(range(len(names)))
        acc = 100.0 * sum(int(p == t) for p, t in zip(preds, targs)) / len(targs)
        macro = f1_score(targs, preds, average='macro', labels=ids, zero_division=0) * 100
        rep = classification_report(targs, preds, target_names=names, labels=ids,
                                    zero_division=0, output_dict=True)
        cm = confusion_matrix(targs, preds, labels=ids)
        return {"accuracy": acc, "macro_f1": macro,
                "classification_report": rep, "confusion_matrix": cm.tolist()}

    pooled_technique = pooled_head(oof_preds_t, oof_labels_t, TECHNIQUE_NAMES)
    pooled_direction = pooled_head(oof_preds_d, oof_labels_d, DIRECTION_NAMES)
    pooled_other = pooled_head(oof_preds_o, oof_labels_o, OTHER_NAMES)

    # ---- COMBINED SUMMARY JSON ----
    summary = {
        "model": MODEL_TAG,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_folds": N_FOLDS,
        "class_names": class_names_5,

        # 5-class reconstructed (PRIMARY, schema matches baseline R(2+1)D)
        "per_fold": {
            "accuracy": fold_acc_5class.tolist(),
            "macro_f1": fold_macro_f1_5.tolist(),
            "weighted_f1": fold_weighted_f1_5.tolist(),
        },
        "cross_fold_mean_std": {
            "accuracy_mean": float(fold_acc_5class.mean()), "accuracy_std": float(fold_acc_5class.std()),
            "macro_f1_mean": float(fold_macro_f1_5.mean()), "macro_f1_std": float(fold_macro_f1_5.std()),
            "weighted_f1_mean": float(fold_weighted_f1_5.mean()), "weighted_f1_std": float(fold_weighted_f1_5.std()),
        },
        "pooled_oof": {
            "accuracy": oof_acc_5,
            "macro_f1": oof_macro_f1_5,
            "classification_report": oof_report_dict_5,
            "confusion_matrix": oof_cm_5.tolist(),
        },

        # Per-head diagnostics (DualHead-specific)
        "per_head_diagnostics": {
            "technique": {
                "class_names": list(TECHNIQUE_NAMES),
                "per_fold_accuracy": fold_acc_technique.tolist(),
                "cross_fold_mean_std": {
                    "accuracy_mean": float(fold_acc_technique.mean()),
                    "accuracy_std": float(fold_acc_technique.std()),
                },
                "pooled_oof": pooled_technique,
            },
            "direction": {
                "class_names": list(DIRECTION_NAMES),
                "per_fold_accuracy": fold_acc_direction.tolist(),
                "cross_fold_mean_std": {
                    "accuracy_mean": float(fold_acc_direction.mean()),
                    "accuracy_std": float(fold_acc_direction.std()),
                },
                "pooled_oof": pooled_direction,
            },
            "other": {
                "class_names": list(OTHER_NAMES),
                "per_fold_accuracy": fold_acc_other.tolist(),
                "cross_fold_mean_std": {
                    "accuracy_mean": float(fold_acc_other.mean()),
                    "accuracy_std": float(fold_acc_other.std()),
                },
                "pooled_oof": pooled_other,
            },
        },
    }
    with open(os.path.join(RESULTS_DIR, "SUMMARY_5fold.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Per-fold summary CSV (drop-in for thesis table) ----
    with open(os.path.join(RESULTS_DIR, "summary_per_fold.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "accuracy_5class", "macro_f1_5class", "weighted_f1_5class",
                    "acc_technique", "acc_direction", "acc_other"])
        for i in range(N_FOLDS):
            w.writerow([i + 1, f"{fold_acc_5class[i]:.2f}", f"{fold_macro_f1_5[i]:.2f}",
                        f"{fold_weighted_f1_5[i]:.2f}", f"{fold_acc_technique[i]:.2f}",
                        f"{fold_acc_direction[i]:.2f}", f"{fold_acc_other[i]:.2f}"])
        w.writerow(["mean", f"{fold_acc_5class.mean():.2f}", f"{fold_macro_f1_5.mean():.2f}",
                    f"{fold_weighted_f1_5.mean():.2f}", f"{fold_acc_technique.mean():.2f}",
                    f"{fold_acc_direction.mean():.2f}", f"{fold_acc_other.mean():.2f}"])
        w.writerow(["std", f"{fold_acc_5class.std():.2f}", f"{fold_macro_f1_5.std():.2f}",
                    f"{fold_weighted_f1_5.std():.2f}", f"{fold_acc_technique.std():.2f}",
                    f"{fold_acc_direction.std():.2f}", f"{fold_acc_other.std():.2f}"])

    # ---- Pooled OOF per-class metrics CSV (5-class) ----
    with open(os.path.join(RESULTS_DIR, "oof_per_class_metrics_5class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names_5:
            row = oof_report_dict_5.get(name, {})
            w.writerow([name, row.get("precision", 0), row.get("recall", 0),
                        row.get("f1-score", 0), row.get("support", 0)])

    # ---- Pooled OOF 5-class confusion matrix CSV ----
    with open(os.path.join(RESULTS_DIR, "oof_confusion_matrix_5class.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual\\predicted"] + class_names_5)
        for r, name in enumerate(class_names_5):
            w.writerow([name] + oof_cm_5[r].tolist())

    # ---- Raw OOF predictions (5-class + per-head, recomputable downstream) ----
    with open(os.path.join(RESULTS_DIR, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index",
                    "true_5class_id", "true_5class_name", "pred_5class_id", "pred_5class_name",
                    "true_technique_id", "pred_technique_id",
                    "true_direction_id", "pred_direction_id",
                    "true_other_id", "pred_other_id"])
        for i in range(len(oof_labels_5)):
            w.writerow([i,
                        oof_labels_5[i], class_names_5[oof_labels_5[i]],
                        oof_preds_5[i], class_names_5[oof_preds_5[i]],
                        oof_labels_t[i], oof_preds_t[i],
                        oof_labels_d[i], oof_preds_d[i],
                        oof_labels_o[i], oof_preds_o[i]])

    # ---- Plain-text OOF reports (5-class + per head) ----
    with open(os.path.join(RESULTS_DIR, "oof_classification_report.txt"), "w") as f:
        f.write(f"Model: {MODEL_TAG}\n")
        f.write(f"Pooled OOF 5-class accuracy: {oof_acc_5:.2f}%\n")
        f.write(f"Pooled OOF 5-class macro-F1: {oof_macro_f1_5:.2f}%\n\n")
        f.write("=== 5-class reconstructed report ===\n")
        f.write(oof_report_text_5)
        f.write("\n\n=== Technique-head report ===\n")
        f.write(classification_report(oof_labels_t, oof_preds_t, target_names=TECHNIQUE_NAMES,
                                       labels=list(range(len(TECHNIQUE_NAMES))), zero_division=0))
        f.write("\n\n=== Direction-head report ===\n")
        f.write(classification_report(oof_labels_d, oof_preds_d, target_names=DIRECTION_NAMES,
                                       labels=list(range(len(DIRECTION_NAMES))), zero_division=0))
        f.write("\n\n=== Other-gate report ===\n")
        f.write(classification_report(oof_labels_o, oof_preds_o, target_names=OTHER_NAMES,
                                       labels=list(range(len(OTHER_NAMES))), zero_division=0))

    # ---- Pooled OOF 5-class confusion matrix figure ----
    plt.figure(figsize=(10, 8))
    sns.heatmap(oof_cm_5, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names_5, yticklabels=class_names_5)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title('Pooled Out-of-Fold Confusion Matrix (Reconstructed 5-class, 5-fold CV)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_R2Plus1DDualHead_oof_5fold.png"), dpi=300)
    plt.close()

    print(f"\nAll metrics, reports, matrices and raw predictions saved to: {RESULTS_DIR}")
    print("Key files: SUMMARY_5fold.json, summary_per_fold.csv,")
    print("           oof_per_class_metrics_5class.csv, oof_confusion_matrix_5class.csv,")
    print("           oof_predictions.csv, oof_classification_report.txt,")
    print("           fold*_results.json")