# --- 1. STANDARD LIBRARY IMPORTS ---
import os
import random
import json
import csv
import datetime
from collections import Counter

# --- 2. THIRD-PARTY LIBRARIES ---
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib.pyplot as plt
import seaborn as sns

# --- 3. LOCAL MODULE IMPORTS ---
from dataset import VolleyballDataset
from SlowFast_model import VolleyballSlowFastModel

torch.manual_seed(16)
np.random.seed(16)
random.seed(16)
torch.cuda.manual_seed_all(16)

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = "/workspace/Master_Dataset_Extracted"

NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 8

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 20
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4

N_FOLDS = 5

MODEL_TAG = "SlowFast"
RESULTS_DIR = "/workspace/results_SlowFast_5fold"


def compute_class_weights(dataset, train_indices, device):
    """
    Computes inverse-frequency class weights from the training subset only,
    so validation labels never influence the loss function.
    Returns a tensor of shape (num_classes,) on the correct device.
    """
    train_labels = [dataset.labels[i] for i in train_indices]
    counts = Counter(train_labels)
    num_classes = len(dataset.class_names)
    total = len(train_labels)
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]

    print("\nClass weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(dataset.class_names, weights)):
        print(f"  [{i}] {name}: count={counts.get(i, 0)}, weight={w:.4f}")

    weights_meta = {dataset.class_names[c]: {"count": counts.get(c, 0), "weight": weights[c]}
                    for c in range(num_classes)}

    return torch.tensor(weights, dtype=torch.float32).to(device), weights_meta


def run_epoch(model, loader, criterion, is_train, device, optimizer=None):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)

            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:
                        videos[i] = torch.flip(videos[i], dims=[3])

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy, all_preds, all_labels


def train_one_fold(fold_idx, train_indices, val_indices,
                   full_dataset_train, full_dataset_val, groups, device):
    """
    Trains and evaluates a single fold.
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

    class_weights, class_weights_meta = compute_class_weights(full_dataset_train, train_indices, device)

    num_classes = len(full_dataset_train.class_names)
    model = VolleyballSlowFastModel(num_classes=num_classes, freeze_backbone=True, dropout_p=0.5, alpha=4).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    ckpt_path = f"/workspace/SlowFast_best_fold{fold_idx}.pth"

    # --- STAGE 1 ---
    print(f"\n=== [Fold {fold_idx + 1}] STAGE 1: Training head only for {STAGE1_EPOCHS} epochs ===")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_S1)
    best_val_acc = 0.0

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, device=device,
                                                optimizer=optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, is_train=False, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Fold {fold_idx + 1} | S1 {epoch + 1}/{STAGE1_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- STAGE 2 ---
    print(f"\n=== [Fold {fold_idx + 1}] STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    model.unfreeze_backbone()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, device=device,
                                                optimizer=optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, is_train=False, device=device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[Fold {fold_idx + 1} | S2 {epoch + 1}/{STAGE2_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- FINAL EVALUATION ---
    print(f"\n=== [Fold {fold_idx + 1}] FINAL EVALUATION (best checkpoint) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    _, final_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion, is_train=False, device=device)

    class_names = full_dataset_train.class_names
    label_ids = list(range(num_classes))

    macro_f1 = f1_score(final_labels, final_preds, average='macro', labels=label_ids, zero_division=0) * 100
    weighted_f1 = f1_score(final_labels, final_preds, average='weighted', labels=label_ids, zero_division=0) * 100
    print(
        f"[Fold {fold_idx + 1}] Final Val Accuracy: {final_acc:.2f}% | Macro-F1: {macro_f1:.2f}% | Weighted-F1: {weighted_f1:.2f}%")

    report_dict = classification_report(final_labels, final_preds, target_names=class_names, labels=label_ids,
                                        zero_division=0, output_dict=True)
    print("\nPer-class report:")
    print(classification_report(final_labels, final_preds, target_names=class_names, labels=label_ids, zero_division=0))

    cm = confusion_matrix(final_labels, final_preds, labels=label_ids)

    # ---- PARITY FIX: Save fold{idx}_per_class_metrics.csv ----
    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_per_class_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names:
            row = report_dict.get(name, {})
            w.writerow([name, row.get("precision", 0), row.get("recall", 0),
                        row.get("f1-score", 0), row.get("support", 0)])

    # ---- PERSIST FOLD RESULTS ----
    fold_payload = {
        "model": MODEL_TAG,
        "fold": fold_idx,
        "hyperparameters": {
            "num_frames": NUM_FRAMES, "img_size": IMG_SIZE, "batch_size": BATCH_SIZE,
            "stage1_epochs": STAGE1_EPOCHS, "stage2_epochs": STAGE2_EPOCHS,
            "lr_s1": LEARNING_RATE_S1, "lr_s2": LEARNING_RATE_S2,
            "weight_decay": 1e-4, "label_smoothing": 0.1, "dropout_p": 0.5,
        },
        "train_matches": train_matches,
        "val_matches": val_matches,
        "class_names": class_names,
        "class_weights": class_weights_meta,
        "final_val_accuracy": final_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "best_val_acc_during_training": best_val_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "history": history,
        "val_predictions": final_preds,
        "val_labels": final_labels,
    }
    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_results.json"), "w") as f:
        json.dump(fold_payload, f, indent=2)

    with open(os.path.join(RESULTS_DIR, f"fold{fold_idx}_confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual\\predicted"] + class_names)
        for r, name in enumerate(class_names):
            w.writerow([name] + cm[r].tolist())

    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title(f"SlowFast Accuracy (Fold {fold_idx + 1})")
    plt.xlabel("Epochs");
    plt.ylabel("Accuracy (%)");
    plt.ylim(0, 100);
    plt.legend();
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title(f"SlowFast Loss (Fold {fold_idx + 1})")
    plt.xlabel("Epochs");
    plt.ylabel("Loss");
    plt.legend();
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"SlowFast_learning_curve_fold{fold_idx}.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual setting action');
    plt.xlabel('Predicted setting action')
    plt.title(f'Confusion Matrix - Fold {fold_idx + 1} (SlowFast)')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_SlowFast_fold{fold_idx}.png"), dpi=300)
    plt.close()

    return final_acc, macro_f1, weighted_f1, final_preds, final_labels


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"All results will be written to: {RESULTS_DIR}")

    # --- 2. DATA PREPARATION ---
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
    ])

    full_dataset_train = VolleyballDataset(root_dir=ROOT_DIR, transform=train_transform, num_frames=NUM_FRAMES)
    full_dataset_val = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)

    groups = []
    for path in full_dataset_train.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)

    fold_accuracies, fold_macro_f1, fold_weighted_f1 = [], [], []
    oof_preds, oof_labels = [], []

    # ---- LOOP OVER ALL FIVE FOLDS ----
    for fold_idx, (train_indices, val_indices) in enumerate(sgkf.split(
            X=np.zeros(len(full_dataset_train)),
            y=full_dataset_train.labels,
            groups=groups)):
        final_acc, macro_f1, weighted_f1, fold_preds, fold_labels = train_one_fold(
            fold_idx, train_indices, val_indices,
            full_dataset_train, full_dataset_val, groups, device
        )

        fold_accuracies.append(final_acc)
        fold_macro_f1.append(macro_f1)
        fold_weighted_f1.append(weighted_f1)
        oof_preds.extend(fold_preds)
        oof_labels.extend(fold_labels)

    # ===== AGGREGATE ACROSS FOLDS =====
    fold_accuracies = np.array(fold_accuracies)
    fold_macro_f1 = np.array(fold_macro_f1)
    fold_weighted_f1 = np.array(fold_weighted_f1)

    class_names = full_dataset_train.class_names
    label_ids = list(range(len(class_names)))

    print("\n\n========= 5-FOLD CROSS-VALIDATION SUMMARY =========")
    for i in range(N_FOLDS):
        print(
            f"  Fold {i + 1}: Acc = {fold_accuracies[i]:.2f}% | Macro-F1 = {fold_macro_f1[i]:.2f}% | Weighted-F1 = {fold_weighted_f1[i]:.2f}%")
    print("---------------------------------------------------")
    print(f"  Accuracy:    {fold_accuracies.mean():.2f}% +/- {fold_accuracies.std():.2f}")
    print(f"  Macro-F1:    {fold_macro_f1.mean():.2f}% +/- {fold_macro_f1.std():.2f}")
    print(f"  Weighted-F1: {fold_weighted_f1.mean():.2f}% +/- {fold_weighted_f1.std():.2f}")
    print("===================================================")

    # ===== POOLED OUT-OF-FOLD REPORT =====
    print("\n========= POOLED OUT-OF-FOLD CLASSIFICATION REPORT =========")
    oof_report_text = classification_report(oof_labels, oof_preds, target_names=class_names, labels=label_ids,
                                            zero_division=0)
    print(oof_report_text)

    oof_report_dict = classification_report(oof_labels, oof_preds, target_names=class_names, labels=label_ids,
                                            zero_division=0, output_dict=True)
    oof_macro_f1 = f1_score(oof_labels, oof_preds, average='macro', labels=label_ids, zero_division=0) * 100
    oof_accuracy = 100.0 * sum(int(p == t) for p, t in zip(oof_preds, oof_labels)) / len(oof_labels)
    oof_cm = confusion_matrix(oof_labels, oof_preds, labels=label_ids)

    # ---- PARITY FIX: cross_fold_mean_std includes weighted metrics ----
    summary = {
        "model": MODEL_TAG,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_folds": N_FOLDS,
        "class_names": class_names,
        "per_fold": {
            "accuracy": fold_accuracies.tolist(), "macro_f1": fold_macro_f1.tolist(),
            "weighted_f1": fold_weighted_f1.tolist(),
        },
        "cross_fold_mean_std": {
            "accuracy_mean": float(fold_accuracies.mean()), "accuracy_std": float(fold_accuracies.std()),
            "macro_f1_mean": float(fold_macro_f1.mean()), "macro_f1_std": float(fold_macro_f1.std()),
            "weighted_f1_mean": float(fold_weighted_f1.mean()), "weighted_f1_std": float(fold_weighted_f1.std()),
        },
        "pooled_oof": {
            "accuracy": oof_accuracy, "macro_f1": oof_macro_f1,
            "classification_report": oof_report_dict, "confusion_matrix": oof_cm.tolist(),
        },
    }
    with open(os.path.join(RESULTS_DIR, "SUMMARY_5fold.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- PARITY FIX: summary_per_fold.csv appends training mean and std rows ----
    with open(os.path.join(RESULTS_DIR, "summary_per_fold.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fold", "accuracy", "macro_f1", "weighted_f1"])
        for i in range(N_FOLDS):
            w.writerow([i + 1, f"{fold_accuracies[i]:.2f}", f"{fold_macro_f1[i]:.2f}", f"{fold_weighted_f1[i]:.2f}"])
        w.writerow(
            ["mean", f"{fold_accuracies.mean():.2f}", f"{fold_macro_f1.mean():.2f}", f"{fold_weighted_f1.mean():.2f}"])
        w.writerow(
            ["std", f"{fold_accuracies.std():.2f}", f"{fold_macro_f1.std():.2f}", f"{fold_weighted_f1.std():.2f}"])

    # ---- PARITY FIX: write oof_per_class_metrics.csv file separately ----
    with open(os.path.join(RESULTS_DIR, "oof_per_class_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "precision", "recall", "f1_score", "support"])
        for name in class_names:
            row = oof_report_dict.get(name, {})
            w.writerow([name, row.get("precision", 0), row.get("recall", 0),
                        row.get("f1-score", 0), row.get("support", 0)])

    with open(os.path.join(RESULTS_DIR, "oof_confusion_matrix.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["actual\\predicted"] + class_names)
        for r, name in enumerate(class_names):
            w.writerow([name] + oof_cm[r].tolist())

    with open(os.path.join(RESULTS_DIR, "oof_predictions.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "true_label_id", "true_label_name", "pred_label_id", "pred_label_name"])
        for i, (t, p) in enumerate(zip(oof_labels, oof_preds)):
            w.writerow([i, t, class_names[t], p, class_names[p]])

    # ---- PARITY FIX: write oof_classification_report.txt plain-text dump ----
    with open(os.path.join(RESULTS_DIR, "oof_classification_report.txt"), "w") as f:
        f.write(f"Model: {MODEL_TAG}\n")
        f.write(f"Pooled OOF accuracy: {oof_accuracy:.2f}%\n")
        f.write(f"Pooled OOF macro-F1: {oof_macro_f1:.2f}%\n\n")
        f.write(oof_report_text)

    plt.figure(figsize=(10, 8))
    sns.heatmap(oof_cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual setting action');
    plt.xlabel('Predicted setting action')
    plt.title('Pooled Out-of-Fold Confusion Matrix (5-fold CV) - SlowFast')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_SlowFast_oof_5fold.png"), dpi=300)
    plt.close()

    print(f"\nAll metrics, reports, matrices and raw predictions saved to: {RESULTS_DIR}")