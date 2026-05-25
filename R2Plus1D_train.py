import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import VolleyballDataset
from R2Plus1D_model import VolleyballR2Plus1DModel
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
import os

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = "/workspace/Master_Dataset_Extracted"

NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 8

STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 15
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4


def compute_class_weights(dataset, train_indices, device):
    """
    Computes inverse-frequency class weights from the training subset only,
    so validation labels never influence the loss function.
    Returns a tensor of shape (num_classes,) on the correct device.
    """
    train_labels = [dataset.labels[i] for i in train_indices]
    counts = Counter(train_labels)
    num_classes = len(dataset.class_names)
    # weight_c = total_train_samples / (num_classes * count_c)
    total = len(train_labels)
    weights = [total / (num_classes * counts.get(c, 1)) for c in range(num_classes)]
    print("\nClass weights (inverse frequency):")
    for i, (name, w) in enumerate(zip(dataset.class_names, weights)):
        print(f"  [{i}] {name}: count={counts.get(i, 0)}, weight={w:.4f}")
    return torch.tensor(weights, dtype=torch.float32).to(device)


def run_epoch(model, loader, criterion, is_train, device, optimizer=None):
    model.train() if is_train else model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)

            # Horizontal flip augmentation (training only) — plain loop avoids
            # bool-tensor fancy-indexing which confuses PyCharm's type checker
            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:
                        videos[i] = torch.flip(videos[i], dims=[3])  # Width dim

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


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. DATA PREPARATION ---

    # Training transform: color jitter + scale jitter (RandomResizedCrop)
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # Validation transform: deterministic center crop only (no augmentation)
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
    ])

    # Load full dataset instances sharing the same structural files
    full_dataset_train = VolleyballDataset(root_dir=ROOT_DIR, transform=train_transform, num_frames=NUM_FRAMES)
    full_dataset_val = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)

    # Programmatically extract match identifier group strings from the filenames
    groups = []
    for path in full_dataset_train.video_files:
        filename = os.path.basename(path)
        # Splits on either '_set_' (XML clipping) or '_frame_' (CSV clipping) to isolate the match prefix
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    # StratifiedGroupKFold splits into 5 chunks (yielding a perfect 80% / 20% split)
    # It balances class representations while preventing any group/match cross-contamination
    sgkf = StratifiedGroupKFold(n_splits=5)

    # Extract indices belonging to the first split fold configuration
    train_indices, val_indices = next(sgkf.split(
        X=np.zeros(len(full_dataset_train)),
        y=full_dataset_train.labels,
        groups=groups
    ))

    # Thesis Verification Summary: Print exactly how matches were distributed
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

    # --- 3. CLASS WEIGHTS ---
    class_weights = compute_class_weights(full_dataset_train, train_indices, device)

    # --- 4. MODEL + LOSS (with class weights) ---
    num_classes = len(full_dataset_train.class_names)
    model = VolleyballR2Plus1DModel(num_classes=num_classes, freeze_backbone=True, dropout_p=0.5).to(device)

    # Weighted cross-entropy penalises errors on rare classes more heavily
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # --- 5. STAGE 1: Train only the FC head ---
    print(f"\n=== STAGE 1: Training head only for {STAGE1_EPOCHS} epochs ===")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_S1)

    best_val_acc = 0.0

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, device=device, optimizer=optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, is_train=False, device=device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[S1 Epoch {epoch + 1}/{STAGE1_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/workspace/R2Plus1D_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- 6. STAGE 2: Unfreeze and fine-tune ---
    print(f"\n=== STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")
    model.load_state_dict(torch.load("/workspace/R2Plus1D_best.pth", map_location=device, weights_only=True))
    model.unfreeze_backbone()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S2, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, device=device, optimizer=optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, is_train=False, device=device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[S2 Epoch {epoch + 1}/{STAGE2_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/workspace/R2Plus1D_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- 7. FINAL EVALUATION ---
    print("\n=== FINAL EVALUATION (best checkpoint) ===")
    model.load_state_dict(torch.load("/workspace/R2Plus1D_best.pth", map_location=device, weights_only=True))
    _, final_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion, is_train=False, device=device)

    print(f"Final Val Accuracy: {final_acc:.2f}%")
    print("\nPer-class report:")
    present_classes = np.unique(final_labels).astype(int)
    present_names = [full_dataset_train.class_names[i] for i in present_classes]
    print(classification_report(final_labels, final_preds, target_names=present_names, labels=present_classes))

    # --- 8. LEARNING CURVES ---
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title("R2+1D Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title("R2+1D Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/workspace/R2Plus1D_learning_curve_first_run.png", dpi=300)
    print("\nGraph saved as 'R2Plus1D_learning_curve_first_run.png'")

    # --- 9. CONFUSION MATRIX ---
    print("Erstelle Confusion Matrix für R2Plus1D...")
    cm = confusion_matrix(final_labels, final_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=full_dataset_train.class_names, yticklabels=full_dataset_train.class_names)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title('Confusion Matrix - Validation data')
    plt.tight_layout()
    plt.savefig("/workspace/confusion_matrix_R2Plus1D_first_run.png", dpi=300)
    print("Confusion Matrix als 'confusion_matrix_R2Plus1D_first_run.png' gespeichert!")