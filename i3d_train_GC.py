import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import VolleyballDataset
from i3d_model import VolleyballI3DModel
import matplotlib.pyplot as plt
import numpy as np

# --- 1. SETUP & HYPERPARAMETERS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "./extracted_sets"

# I3D needs 16 frames at 112x112 — for Kinetics pretraining
NUM_FRAMES = 16
IMG_SIZE = 112
BATCH_SIZE = 128

# Two-stage training
STAGE1_EPOCHS = 5  # Frozen backbone, train head only
STAGE2_EPOCHS = 20  # Full fine-tuning


#one epoch of training
def run_epoch(model, loader, criterion, is_train, optimizer=None):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)

            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:  # 50% Chance
                        # videos[i] hat die Shape (Frames, Channels, Height, Width)
                        # Width ist die Dimension 3. spiegeln also horizontal
                        videos[i] = torch.flip(videos[i], dims=[3])

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
    # --- 2. DATA PREPARATION ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                             std=[0.22803, 0.22145, 0.216989])
    ])

    full_dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=transform, num_frames=NUM_FRAMES)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    gen = torch.Generator().manual_seed(42)
    train_db, val_db = random_split(full_dataset, [train_size, val_size], generator=gen)

    # increase num_workers for faster data loading!
    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=12, pin_memory=True)

    # --- 3. MODEL ---
    num_classes = len(full_dataset.class_names)
    model = VolleyballI3DModel(num_classes=num_classes, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # Initialize history tracker
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    # --- 4. STAGE 1: Train only the FC head ---
    print(f"\n=== STAGE 1: Training head only for {STAGE1_EPOCHS} epochs ===")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    best_val_acc = 0.0

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, optimizer=optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, is_train=False)

        # Save metrics for current epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[S1 Epoch {epoch + 1}/{STAGE1_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/Uni-LI/MT/i3d_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- 5. STAGE 2: Unfreeze and fine-tune the whole network ---
    print(f"\n=== STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")

    # Load the best Stage 1 weights before fine-tuning
    model.load_state_dict(torch.load("/content/drive/MyDrive/Uni-LI/MT/i3d_best.pth", map_location=device, weights_only=True))
    model.unfreeze_backbone()

    # Lower LR + weight decay for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc, _, _ = run_epoch(model, train_loader, criterion, is_train=True, optimizer=optimizer)
        val_loss, val_acc, val_preds, val_labels = run_epoch(model, val_loader, criterion, is_train=False)

        scheduler.step(val_loss)

        # Save metrics for this epoch
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[S2 Epoch {epoch + 1}/{STAGE2_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "/content/drive/MyDrive/Uni-LI/MT/i3d_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    # --- 6. FINAL EVALUATION with per-class breakdown ---
    print("\n=== FINAL EVALUATION (best checkpoint) ===")
    model.load_state_dict(torch.load("/content/drive/MyDrive/Uni-LI/MT/i3d_best.pth", map_location=device, weights_only=True))
    _, final_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion, is_train=False)

    print(f"Final Val Accuracy: {final_acc:.2f}%")
    print("\nPer-class report:")
    present_classes = np.unique(final_labels).astype(int)
    present_names = [full_dataset.class_names[i] for i in present_classes]
    print(classification_report(final_labels, final_preds, target_names=present_names, labels=present_classes))

    # --- 7. PLOTTING THE LEARNING CURVES ---
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label="Train Accuracy", marker='o')
    plt.plot(epochs_range, history["val_acc"], label="Val Accuracy", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title("I3D Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0,100)

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title("I3D Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/Uni-LI/MT/i3d_learning_curve.png", dpi=300)
    print("\nGraph saved as 'i3d_learning_curve.png'")

    # --- 8. CONFUSION MATRIX ---
    print("Erstelle Confusion Matrix für i3D...")
    cm = confusion_matrix(final_labels, final_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.class_names,
                yticklabels=full_dataset.class_names)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title('Confusion Matrix - Validation data')
    plt.tight_layout()
    plt.savefig("/content/drive/MyDrive/Uni-LI/MT/confusion_matrix_i3D.png", dpi=300)
    print("Confusion Matrix als 'confusion_matrix_i3D.png' gespeichert!")