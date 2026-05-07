import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from dataset import VolleyballDataset
from baseline_model import VolleyballBaselineModel
import numpy as np

# ==========================================
# --- 1. SETUP & HYPERPARAMETERS ---
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = r"D:\Master_Dataset_Extracted"

# Hardware-Limit der L4 GPU (wegen 60 Frames)
BATCH_SIZE = 4
NUM_WORKERS = 4  # Reduziert für L4, um System-RAM Crashes zu vermeiden

# Two-stage training (wie beim I3D)
STAGE1_EPOCHS = 5  # Frozen backbone, train head only
STAGE2_EPOCHS = 20  # Full fine-tuning
LEARNING_RATE_S1 = 1e-3
LEARNING_RATE_S2 = 1e-4


# ==========================================
# --- 2. HILFSFUNKTION FÜR DEN TRAININGSLOOP ---
# ==========================================
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

            # Data Augmentation (Horizontaler Flip) nur im Training
            if is_train:
                for i in range(videos.size(0)):
                    if torch.rand(1).item() < 0.5:
                        videos[i] = torch.flip(videos[i], dims=[3])

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(videos)
                loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Für Evaluation sammeln
            if not is_train:
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    accuracy = 100 * correct / total

    if not is_train:
        return avg_loss, accuracy, all_preds, all_labels
    return avg_loss, accuracy


# ==========================================
# --- 3. HAUPTPROGRAMM ---
# ==========================================
if __name__ == '__main__':
    # --- DATA PREPARATION ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=transform, num_frames=60)

    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_db, val_db = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # --- MODEL, LOSS, OPTIMIZER ---
    model = VolleyballBaselineModel(num_classes=len(full_dataset.class_names)).to(device)

    # Standard-Loss (ohne Klassengewichte für den ersten Testlauf)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    # ==========================================
    # --- 4. STAGE 1: TRAINING HEAD ONLY ---
    # ==========================================
    print(f"\n=== STAGE 1: Training head only for {STAGE1_EPOCHS} epochs ===")

    # Optimizer kennt nur die Parameter, die nicht eingefroren sind (LSTM & FC)
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE_S1)

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, is_train=True, optimizer=optimizer)
        val_loss, val_acc, _, _ = run_epoch(model, val_loader, criterion, is_train=False)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "baseline_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[S1 Epoch {epoch + 1}/{STAGE1_EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    # ==========================================
    # --- 5. STAGE 2: FULL FINE-TUNING ---
    # ==========================================
    print(f"\n=== STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")

    # WICHTIG: Das beste Modell aus Phase 1 laden, bevor wir das ResNet freigeben
    model.load_state_dict(
        torch.load("baseline_best.pth", map_location=device, weights_only=True))
    model.unfreeze_backbone()

    # Neuer Optimizer mit viel kleinerer Lernrate für das gesamte Netzwerk
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_S2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, is_train=True, optimizer=optimizer)
        val_loss, val_acc, final_preds, final_labels = run_epoch(model, val_loader, criterion, is_train=False)

        # Scheduler schaut, ob der Val-Loss noch sinkt
        scheduler.step(val_loss)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), "baseline_best.pth")
            print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"[S2 Epoch {epoch + 1}/{STAGE2_EPOCHS}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    torch.save(model.state_dict(), "volleyball_model_final.pth")
    print("\nTraining abgeschlossen. Letztes Modell unter 'volleyball_model_final.pth' gespeichert.")

    # ==========================================
    # --- 6. GRAPHEN & EVALUATION ---
    # ==========================================
    print("\nErstelle Lernkurven-Graph...")
    total_epochs = STAGE1_EPOCHS + STAGE2_EPOCHS
    epochs_range = range(1, total_epochs + 1)

    plt.figure(figsize=(12, 5))

    # Plot 1: Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_acc"], label='Training Accuracy', marker='o')
    plt.plot(epochs_range, history["val_acc"], label='Validation Accuracy', marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # Plot 2: Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_loss"], label='Training Loss', marker='o')
    plt.plot(epochs_range, history["val_loss"], label='Validation Loss', marker='o')
    plt.axvline(x=STAGE1_EPOCHS, color='gray', linestyle='--', label='Unfreeze point')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("learning_curve_baseline.png", dpi=300)
    print("Graph erfolgreich als 'learning_curve_baseline.png' gespeichert!")

    # --- FINAL EVALUATION (Best Checkpoint) ---
    print("\nLade bestes Modell für die finale Evaluation...")
    model.load_state_dict(
        torch.load("baseline_best.pth", map_location=device, weights_only=True))
    _, _, final_preds, final_labels = run_epoch(model, val_loader, criterion, is_train=False)

    print("\nPer-class report:")
    present_classes = np.unique(final_labels).astype(int)
    present_names = [full_dataset.class_names[i] for i in present_classes]
    print(classification_report(final_labels, final_preds, target_names=present_names, labels=present_classes))

    print("Erstelle Confusion Matrix für baseline...")
    cm = confusion_matrix(final_labels, final_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=full_dataset.class_names,
                yticklabels=full_dataset.class_names)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title('Confusion Matrix - Validation Data')
    plt.tight_layout()
    plt.savefig("confusion_matrix_baseline.png", dpi=300)
    print("Confusion Matrix als 'confusion_matrix_baseline.png' gespeichert!")