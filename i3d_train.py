import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from sklearn.metrics import classification_report
from dataset import VolleyballDataset
from i3d_model import VolleyballI3DModel

# --- 1. SETUP & HYPERPARAMETERS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = r"D:\Master_Dataset_Extracted"

# I3D uses 16 frames at 112x112 — memory-efficient and matches Kinetics pretraining
NUM_FRAMES = 16
IMG_SIZE = 112
BATCH_SIZE = 4

# Two-stage training
STAGE1_EPOCHS = 5   # Frozen backbone, train head only
STAGE2_EPOCHS = 15  # Full fine-tuning

# --- 2. DATA PREPARATION ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645],
                         std=[0.22803, 0.22145, 0.216989])
    # Note: These are the Kinetics-400 stats, correct for R3D pretraining
    # (Different from ImageNet stats used in baseline!)
])

full_dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=transform, num_frames=NUM_FRAMES)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_db, val_db = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# --- 3. MODEL ---
num_classes = len(full_dataset.class_names)
model = VolleyballI3DModel(num_classes=num_classes, freeze_backbone=True).to(device)
criterion = nn.CrossEntropyLoss()

# --- 4. HELPER: one epoch of training ---
def run_epoch(loader, is_train, optimizer=None):
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for videos, labels in loader:
            videos, labels = videos.to(device), labels.to(device)
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


# --- 5. STAGE 1: Train only the FC head ---
print(f"\n=== STAGE 1: Training head only for {STAGE1_EPOCHS} epochs ===")
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

best_val_acc = 0.0

for epoch in range(STAGE1_EPOCHS):
    train_loss, train_acc, _, _ = run_epoch(train_loader, is_train=True, optimizer=optimizer)
    val_loss, val_acc, val_preds, val_labels = run_epoch(val_loader, is_train=False)

    print(f"[S1 Epoch {epoch+1}/{STAGE1_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "i3d_best.pth")
        print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")


# --- 6. STAGE 2: Unfreeze and fine-tune the whole network ---
print(f"\n=== STAGE 2: Full fine-tuning for {STAGE2_EPOCHS} epochs ===")
model.unfreeze_backbone()

# Lower LR + weight decay for fine-tuning
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5, verbose=True)

for epoch in range(STAGE2_EPOCHS):
    train_loss, train_acc, _, _ = run_epoch(train_loader, is_train=True, optimizer=optimizer)
    val_loss, val_acc, val_preds, val_labels = run_epoch(val_loader, is_train=False)

    scheduler.step(val_loss)

    print(f"[S2 Epoch {epoch+1}/{STAGE2_EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "i3d_best.pth")
        print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

# --- 7. FINAL EVALUATION with per-class breakdown ---
print("\n=== FINAL EVALUATION (best checkpoint) ===")
model.load_state_dict(torch.load("i3d_best.pth", map_location=device, weights_only=True))
_, final_acc, final_preds, final_labels = run_epoch(val_loader, is_train=False)

print(f"Final Val Accuracy: {final_acc:.2f}%")
print("\nPer-class report:")
print(classification_report(final_labels, final_preds, target_names=full_dataset.class_names))