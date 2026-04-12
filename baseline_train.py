import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from dataset import VolleyballDataset
from baseline_model import VolleyballBaselineModel

# --- 1. SETUP & HYPERPARAMETERS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3  # Auf 10 erhöht für den trainingslauf 2 - auf 3 reduziert für testlauf 3
LEARNING_RATE = 0.001
ROOT_DIR = r"D:\Master_Dataset_Extracted"

# --- 2. DATA PREPARATION ---
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

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODEL, LOSS, OPTIMIZER ---
model = VolleyballBaselineModel(num_classes=len(full_dataset.class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- LISTEN FÜR DEN GRAPHEN ---
history_train_loss, history_val_loss = [], []
history_train_acc, history_val_acc = [], []

# --- 4. TRAINING LOOP ---
print(f"Starte Training auf {device} für {EPOCHS} Epochen mit {len(full_dataset)} Clips...")

best_val_acc = 0.0

for epoch in range(EPOCHS):
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        for i in range(videos.size(0)):
            if torch.rand(1).item() < 0.5:  # 50% Wahrscheinlichkeit
                videos[i] = torch.flip(videos[i], dims=[3])

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # VALIDATION PHASE
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for videos, labels in val_loader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    # EPOCHEN-WERTE BERECHNEN UND SPEICHERN
    ep_train_loss = train_loss / len(train_loader)
    ep_val_loss = val_loss / len(val_loader)
    ep_train_acc = 100 * correct_train / total_train
    ep_val_acc = 100 * correct_val / total_val
    if ep_val_acc > best_val_acc:
        best_val_acc = ep_val_acc
        torch.save(model.state_dict(), "baseline_best.pth")
        print(f"  -> Checkpoint saved (best val acc: {best_val_acc:.2f}%)")

    history_train_loss.append(ep_train_loss)
    history_val_loss.append(ep_val_loss)
    history_train_acc.append(ep_train_acc)
    history_val_acc.append(ep_val_acc)

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {ep_train_loss:.4f} | Train Acc: {ep_train_acc:.2f}%")
    print(f"Val Loss:   {ep_val_loss:.4f} | Val Acc:   {ep_val_acc:.2f}%")

# MODELL SPEICHERN
torch.save(model.state_dict(), "volleyball_model_final.pth")
print("\nTraining abgeschlossen. Modell unter 'volleyball_model_final.pth' gespeichert.")

# --- GRAPHEN ZEICHNEN UND SPEICHERN ---
print("Erstelle Lernkurven-Graph...")
epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Plot 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history_train_acc, label='Training Accuracy', marker='o')
plt.plot(epochs_range, history_val_acc, label='Validation Accuracy', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Plot 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history_train_loss, label='Training Loss', marker='o')
plt.plot(epochs_range, history_val_loss, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Graph als Bild speichern
plt.tight_layout()
plt.savefig("learning_curve.png", dpi=300)
print("Graph erfolgreich als 'learning_curve_baseline.png' gespeichert!")

# --- FINAL EVALUATION (Confusion Matrix Daten sammeln) ---
print("\nLade bestes Modell für die Confusion Matrix...")
model.load_state_dict(torch.load("baseline_best.pth", map_location=device, weights_only=True))
model.eval()

final_labels = []
final_preds = []

with torch.no_grad():
    for videos, labels in val_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        _, predicted = torch.max(outputs.data, 1)

        # Die Tensoren von der Grafikkarte holen und in normale Python-Listen packen
        final_labels.extend(labels.cpu().tolist())
        final_preds.extend(predicted.cpu().tolist())

print("Erstelle Confusion Matrix für baseline...")
cm = confusion_matrix(final_labels, final_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=full_dataset.class_names,
            yticklabels=full_dataset.class_names)
plt.ylabel('Tatsächliches Zuspiel')
plt.xlabel('Vorhergesagtes Zuspiel')
plt.title('Confusion Matrix - Validierungsdaten')
plt.tight_layout()
plt.savefig("confusion_matrix_baseline.png", dpi=300)
print("Confusion Matrix als 'confusion_matrix.png' gespeichert!")