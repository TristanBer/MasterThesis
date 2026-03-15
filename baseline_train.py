import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from dataset import VolleyballDataset
from baseline_model import VolleyballBaselineModel

# --- 1. SETUP & HYPERPARAMETERS ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 4
EPOCHS = 3
LEARNING_RATE = 0.001
ROOT_DIR = r"D:\Master_Dataset_Extracted"

# --- 2. DATA PREPARATION ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

full_dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=transform)

# Split: 80% Training, 20% Validierung
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_db, val_db = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_db, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_db, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. MODEL, LOSS, OPTIMIZER ---
model = VolleyballBaselineModel(num_classes=len(full_dataset.class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 4. TRAINING LOOP ---
print(f"Starte Training auf {device} für {EPOCHS} Epochen...")

for epoch in range(EPOCHS):
    # TRAINING PHASE
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)

        outputs = model(videos)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    # VALIDATION PHASE (Kein Gradienten-Update)
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

    # STATISTIKEN AUSGEBEN
    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {100 * correct_train / total_train:.2f}%")
    print(f"Val Loss:   {val_loss / len(val_loader):.4f} | Val Acc:   {100 * correct_val / total_val:.2f}%")

# MODELL SPEICHERN
torch.save(model.state_dict(), "volleyball_model_final.pth")
print("\nTraining abgeschlossen. Modell unter 'volleyball_model_final.pth' gespeichert.")