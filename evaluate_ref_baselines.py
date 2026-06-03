import os
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = r"D:\Master_Dataset_Extracted"
IMG_SIZE = 224
NUM_FRAMES = 60  # Nutzt die vollen 60 Frames der gereinigten Clips
BATCH_SIZE = 8

from dataset import VolleyballDataset


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Confusion Matrix saved as '{filename}'")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Device: {device}")

    # --- 2. DATA PREPARATION (Korrigiert auf ImageNet-Stats für ResNet-18 Konsistenz) ---
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Laden des Datasets
    dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)

    # Match-Gruppen extrahieren für den sauberen Split
    groups = []
    for path in dataset.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    sgkf = StratifiedGroupKFold(n_splits=5)
    train_indices, val_indices = next(sgkf.split(np.zeros(len(dataset)), dataset.labels, groups=groups))

    train_matches = sorted(list(set([groups[i] for i in train_indices])))
    val_matches = sorted(list(set([groups[i] for i in val_indices])))

    print("\n================ DATA LEAKAGE VERIFICATION ================")
    print(f"Training Matches: {train_matches}")
    print(f"Validation Matches: {val_matches}")
    print(f"Schnittmenge (Muss leer sein): {set(train_matches).intersection(set(val_matches))}")
    print("===========================================================\n")

    # --- 3. BASELINE 1 & 2: HEURISTISCHE MODELLE ---
    train_labels = np.array([dataset.labels[i] for i in train_indices])
    val_labels = np.array([dataset.labels[i] for i in val_indices])

    # Bestimme die Mehrheitsklasse im Training (Zero-R)
    counts = Counter(train_labels)
    majority_class = counts.most_common(1)[0][0]

    # 1. Zero-R Vorhersage
    zeror_preds = np.full_like(val_labels, majority_class)
    print("=== 5.1.1 Heuristic Majority-Class Baseline (Zero-R) ===")
    print(f"Mehrheitsklasse im Training: {dataset.class_names[majority_class]}")
    print(f"Validation Accuracy: {accuracy_score(val_labels, zeror_preds) * 100:.2f}%")
    print(classification_report(val_labels, zeror_preds, target_names=dataset.class_names, zero_division=0))
    plot_confusion_matrix(val_labels, zeror_preds, dataset.class_names,
                          'Confusion Matrix - Zero-R Baseline', 'confusion_matrix_zero_r.png')

    # 2. Stratified Distribution Vorhersage
    total_train = len(train_labels)
    class_probs = [counts[cls] / total_train for cls in range(len(dataset.class_names))]

    strat_preds = np.random.choice(len(dataset.class_names), size=len(val_labels), p=class_probs)
    print("\n=== 5.1.2 Heuristic Stratified Distribution Baseline ===")
    print(f"Validation Accuracy: {accuracy_score(val_labels, strat_preds) * 100:.2f}%")
    print(classification_report(val_labels, strat_preds, target_names=dataset.class_names, zero_division=0))
    plot_confusion_matrix(val_labels, strat_preds, dataset.class_names,
                          'Confusion Matrix - Stratified Baseline', 'confusion_matrix_stratified.png')

    # --- 4. FEATURE EXTRACTION FÜR BASELINE 3 & 4 (SPATIAL-ONLY) ---
    print("\nInitialisiere ResNet-18 Feature Extractor...")
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    central_features_all = []
    mean_features_all = []
    labels_all = []

    print("Starte Feature-Extraktion (Zentralbild & Zeit-Mittelwert)...")
    central_idx = NUM_FRAMES // 2  # Frame Index 30 (Zentraler Ballkontakt)

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            # b=Batch (16), f=Frames (60), ch=Channels (3), h=224, w=224
            b, f, ch, h, w = videos.shape
            videos = videos.to(device)

            # --- 1. Extraktion für Baseline 3: Nur das zentrale Frame (Sehr leichtgewichtig) ---
            central_frame = videos[:, central_idx, :, :, :]  # (B, C, H, W)
            c_feat = feature_extractor(central_frame).view(b, -1)  # (B, 512)
            central_features_all.append(c_feat.cpu().numpy())

            # --- 2. Extraktion für Baseline 4: Zeitlicher Mittelwert (Video für Video statt alle auf einmal) ---
            batch_mean_feats = []
            for i in range(b):
                video_frames = videos[i]  # Holt die 60 Frames von Video i -> (60, 3, 224, 224)

                # Jage nur die 60 Frames dieses einzelnen Videos durch das ResNet
                feats = feature_extractor(video_frames).view(f, -1)  # (60, 512)

                # Berechne den zeitlichen Mittelwert für dieses eine Video direkt auf der CPU
                mean_feat = torch.mean(feats, dim=0)  # (512,)
                batch_mean_feats.append(mean_feat.unsqueeze(0))  # (1, 512)

            # Staple die 16 gemittelten Vektoren wieder zum Batch zusammen
            batch_mean_feats = torch.cat(batch_mean_feats, dim=0)  # (B, 512)
            mean_features_all.append(batch_mean_feats.cpu().numpy())

            labels_all.append(labels.numpy())

            if (batch_idx + 1) % 5 == 0:
                print(f"  Fortschritt: Batch {batch_idx + 1}/{len(loader)} erfolgreich im RAM verarbeitet.")

    X_central = np.concatenate(central_features_all, axis=0)
    X_mean = np.concatenate(mean_features_all, axis=0)
    Y_all = np.concatenate(labels_all, axis=0)

    # Splitting der extrahierten Features in Train und Val
    X_central_train, X_central_val = X_central[train_indices], X_central[val_indices]
    X_mean_train, X_mean_val = X_mean[train_indices], X_mean[val_indices]
    Y_train, Y_val = Y_all[train_indices], Y_all[val_indices]

    # --- 5. BASELINE 3 EVALUIERUNG: CENTRAL FRAME ---
    print("\n=== 5.1.3 Spatial-Only Central Frame Classification ===")
    clf_central = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_central.fit(X_central_train, Y_train)
    central_preds = clf_central.predict(X_central_val)

    print(f"Validation Accuracy: {accuracy_score(Y_val, central_preds) * 100:.2f}%")
    print(classification_report(Y_val, central_preds, target_names=dataset.class_names))
    plot_confusion_matrix(Y_val, central_preds, dataset.class_names,
                          'Confusion Matrix - Central Frame Baseline', 'confusion_matrix_central_frame.png')

    # --- 6. BASELINE 4 EVALUIERUNG: MEAN-POOLED FEATURES ---
    print("\n=== 5.1.4 Temporally Aggregated Frame Feature Classification ===")
    clf_mean = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf_mean.fit(X_mean_train, Y_train)
    mean_preds = clf_mean.predict(X_mean_val)

    print(f"Validation Accuracy: {accuracy_score(Y_val, mean_preds) * 100:.2f}%")
    print(classification_report(Y_val, mean_preds, target_names=dataset.class_names))
    plot_confusion_matrix(Y_val, mean_preds, dataset.class_names,
                          'Confusion Matrix - Mean Pooled Baseline', 'confusion_matrix_mean_pooled.png')

    print("\n--- All reference baselines evaluated successfully! ---")