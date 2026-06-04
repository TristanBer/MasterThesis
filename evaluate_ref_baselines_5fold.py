import os
import json
import csv
import datetime
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from collections import Counter
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import VolleyballDataset

# --- 1. SETUP & HYPERPARAMETERS ---
ROOT_DIR = "/workspace/Master_Dataset_Extracted"
RESULTS_DIR = "/workspace/results_baselines_5fold"
IMG_SIZE = 224
NUM_FRAMES = 60
BATCH_SIZE = 8
N_FOLDS = 5

BASELINE_MODELS = [
    "Zero_R",
    "Stratified",
    "Central_Frame",
    "Mean_Pooled"
]


def plot_confusion_matrix(y_true, y_pred, class_names, title, filename):
    """Generates and saves a confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual setting action')
    plt.xlabel('Predicted setting action')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Nutze Device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"All baseline results will be written to: {RESULTS_DIR}")

    # --- 2. DATA PREPARATION ---
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(IMG_SIZE * 1.15)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)
    class_names = dataset.class_names
    label_ids = list(range(len(class_names)))

    groups = []
    for path in dataset.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split('_set_')[0].split('_frame_')[0]
        groups.append(match_prefix)

    # --- 3. GLOBAL FEATURE EXTRACTION (Runs Once) ---
    print("\nInitialisiere ResNet-18 Feature Extractor (ImageNet pre-trained)...")
    resnet = models.resnet18(weights='IMAGENET1K_V1')
    feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    feature_extractor.eval()

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    central_features_all = []
    mean_features_all = []
    labels_all = []

    print("Starte globale Feature-Extraktion (Zentralbild & Zeit-Mittelwert)...")
    central_idx = NUM_FRAMES // 2

    with torch.no_grad():
        for batch_idx, (videos, labels) in enumerate(loader):
            b, f, ch, h, w = videos.shape
            videos = videos.to(device)

            # Central Frame Extraction
            central_frame = videos[:, central_idx, :, :, :]
            c_feat = feature_extractor(central_frame).view(b, -1)
            central_features_all.append(c_feat.cpu().numpy())

            # Mean Pooled Extraction
            batch_mean_feats = []
            for i in range(b):
                video_frames = videos[i]
                feats = feature_extractor(video_frames).view(f, -1)
                mean_feat = torch.mean(feats, dim=0)
                batch_mean_feats.append(mean_feat.unsqueeze(0))

            batch_mean_feats = torch.cat(batch_mean_feats, dim=0)
            mean_features_all.append(batch_mean_feats.cpu().numpy())
            labels_all.append(labels.numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f"  Fortschritt: Batch {batch_idx + 1}/{len(loader)} verarbeitet.")

    X_central = np.concatenate(central_features_all, axis=0)
    X_mean = np.concatenate(mean_features_all, axis=0)
    Y_all = np.concatenate(labels_all, axis=0)

    print("\nFeature-Extraktion abgeschlossen. Starte 5-Fold Cross-Validation...")

    # --- 4. 5-FOLD CROSS VALIDATION SETUP ---
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS)

    # Data structures to hold results across all folds and baselines
    results = {
        model: {
            "fold_accuracies": [],
            "fold_macro_f1": [],
            "fold_weighted_f1": [],
            "oof_preds": np.zeros_like(Y_all),
            "oof_labels": np.zeros_like(Y_all)
        } for model in BASELINE_MODELS
    }

    # --- 5. 5-FOLD LOOP ---
    for fold_idx, (train_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(dataset)), Y_all, groups=groups)):
        print(f"\n############## FOLD {fold_idx + 1}/{N_FOLDS} ##############")

        train_matches = sorted(list(set([groups[i] for i in train_idx])))
        val_matches = sorted(list(set([groups[i] for i in val_idx])))
        print(f"Train Matches ({len(train_matches)}): {train_matches}")
        print(f"Val Matches ({len(val_matches)}): {val_matches}")

        # Split data
        Y_train, Y_val = Y_all[train_idx], Y_all[val_idx]
        X_c_train, X_c_val = X_central[train_idx], X_central[val_idx]
        X_m_train, X_m_val = X_mean[train_idx], X_mean[val_idx]

        fold_preds = {}

        # 5.1 Baseline 1: Zero-R
        majority_class = Counter(Y_train).most_common(1)[0][0]
        fold_preds["Zero_R"] = np.full_like(Y_val, majority_class)

        # 5.2 Baseline 2: Stratified Distribution
        counts = Counter(Y_train)
        class_probs = [counts.get(cls, 0) / len(Y_train) for cls in range(len(class_names))]
        fold_preds["Stratified"] = np.random.choice(len(class_names), size=len(Y_val), p=class_probs)

        # 5.3 Baseline 3: Central Frame Logistic Regression
        clf_central = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf_central.fit(X_c_train, Y_train)
        fold_preds["Central_Frame"] = clf_central.predict(X_c_val)

        # 5.4 Baseline 4: Mean-Pooled Logistic Regression
        clf_mean = LogisticRegression(max_iter=1000, class_weight='balanced')
        clf_mean.fit(X_m_train, Y_train)
        fold_preds["Mean_Pooled"] = clf_mean.predict(X_m_val)

        # Evaluate and Store Fold Results
        for model in BASELINE_MODELS:
            preds = fold_preds[model]
            acc = accuracy_score(Y_val, preds) * 100
            macro = f1_score(Y_val, preds, average='macro', labels=label_ids, zero_division=0) * 100
            weighted = f1_score(Y_val, preds, average='weighted', labels=label_ids, zero_division=0) * 100

            results[model]["fold_accuracies"].append(acc)
            results[model]["fold_macro_f1"].append(macro)
            results[model]["fold_weighted_f1"].append(weighted)

            # Map predictions back to original global indices for accurate OOF tracking
            results[model]["oof_preds"][val_idx] = preds
            results[model]["oof_labels"][val_idx] = Y_val

            print(f"  [{model}] Fold {fold_idx + 1} Acc: {acc:.2f}% | Macro F1: {macro:.2f}%")

    # --- 6. AGGREGATE POOLED OOF METRICS & EXPORT ---
    print("\n\n========= BASELINES 5-FOLD CROSS-VALIDATION SUMMARY =========")

    summary_payload = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_folds": N_FOLDS,
        "class_names": class_names,
        "models": {}
    }

    for model in BASELINE_MODELS:
        print(f"\n--- {model.upper()} ---")
        fold_accs = np.array(results[model]["fold_accuracies"])
        fold_macros = np.array(results[model]["fold_macro_f1"])
        fold_weighteds = np.array(results[model]["fold_weighted_f1"])

        oof_preds = results[model]["oof_preds"]
        oof_labels = results[model]["oof_labels"]

        oof_acc = accuracy_score(oof_labels, oof_preds) * 100
        oof_macro = f1_score(oof_labels, oof_preds, average='macro', labels=label_ids, zero_division=0) * 100
        oof_weighted = f1_score(oof_labels, oof_preds, average='weighted', labels=label_ids, zero_division=0) * 100

        print(f"Mean Fold Accuracy: {fold_accs.mean():.2f}% +/- {fold_accs.std():.2f}")
        print(f"Pooled OOF Accuracy: {oof_acc:.2f}% | Macro F1: {oof_macro:.2f}%")

        oof_report_dict = classification_report(oof_labels, oof_preds, target_names=class_names,
                                                labels=label_ids, zero_division=0, output_dict=True)
        oof_cm = confusion_matrix(oof_labels, oof_preds, labels=label_ids)

        # Populate JSON Dictionary
        summary_payload["models"][model] = {
            "cross_fold_mean_std": {
                "accuracy_mean": float(fold_accs.mean()), "accuracy_std": float(fold_accs.std()),
                "macro_f1_mean": float(fold_macros.mean()), "macro_f1_std": float(fold_macros.std()),
                "weighted_f1_mean": float(fold_weighteds.mean()), "weighted_f1_std": float(fold_weighteds.std()),
            },
            "pooled_oof": {
                "accuracy": oof_acc,
                "macro_f1": oof_macro,
                "weighted_f1": oof_weighted,
                "classification_report": oof_report_dict,
                "confusion_matrix": oof_cm.tolist(),
            }
        }

        # Save Plots
        plot_confusion_matrix(oof_labels, oof_preds, class_names,
                              f'Pooled OOF Confusion Matrix - {model}',
                              os.path.join(RESULTS_DIR, f'confusion_matrix_oof_{model}.png'))

        # Save Model-Specific CSVs
        # 1. Per-class metrics
        with open(os.path.join(RESULTS_DIR, f"{model}_oof_per_class_metrics.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["class", "precision", "recall", "f1_score", "support"])
            for name in class_names:
                row = oof_report_dict.get(name, {})
                w.writerow([name, row.get("precision", 0), row.get("recall", 0),
                            row.get("f1-score", 0), row.get("support", 0)])

        # 2. Confusion Matrix
        with open(os.path.join(RESULTS_DIR, f"{model}_oof_confusion_matrix.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["actual\\predicted"] + class_names)
            for r, name in enumerate(class_names):
                w.writerow([name] + oof_cm[r].tolist())

        # 3. Raw Predictions
        with open(os.path.join(RESULTS_DIR, f"{model}_oof_predictions.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "true_label_id", "true_label_name", "pred_label_id", "pred_label_name"])
            for i, (t, p) in enumerate(zip(oof_labels, oof_preds)):
                w.writerow([i, t, class_names[t], p, class_names[p]])

    # Export Master JSON Summary
    with open(os.path.join(RESULTS_DIR, "SUMMARY_baselines_5fold.json"), "w") as f:
        json.dump(summary_payload, f, indent=2)

    # Export Cross-Model Fold Summary CSV
    with open(os.path.join(RESULTS_DIR, "summary_per_fold_all_baselines.csv"), "w", newline="") as f:
        w = csv.writer(f)
        headers = ["model", "fold", "accuracy", "macro_f1", "weighted_f1"]
        w.writerow(headers)

        for model in BASELINE_MODELS:
            fold_accs = results[model]["fold_accuracies"]
            fold_macros = results[model]["fold_macro_f1"]
            fold_weights = results[model]["fold_weighted_f1"]

            for i in range(N_FOLDS):
                w.writerow([model, i + 1, f"{fold_accs[i]:.2f}", f"{fold_macros[i]:.2f}", f"{fold_weights[i]:.2f}"])

            # Add a summary block per model at the bottom of the folds
            w.writerow([model, "mean", f"{np.mean(fold_accs):.2f}", f"{np.mean(fold_macros):.2f}",
                        f"{np.mean(fold_weights):.2f}"])
            w.writerow(
                [model, "std", f"{np.std(fold_accs):.2f}", f"{np.std(fold_macros):.2f}", f"{np.std(fold_weights):.2f}"])
            w.writerow([])  # blank line separator

    print(f"\nAll baseline metrics, reports, matrices, and raw predictions saved to: {RESULTS_DIR}")
    print("Key files: SUMMARY_baselines_5fold.json, summary_per_fold_all_baselines.csv")