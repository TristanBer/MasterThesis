import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, confusion_matrix
from dataset import VolleyballDataset
from bootstrap_CI import bootstrap_metrics_ci, print_ci_report, plot_ci_results

# ============================================================
# CONFIG  —  edit this block only
# ============================================================
ROOT_DIR = "/workspace/Master_Dataset_Extracted"
MODEL_KEY = "i3d"   # one of: baseline | i3d | r2plus1d | x3d | videoswin
N_SPLITS = 5        # MUST match the value used in your training script
N_BOOTSTRAPS = 10000
CI_LEVEL = 95

# Per-model settings: input geometry, normalisation, checkpoint path, loader.
# These mirror the values hard-coded in each *_train.py file. Verify each
# checkpoint path points to the file you actually saved during training.
MODEL_CONFIGS = {
    "i3d": {
        "num_frames": 16,
        "img_size": 112,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "ckpt": "/workspace/i3d_best.pth",
    },
    "baseline": {
        "num_frames": 60,
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "ckpt": "/workspace/baseline_best.pth",
    },
    "r2plus1d": {
        "num_frames": 16,
        "img_size": 112,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "ckpt": "/workspace/R2Plus1D_best.pth",
    },
    "x3d": {
        "num_frames": 16,
        "img_size": 112,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "ckpt": ".workspace/X3D_best.pth",
    },
    "videoswin": {
        "num_frames": 16,
        "img_size": 112,
        "mean": [0.43216, 0.394666, 0.37645],
        "std": [0.22803, 0.22145, 0.216989],
        "ckpt": "/workspace/VideoSwin_best.pth",
    },
}
# ============================================================


def build_model(model_key, num_classes, device):
    """Instantiate the requested architecture with eval-time settings."""
    if model_key == "i3d":
        from i3d_model import VolleyballI3DModel
        model = VolleyballI3DModel(num_classes=num_classes, freeze_backbone=False, dropout_p=0.5)
    elif model_key == "CNNBiLSTM":
        from CNN_BiLSTM_model import VolleyballCNNBiLSTMModel
        model = VolleyballCNNBiLSTMModel(num_classes=num_classes, dropout_p=0.5)
    elif model_key == "R(2+1)D":
        from R2Plus1D_model import VolleyballR2Plus1DModel
        model = VolleyballR2Plus1DModel(num_classes=num_classes)
    elif model_key == "x3d":
        from X3D_model import VolleyballX3DModel
        model = VolleyballX3DModel(num_classes=num_classes)
    elif model_key == "videoswin":
        from VideoSwin_model import VolleyballVideoSwinModel
        model = VolleyballVideoSwinModel(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown MODEL_KEY: {model_key}")
    return model.to(device)


@torch.no_grad()
def collect_predictions(model, loader, device):
    """Single forward pass over the validation set. Returns preds, labels."""
    model.eval()
    all_preds, all_labels = [], []
    for videos, labels in loader:
        videos, labels = videos.to(device), labels.to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(videos)
        else:
            outputs = model(videos)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
    return all_preds, all_labels


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MODEL_CONFIGS[MODEL_KEY]
    print(f"Evaluating model '{MODEL_KEY}' on device: {device}")

    # --- Validation transform: deterministic, identical to training script ---
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(cfg["img_size"] * 1.15)),
        transforms.CenterCrop(cfg["img_size"]),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"]),
    ])

    full_dataset_val = VolleyballDataset(
        root_dir=ROOT_DIR, transform=val_transform, num_frames=cfg["num_frames"]
    )

    # --- Reproduce the SAME split used in training ---
    groups = []
    for path in full_dataset_val.video_files:
        filename = os.path.basename(path)
        match_prefix = filename.split("_set_")[0].split("_frame_")[0]
        groups.append(match_prefix)

    sgkf = StratifiedGroupKFold(n_splits=N_SPLITS)
    _, val_indices = next(sgkf.split(
        X=np.zeros(len(full_dataset_val)),
        y=full_dataset_val.labels,
        groups=groups,
    ))

    val_matches = sorted({groups[i] for i in val_indices})
    print(f"\nValidation matches ({len(val_matches)}): {val_matches}")
    print(f"Validation samples: {len(val_indices)}\n")

    val_db = Subset(full_dataset_val, val_indices)
    val_loader = DataLoader(val_db, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    # --- Load checkpoint (no training) ---
    num_classes = len(full_dataset_val.class_names)
    model = build_model(MODEL_KEY, num_classes, device)
    if not os.path.exists(cfg["ckpt"]):
        raise FileNotFoundError(
            f"Checkpoint not found: {cfg['ckpt']}. "
            "Update the path in MODEL_CONFIGS to your saved .pth file."
        )
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=device, weights_only=True), strict=False)
    print(f"Loaded checkpoint: {cfg['ckpt']}")

    # --- Forward pass + point-estimate report ---
    final_preds, final_labels = collect_predictions(model, val_loader, device)
    final_labels_arr = np.asarray(final_labels)
    present_classes = np.unique(final_labels_arr).astype(int)
    present_names = [full_dataset_val.class_names[i] for i in present_classes]

    acc = 100 * np.mean(np.asarray(final_preds) == final_labels_arr)
    print(f"\nPoint-estimate validation accuracy: {acc:.2f}%")
    print("\nPer-class report:")
    print(classification_report(final_labels, final_preds,
                                target_names=present_names, labels=present_classes))

    # --- Bootstrap confidence intervals ---
    ci_results = bootstrap_metrics_ci(
        y_true=final_labels,
        y_pred=final_preds,
        class_names=full_dataset_val.class_names,
        n_bootstraps=N_BOOTSTRAPS,
        ci=CI_LEVEL,
    )
    print_ci_report(ci_results, ci=CI_LEVEL)

    plot_ci_results(
        ci_results,
        model_name=MODEL_KEY.upper(),
        ci=CI_LEVEL,
        save_path=f"./{MODEL_KEY}_ci"  # saves i3d_ci_overall.png + i3d_ci_perclass.png
    )

if __name__ == "__main__":
    main()