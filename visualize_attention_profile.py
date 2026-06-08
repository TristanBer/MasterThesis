import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedGroupKFold
import os

from dataset import VolleyballDataset
from Temporal_Attention_model import R2Plus1DTemporalAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ROOT_DIR = "/workspace/Master_Dataset_Extracted"
CHECKPOINT_DIR = "/workspace/results_TemporalAttn_5fold"
OUTPUT_DIR = "/workspace/results_TemporalAttn_5fold"
NUM_FRAMES = 32
IMG_SIZE = 224
BATCH_SIZE = 16

val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(int(IMG_SIZE * 1.15)),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989])
])

full_dataset = VolleyballDataset(root_dir=ROOT_DIR, transform=val_transform, num_frames=NUM_FRAMES)
class_names = full_dataset.class_names

groups = []
for path in full_dataset.video_files:
    filename = os.path.basename(path)
    match_prefix = filename.split("_set_")[0].split("_frame_")[0]
    groups.append(match_prefix)

sgkf = StratifiedGroupKFold(n_splits=5)
global_class_attentions = {i: [] for i in range(5)}

fold_num = -1
for train_idx, val_idx in sgkf.split(np.zeros(len(full_dataset)), full_dataset.labels, groups):
    fold_num += 1

    checkpoint_path = f"{CHECKPOINT_DIR}/TemporalAttn_best_fold{fold_num}.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Fold {fold_num}: not found at {checkpoint_path}, skipping.")
        continue

    model = R2Plus1DTemporalAttention(num_classes=5, freeze_backbone=False).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()

    val_subset = Subset(full_dataset, val_idx)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    class_attentions = {i: [] for i in range(5)}

    with torch.no_grad():
        for videos, labels in val_loader:
            videos = videos.to(device)
            _, attn_weights = model(videos, return_attention=True)
            for b in range(videos.size(0)):
                class_idx = labels[b].item()
                attn = attn_weights[b, 0, :].cpu().numpy()
                class_attentions[class_idx].append(attn)
                global_class_attentions[class_idx].append(attn)

    # Per-fold plot
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
    fig.suptitle(f"Temporal Attention Profiles - Fold {fold_num}", fontsize=13)

    for i, name in enumerate(class_names):
        if class_attentions[i]:
            all_attn = np.array(class_attentions[i])
            mean_attn = np.mean(all_attn, axis=0)
            std_attn = np.std(all_attn, axis=0)
            x = range(len(mean_attn))
            axes[i].plot(x, mean_attn, linewidth=2, color="blue")
            axes[i].fill_between(x, mean_attn - std_attn, mean_attn + std_attn, alpha=0.2, color="blue")
            axes[i].set_title(f"{name}\n(n={len(class_attentions[i])})", fontsize=9)
            axes[i].set_xlabel("Temporal position")
            axes[i].set_ylabel("Attention weight")
            axes[i].set_ylim(0, max(mean_attn.max() * 1.3, 0.01))
        else:
            axes[i].text(0.5, 0.5, "No samples", ha="center", va="center", transform=axes[i].transAxes)
            axes[i].set_title(f"{name}\n(n=0)", fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/attention_profiles_fold{fold_num}.png", dpi=300)
    plt.close()
    print(f"Fold {fold_num} saved")

# Global average across all folds
fig, axes = plt.subplots(1, 5, figsize=(18, 3.5))
fig.suptitle("Temporal Attention Profiles - Averaged Across All Folds", fontsize=13)

for i, name in enumerate(class_names):
    if global_class_attentions[i]:
        all_attn = np.array(global_class_attentions[i])
        mean_attn = np.mean(all_attn, axis=0)
        std_attn = np.std(all_attn, axis=0)
        x = range(len(mean_attn))
        axes[i].plot(x, mean_attn, linewidth=2, color="darkblue")
        axes[i].fill_between(x, mean_attn - std_attn, mean_attn + std_attn, alpha=0.2, color="blue")
        axes[i].set_title(f"{name}\n(n={len(global_class_attentions[i])})", fontsize=9)
        axes[i].set_xlabel("Temporal position")
        axes[i].set_ylabel("Attention weight")
        axes[i].set_ylim(0, max(mean_attn.max() * 1.3, 0.01))
    else:
        axes[i].text(0.5, 0.5, "No samples", ha="center", va="center", transform=axes[i].transAxes)
        axes[i].set_title(f"{name}\n(n=0)", fontsize=9)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/attention_profiles_global.png", dpi=300)
plt.close()
print("\nGlobal visualization saved")
print("All visualizations complete!")