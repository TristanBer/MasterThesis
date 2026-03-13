import torch
from torch.utils.data import Dataset
import cv2
import os
import numpy as np
from torchvision import transforms

class VolleyballDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: Pfad zu 'Master_Dataset_Extracted'
        transform: PyTorch Transformationen (Normalisierung, etc.)
        """
        self.root_dir = root_dir
        self.transform = transform
        self.video_files = []
        self.labels = []

        # Mapping erstellen (alphabetisch sortiert)
        # 0: bump_set_backward, 1: bump_set_forward, etc.
        self.class_names = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.class_names)}

        for label_name, label_idx in self.class_to_idx.items():
            dir_path = os.path.join(root_dir, label_name)
            for vid in os.listdir(dir_path):
                if vid.endswith(('.mp4', '.avi')):
                    self.video_files.append(os.path.join(dir_path, vid))
                    self.labels.append(label_idx)

        print(f"Dataset geladen: {len(self.video_files)} Videos in {len(self.class_names)} Klassen.")

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video_path = self.video_files[idx]
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)
        frames = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Konvertiere von BGR (OpenCV) zu RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # WICHTIG: ResNet erwartet Bilder im Bereich [0, 1] und normalisiert
            if self.transform:
                frame = self.transform(frame)
            else:
                # Fallback: Einfache Konvertierung zu Tensor
                frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0

            frames.append(frame)

        cap.release()

        # Stack zu (Frames, Channels, Height, Width)
        # Beispiel: (60, 3, 224, 224)
        video_tensor = torch.stack(frames)

        return video_tensor, label

if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    # Das sind die Standard-Werte für ResNet
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Pfad zu deinem neuen Master-Ordner auf der SSD
    root = r"D:\Master_Dataset_Extracted"

    # Dataset instanziieren
    dataset = VolleyballDataset(root_dir=root, transform=transform)

    # Den ersten Clip laden
    video_tensor, label = dataset[0]

    print(f"Shape des Video-Tensors: {video_tensor.shape}")
    # Erwartet: [60, 3, 224, 224] -> (Frames, Channels, Height, Width)

    print(f"Label des Clips: {label} (Klasse: {dataset.class_names[label]})")