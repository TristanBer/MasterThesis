import torch
import cv2
import os
from torchvision import transforms
from baseline_model import VolleyballBaselineModel

# --- 1. SETUP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "volleyball_model_final.pth"

# HIER EINFACH DEN PFAD ZU EINEM BELIEBIGEN CLIP EINTRAGEN
VIDEO_TO_TEST = r"D:\Master_Dataset_Extracted\overhead_set_forward\SAG_FROE_set_7.mp4"

# WICHTIG: Diese Liste muss exakt der alphabetischen Reihenfolge deiner Ordner entsprechen!
# Schau in deinen Master_Dataset_Extracted Ordner und trage sie hier ein:
CLASS_NAMES = [
    'Others',
    'bump_set_backward',
    'bump_set_forward',
    'overhead_set_backward',
    'overhead_set_forward',
    ]

def predict_video(video_path):
    print(f"Lade Modell von: {MODEL_PATH}...")
    model = VolleyballBaselineModel(num_classes=len(CLASS_NAMES)).to(device)

    # weights_only=True ist ein Sicherheitsfeature in neueren PyTorch-Versionen
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()  # Modell in den Vorhersage-Modus schalten (wichtig!)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print(f"Analysiere Video: {video_path}...")
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(transform(frame))

    cap.release()

    if len(frames) == 0:
        print("Fehler: Video konnte nicht gelesen werden.")
        return

    # Tensor bauen: Aus (60, 3, 224, 224) wird (1, 60, 3, 224, 224)
    # Die "1" steht für die Batch-Size (wir testen ja nur 1 Video)
    video_tensor = torch.stack(frames).unsqueeze(0).to(device)

    # Vorhersage machen
    with torch.no_grad():
        outputs = model(video_tensor)
        # Softmax wandelt die rohen Zahlen in Prozent-Wahrscheinlichkeiten um
        probabilities = torch.nn.functional.softmax(outputs, dim=1)

        predicted_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_idx].item() * 100

    print("\n--- ERGEBNIS ---")
    print(f"Vorhersage: {CLASS_NAMES[predicted_idx]}")
    print(f"Sicherheit: {confidence:.2f}%")

    # Optional: Alle Wahrscheinlichkeiten anzeigen
    print("\nDetails:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {probabilities[0][i].item() * 100:.2f}%")


if __name__ == "__main__":
    predict_video(VIDEO_TO_TEST)