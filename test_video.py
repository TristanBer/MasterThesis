import torch
import cv2
import numpy as np
from torchvision import transforms
from baseline_model import VolleyballBaselineModel
from i3d_model import VolleyballI3DModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = [
    'Others',
    'bump_set_backward',
    'bump_set_forward',
    'overhead_set_backward',
    'overhead_set_forward',
]

# --- CHANGE THESE TWO LINES TO SWITCH MODELS ---
MODEL_TYPE = "i3d"   # "baseline" or "i3d"
VIDEO_TO_TEST = r"D:\Master_Dataset_Extracted\overhead_set_forward\SAG_FROE_set_7.mp4"
# ------------------------------------------------

CONFIGS = {
    "baseline": {
        "model_path": "volleyball_model_final.pth",
        "num_frames": 60,
        "img_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std":  [0.229, 0.224, 0.225],
    },
    "i3d": {
        "model_path": "i3d_best.pth",
        "num_frames": 16,
        "img_size": 112,
        "mean": [0.43216, 0.394666, 0.37645],
        "std":  [0.22803, 0.22145, 0.216989],
    }
}

def predict_video(video_path, model_type):
    cfg = CONFIGS[model_type]

    # Load correct model
    if model_type == "baseline":
        model = VolleyballBaselineModel(num_classes=len(CLASS_NAMES)).to(device)
    else:
        model = VolleyballI3DModel(num_classes=len(CLASS_NAMES)).to(device)

    model.load_state_dict(torch.load(cfg["model_path"], map_location=device, weights_only=True))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((cfg["img_size"], cfg["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, cfg["num_frames"], dtype=int)

    frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(transform(frame))
    cap.release()

    if len(frames) == 0:
        print("Error: video could not be read.")
        return

    video_tensor = torch.stack(frames).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][predicted_idx].item() * 100

    print(f"\n--- RESULT ({model_type.upper()}) ---")
    print(f"Prediction : {CLASS_NAMES[predicted_idx]}")
    print(f"Confidence : {confidence:.2f}%")
    print("\nAll probabilities:")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"  {cls}: {probs[0][i].item() * 100:.2f}%")


if __name__ == "__main__":
    predict_video(VIDEO_TO_TEST, MODEL_TYPE)