import cv2
import xml.etree.ElementTree as ET
import os

# --- KONFIGURATION ---
SSD_PATH = r"D:"
GLOBAL_OUTPUT_DIR = os.path.join(SSD_PATH, "Master_Dataset_Extracted")
VIDEO_DIR = r"D:\BeachVolleyballData"

# Alle 5 Spiele, die als XML vorliegen
GAMES_TO_PROCESS = [
    #("KUN-GRÜ", "2025-07-13 KUN-GRÜ.mp4", "KUN_GRÜ"),
    #("HAR-AHR", "2025-09-05 HAR-AHR.mp4", "HAR_AHR"),
    #("SAG-FROE", "2025-09-05 SAG-FROE.mp4", "SAG_FROE"),
    #("KUH - KIR", "2025-08-01 KUH-KIR.mp4", "KUH_KIR"),
    ("BOR-BEH", "2025-07-11 BOR-BEH.mp4", "BOR_BEH")
]

TARGET_SIZE = (448, 448)
TARGET_FPS = 30
MAX_FRAMES = 60


def extract_all_games():
    if not os.path.exists(GLOBAL_OUTPUT_DIR):
        os.makedirs(GLOBAL_OUTPUT_DIR)
        print(f"Zentraler Ordner erstellt: {GLOBAL_OUTPUT_DIR}")

    for xml_folder, video_name, game_prefix in GAMES_TO_PROCESS:
        print(f"\n--- Starte Bearbeitung von XML-Spiel: {game_prefix} ---")

        video_path = os.path.join(VIDEO_DIR, video_name)
        xml_path = os.path.join(SSD_PATH, "annotated_games", xml_folder, "annotations.xml")

        # NEUER FALLBACK: Schau zuerst nach, ob das Video direkt im spezifischen Spielordner liegt (z.B. bei BOR-BEH)
        local_video_path = os.path.join(SSD_PATH, "annotated_games", xml_folder, video_name)
        if os.path.exists(local_video_path):
            video_path = local_video_path
            print(f"-> Video direkt im lokalen Spielordner gefunden: {video_path}")

        # Alter Fallback: Suche im Haupt-Video-Ordner, falls oben nichts gefunden wurde
        elif not os.path.exists(video_path) and os.path.exists(VIDEO_DIR):
            for f in os.listdir(VIDEO_DIR):
                if xml_folder.replace(" ", "").lower() in f.replace(" ", "").lower() and f.endswith(".mp4"):
                    video_path = os.path.join(VIDEO_DIR, f)
                    print(f"-> Video automatisch im Hauptordner korrigiert zu: {f}")
                    break

        if not os.path.exists(video_path) or not os.path.exists(xml_path):
            print(f"WARNUNG: Video oder XML fehlt für {game_prefix}. Überspringe...")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        cap = cv2.VideoCapture(video_path)

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0: orig_fps = 30

        source_fps = round(orig_fps)
        frame_step = 2 if source_fps == 60 else 1

        for track in root.findall('track'):
            label = track.get('label', 'unknown')
            track_id = track.get('id', '0')

            first_box = track.find('box')
            if first_box is None: continue

            attr = first_box.find(".//attribute[@name='type']")
            if attr is not None:
                label = attr.text

            label = label.replace(" ", "_")
            label_dir = os.path.join(GLOBAL_OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            start_frame = int(first_box.get('frame'))
            offset = int(orig_fps * 0.5)
            actual_start = max(0, start_frame - offset)

            output_name = f"{game_prefix}_set_{track_id}.mp4"
            output_path = os.path.join(label_dir, output_name)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

            for _ in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret: break
                out.write(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_CUBIC))

                if frame_step == 2:
                    cap.grab()

            out.release()
            print(f"  Saved: {label}/{output_name} [Source FPS: {source_fps}]")

        cap.release()

    print("\n--- Alle XML-Spiele erfolgreich zusammengeführt! ---")


if __name__ == "__main__":
    extract_all_games()