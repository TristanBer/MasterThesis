import cv2
import xml.etree.ElementTree as ET
import os

# --- KONFIGURATION ---
SSD_PATH = r"D:"\
# Der zentrale Ordner für alle deine Clips (über alle Spiele hinweg)
GLOBAL_OUTPUT_DIR = os.path.join(SSD_PATH, "Master_Dataset_Extracted")

# Hier fügst du einfach neue Spiele hinzu, sobald du die XMLs hast
# Struktur: (Ordnername_der_XML, Video_Dateiname, Spiel_Kürzel_für_Datei)
GAMES_TO_PROCESS = [
    ("BOR - BEH", "2025-07-11 BOR-BEH.mp4", "BOR_BEH"),
    # ("KUH - KIR", "2025-08-01 KUH-KIR.mp4", "KUH_KIR"), # Beispiel für später
]

TARGET_SIZE = (448, 448)
TARGET_FPS = 30
MAX_FRAMES = 60


def extract_all_games():
    # Ordner erstellen falls er noch nicht existiert
    if not os.path.exists(GLOBAL_OUTPUT_DIR):
        os.makedirs(GLOBAL_OUTPUT_DIR)
        print(f"Zentraler Ordner erstellt: {GLOBAL_OUTPUT_DIR}")

    for xml_folder, video_name, game_prefix in GAMES_TO_PROCESS:
        print(f"\n--- Starte Bearbeitung von: {game_prefix} ---")

        video_path = os.path.join(SSD_PATH, xml_folder, video_name)
        xml_path = os.path.join(SSD_PATH, xml_folder, "annotations.xml")

        if not os.path.exists(video_path) or not os.path.exists(xml_path):
            print(f"WARNUNG: Video oder XML fehlt für {game_prefix}. Überspringe...")
            continue

        tree = ET.parse(xml_path)
        root = tree.getroot()
        cap = cv2.VideoCapture(video_path)
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        if orig_fps == 0: orig_fps = 30

        for track in root.findall('track'):
            label = track.get('label', 'unknown')
            track_id = track.get('id', '0')

            first_box = track.find('box')
            if first_box is None: continue

            attr = first_box.find(".//attribute[@name='type']")
            if attr is not None:
                label = attr.text

            # Ordnerstruktur im globalen Verzeichnis
            label = label.replace(" ", "_")
            label_dir = os.path.join(GLOBAL_OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            # Zeitlicher Offset (0.5s Vorlauf)
            start_frame = int(first_box.get('frame'))
            offset = int(orig_fps * 0.5)
            actual_start = max(0, start_frame - offset)

            # Eindeutiger Dateiname: SPIEL_ID.mp4
            output_name = f"{game_prefix}_set_{track_id}.mp4"
            output_path = os.path.join(label_dir, output_name)

            # Codec & Writer (mp4v für maximale Stabilität)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

            for _ in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret: break
                out.write(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_CUBIC))

            out.release()
            print(f"  Saved: {label}/{output_name}")

        cap.release()

    print("\n--- Alle Spiele erfolgreich im Master-Dataset zusammengeführt! ---")


if __name__ == "__main__":
    extract_all_games()