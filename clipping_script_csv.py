import cv2
import os
import csv

# --- KONFIGURATION ---
SSD_PATH = "D:\\"
ANNOTATED_GAMES_DIR = os.path.join(SSD_PATH, "annotated_games")
GLOBAL_OUTPUT_DIR = os.path.join(SSD_PATH, "Master_Dataset_Extracted")
VIDEO_DIR = os.path.join(SSD_PATH, "BeachVolleyballData")

# HIER DAS GEWÜNSCHTE SPIEL EINTRAGEN (Ordnername in "annotated_games")
TARGET_GAME = "BEH-BOC"

TARGET_SIZE = (448, 448)
TARGET_FPS = 30
MAX_FRAMES = 60


def extract_single_csv_game():
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

    if not os.path.exists(ANNOTATED_GAMES_DIR):
        print(f"FEHLER: Ordner mit annotierten Spielen existiert nicht -> {ANNOTATED_GAMES_DIR}")
        return

    # Pfad zum spezifischen Spiel ordner
    folder_path = os.path.join(ANNOTATED_GAMES_DIR, TARGET_GAME)

    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        print(f"FEHLER: Das angegebene Spiel '{TARGET_GAME}' wurde nicht in {ANNOTATED_GAMES_DIR} gefunden.")
        return

    # CSV-Datei im Ordner suchen
    csv_file = None
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            csv_file = file
            break

    if not csv_file:
        print(f"WARNUNG: Keine CSV-Datei im Ordner {TARGET_GAME} gefunden.")
        return

    csv_path = os.path.join(folder_path, csv_file)
    print(f"\n--- Starte Bearbeitung von CSV-Spiel: {TARGET_GAME} ({csv_file}) ---")

    with open(csv_path, mode='r', encoding='cp1252') as file:
        reader = csv.DictReader(file, delimiter=';')

        for row_idx, row in enumerate(reader):
            # SICHERHEITS-CHECK: Falls die Zeile komplett leer ist oder Spalten fehlen
            if not row.get('Dateiname') or not row.get('Start_frame'):
                continue

            video_name = row['Dateiname'].strip()
            start_frame_str = row['Start_frame'].strip()
            label = row['Label'].strip() if row.get('Label') else 'unknown'

            # Falls die Felder Text-Inhalte wie "" nach dem Strippen haben
            if not video_name or not start_frame_str:
                continue

            # SICHERHEITS-CHECK: Kann der Frame in eine Zahl umgewandelt werden?
            try:
                kontakt_frame = int(start_frame_str)
            except ValueError:
                print(f"  Hinweis: Zeile {row_idx + 1} übersprungen (Ungültiger Frame-Wert: '{start_frame_str}')")
                continue

            video_path = os.path.join(VIDEO_DIR, video_name)

            if not os.path.exists(video_path):
                print(f"  FEHLER: Video nicht gefunden -> {video_path}")
                continue

            cap = cv2.VideoCapture(video_path)
            source_fps = round(cap.get(cv2.CAP_PROP_FPS))

            if source_fps == 60:
                actual_start = max(0, kontakt_frame - 60)
                frame_step = 2
            else:
                actual_start = max(0, kontakt_frame - 30)
                frame_step = 1

            label = label.replace(" ", "_")
            label_dir = os.path.join(GLOBAL_OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            prefix = video_name.replace(".mp4", "").replace(" ", "_")
            output_name = f"{prefix}_frame_{kontakt_frame}.mp4"
            output_path = os.path.join(label_dir, output_name)

            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            for _ in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret: break

                out.write(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_CUBIC))

                if frame_step == 2:
                    cap.grab()

            out.release()
            cap.release()
            print(f"  Gespeichert ({row_idx + 1}): {label}/{output_name} [Source FPS: {source_fps}]")

    print(f"\n--- Spiel '{TARGET_GAME}' erfolgreich extrahiert! ---")


if __name__ == "__main__":
    extract_single_csv_game()