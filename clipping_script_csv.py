import cv2
import os
import csv

# --- KONFIGURATION ---
SSD_PATH = "D:\\"
GLOBAL_OUTPUT_DIR = os.path.join(SSD_PATH, "Master_Dataset_Extracted")

# Pfad zu deiner neuen Test-CSV-Datei
CSV_PATH = r"D:\KUH-PON\KUH-PON_csv_annotation.csv"

# Ordner, in dem die Original-Videos liegen
# WICHTIG: Falls dein KUH-PON.mp4 nicht hier, sondern direkt in D:\KUH-PON liegt,
# ändere diesen Pfad entsprechend auf r"D:\KUH-PON"
VIDEO_DIR = os.path.join(SSD_PATH, "BeachVolleyballData")

TARGET_SIZE = (448, 448)
TARGET_FPS = 30
MAX_FRAMES = 60
OFFSET_FRAMES = 30  # Startet 1 Sekunde vor dem Ballkontakt


def extract_from_csv():
    os.makedirs(GLOBAL_OUTPUT_DIR, exist_ok=True)

    # 'utf-8-sig' fängt eventuelle Formatierungsprobleme ab, delimiter=';' ist Standard im deutschen Excel
    with open(CSV_PATH, mode='r', encoding='utf-8-sig') as file:
        reader = csv.DictReader(file, delimiter=';')

        for row_idx, row in enumerate(reader):
            video_name = row['Dateiname'].strip()

            kontakt_frame = int(row['Start_frame'].strip())
            label = row['Label'].strip()

            video_path = os.path.join(VIDEO_DIR, video_name)

            if not os.path.exists(video_path):
                print(f"FEHLER: Video nicht gefunden -> {video_path}")
                continue

            label_dir = os.path.join(GLOBAL_OUTPUT_DIR, label)
            os.makedirs(label_dir, exist_ok=True)

            # Einzigartiger Dateiname (z.B. KUH-PON_frame_16654.mp4)
            prefix = video_name.replace(".mp4", "").replace(" ", "_")
            output_name = f"{prefix}_frame_{kontakt_frame}.mp4"
            output_path = os.path.join(label_dir, output_name)

            # Startframe berechnen (darf nicht ins Negative rutschen)
            actual_start = max(0, kontakt_frame - OFFSET_FRAMES)

            # Video öffnen und zum Startframe springen
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start)

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, TARGET_FPS, TARGET_SIZE)

            for _ in range(MAX_FRAMES):
                ret, frame = cap.read()
                if not ret: break
                out.write(cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_CUBIC))

            out.release()
            cap.release()
            print(f"Gespeichert ({row_idx + 1}): {label}/{output_name}")

    print("\n--- CSV-Extraktion erfolgreich beendet! ---")


if __name__ == "__main__":
    extract_from_csv()