import cv2
import xml.etree.ElementTree as ET
import os
import shutil

# --- EINSTELLUNGEN ---
VIDEO_FILE = r"D:\BOR - BEH\2025-07-11 BOR-BEH.mp4"
XML_FILE = r"D:\BOR - BEH\annotations.xml"
OUTPUT_DIR = r"D:\BOR - BEH\extracted_sets"

# ORDNER BEREINIGEN (Damit die Statistik nicht doppelt zählt)
if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

# XML laden
tree = ET.parse(XML_FILE)
root = tree.getroot()

# Video laden
cap = cv2.VideoCapture(VIDEO_FILE)
if not cap.isOpened():
    print(f"Fehler: Konnte Video nicht öffnen unter {VIDEO_FILE}")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Starte Extraktion mit {fps} FPS...")

# Tracks verarbeiten
for track in root.findall('track'):
    track_id = track.get('id')

    # Das richtige Attribut finden
    specific_label = "unknown"
    for attr in track.findall('.//attribute'):
        if attr.get('name') == 'type':  # Wir suchen gezielt nach dem 'type' Attribut
            specific_label = attr.text
            break

    # Unterordner erstellen
    label_dir = os.path.join(OUTPUT_DIR, specific_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)

    # Frames extrahieren
    boxes = track.findall('box')
    if not boxes: continue

    start_frame = int(boxes[0].get('frame'))
    end_frame = int(boxes[-1].get('frame'))

    output_filename = os.path.join(label_dir, f"track_{track_id}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret: break
        out.write(frame)

    out.release()
    print(f"Clip erstellt: {specific_label} (ID: {track_id})")

cap.release()

# --- STATISTIK ---
print("\n" + "=" * 35)
print("ZUSAMMENFASSUNG DER EXTRAKTION")
print("=" * 35)
total = 0
for folder in os.listdir(OUTPUT_DIR):
    folder_path = os.path.join(OUTPUT_DIR, folder)
    if os.path.isdir(folder_path):
        count = len([f for f in os.listdir(folder_path) if f.endswith('.mp4')])
        print(f"- {folder:25}: {count} Clips")
        total += count
print("=" * 35)
print(f"GESAMT: {total} Clips")