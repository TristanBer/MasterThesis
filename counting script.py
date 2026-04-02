import os
import cv2


def count_video_fps(folder_path):
    count_30 = 0
    count_60 = 0
    count_other = 0
    total_mp4s = 0

    # Go through every file in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is an mp4
        if filename.lower().endswith('.mp4'):
            total_mp4s += 1
            file_path = os.path.join(folder_path, filename)

            # Open the video file to read metadata
            cap = cv2.VideoCapture(file_path)

            if not cap.isOpened():
                print(f"Warning: Could not open {filename}")
                continue

            # Extract the raw FPS value
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()  # Close the file immediately

            # Video framerates are rarely exact integers (e.g., 29.97 or 59.94).
            # Rounding ensures we accurately group them into 30 or 60.
            rounded_fps = round(fps)

            if rounded_fps == 30:
                count_30 += 1
            elif rounded_fps == 60:
                count_60 += 1
            else:
                count_other += 1

    # Print the final results
    print("--- Video FPS Count Results ---")
    print(f"Total .mp4 files processed: {total_mp4s}")
    print(f"Videos at 30 FPS: {count_30}")
    print(f"Videos at 60 FPS: {count_60}")
    print(f"Videos at other framerates: {count_other}")


# --- HOW TO USE ---
# Replace the path below with the actual path to your folder
folder_directory = r"D:\BeachVolleyballData"
count_video_fps(folder_directory)