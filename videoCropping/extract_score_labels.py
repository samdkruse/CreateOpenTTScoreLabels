import cv2
import easyocr
from pathlib import Path

# Coordinates per game (your crop values)
scoreboard_coords = {
    "game_1.mp4": (875, 363, 127, 90),
    "game_2.mp4": (957, 489, 120, 69),
    "game_3.mp4": (842, 421, 120, 63),
    "game_4.mp4": (851, 413, 124, 61),
    "game_5.mp4": (885, 395, 130, 51),
}

# OCR reader
reader = easyocr.Reader(['en'], gpu=True)

# Parameters
video_dir = Path("/Users/sdk/Downloads")
frame_interval = 10  # Only check every Nth frame for speed

# Loop through all videos
for video_file in scoreboard_coords:
    path = video_dir / video_file
    x, y, w, h = scoreboard_coords[video_file]
    print(f"\n🔍 Processing {video_file}")

    cap = cv2.VideoCapture(str(path))
    frame_id = 0
    last_score = None
    labels = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % frame_interval == 0:
            cropped = frame[y:y+h, x:x+w]
            result = reader.readtext(cropped)
            texts = [t[1] for t in result if t[2] > 0.5]  # high confidence only

            for text in texts:
                text = text.replace(" ", "").replace("-", "–")  # normalize
                if text != last_score and "–" in text and len(text) <= 5:
                    print(f"[{frame_id}] Score: {text}")
                    labels[frame_id] = text
                    last_score = text
                    break

        frame_id += 1

    cap.release()

    # Save output
    out_path = Path(f"scores_{Path(video_file).stem}.json")
    with open(out_path, "w") as f:
        import json
        json.dump(labels, f, indent=2)

    print(f"✅ Saved: {out_path}")
