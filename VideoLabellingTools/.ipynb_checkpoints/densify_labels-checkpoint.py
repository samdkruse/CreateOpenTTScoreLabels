#!/usr/bin/env python3
import csv
import re
from pathlib import Path

# --- Configuration ---
scores_dir = Path("./labels/cleaned_scores")  # Input: sparse CSVs
output_dir = Path("./labels/densified_scores")  # Output: densified CSVs
DEFAULT_FRAME_MAX = 36000  # Set to the max frame for your videos
# --- End Configuration ---

output_dir.mkdir(exist_ok=True)

SCORE_RX = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")

def parse_score(cell: str):
    """Return (p1, p2) from 'N-M'; blanks/invalid -> (0,0)."""
    m = SCORE_RX.match(cell or "")
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

for csv_path in scores_dir.glob("*_scores.csv"):
    # 1. Read sparse score data
    frames = []
    labels = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                fr = int(row["frame"])
                frames.append(fr)
                labels[fr] = row["score"]
            except (ValueError, KeyError):
                continue

    frames.sort()
    if not frames:
        print(f"⚠️ Skipping {csv_path.name}: no valid rows.")
        continue

    first_frame = frames[0]
    last_frame = frames[-1]

    # 2. Densify using forward-fill across full range
    dense_labels = {}

    # Fill pre-label range (e.g. frame 0 to first labeled frame)
    for fr in range(0, first_frame):
        dense_labels[fr] = labels[first_frame]  # or use '0-0' if you prefer

    # Fill between labels
    for i in range(len(frames) - 1):
        start_frame = frames[i]
        end_frame = frames[i + 1]
        score = labels[start_frame]
        for fr in range(start_frame, end_frame):
            dense_labels[fr] = score

    # Fill from last label to DEFAULT_FRAME_MAX
    for fr in range(last_frame, DEFAULT_FRAME_MAX + 1):
        dense_labels[fr] = labels[last_frame]

    # 3. Write the densified CSV
    out_path = output_dir / csv_path.name
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame", "score"])
        writer.writeheader()
        for fr in sorted(dense_labels.keys()):
            writer.writerow({"frame": fr, "score": dense_labels[fr]})

    print(f"✅ REDensified {csv_path.name} -> {out_path.name} | total frames: {len(dense_labels)}")
