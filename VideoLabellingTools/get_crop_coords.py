#!/usr/bin/env python3
"""
Pick scoreboard ROIs for a single video and emit JSON in your format.

Usage:
  python get_crop_coords.py /path/to/game_1.mp4

Behavior:
  - Seeks to the middle frame for selection (change MID_SELECT to fix a frame).
  - Lets you select ROIs for left_player and right_player via cv2.selectROI.
  - Prints the JSON snippet and saves it to:
      ./labels/scoreboard_coordinates/<video_stem>_coords.json
  - Top-level key is the filename, value is a list with one {start, end, left_player, right_player}.
"""

import argparse
import json
from pathlib import Path
import sys
import cv2

MID_SELECT = True         # if False, change SELECT_FRAME to a specific index
SELECT_FRAME = 8740       # only used if MID_SELECT=False

def read_select_frame(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"❌ Could not open video: {video_path}", file=sys.stderr)
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 0.0
    idx   = (total // 2) if MID_SELECT else min(SELECT_FRAME, max(0, total - 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        print("❌ Failed to read a frame.", file=sys.stderr)
        sys.exit(1)

    return idx, total, fps, frame

def main():
    ap = argparse.ArgumentParser(description="Pick scoreboard ROIs and emit JSON.")
    ap.add_argument("video", type=Path, help="Path to the video file (e.g., game_1.mp4)")
    args = ap.parse_args()

    video_path = args.video
    if not video_path.exists():
        print(f"❌ File not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    frame_idx, total, fps, frame = read_select_frame(video_path)
    print(f"\n🎥 {video_path}  (frames≈{total}, fps≈{fps:.2f}, select frame {frame_idx})")

    # Select left player's ROI
    print("👤 Select ROI for Player 1 (left_player). Press ENTER/SPACE to confirm; C to cancel.")
    cv2.namedWindow("Player 1 Score", cv2.WINDOW_NORMAL)
    roi1 = cv2.selectROI("Player 1 Score", frame, showCrosshair=True)
    cv2.destroyWindow("Player 1 Score")

    # Select right player's ROI
    print("👤 Select ROI for Player 2 (right_player). Press ENTER/SPACE to confirm; C to cancel.")
    cv2.namedWindow("Player 2 Score", cv2.WINDOW_NORMAL)
    roi2 = cv2.selectROI("Player 2 Score", frame, showCrosshair=True)
    cv2.destroyWindow("Player 2 Score")

    (x1, y1, w1, h1) = [int(v) for v in roi1]
    (x2, y2, w2, h2) = [int(v) for v in roi2]

    if w1 == 0 or h1 == 0 or w2 == 0 or h2 == 0:
        print("❌ One of the ROIs is empty (canceled). Aborting.", file=sys.stderr)
        sys.exit(1)

    print(f"\n✅ Player 1 ROI: x={x1}, y={y1}, width={w1}, height={h1}")
    print(f"✅ Player 2 ROI: x={x2}, y={y2}, width={w2}, height={h2}")

    # Exact format you specified: left_player/right_player, end=null
    file_key = video_path.name  # e.g., "game_1.mp4"
    data = {
        file_key: [
            {
                "start": 0,
                "end": None,
                "left_player":  [x1, y1, w1, h1],
                "right_player": [x2, y2, w2, h2]
            }
        ]
    }

    text = json.dumps(data, indent=2)
    print("\n--- JSON (copy/paste into your master file) ---")
    print(text)

    out_dir = Path("./labels/scoreboard_coordinates")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{video_path.stem}_coords.json"
    out_path.write_text(text, encoding="utf-8")
    print(f"\n💾 Saved JSON to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
