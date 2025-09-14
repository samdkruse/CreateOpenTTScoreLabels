#!/usr/bin/env python3
import argparse
import csv
import re
import cv2
import easyocr
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# === Helpers for range-aware coords ===
def load_scoreboard_ranges(path: Path) -> Dict[str, List[dict]]:
    """Load game coordinate ranges from JSON, normalize, and sort."""
    with open(path, "r") as f:
        data = json.load(f)
    for game, segs in data.items():
        for s in segs:
            if s.get("end") is None:
                s["end"] = 10**12  # effectively "until end"
        segs.sort(key=lambda s: (int(s["start"]), int(s["end"])))
    return data

def coords_for_frame(game_ranges: Dict[str, List[dict]],
                     filename: str,
                     frame_idx: int) -> Optional[Dict[str, Tuple[int,int,int,int]]]:
    """Return the ROI coordinates for a given frame of a video."""
    segs = game_ranges.get(filename, [])
    for s in segs:
        if s["start"] <= frame_idx <= s["end"]:
            return {
                "left_player": tuple(s["left_player"]),
                "right_player": tuple(s["right_player"]),
            }
    return None

# === Load range-aware coordinates (after helpers/imports exist) ===
scoreboard_coords = load_scoreboard_ranges(
    Path("./labels/scoreboard_coordinates/test_scoreboard_coordinates.json")
)

# === Paths & defaults ===
VIDEO_DIR = Path("/Users/sdk/Downloads")
LABELS_DIR = Path("./labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)
(LABELS_DIR / "scores").mkdir(parents=True, exist_ok=True)

# === EasyOCR ===
reader = easyocr.Reader(['en'], gpu=False)
CONF_TH  = 0.35
MAX_DIG  = 2
DIGIT_RX = re.compile(r"^\d{1,%d}$" % MAX_DIG)
CHAR_FIX = str.maketrans({"O":"0","o":"0","I":"1","l":"1","|":"1","S":"5","B":"8"})

def best_numeric_text(result):
    """Pick best 1–2 digit string from EasyOCR result."""
    candidates = []
    for _, text, conf in result:
        if conf < CONF_TH:
            continue
        t = text.strip().translate(CHAR_FIX)
        t = "".join(ch for ch in t if ch.isdigit())
        if t and len(t) <= MAX_DIG and DIGIT_RX.match(t):
            candidates.append((t, conf))
    if not candidates:
        return ""
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[0][0]

def smooth_for_ocr(img_bgr, upscale=4, ksize=3, sigma=0.8):
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(up, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

def ocr_both_players(frame_idx, frame, filename):
    """Runs OCR for both players. Returns (left_player, right_player)."""
    coords = coords_for_frame(scoreboard_coords, filename, frame_idx)
    if coords is None:
        # Make an explicit empty return to avoid unpack errors upstream.
        return "", ""

    ocr_digits = {}
    for player, (x, y, w, h) in coords.items():
        crop = frame[y:y+h, x:x+w]
        proc = smooth_for_ocr(crop, upscale=6, ksize=3, sigma=0.6)
        result = reader.readtext(proc)
        digits = best_numeric_text(result)
        ocr_digits[player] = digits

    left_player  = ocr_digits.get("left_player", "")
    right_player = ocr_digits.get("right_player", "")
    return left_player, right_player

def process_video(filename: str, step_size: int):
    path = VIDEO_DIR / filename
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"❌ Could not open {filename}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 120.0
    game         = Path(filename).stem
    out_csv      = LABELS_DIR / "scores" / f"{game}_scores.csv"

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "score"])

        print(f"\n🎬 {filename}  (frames={total_frames}, fps≈{fps:.2f}, step={step_size})")
        frame_idx = 0
        while frame_idx < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = cap.read()
            if not ok:
                break

            left_player, right_player = ocr_both_players(frame_idx, frame, filename)
            score_str = f"{left_player}-{right_player}" if (left_player and right_player) else ""
            writer.writerow([frame_idx, score_str])

            frame_idx += step_size

    cap.release()
    print(f"✅ Labels written → {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Scoreboard OCR label generator")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--step-size", type=int, help="Frame step size for labeling (e.g., 1000)")
    args = ap.parse_args()

    for fname in scoreboard_coords.keys():
        if (VIDEO_DIR / fname).exists():
            process_video(fname, step_size=args.step_size)
        else:
            print(f"⚠️  Missing file: {VIDEO_DIR / fname}")

if __name__ == "__main__":
    main()
