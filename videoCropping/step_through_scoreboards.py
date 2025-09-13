#!/usr/bin/env python3
import argparse
import csv
import re
import cv2
import easyocr
import numpy as np
from pathlib import Path


scoreboard_coords = {
    "game_1.mp4": {"left_player": (917, 399, 20, 24), "player_2": (941, 400, 19, 23)},
    "game_2.mp4": {"left_player": (999, 506, 23, 25), "player_2": (1024, 508, 21, 23)},
    "game_3.mp4": {"left_player": (877, 435, 23, 26), "player_2": (900, 433, 22, 28)},
    "game_4.mp4": {"left_player": (905, 433, 22, 26), "player_2": (928, 432, 24, 26)},
    "game_5.mp4": {"left_player": (934, 411, 23, 26), "player_2": (958, 410, 24, 25)},
}


# === Paths & defaults ===
VIDEO_DIR = Path("/Users/sdk/Downloads")
LABELS_DIR = Path("./labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)

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
    """Bicubic upsample + mild Gaussian blur (helps blocky pixels)."""
    h, w = img_bgr.shape[:2]
    up = cv2.resize(img_bgr, (w*upscale, h*upscale), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(up, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)


def ocr_both_players(frame, coords):
    """Runs OCR for both players. Returns (left_player, right_player])."""
    ocr_digits = {}
    for player, (x, y, w, h) in coords.items():
        crop = frame[y:y+h, x:x+w]
        proc = smooth_for_ocr(crop, upscale=6, ksize=3, sigma=0.6)
        result = reader.readtext(proc)
        digits = best_numeric_text(result)
        ocr_digits[player] = digits

    left_player = ocr_digits.get("left_player", "")
    right_player = ocr_digits.get("right_player", "")
    return left_player, right_player

def process_video(filename: str, step_size: int):
    path = VIDEO_DIR / filename

    coords = scoreboard_coords.get(filename)
    if coords is None:
        print(f"⚠️  No scoreboard coords for {filename}, skipping.")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"❌ Could not open {filename}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 120.0
    stem         = Path(filename).stem
    out_csv      = LABELS_DIR / f"{stem}_scores.csv"

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

            left_player, right_player = ocr_both_players(frame, coords)
            score_str = f"{left_player}-{right_player}" if (left_player and right_player) else ""
            writer.writerow([frame_idx, score_str])

            frame_idx += step_size

    cap.release()
    print(f"✅ Labels written → {out_csv}")

def main():
    ap = argparse.ArgumentParser(description="Scoreboard OCR label generator")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--step-size", type=int,
                    help="Frame step size for labeling (e.g., 1000)")
    args = ap.parse_args()

    # --step-size mode
    for fname in scoreboard_coords.keys():
        if (VIDEO_DIR / fname).exists():
            process_video(fname, step_size=args.step_size, debug=args.debug)
        else:
            print(f"⚠️  Missing file: {VIDEO_DIR / fname}")

if __name__ == "__main__":
    main()
