#!/usr/bin/env python3
import argparse
import csv
import re
import cv2
import easyocr
import numpy as np
from pathlib import Path

# === Dual scoreboard crop coordinates (player_1 = left, player_2 = right) ===
# scoreboard_coords = {
#     "game_1.mp4": {"player_1": (915, 398, 24, 26), "player_2": (940, 399, 21, 26)},
#     "game_2.mp4": {"player_1": (999, 507, 22, 24), "player_2": (1024, 508, 21, 24)},
#     "game_3.mp4": {"player_1": (877, 435, 22, 27), "player_2": (901, 435, 23, 26)},
#     "game_4.mp4": {"player_1": (905, 433, 23, 27), "player_2": (927, 433, 25, 27)},
#     "game_5.mp4": {"player_1": (932, 410, 27, 28), "player_2": (957, 410, 25, 28)},
# }

scoreboard_coords = {
    "game_1.mp4": {"player_1": (917, 399, 20, 24), "player_2": (941, 400, 19, 23)},
    "game_2.mp4": {"player_1": (999, 506, 23, 25), "player_2": (1024, 508, 21, 23)},
    "game_3.mp4": {"player_1": (877, 435, 23, 26), "player_2": (900, 433, 22, 28)},
    "game_4.mp4": {"player_1": (905, 433, 22, 26), "player_2": (928, 432, 24, 26)},
    "game_5.mp4": {"player_1": (934, 411, 23, 26), "player_2": (958, 410, 24, 25)},
}


# === Paths & defaults ===
VIDEO_DIR = Path("/Users/sdk/Downloads")
LABELS_DIR = Path("./labels")
DEBUG_FRAMES_DIR = Path("./debug_frames")  # only used in --debug or --frameid
LABELS_DIR.mkdir(parents=True, exist_ok=True)
DEBUG_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

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

def render_overlay(crop, label, digits, target_height=250):
    """Debug pane: enlarged crop with text bar."""
    h, w = crop.shape[:2]
    scale = max(1, int(round(target_height / max(1, h))))
    disp  = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
    bar_h = 28
    cv2.rectangle(disp, (0, 0), (disp.shape[1], bar_h), (0, 0, 0), thickness=-1)
    text  = f"{label}: {digits if digits else '∅'}"
    cv2.putText(disp, text, (6, bar_h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return disp

def as_color_u8(img):
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img

def concat_side_by_side(rendered_left, rendered_right, spacer_w=60):
    """Safe hconcat two debug panes with a white spacer."""
    rendered = [as_color_u8(rendered_left), as_color_u8(rendered_right)]
    h_disp = max(r.shape[0] for r in rendered)
    rendered = [
        cv2.resize(r, (int(round(r.shape[1] * h_disp / r.shape[0])), h_disp),
                   interpolation=cv2.INTER_NEAREST)
        for r in rendered
    ]
    gap = np.full((h_disp, spacer_w, 3), 255, dtype=np.uint8)
    return cv2.hconcat([rendered[0], gap, rendered[1]])

def ocr_two_players(frame, coords, debug=False):
    """Runs OCR for both players. Returns (p1, p2, [debug panes])."""
    ocr_digits = {}
    panes = []
    for player, (x, y, w, h) in coords.items():
        crop = frame[y:y+h, x:x+w]
        proc = smooth_for_ocr(crop, upscale=6, ksize=3, sigma=0.6)
        result = reader.readtext(proc)
        digits = best_numeric_text(result)
        ocr_digits[player] = digits
        if debug:
            panes.append(render_overlay(proc, player, digits, target_height=260))
    p1 = ocr_digits.get("player_1", "")
    p2 = ocr_digits.get("player_2", "")
    return p1, p2, panes

def process_video(filename: str, step_size: int, debug: bool):
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
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stem         = Path(filename).stem
    out_csv      = LABELS_DIR / f"{stem}_newCoordinates.csv"

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

            p1, p2, panes = ocr_two_players(frame, coords, debug=debug)
            score_str = f"{p1}-{p2}" if (p1 and p2) else ""
            writer.writerow([frame_idx, score_str])

            if debug and len(panes) == 2:
                combined = concat_side_by_side(panes[0], panes[1])
                win = f"{filename} :: frame {frame_idx}  score='{score_str or '∅'}'"
                cv2.namedWindow(win, cv2.WINDOW_NORMAL)
                cv2.imshow(win, combined)
                k = cv2.waitKey(0)
                cv = chr(k & 0xFF).lower() if k != -1 else ''
                cv2.destroyWindow(win)
                if cv in ('q', '\x1b'):
                    break
                if cv == 's':
                    debug_path = DEBUG_FRAMES_DIR / f"{stem}_f{frame_idx}.png"
                    cv2.imwrite(str(debug_path), combined)
                    print(f"🖼️  saved {debug_path}")

            frame_idx += step_size

    cap.release()
    print(f"✅ Labels written → {out_csv}")

def show_single_frame(frame_spec: str):
    """
    frame_spec format: '<video_name>-<frame_index>' e.g. 'game_1.mp4-1000'
    Displays that frame with OCR result until 'q'/ESC is pressed.
    """
    try:
        video_name, frame_str = frame_spec.split("-", 1)
        frame_idx = int(frame_str)
    except Exception:
        print("❌ --frameid must look like 'game_1.mp4-1000'")
        return

    coords = scoreboard_coords.get(video_name)
    if coords is None:
        print(f"⚠️  No scoreboard coords for {video_name}")
        return

    path = VIDEO_DIR / video_name
    if not path.exists():
        print(f"❌ File not found: {path}")
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        print(f"❌ Could not open {video_name}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"❌ Frame {frame_idx} out of range (0..{max(0,total_frames-1)})")
        cap.release()
        return

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    if not ok:
        print("❌ Failed to read that frame.")
        cap.release()
        return

    p1, p2, panes = ocr_two_players(frame, coords, debug=True)
    score_str = f"{p1}-{p2}" if (p1 and p2) else "∅"
    if len(panes) == 2:
        combined = concat_side_by_side(panes[0], panes[1])
    else:
        combined = np.zeros((300, 800, 3), np.uint8)
        cv2.putText(combined, "No panes to display", (30, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2, cv2.LINE_AA)

    win = f"{video_name} :: frame {frame_idx}  score='{score_str}'"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, combined)
    # Wait until 'q' or ESC
    while True:
        k = cv2.waitKey(0)
        if k == -1:
            continue
        c = chr(k & 0xFF).lower()
        if c == 'q' or k == 27:
            break
    cv2.destroyWindow(win)
    cap.release()

def main():
    ap = argparse.ArgumentParser(description="Scoreboard OCR label generator")
    mx = ap.add_mutually_exclusive_group(required=True)
    mx.add_argument("--step-size", type=int,
                    help="Frame step size for labeling (e.g., 1000)")
    mx.add_argument("--frameid", type=str,
                    help="Show a single frame: '<video>-<frame>' e.g. 'game_1.mp4-1000'")
    ap.add_argument("--debug", action="store_true",
                    help="Enable interactive debug viewer (only used with --step-size)")
    args = ap.parse_args()

    if args.frameid:
        show_single_frame(args.frameid)
        return

    # --step-size mode
    for fname in scoreboard_coords.keys():
        if (VIDEO_DIR / fname).exists():
            process_video(fname, step_size=args.step_size, debug=args.debug)
        else:
            print(f"⚠️  Missing file: {VIDEO_DIR / fname}")

if __name__ == "__main__":
    main()
