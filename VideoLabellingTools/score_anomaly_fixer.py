#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from dearpygui import dearpygui as dpg

# =========================
# Helpers
# =========================
def frame_to_rgba_flat(frame_bgr: np.ndarray):
    """OpenCV BGR -> Dear PyGui RGBA float32 flat array [0..1], plus width/height."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = rgb / 255.0
    rgba[:, :, 3] = 1.0
    return rgba.flatten(), w, h

def crop_frame(frame_bgr: np.ndarray, roi):
    x, y, w, h = roi
    return frame_bgr[y:y+h, x:x+w]

def load_scores_list(csv_path: Path):
    """Return sorted list[(frame, score)] from a frame,score CSV. Missing file -> []."""
    out = []
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fr = int(row.get("frame", ""))
                except Exception:
                    continue
                out.append((fr, row.get("score", "")))
    out.sort(key=lambda x: x[0])
    return out

def load_scores_map(csv_path: Path):
    """Return (map[int->score], sorted_list[(frame,score)]) for editing."""
    mp = {}
    if csv_path.exists():
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    fr = int(row.get("frame", ""))
                except Exception:
                    continue
                mp[fr] = row.get("score", "")
    lst = sorted(mp.items(), key=lambda x: x[0])
    return mp, lst

def save_scores_map(csv_path: Path, mp: dict[int, str]):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    items = sorted(mp.items(), key=lambda x: x[0])
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "score"])
        for fr, sc in items:
            w.writerow([fr, sc])

def nearest_before_after(scores_list, idx: int):
    """Given sorted list[(frame,score)], return nearest before (<=) and after (>=)."""
    if not scores_list:
        return None, None
    before = None
    after = None
    for f, s in scores_list:
        if f <= idx:
            before = (f, s)
        if f >= idx:
            after = (f, s)
            break
    return before, after

def nearest_before_only(scores_list, idx: int):
    """Nearest prior (<= idx) score; returns (frame,score) or None."""
    if not scores_list:
        return None
    before = None
    for f, s in scores_list:
        if f <= idx:
            before = (f, s)
        else:
            break
    return before

def coords_for_frame(segments, frame_idx: int):
    """Pick segment whose start <= frame <= end (end None -> infinity)."""
    for seg in segments:
        s = int(seg["start"])
        e = int(seg["end"]) if seg["end"] is not None else 10**12
        if s <= frame_idx <= e:
            return tuple(seg["left_player"]), tuple(seg["right_player"]), seg
    return None, None, None

def fmt_seg(seg):
    if not seg:
        return "No matching segment for this frame."
    s = int(seg["start"])
    e = "∞" if seg["end"] is None else int(seg["end"])
    lp = seg["left_player"]; rp = seg["right_player"]
    return f"range: [{s} .. {e}]\nleft_player: {lp}\nright_player: {rp}"

# =========================
# CLI
# =========================
ap = argparse.ArgumentParser(description="Video + Crops + OCR (range-aware) with cleaned edits (inline).")
ap.add_argument("video", type=Path, help="Path to an .mp4 video (e.g., game_5.mp4)")
ap.add_argument(
    "--coords",
    type=Path,
    default=Path("/Users/sdk/Desktop/code/pingpong/VideoLabellingTools/labels/scoreboard_coordinates/game_scoreboard_coordinates.json"),
    help="Path to JSON with scoreboard coords (range-aware).",
)
ap.add_argument(
    "--scores_dir",
    type=Path,
    default=Path("/Users/sdk/Desktop/code/pingpong/VideoLabellingTools/labels/scores"),
    help="Directory with original *_scores.csv",
)
ap.add_argument(
    "--cleaned_dir",
    type=Path,
    default=Path("/Users/sdk/Desktop/code/pingpong/VideoLabellingTools/labels/cleaned_scores"),
    help="Directory for cleaned (editable) *_scores.csv",
)
args = ap.parse_args()

if not args.video.exists():
    raise FileNotFoundError(f"Video not found: {args.video}")
if not args.coords.exists():
    raise FileNotFoundError(f"Coords JSON not found: {args.coords}")

# =========================
# Load coords (range-aware)
# =========================
with open(args.coords, "r") as f:
    coords_data = json.load(f)
video_key = args.video.name
if video_key not in coords_data:
    raise KeyError(f"No entry for {video_key} in {args.coords}")
segments = coords_data[video_key]

# =========================
# Load scores (original + cleaned)
# =========================
stem = args.video.stem
orig_csv_path = args.scores_dir / f"{stem}_scores.csv"
orig_scores = load_scores_list(orig_csv_path)  # sorted list

args.cleaned_dir.mkdir(parents=True, exist_ok=True)
cleaned_csv_path = args.cleaned_dir / f"{stem}_scores.csv"
# Make cleaned copy if needed
if orig_csv_path.exists() and not cleaned_csv_path.exists():
    shutil.copyfile(orig_csv_path, cleaned_csv_path)

cleaned_map, cleaned_list = load_scores_map(cleaned_csv_path)  # editable

# =========================
# Video capture
# =========================
cap = cv2.VideoCapture(str(args.video))
if not cap.isOpened():
    raise RuntimeError(f"Could not open {args.video}")
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
current_idx = 0

def read_packet(idx: int):
    """Read frame idx and produce full + crops as RGBA flat data. Returns dict or None."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    left_roi, right_roi, seg = coords_for_frame(segments, idx)
    full_flat, fw, fh = frame_to_rgba_flat(frame)
    if left_roi and right_roi:
        lc = crop_frame(frame, left_roi)
        rc = crop_frame(frame, right_roi)
        left_flat, lw, lh = frame_to_rgba_flat(lc)
        right_flat, rw, rh = frame_to_rgba_flat(rc)
    else:
        ph = np.zeros((1, 1, 3), dtype=np.uint8)
        left_flat, lw, lh = frame_to_rgba_flat(ph)
        right_flat, rw, rh = frame_to_rgba_flat(ph)
        seg = None
    return {
        "full":  (full_flat, fw, fh),
        "left":  (left_flat, lw, lh),
        "right": (right_flat, rw, rh),
        "seg":   seg
    }

# Initial packet
packet = read_packet(current_idx)
if packet is None:
    raise RuntimeError("Could not read initial frame.")
(full_flat, fw, fh) = packet["full"]
(left_flat, lw, lh) = packet["left"]
(right_flat, rw, rh) = packet["right"]

# Track current texture sizes to recreate on ROI changes
current_sizes = {
    "full":  (fw, fh),
    "left":  (lw, lh),
    "right": (rw, rh),
}

# =========================
# Dear PyGui setup
# =========================
dpg.create_context()
dpg.create_viewport(title="Video + Crops + OCR (Inline Edit)", width=fw//2 + 720, height=fh//2 + lh + 460)

with dpg.texture_registry(show=False):
    dpg.add_dynamic_texture(fw, fh, full_flat, tag="video_tex")
    dpg.add_dynamic_texture(lw, lh, left_flat, tag="left_tex")
    dpg.add_dynamic_texture(rw, rh, right_flat, tag="right_tex")

def recreate_texture(tag: str, data_flat, w: int, h: int):
    if dpg.does_item_exist(tag):
        dpg.delete_item(tag)
    with dpg.texture_registry(show=False):
        dpg.add_dynamic_texture(w, h, data_flat, tag=tag)

def ensure_texture(tag: str, data_flat, w: int, h: int, which: str):
    """Recreate if size changed; otherwise set value."""
    prev = current_sizes.get(which)
    if (prev is None) or (prev != (w, h)) or (not dpg.does_item_exist(tag)):
        recreate_texture(tag, data_flat, w, h)
        current_sizes[which] = (w, h)
        return True
    else:
        dpg.set_value(tag, data_flat)
        return False

with dpg.window(label="Video Review", tag="main"):
    dpg.add_text(f"Video: {args.video.name}")
    dpg.add_separator()

    # Header: frame info + full image
    dpg.add_text("", tag="frame_label")
    dpg.add_image("video_tex", width=fw//2, height=fh//2, tag="full_image_item")

    dpg.add_separator()
    # Crops + side panel
    with dpg.group(horizontal=True):
        with dpg.group():
            dpg.add_text("Scoreboard Crops")
            with dpg.group(horizontal=True):
                dpg.add_image("left_tex",  width=lw*2, height=lh*2, tag="left_image_item")
                dpg.add_image("right_tex", width=rw*2, height=rh*2, tag="right_image_item")
        with dpg.group():
            dpg.add_text("ROI Segment (active)")
            dpg.add_text("", tag="coords_label")
            dpg.add_spacer(height=6)
            dpg.add_text("OCR (Original)")
            dpg.add_text("", tag="ocr_label_orig")
            dpg.add_spacer(height=6)
            dpg.add_text("OCR (Cleaned)")
            dpg.add_text("", tag="ocr_label_cleaned")

    dpg.add_separator()
    # Navigation + Actions (inline editing)
    dpg.add_text("Navigation / Actions")
    dpg.add_input_int(label="Step Size", default_value=100, min_value=1, width=100, tag="step_size_input")
    dpg.add_same_line()
    dpg.add_button(label="Next Point of Interest", callback=lambda: goto_next_poi())
    dpg.add_same_line()
    dpg.add_button(label="Apply Inline Edit", callback=lambda: apply_inline_edit())
    dpg.add_input_text(label="Score for Current Frame", tag="inline_score_input", hint="e.g., 7-5", width=150)
    dpg.add_spacer(height=6)
    dpg.add_button(label="Use Last Known Score: (computing...)", tag="last_known_btn", callback=lambda: use_last_known_score())

# =========================
# UI update functions
# =========================
def update_score_labels(idx: int):
    # Original
    b1, a1 = nearest_before_after(orig_scores, idx)
    if b1 and a1 and b1 != a1:
        dpg.set_value("ocr_label_orig", f"{b1[0]}: {b1[1]}\n{a1[0]}: {a1[1]}")
    elif b1:
        dpg.set_value("ocr_label_orig", f"{b1[0]}: {b1[1]}")
    elif a1:
        dpg.set_value("ocr_label_orig", f"{a1[0]}: {a1[1]}")
    else:
        dpg.set_value("ocr_label_orig", "(no OCR data)")

    # Cleaned
    global cleaned_list
    b2, a2 = nearest_before_after(cleaned_list, idx)
    if b2 and a2 and b2 != a2:
        dpg.set_value("ocr_label_cleaned", f"{b2[0]}: {b2[1]}\n{a2[0]}: {a2[1]}")
    elif b2:
        dpg.set_value("ocr_label_cleaned", f"{b2[0]}: {b2[1]}")
    elif a2:
        dpg.set_value("ocr_label_cleaned", f"{a2[0]}: {a2[1]}")
    else:
        dpg.set_value("ocr_label_cleaned", "(no cleaned data)")

def compute_last_known_label(idx: int):
    """Use ONLY the original OCR CSV to find the nearest prior score (<= idx)."""
    # orig_scores is a sorted list of (frame, score)
    if not orig_scores:
        return None
    before = None
    for f, s in orig_scores:
        if f <= idx:
            before = (f, s)
        else:
            break
    return before  # (frame, score) or None

def update_last_known_button(idx: int):
    """Update the button label to display the last known ORIGINAL score."""
    pair = compute_last_known_label(idx)
    if pair and pair[1]:
        dpg.configure_item("last_known_btn", label=f"Use Last Known Score: {pair[1]}")
        dpg.enable_item("last_known_btn")
    else:
        dpg.configure_item("last_known_btn", label="Use Last Known Score: (none)")
        dpg.disable_item("last_known_btn")

def update_inline_field(idx: int):
    """Prefill inline field with existing cleaned score for EXACT frame if present."""
    existing = ""
    # exact in cleaned?
    for f, s in cleaned_list:
        if f == idx:
            existing = s or ""
            break
        if f > idx:
            break
    dpg.set_value("inline_score_input", existing)

def update_display(idx: int):
    pkt = read_packet(idx)
    if pkt is None:
        return
    (full_flat, fw, fh)   = pkt["full"]
    (left_flat, lw, lh)   = pkt["left"]
    (right_flat, rw, rh)  = pkt["right"]
    seg                   = pkt["seg"]

    # Full
    ensure_texture("video_tex", full_flat, fw, fh, "full")

    # Crops (resize if needed)
    left_recreated  = ensure_texture("left_tex",  left_flat,  lw, lh, "left")
    right_recreated = ensure_texture("right_tex", right_flat, rw, rh, "right")
    if left_recreated:
        dpg.configure_item("left_image_item",  width=lw*2,  height=lh*2)
    if right_recreated:
        dpg.configure_item("right_image_item", width=rw*2,  height=rh*2)

    # Labels
    dpg.set_value("frame_label", f"Frame {idx}/{total_frames-1}")
    dpg.set_value("coords_label", fmt_seg(seg))

    # OCR panels + buttons
    update_score_labels(idx)
    update_last_known_button(idx)
    update_inline_field(idx)

# =========================
# Navigation & POI
# =========================
def step(delta: int):
    global current_idx
    step_sz = dpg.get_value("step_size_input") or 1
    new_idx = current_idx + delta * step_sz
    if 0 <= new_idx < total_frames:
        current_idx = new_idx
        update_display(current_idx)

def current_effective_score(scores_list, idx: int):
    if not scores_list:
        return ""
    b, a = nearest_before_after(scores_list, idx)
    if b and b[0] == idx:
        return b[1]
    if b:
        return b[1]
    if a:
        return a[1]
    return ""

def goto_next_poi():
    global current_idx
    # Prefer cleaned if available, else original
    base = cleaned_list if cleaned_list else orig_scores
    if not base:
        return
    cur_score = current_effective_score(base, current_idx)
    for f, s in base:
        if f > current_idx and s != cur_score:
            current_idx = f
            update_display(current_idx)
            return
    # no next change

# =========================
# Inline editing actions
# =========================
def apply_inline_edit():
    """Write inline input score to cleaned CSV for current frame, update UI."""
    global cleaned_map, cleaned_list
    val = (dpg.get_value("inline_score_input") or "").strip()
    if val:
        cleaned_map[current_idx] = val
    else:
        cleaned_map.pop(current_idx, None)
    save_scores_map(cleaned_csv_path, cleaned_map)
    # reload sorted list
    _, new_list = load_scores_map(cleaned_csv_path)
    cleaned_list[:] = new_list
    update_display(current_idx)

def use_last_known_score():
    """Copy nearest prior score from ORIGINAL CSV into the cleaned CSV at the current frame."""
    global cleaned_map, cleaned_list
    pair = compute_last_known_label(current_idx)  # now original-only
    if not pair:
        return
    val = pair[1]
    cleaned_map[current_idx] = val
    save_scores_map(cleaned_csv_path, cleaned_map)
    _, new_list = load_scores_map(cleaned_csv_path)
    cleaned_list[:] = new_list
    dpg.set_value("inline_score_input", val)
    update_display(current_idx)

# =========================
# Key handlers & kickoff
# =========================
with dpg.handler_registry():
    dpg.add_key_press_handler(dpg.mvKey_Right, callback=lambda: step(+1))
    dpg.add_key_press_handler(dpg.mvKey_Left,  callback=lambda: step(-1))

# Initial draw
update_display(current_idx)

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("main", True)
dpg.start_dearpygui()
dpg.destroy_context()
