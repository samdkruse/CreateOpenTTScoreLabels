#!/usr/bin/env python3
"""
splice_game.py

Given a video and its score CSV, cut them into two parts at a switch frame index.
Outputs <prefix>a.mp4 / <prefix>a_scores.csv and <prefix>b.mp4 / <prefix>b_scores.csv.

Example:
  python splice_game.py \
    --video /data/game_1.mp4 \
    --csv /data/labels/scores/game_1_scores.csv \
    --switch-frame 54321 \
    --out-prefix /data/game_1


      python splice_game.py \
    --video /opt/dlami/nvme/opentt_data/trainingdata/video/game_1.mp4 \
    --csv /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_1_scores.csv \
    --switch-frame 25400 \
    --out-prefix /opt/dlami/nvme/opentt_data/trainingdata/video/game_1



    /opt/dlami/nvme/opentt_data/trainingdata/video/game_1.mp4
    /opt/dlami/nvme/opentt_data/trainingdata/video/game_2.mp4

    /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_1_scores.csv
    /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_2_scores.csv
    25400
    149900
"""

import argparse
import csv
import subprocess
from pathlib import Path

import cv2


def get_fps_and_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"❌ Could not open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def run_ffmpeg_cut(video_path: Path, fps: float, switch_frame: int, out_prefix: Path):
    """
    Cut video into two segments: [0, switch) and [switch, end].
    Uses frame counts converted to seconds with FPS.
    """
    switch_time = switch_frame / fps

    out_a = out_prefix.with_name(out_prefix.stem + "a.mp4")
    out_b = out_prefix.with_name(out_prefix.stem + "b.mp4")

    # First segment
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", "0",
        "-to", f"{switch_time}",
        "-c", "copy",
        str(out_a)
    ], check=True)

    # Second segment
    subprocess.run([
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-ss", f"{switch_time}",
        "-c", "copy",
        str(out_b)
    ], check=True)

    return out_a, out_b


def split_csv(csv_path: Path, switch_frame: int, out_prefix: Path):
    out_a = out_prefix.with_name(out_prefix.stem + "a_scores.csv")
    out_b = out_prefix.with_name(out_prefix.stem + "b_scores.csv")

    rows_a, rows_b = [], []

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fr = int(row["frame"])
            if fr < switch_frame:
                rows_a.append(row)
            else:
                row_b = row.copy()
                row_b["frame"] = str(fr - switch_frame)  # reindex second half
                rows_b.append(row_b)

    if rows_a:
        with open(out_a, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "score"])
            writer.writeheader()
            writer.writerows(rows_a)

    if rows_b:
        with open(out_b, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "score"])
            writer.writeheader()
            writer.writerows(rows_b)

    return out_a, out_b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", type=Path, required=True, help="Input MP4 file")
    ap.add_argument("--csv", type=Path, required=True, help="Score CSV file")
    ap.add_argument("--switch-frame", type=int, required=True, help="Frame index where sides switch")
    ap.add_argument("--out-prefix", type=Path, required=True, help="Output prefix (no extension)")
    args = ap.parse_args()

    fps, total = get_fps_and_frames(args.video)
    if args.switch_frame <= 0 or args.switch_frame >= total:
        raise ValueError(f"Switch frame {args.switch_frame} is outside video range (0–{total})")

    print(f"🎬 Video: {args.video}, fps={fps:.2f}, total={total}")
    print(f"✂️ Cutting at frame {args.switch_frame} (≈{args.switch_frame/fps:.2f}s)")

    out_vids = run_ffmpeg_cut(args.video, fps, args.switch_frame, args.out_prefix)
    out_csvs = split_csv(args.csv, args.switch_frame, args.out_prefix)

    print(f"✅ Written: {out_vids[0]}, {out_vids[1]}")
    print(f"✅ Written: {out_csvs[0]}, {out_csvs[1]}")


if __name__ == "__main__":
    main()
