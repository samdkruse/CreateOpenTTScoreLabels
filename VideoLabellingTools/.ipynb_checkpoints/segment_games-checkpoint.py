#!/usr/bin/env python3


"""
python3 segment_games.py \
  --csv /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_2a_scores.csv \
  --out_json /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/game_segments/game_2a_segments.json
python3 segment_games.py \
  --csv /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_2b_scores.csv \
  --out_json /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/game_segments/game_2b_segments.json
  python3 segment_games.py \
  --csv /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_3_scores.csv \
  --out_json /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/game_segments/game_3_segments.json
    python3 segment_games.py \
  --csv /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/densified_scores/game_4_scores.csv \
  --out_json /opt/dlami/nvme/opentt_data/CreateOpenTTScoreLabels/VideoLabellingTools/labels/game_segments/game_4_segments.json


reads a dense frame,score CSV,

finds game boundaries using your rule (≥11 and lead ≥2) and resets (score drops),

outputs a JSON list of segments with [start_frame, end_frame], winner, and final score,

prints a readable summary.
"""
import argparse, csv, json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

def parse_score(s: str) -> Tuple[Optional[int], Optional[int]]:
    s = (s or "").strip()
    if "-" not in s: return None, None
    try:
        l, r = s.split("-")
        return int(l), int(r)
    except:
        return None, None

def read_dense_csv(csv_path: Path):
    frames, f2s = [], {}
    with open(csv_path, newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            try:
                fr = int(row["frame"])
            except:
                continue
            l, r = parse_score(row.get("score", ""))
            if l is None: continue
            frames.append(fr)
            f2s[fr] = (l, r)
    frames.sort()
    return frames, f2s

def is_win(l: int, r: int, min_points: int, lead: int) -> bool:
    return (l >= min_points or r >= min_points) and abs(l - r) >= lead

def segment_games(frames: List[int], f2s: Dict[int, Tuple[int,int]],
                  min_points: int = 11, lead: int = 2):
    """Return list of segments: dict(start,end,won,winner,final_score)."""
    if not frames: return []

    segs = []
    start = frames[0]
    last_win_frame = None

    prev_fr = frames[0]
    prev_l, prev_r = f2s[prev_fr]

    for fr in frames[1:]:
        l, r = f2s[fr]

        # Track last frame that met the win condition
        if is_win(l, r, min_points, lead):
            last_win_frame = fr

        # Detect reset: any score drop vs previous frame
        if l < prev_l or r < prev_r:
            end = prev_fr
            final_l, final_r = f2s[end]
            winner = "left" if final_l > final_r else ("right" if final_r > final_l else None)
            won = last_win_frame is not None and last_win_frame <= end
            segs.append({
                "start": start,
                "end": end,
                "won": bool(won),
                "winner": winner if won else None,
                "final_score": [final_l, final_r]
            })
            # start next segment at reset frame
            start = fr
            last_win_frame = None

        prev_fr, prev_l, prev_r = fr, l, r

    # close final segment
    end = frames[-1]
    final_l, final_r = f2s[end]
    winner = "left" if final_l > final_r else ("right" if final_r > final_l else None)
    won = last_win_frame is not None and last_win_frame <= end
    segs.append({
        "start": start,
        "end": end,
        "won": bool(won),
        "winner": winner if won else None,
        "final_score": [final_l, final_r]
    })
    return segs

def main():
    ap = argparse.ArgumentParser(description="Detect game segments in a dense score CSV.")
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out_json", type=Path, required=False,
                    help="Optional path to write JSON segments.")
    ap.add_argument("--min_points", type=int, default=11)
    ap.add_argument("--lead", type=int, default=2)
    args = ap.parse_args()

    frames, f2s = read_dense_csv(args.csv)
    if not frames:
        print("No data.")
        return

    segs = segment_games(frames, f2s, args.min_points, args.lead)

    # Pretty print
    print(f"Found {len(segs)} segment(s):")
    for i, s in enumerate(segs, 1):
        print(f"  #{i}: [{s['start']} .. {s['end']}]  "
              f"won={s['won']} winner={s['winner']} final={tuple(s['final_score'])}")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(segs, f, indent=2)
        print(f"Wrote segments → {args.out_json}")

if __name__ == "__main__":
    main()
