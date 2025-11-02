#!/usr/bin/env python3
"""
Generate noise manifest with train/val/test split (file-level)
"""
import argparse
import re
from pathlib import Path
import soundfile as sf
import csv

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate noise manifest with file-level split assignment"
    )
    parser.add_argument("--input-dir", type=str, required=True,
                        help="Directory with purenoise_#_#####.wav files")
    parser.add_argument("--output", type=str, default="manifests/noise_manifest.csv",
                        help="Output CSV manifest path")
    parser.add_argument("--sr", type=int, default=44100,
                        help="Sample rate (for 500ms half calculation)")
    return parser.parse_args()

def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ✅ File-level split (Plan A: 3-1-1)
    split_map = {
        1: "train",
        2: "train",
        3: "train",
        4: "val",
        5: "test",
    }

    rows = []
    wav_files = sorted(in_dir.glob("purenoise_*.wav"))
    if len(wav_files) == 0:
        raise FileNotFoundError(f"No purenoise_*.wav files in {in_dir}")

    print(f"Found {len(wav_files)} noise files")

    for wav_path in wav_files:
        m = re.match(r"purenoise_(\d+)_(\d+)\.wav", wav_path.name)
        if not m:
            continue
        
        parent_id = int(m.group(1))
        seg_id = int(m.group(2))
        split = split_map.get(parent_id, "train")

        # ✅ Read waveform and validate
        y, sr = sf.read(wav_path)
        
        # ✅ Ensure exactly 1 second
        if len(y) < args.sr:
            print(f"⚠️ Skipping {wav_path.name}: too short ({len(y)} samples)")
            continue
        
        y = y[:args.sr]  # ✅ Trim to exactly 1 second

        # ✅ Split into two 500ms halves
        half_len = args.sr // 2
        halves = [("a", 0, half_len), ("b", half_len, args.sr)]
        
        for half, start, end in halves:
            rows.append({
                "segment_id": f"{parent_id}_{seg_id:05d}_{half}",
                "parent_id": parent_id,
                "original_seg_id": seg_id,  # ✅ Added
                "split": split,
                "half": half,
                "path": str(wav_path),
                "start": start,
                "end": end,
                "duration_ms": 500.0  # ✅ Added
            })

    # ✅ Write CSV
    if not rows:
        raise ValueError("No valid segments found!")
    
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✅ Generated manifest: {out_path}")
    print(f"Total half-segments: {len(rows)} ({len(rows)//2} × 1s files)")
    
    # ✅ Print split statistics
    split_counts = {}
    for r in rows:
        split_counts[r["split"]] = split_counts.get(r["split"], 0) + 1
    
    print("\nSplit distribution:")
    for k, v in sorted(split_counts.items()):
        print(f"  {k}: {v} half-segments ({v//2} 1s files)")

if __name__ == "__main__":
    main()