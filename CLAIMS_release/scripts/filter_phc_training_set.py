#!/usr/bin/env python3
"""
Filter PHC training assets using alignment-retained video ids.

This script keeps only samples whose stem appears in a retained-video list, or
whose verdict is not `mismatch` in `alignment_summary.csv`.
It can filter:
1. the merged PHC training pkl
2. the per-sample `single_dict/` directory
"""

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Dict, Set

import joblib


def load_stems_from_video_csv(csv_path: Path) -> Set[str]:
    stems: Set[str] = set()
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get("video_name") or "").strip()
            if name:
                stems.add(Path(name).stem)
    if not stems:
        raise ValueError(f"No video names found in {csv_path}")
    return stems


def load_non_mismatch_stems(summary_csv_path: Path) -> Set[str]:
    stems: Set[str] = set()
    with open(summary_csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = (row.get("video_name") or "").strip()
            verdict = (row.get("verdict") or "").strip().lower()
            if video_name and verdict and verdict != "mismatch":
                stems.add(Path(video_name).stem)
    if not stems:
        raise ValueError(f"No non-mismatch rows found in {summary_csv_path}")
    return stems


def filter_full_pkl(input_pkl: Path, output_pkl: Path, accepted_stems: Set[str]) -> Dict[str, int]:
    data = joblib.load(input_pkl)
    if not isinstance(data, dict):
        raise TypeError(f"Expected dict in {input_pkl}, got {type(data)}")

    filtered = {key: value for key, value in data.items() if Path(key).stem in accepted_stems}
    output_pkl.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(filtered, output_pkl)
    return {"input_count": len(data), "kept_count": len(filtered)}


def filter_single_dict(input_dir: Path, output_dir: Path, accepted_stems: Set[str]) -> Dict[str, int]:
    input_files = sorted(input_dir.glob("*.pkl"))
    output_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    for file_path in input_files:
        if file_path.stem in accepted_stems:
            shutil.copy2(file_path, output_dir / file_path.name)
            kept += 1

    return {"input_count": len(input_files), "kept_count": kept}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter PHC training assets using retained video ids.")
    parser.add_argument("--accepted-videos-csv", type=Path)
    parser.add_argument("--alignment-summary-csv", type=Path)
    parser.add_argument("--input-pkl", type=Path, required=True)
    parser.add_argument("--output-pkl", type=Path, required=True)
    parser.add_argument("--input-single-dir", type=Path)
    parser.add_argument("--output-single-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--kept-videos-csv", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if bool(args.accepted_videos_csv) == bool(args.alignment_summary_csv):
        raise ValueError("Provide exactly one of --accepted-videos-csv or --alignment-summary-csv")

    if args.accepted_videos_csv:
        kept_stems = load_stems_from_video_csv(args.accepted_videos_csv)
        keep_source = str(args.accepted_videos_csv)
        keep_rule = "listed_in_video_csv"
    else:
        kept_stems = load_non_mismatch_stems(args.alignment_summary_csv)
        keep_source = str(args.alignment_summary_csv)
        keep_rule = "verdict_not_mismatch"

    full_stats = filter_full_pkl(args.input_pkl, args.output_pkl, kept_stems)
    summary = {
        "keep_rule": keep_rule,
        "keep_source": keep_source,
        "kept_count": len(kept_stems),
        "full_pkl": {
            "input_path": str(args.input_pkl),
            "output_path": str(args.output_pkl),
            **full_stats,
        },
    }

    if args.input_single_dir and args.output_single_dir:
        single_stats = filter_single_dict(args.input_single_dir, args.output_single_dir, kept_stems)
        summary["single_dict"] = {
            "input_dir": str(args.input_single_dir),
            "output_dir": str(args.output_single_dir),
            **single_stats,
        }

    if args.kept_videos_csv:
        args.kept_videos_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(args.kept_videos_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name"])
            writer.writeheader()
            for stem in sorted(kept_stems):
                writer.writerow({"video_name": stem})
        summary["kept_videos_csv"] = str(args.kept_videos_csv)

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
