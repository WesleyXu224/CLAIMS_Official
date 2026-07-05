import argparse
import csv
import os
from difflib import SequenceMatcher


def find_best_match(video_name, metrics_dict):
    if video_name in metrics_dict:
        return metrics_dict[video_name]
    for full_name, data in metrics_dict.items():
        if full_name.startswith(video_name) or video_name.startswith(full_name):
            return data

    best_match = None
    best_ratio = 0.95
    for full_name, data in metrics_dict.items():
        ratio = SequenceMatcher(None, video_name, full_name).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = data
    return best_match


def merge_csv_files(feedback_csv, metrics_csv, output_csv):
    metrics_dict = {}
    with open(metrics_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = row.get("video_name", row.get("name", ""))
            try:
                num_frames = int(row.get("num_frames", 0) or 0)
            except (TypeError, ValueError):
                num_frames = 0
            metrics_dict[video_name] = {
                "sample_index": row.get("sample_index", ""),
                "name": row.get("name", ""),
                "num_frames": row.get("num_frames", ""),
                "mpjpe_g": row.get("mpjpe_g", ""),
                "mpjpe_l": row.get("mpjpe_l", ""),
                "mpjpe_pa": row.get("mpjpe_pa", ""),
                "accel_dist": row.get("accel_dist", ""),
                "vel_dist": row.get("vel_dist", ""),
                "success": "YES" if num_frames >= 180 else ("NO" if num_frames else row.get("success", "")),
            }

    with open(feedback_csv, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames + ["sample_index", "name", "num_frames", "mpjpe_g", "mpjpe_l", "mpjpe_pa", "accel_dist", "vel_dist", "success"]
        rows = []
        for row in reader:
            metrics = find_best_match(row["video_name"], metrics_dict)
            if metrics:
                row.update(metrics)
            else:
                row.update({
                    "sample_index": "",
                    "name": "",
                    "num_frames": "",
                    "mpjpe_g": "",
                    "mpjpe_l": "",
                    "mpjpe_pa": "",
                    "accel_dist": "",
                    "vel_dist": "",
                    "success": "",
                })
            rows.append(row)

    with open(output_csv, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Merge VLM feedback CSV with PHC metrics CSV.")
    parser.add_argument("feedback_csv")
    parser.add_argument("metrics_csv")
    parser.add_argument("output_csv")
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    merge_csv_files(args.feedback_csv, args.metrics_csv, args.output_csv)


if __name__ == "__main__":
    main()
