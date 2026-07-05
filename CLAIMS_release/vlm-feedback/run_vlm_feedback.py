import argparse
import csv
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from motion_analyzer import MotionAnalyzerV2
from merge_feedback_with_metrics import merge_csv_files
from video_processor import DifficultyVideoProcessor


def normalize_prompt_text(text: str) -> str:
    normalized = text.lower().replace("_", " ").replace("→", " ")
    normalized = re.sub(r"[^a-z0-9\s]", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_prompt_manifest(manifest_path: str) -> List[Dict[str, str]]:
    if not manifest_path or not os.path.exists(manifest_path):
        return []

    records: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            prompt_name = (row.get("prompt_name") or "").strip()
            category = (row.get("category") or "").strip()
            if not prompt_name or not category:
                continue
            records.append({
                "prompt_name": prompt_name,
                "category": category,
                "normalized_prompt": normalize_prompt_text(
                    row.get("normalized_prompt") or prompt_name
                ),
            })
    return records


def resolve_config_path(config_path: str, base_dir: Optional[Path]) -> str:
    if not config_path:
        return ""
    path = Path(config_path)
    if path.is_absolute() or base_dir is None:
        return str(path)
    return str((base_dir / path).resolve())


def load_allowed_video_names(allowed_videos_csv: str) -> List[str]:
    if not allowed_videos_csv or not os.path.exists(allowed_videos_csv):
        return []

    allowed: List[str] = []
    with open(allowed_videos_csv, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            video_name = (row.get("video_name") or "").strip()
            if video_name:
                allowed.append(Path(video_name).stem)
    return allowed


def infer_category_from_manifest(text: str, manifest_records: List[Dict[str, str]]) -> str:
    if not text or not manifest_records:
        return ""

    normalized_text = normalize_prompt_text(text)
    if not normalized_text:
        return ""

    for record in manifest_records:
        if normalized_text == record["normalized_prompt"]:
            return record["category"]

    best_match = ("", 0)
    for record in manifest_records:
        candidate = record["normalized_prompt"]
        if not candidate:
            continue
        if normalized_text in candidate or candidate in normalized_text:
            score = min(len(normalized_text), len(candidate))
            if score > best_match[1]:
                best_match = (record["category"], score)

    return best_match[0]


def infer_category_from_name(text: str) -> str:
    lower = text.lower()
    if any(token in lower for token in ["pirouette", "grand", "pas de", "fouett", "dancer", "allegro"]):
        return "dance"
    if any(token in lower for token in ["vault", "tsukahara", "gymnast", "cartwheel", "aerial"]):
        return "gymnastics"
    if any(token in lower for token in ["palm strike", "bow stance", "practitioner", "dragon", "tiger"]):
        return "martial_arts"
    if any(token in lower for token in ["fighter", "combatant", "hook punch", "roundhouse", "flying knee"]):
        return "combat"
    return "sport"


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(output_dir: str, log_level: str) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"vlm_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("vlm_feedback")
    logger.info("Log file: %s", log_path)
    return logger


def process_single_video_for_provider(
    video_path: str,
    config: dict,
    output_dir: str,
    provider_name: str,
    logger: logging.Logger,
) -> bool:
    video_name = Path(video_path).stem
    provider_output_dir = os.path.join(output_dir, provider_name, video_name)
    os.makedirs(provider_output_dir, exist_ok=True)

    processor = DifficultyVideoProcessor(config)
    analyzer = MotionAnalyzerV2(config, provider_name=provider_name)
    temp_frames_dir = None

    try:
        logger.info("[%s] Processing video: %s", provider_name, video_name)
        video_result = processor.process_video(video_path, provider_output_dir)
        temp_frames_dir = video_result["temp_frames_dir"]

        motion_result = analyzer.analyze_motion(
            frame_paths=video_result["extracted_frames"],
            video_name=video_name,
            output_dir=provider_output_dir,
        )

        output_json = os.path.join(provider_output_dir, f"{video_name}_analysis.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(motion_result, f, indent=2, ensure_ascii=False)

        logger.info("[%s] Saved analysis: %s", provider_name, output_json)
        return True
    except Exception as e:
        logger.error("[%s] Failed processing %s: %s", provider_name, video_name, e, exc_info=True)
        return False
    finally:
        if temp_frames_dir:
            processor.cleanup_temp_frames(temp_frames_dir)


def merge_provider_json(provider_root: str, output_json_path: str):
    provider_path = Path(provider_root)
    merged: List[Dict] = []
    for subdir in sorted(provider_path.iterdir()):
        if not subdir.is_dir():
            continue
        json_files = sorted(subdir.glob("*_analysis.json"))
        if not json_files:
            continue
        with open(json_files[0], "r", encoding="utf-8") as f:
            merged.append(json.load(f))
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)


def merge_dual_json_to_csv(
    gpt_json_path: str,
    qwen_json_path: str,
    output_csv_path: str,
    prompt_manifest_path: str = "",
):
    with open(gpt_json_path, "r", encoding="utf-8") as f:
        gpt_data = json.load(f)
    with open(qwen_json_path, "r", encoding="utf-8") as f:
        qwen_data = json.load(f)

    assert len(gpt_data) == len(qwen_data), "Provider result sizes do not match."
    manifest_records = load_prompt_manifest(prompt_manifest_path)

    rows = []
    for gpt_item, qwen_item in zip(gpt_data, qwen_data):
        gpt_name = gpt_item.get("video_name", "")
        qwen_name = qwen_item.get("video_name", "")
        video_name = qwen_name if len(qwen_name) >= len(gpt_name) else gpt_name
        prompt_name = video_name.replace("_", " ")
        category = (
            infer_category_from_manifest(prompt_name, manifest_records)
            or infer_category_from_manifest(video_name, manifest_records)
            or infer_category_from_name(video_name)
        )

        gpt_feedback = gpt_item.get("feedback", {}) or {}
        gpt_analysis = gpt_item.get("analysis", {}) or {}

        qwen_feedback = qwen_item.get("feedback", {}) or {}
        qwen_analysis = qwen_item.get("analysis", {}) or {}

        rows.append({
            "video_name": video_name,
            "category": category,
            "prompt_name": prompt_name,
            "gpt4o_difficulty_score": gpt_item.get("difficulty_score", ""),
            "gpt4o_action_name": gpt_item.get("action_name", ""),
            "gpt4o_action_sequence": gpt_analysis.get("action_sequence", ""),
            "gpt4o_technical_complexity": gpt_analysis.get("technical_complexity", ""),
            "gpt4o_movement_intensity": gpt_analysis.get("movement_intensity", ""),
            "gpt4o_balance_requirement": gpt_analysis.get("balance_requirement", ""),
            "gpt4o_continuity": gpt_analysis.get("continuity", ""),
            "gpt4o_scoring_reason": gpt_item.get("scoring_reason", ""),
            "gpt_description": gpt_feedback.get("description", ""),
            "gpt_key_events": gpt_feedback.get("key_events", ""),
            "gpt_dynamism_description": gpt_feedback.get("dynamism_description", ""),
            "gpt_complexity_description": gpt_feedback.get("complexity_description", ""),
            "gpt_difficulty_description": gpt_feedback.get("difficulty_description", ""),
            "gpt_increase_dynamism_suggestion": gpt_feedback.get("increase_dynamism_suggestion", ""),
            "gpt_increase_complexity_suggestion": gpt_feedback.get("increase_complexity_suggestion", ""),
            "gpt_increase_difficulty_suggestion": gpt_feedback.get("increase_difficulty_suggestion", ""),
            "qwen_difficulty_score": qwen_item.get("difficulty_score", ""),
            "qwen_action_name": qwen_item.get("action_name", ""),
            "qwen_action_sequence": qwen_analysis.get("action_sequence", ""),
            "qwen_technical_complexity": qwen_analysis.get("technical_complexity", ""),
            "qwen_movement_intensity": qwen_analysis.get("movement_intensity", ""),
            "qwen_balance_requirement": qwen_analysis.get("balance_requirement", ""),
            "qwen_continuity": qwen_analysis.get("continuity", ""),
            "qwen_scoring_reason": qwen_item.get("scoring_reason", ""),
            "qwen_description": qwen_feedback.get("description", ""),
            "qwen_key_events": qwen_feedback.get("key_events", ""),
            "qwen_dynamism_description": qwen_feedback.get("dynamism_description", ""),
            "qwen_complexity_description": qwen_feedback.get("complexity_description", ""),
            "qwen_difficulty_description": qwen_feedback.get("difficulty_description", ""),
            "qwen_increase_dynamism_suggestion": qwen_feedback.get("increase_dynamism_suggestion", ""),
            "qwen_increase_complexity_suggestion": qwen_feedback.get("increase_complexity_suggestion", ""),
            "qwen_increase_difficulty_suggestion": qwen_feedback.get("increase_difficulty_suggestion", ""),
        })

    fieldnames = list(rows[0].keys()) if rows else []
    with open(output_csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)


def batch_process_videos(
    input_dir: str,
    config: dict,
    output_dir: str,
    logger: logging.Logger,
    allowed_video_names: Optional[List[str]] = None,
    config_dir: Optional[Path] = None,
):
    video_files = sorted(Path(input_dir).glob("*.mp4"))
    if not video_files:
        raise FileNotFoundError(f"No mp4 files found in {input_dir}")

    if allowed_video_names:
        allowed_set = {Path(name).stem for name in allowed_video_names}
        video_files = [path for path in video_files if path.stem in allowed_set]
        logger.info("Filtered VLM feedback input to %d retained videos", len(video_files))
        if not video_files:
            raise ValueError("No videos remain after applying the alignment filter.")

    providers = ["gpt4o", "qwen"]
    report = {"total": len(video_files), "providers": {}, "details": []}

    for provider_name in providers:
        provider_success = 0
        provider_failed = 0
        for idx, video_path in enumerate(video_files, 1):
            logger.info("[%s] Progress %d/%d: %s", provider_name, idx, len(video_files), video_path.name)
            success = process_single_video_for_provider(
                video_path=str(video_path),
                config=config,
                output_dir=output_dir,
                provider_name=provider_name,
                logger=logger,
            )
            if success:
                provider_success += 1
            else:
                provider_failed += 1
        report["providers"][provider_name] = {"success": provider_success, "failed": provider_failed}

    gpt_merged_json = os.path.join(output_dir, "gpt_analysis_merged.json")
    qwen_merged_json = os.path.join(output_dir, "qwen_analysis_merged.json")
    merge_provider_json(os.path.join(output_dir, "gpt4o"), gpt_merged_json)
    merge_provider_json(os.path.join(output_dir, "qwen"), qwen_merged_json)

    merged_feedback_csv = os.path.join(output_dir, "merged_feedback.csv")
    merge_dual_json_to_csv(
        gpt_merged_json,
        qwen_merged_json,
        merged_feedback_csv,
        prompt_manifest_path=resolve_config_path(config.get("prompt_manifest", ""), config_dir),
    )

    final_csv_name = config.get("final_csv_name", "claims_loop0_vlm_feedback.csv")
    final_csv_path = os.path.join(output_dir, final_csv_name)
    metrics_csv = resolve_config_path(config.get("metrics_csv", ""), config_dir)
    if metrics_csv and os.path.exists(metrics_csv):
        merge_csv_files(merged_feedback_csv, metrics_csv, final_csv_path)
    else:
        final_csv_path = merged_feedback_csv

    report["merged_outputs"] = {
        "gpt_json": gpt_merged_json,
        "qwen_json": qwen_merged_json,
        "merged_csv": merged_feedback_csv,
        "final_csv": final_csv_path,
    }

    report_path = os.path.join(output_dir, "batch_analysis_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Saved batch report: %s", report_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Run dual-provider VLM feedback on rendered motion videos.")
    default_config = Path(__file__).with_name("config_difficulty.yaml")
    parser.add_argument("--batch", action="store_true", help="Batch mode.")
    parser.add_argument("--input", type=str, help="Input video directory for batch mode.")
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/vlm_feedback",
        help="Output directory.",
    )
    parser.add_argument(
        "--final-csv-name",
        type=str,
        default="claims_loop0_vlm_feedback.csv",
        help="Final merged CSV file name.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Config file path.",
    )
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument(
        "--allowed-videos-csv",
        type=str,
        default="",
        help="Optional CSV with a video_name column. Only the listed retained videos will be processed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.batch:
        raise ValueError("This release entry currently supports batch mode only. Pass --batch.")
    if not args.input:
        raise ValueError("Batch mode requires --input.")

    config = load_config(args.config)
    config_dir = Path(args.config).resolve().parent
    logger = setup_logging(args.output, args.log_level)
    if args.final_csv_name:
        config["final_csv_name"] = args.final_csv_name
    allowed_video_names = load_allowed_video_names(args.allowed_videos_csv)
    batch_process_videos(
        args.input,
        config,
        args.output,
        logger,
        allowed_video_names=allowed_video_names,
        config_dir=config_dir,
    )


if __name__ == "__main__":
    main()
