"""
Prompt-motion semantic alignment filtering for rendered motion videos.

This script sits between:
1. rendered motion videos
2. downstream VLM feedback

It builds or consumes a `video_name,prompt_name` CSV, evaluates alignment with a
VLM on stitched keyframes, and exports:
- per-video JSON results
- `alignment_summary.csv`
- `accepted_videos.csv`
- `accepted_alignment_inputs.csv`
"""

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from video_processor_alignment import AlignmentVideoProcessor  # noqa: E402


def normalize_text(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]", "", text.lower())


def setup_logging(output_dir: Path, level: str = "INFO") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "prompt_alignment.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("prompt_alignment")
    logger.info("Log file: %s", log_file)
    return logger


class VideoRepository:
    def __init__(self, root: Path, logger: logging.Logger):
        self.root = root
        self.logger = logger
        self._index = self._build_index()

    def _build_index(self) -> Dict[str, Path]:
        mapping: Dict[str, Path] = {}
        for mp4_path in self.root.rglob("*.mp4"):
            stem = mp4_path.stem
            if stem in mapping:
                self.logger.warning(
                    "Duplicate video stem '%s'; keeping first path: %s",
                    stem,
                    mapping[stem],
                )
                continue
            mapping[stem] = mp4_path
        self.logger.info("Indexed %d mp4 files", len(mapping))
        return mapping

    def all_video_stems(self) -> List[str]:
        return sorted(self._index.keys())

    def find(self, video_name: str) -> Optional[Path]:
        if not video_name:
            return None
        stem = Path(video_name).stem
        if stem in self._index:
            return self._index[stem]
        direct = self.root / video_name
        if direct.exists():
            return direct
        candidate = self.root / f"{stem}.mp4"
        if candidate.exists():
            return candidate
        return None


def load_prompt_manifest(manifest_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(manifest_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        prompt_key = next((name for name in fieldnames if "prompt_name" in name), fieldnames[0] if fieldnames else "prompt_name")
        for row in reader:
            prompt_name = (row.get(prompt_key) or "").strip().strip('"')
            if not prompt_name:
                continue
            rows.append(
                {
                    "prompt_name": prompt_name,
                    "normalized_prompt": normalize_text(prompt_name),
                    "category": (row.get("category") or "").strip(),
                }
            )
    return rows


def match_prompt_to_video(video_stem: str, prompt_rows: Sequence[Dict[str, str]]) -> Optional[str]:
    normalized_video = normalize_text(video_stem)
    if not normalized_video:
        return None

    best_prompt = None
    best_score = -1
    for row in prompt_rows:
        prompt = row["prompt_name"]
        normalized_prompt = row["normalized_prompt"]
        if not normalized_prompt:
            continue
        if normalized_video.startswith(normalized_prompt) or normalized_prompt.startswith(normalized_video):
            return prompt
        common_prefix = len(
            __import__("os").path.commonprefix([normalized_video, normalized_prompt])
        )
        if common_prefix > best_score:
            best_score = common_prefix
            best_prompt = prompt
    return best_prompt


def build_alignment_input_csv(
    manifest_path: Path,
    videos_root: Path,
    output_csv: Path,
    logger: logging.Logger,
) -> Path:
    prompt_rows = load_prompt_manifest(manifest_path)
    video_repo = VideoRepository(videos_root, logger)

    records = []
    for video_stem in video_repo.all_video_stems():
        prompt = match_prompt_to_video(video_stem, prompt_rows)
        if not prompt:
            logger.warning("Could not infer prompt for video: %s", video_stem)
            continue
        records.append({"video_name": video_stem, "prompt_name": prompt})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["video_name", "prompt_name"])
        writer.writeheader()
        writer.writerows(records)

    logger.info("Wrote alignment input CSV: %s (%d rows)", output_csv, len(records))
    return output_csv


class PromptAlignmentModel:
    def __init__(self, config: Dict, logger: logging.Logger, provider: str):
        provider_cfg = self._resolve_provider_config(config, provider)
        api_key = provider_cfg.get("api_key")
        api_key_env = provider_cfg.get("api_key_env")
        if not api_key and api_key_env:
            api_key = __import__("os").environ.get(api_key_env, "")

        self.client = OpenAI(
            api_key=api_key,
            base_url=provider_cfg.get("base_url"),
        )
        self.model = provider_cfg.get("model")
        self.max_tokens = provider_cfg.get("max_tokens", 2000)
        self.temperature = provider_cfg.get("temperature", 0.0)
        self.logger = logger

    @staticmethod
    def _resolve_provider_config(config: Dict, provider: str) -> Dict:
        providers = config.get("providers", {})
        if provider in providers:
            return providers[provider]
        if "gpt4o" in config:
            return config["gpt4o"]
        raise KeyError(f"Could not resolve provider config for '{provider}'")

    def _build_prompt(self, video_name: str, prompt_text: str) -> str:
        return f"""
You are a motion alignment evaluator. Compare 60 sequential frames (left to right, about 6 seconds) with the provided motion prompt.

Judge alignment only from visible body motion. Ignore scene, props, missing equipment, imperfect camera view, body-shape differences, rendering artifacts, and environmental differences.

This evaluation should be intentionally recall-oriented and lenient. The goal is to reject only clearly wrong videos, while keeping any video that shows a plausible semantic relation to the prompt.

Prioritize high-level action semantics far more than exact details:
- First identify the broad action family or intent, such as combat / dance / locomotion / jump / turn / stretch / balance / floor movement / athletic skill.
- If the video appears to be in the same broad action family, strongly favor "partial" or "aligned".
- If the video shows any plausible subset, approximation, weakened version, or visually related variant of the prompt, count it as semantically related.
- Treat simplified, incomplete, approximate, lower-amplitude, slower, shorter, partially executed, or loosely ordered versions as acceptable matches.
- Treat differences in timing, ordering, body side, facing direction, style, transition quality, repetition count, and execution quality as minor issues unless they completely change the action family.
- When prompts contain multiple sub-actions, only one central sub-action or one clearly related motion phrase is enough for at least "partial".
- Missing modifiers, adjectives, equipment references, stylistic details, or fine-grained technique terms should almost never cause rejection.
- If the motion is ambiguous but not clearly contradictory, prefer "partial".
- Use "mismatch" very rarely and only when the clip is clearly from a different action family, has essentially no visible core correspondence, or strongly contradicts the prompt.

Bias the decision toward acceptance:
- If there is any reasonable semantic case for relatedness, choose "partial".
- Reserve "mismatch" for only the most obvious failures.
- In borderline cases, score upward rather than downward.

Use these labels:
- aligned: the main action semantics are clearly present, or the clip is a broadly correct and plausible realization of the prompt even with missing details
- partial: the clip is semantically related, shares the same broad action family, or captures at least one central sub-action / plausible variant
- mismatch: the clip is clearly unrelated, clearly contradictory, or belongs to a different action family with no meaningful overlap

Score guidance:
- 80-100: strong semantic match
- 60-79: same broad action family and visibly related core behavior
- 40-59: weak-to-moderate but still acceptable semantic relation; this should still usually map to "partial"
- 20-39: only slight relation, but still not clearly contradictory
- 0-19: clear mismatch; use sparingly

Video Name: {video_name}
Motion Prompt:
\"\"\"{prompt_text}\"\"\"

Return ONLY JSON:
{{
  "video_name": "{video_name}",
  "prompt_name": "string",
  "alignment_result": {{
    "alignment_score": 0-100,
    "verdict": "aligned" | "partial" | "mismatch",
    "frame_observation": "<=200 chars",
    "prompt_overlap": "<=200 chars",
    "missing_elements": "<=200 chars"
  }}
}}
""".strip()

    def _extract_json(self, text: str) -> Dict:
        brace_stack: List[int] = []
        start_idx = -1
        for idx, char in enumerate(text):
            if char == "{":
                if not brace_stack:
                    start_idx = idx
                brace_stack.append(idx)
            elif char == "}":
                if brace_stack:
                    brace_stack.pop()
                    if not brace_stack and start_idx != -1:
                        candidate = text[start_idx : idx + 1]
                        try:
                            return json.loads(candidate)
                        except json.JSONDecodeError:
                            continue
        raise ValueError(f"Could not parse JSON from model output: {text[:500]}")

    def analyze(self, video_name: str, prompt_name: str, stitched_base64: str) -> Dict:
        prompt = self._build_prompt(video_name, prompt_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{stitched_base64}"}},
                ],
            }
        ]
        self.logger.info("Running alignment model %s", self.model)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        content = response.choices[0].message.content
        result = self._extract_json(content)
        result.setdefault("video_name", video_name)
        result.setdefault("prompt_name", prompt_name)
        return result


@dataclass
class AlignmentTask:
    video_name: str
    prompt_text: str


class VideoPromptAlignmentExperiment:
    def __init__(
        self,
        config_path: Path,
        videos_root: Path,
        csv_path: Path,
        output_dir: Path,
        logger: logging.Logger,
        provider: str,
        threshold: float,
    ):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.logger = logger
        self.videos_root = videos_root
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.threshold = threshold
        self.processor = AlignmentVideoProcessor(self.config)
        self.alignment_model = PromptAlignmentModel(self.config, self.logger, provider=provider)
        self.video_repo = VideoRepository(self.videos_root, self.logger)
        self.tasks, self.task_lookup = self._load_tasks()

    def _load_tasks(self) -> Tuple[List[AlignmentTask], Dict[str, AlignmentTask]]:
        tasks: List[AlignmentTask] = []
        lookup: Dict[str, AlignmentTask] = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_name = (row.get("video_name") or "").strip()
                prompt_text = (row.get("prompt_name") or "").strip()
                if not video_name or not prompt_text:
                    self.logger.warning("Skipping row with missing fields: %s", row)
                    continue
                task = AlignmentTask(video_name=video_name, prompt_text=prompt_text)
                tasks.append(task)
                lookup[Path(video_name).stem] = task
        self.logger.info("Loaded %d alignment tasks", len(tasks))
        return tasks, lookup

    def _is_accepted(self, score_value) -> bool:
        score = self._parse_score(score_value)
        return score is not None and score >= self.threshold

    def _write_accept_lists(self, summary_rows: List[Dict]):
        accepted_videos_path = self.output_dir / "accepted_videos.csv"
        accepted_inputs_path = self.output_dir / "accepted_alignment_inputs.csv"

        accepted_rows = [row for row in summary_rows if self._is_accepted(row.get("alignment_score"))]

        with open(accepted_videos_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name"])
            writer.writeheader()
            for row in accepted_rows:
                writer.writerow({"video_name": row["video_name"]})

        with open(accepted_inputs_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_name", "prompt_name"])
            writer.writeheader()
            for row in accepted_rows:
                writer.writerow({"video_name": row["video_name"], "prompt_name": row["prompt_name"]})

        self.logger.info("Accepted videos: %d / %d", len(accepted_rows), len(summary_rows))
        self.logger.info("Wrote accepted list: %s", accepted_videos_path)
        self.logger.info("Wrote accepted input CSV: %s", accepted_inputs_path)

    def run_batch(self, limit: Optional[int], start_index: int, skip_existing: bool):
        summary_rows: List[Dict] = []
        stats_path = self.output_dir / "alignment_stats.csv"
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, "w", encoding="utf-8", newline="") as stats_file:
            stats_writer = csv.DictWriter(stats_file, fieldnames=["video_name", "aligned", "alignment_rate"])
            stats_writer.writeheader()
            stats_total = 0
            stats_aligned = 0

            for idx, task in enumerate(self.tasks):
                if idx < start_index:
                    continue
                if limit is not None and len(summary_rows) >= limit:
                    break

                self.logger.info("\n%s", "=" * 80)
                self.logger.info("Processing task #%d: %s", idx + 1, task.video_name)

                video_path = self.video_repo.find(task.video_name)
                if not video_path:
                    self.logger.error("Missing video file: %s", task.video_name)
                    summary_rows.append(
                        {
                            "video_name": task.video_name,
                            "prompt_name": task.prompt_text,
                            "status": "missing_video",
                            "alignment_score": "",
                            "verdict": "",
                            "prompt_overlap": "",
                            "missing_elements": "",
                        }
                    )
                    continue

                video_output_dir = self.output_dir / task.video_name
                video_output_dir.mkdir(parents=True, exist_ok=True)
                result_path = video_output_dir / f"{task.video_name}_alignment.json"

                if skip_existing and result_path.exists():
                    with open(result_path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                    alignment = result.get("alignment_result", {})
                    summary_rows.append(
                        {
                            "video_name": task.video_name,
                            "prompt_name": task.prompt_text,
                            "status": "skipped",
                            "alignment_score": alignment.get("alignment_score", ""),
                            "verdict": alignment.get("verdict", ""),
                            "prompt_overlap": alignment.get("prompt_overlap", ""),
                            "missing_elements": alignment.get("missing_elements", ""),
                        }
                    )
                    stats_total, stats_aligned = self._record_alignment_stat(
                        stats_writer, stats_file, task.video_name, alignment.get("alignment_score"), stats_total, stats_aligned
                    )
                    continue

                temp_dir = None
                try:
                    video_result = self.processor.process_video(str(video_path), str(video_output_dir))
                    temp_dir = video_result.get("temp_frames_dir")
                    result = self.alignment_model.analyze(
                        video_name=task.video_name,
                        prompt_name=task.prompt_text,
                        stitched_base64=video_result["stitched_image_base64"],
                    )
                    with open(result_path, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False)

                    alignment = result.get("alignment_result", {})
                    summary_rows.append(
                        {
                            "video_name": task.video_name,
                            "prompt_name": task.prompt_text,
                            "status": "success",
                            "alignment_score": alignment.get("alignment_score", ""),
                            "verdict": alignment.get("verdict", ""),
                            "prompt_overlap": alignment.get("prompt_overlap", ""),
                            "missing_elements": alignment.get("missing_elements", ""),
                        }
                    )
                    stats_total, stats_aligned = self._record_alignment_stat(
                        stats_writer, stats_file, task.video_name, alignment.get("alignment_score"), stats_total, stats_aligned
                    )
                except Exception as exc:
                    self.logger.error("Failed processing %s: %s", task.video_name, exc, exc_info=True)
                    summary_rows.append(
                        {
                            "video_name": task.video_name,
                            "prompt_name": task.prompt_text,
                            "status": f"error: {exc}",
                            "alignment_score": "",
                            "verdict": "",
                            "prompt_overlap": "",
                            "missing_elements": "",
                        }
                    )
                finally:
                    if temp_dir:
                        self.processor.cleanup_temp_frames(temp_dir)

        summary_path = self.output_dir / "alignment_summary.csv"
        with open(summary_path, "w", encoding="utf-8", newline="") as f:
            fieldnames = [
                "video_name",
                "prompt_name",
                "status",
                "alignment_score",
                "verdict",
                "prompt_overlap",
                "missing_elements",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

        self.logger.info("Wrote alignment summary: %s", summary_path)
        self._write_accept_lists(summary_rows)

    def run_single(self, video_path: Optional[Path], video_name: Optional[str]):
        if not video_path and not video_name:
            raise ValueError("Single mode requires --video-path or --video-name")

        if video_path:
            if not video_path.exists():
                raise FileNotFoundError(f"Missing mp4 file: {video_path}")
            resolved_path = video_path
            resolved_name = video_path.stem
        else:
            resolved_path = self.video_repo.find(video_name or "")
            if not resolved_path:
                raise FileNotFoundError(f"Could not find video under {self.videos_root}: {video_name}")
            resolved_name = Path(video_name).stem

        task = self.task_lookup.get(resolved_name)
        if not task:
            raise ValueError(f"No prompt found in alignment CSV for video {resolved_name}")

        video_output_dir = self.output_dir / resolved_name
        video_output_dir.mkdir(parents=True, exist_ok=True)
        result_json_path = video_output_dir / f"{resolved_name}_alignment.json"

        temp_dir = None
        try:
            video_result = self.processor.process_video(str(resolved_path), str(video_output_dir))
            temp_dir = video_result.get("temp_frames_dir")
            result = self.alignment_model.analyze(
                video_name=resolved_name,
                prompt_name=task.prompt_text,
                stitched_base64=video_result["stitched_image_base64"],
            )
            with open(result_json_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            self._write_single_stat(resolved_name, result.get("alignment_result", {}).get("alignment_score"))
        finally:
            if temp_dir:
                self.processor.cleanup_temp_frames(temp_dir)

    def _record_alignment_stat(
        self,
        writer: csv.DictWriter,
        stats_file,
        video_name: str,
        score_value,
        total_count: int,
        aligned_count: int,
    ) -> Tuple[int, int]:
        score = self._parse_score(score_value)
        if score is None:
            return total_count, aligned_count

        total_count += 1
        aligned_flag = score >= self.threshold
        if aligned_flag:
            aligned_count += 1
        rate = (aligned_count / total_count) * 100 if total_count else 0.0
        writer.writerow(
            {
                "video_name": video_name,
                "aligned": "yes" if aligned_flag else "no",
                "alignment_rate": f"{rate:.2f}%",
            }
        )
        stats_file.flush()
        return total_count, aligned_count

    def _write_single_stat(self, video_name: str, score_value):
        score = self._parse_score(score_value)
        if score is None:
            return
        stats_path = self.output_dir / "alignment_stats.csv"
        with open(stats_path, "w", encoding="utf-8", newline="") as stats_file:
            writer = csv.DictWriter(stats_file, fieldnames=["video_name", "aligned", "alignment_rate"])
            writer.writeheader()
            aligned_flag = score >= self.threshold
            writer.writerow(
                {
                    "video_name": video_name,
                    "aligned": "yes" if aligned_flag else "no",
                    "alignment_rate": f"{100.0 if aligned_flag else 0.0:.2f}%",
                }
            )

    @staticmethod
    def _parse_score(score_value) -> Optional[float]:
        if score_value is None or score_value == "":
            return None
        try:
            return float(score_value)
        except (TypeError, ValueError):
            return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt-motion semantic alignment filtering for rendered videos.")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config_difficulty.yaml")
    parser.add_argument("--videos-root", type=Path, required=True)
    parser.add_argument("--csv-path", type=Path, help="Alignment input CSV with columns video_name,prompt_name.")
    parser.add_argument("--manifest-path", type=Path, help="Prompt manifest used to auto-build the alignment input CSV.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", type=str, default="qwen", choices=["qwen", "gpt4o"])
    parser.add_argument("--alignment-threshold", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--video-path", type=Path)
    parser.add_argument("--video-name", type=str)
    parser.add_argument("--build-input-csv", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser.parse_args()


def main():
    args = parse_args()
    logger = setup_logging(args.output_dir, args.log_level)

    csv_path = args.csv_path
    if args.build_input_csv:
        if not args.manifest_path:
            raise ValueError("--build-input-csv requires --manifest-path")
        csv_path = build_alignment_input_csv(
            manifest_path=args.manifest_path,
            videos_root=args.videos_root,
            output_csv=args.output_dir / "alignment_input.csv",
            logger=logger,
        )

    if not csv_path:
        raise ValueError("Provide --csv-path or use --build-input-csv with --manifest-path")

    experiment = VideoPromptAlignmentExperiment(
        config_path=args.config,
        videos_root=args.videos_root,
        csv_path=csv_path,
        output_dir=args.output_dir,
        logger=logger,
        provider=args.provider,
        threshold=args.alignment_threshold,
    )

    if args.video_path or args.video_name:
        experiment.run_single(video_path=args.video_path, video_name=args.video_name)
    else:
        experiment.run_batch(limit=args.limit, start_index=args.start_index, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
