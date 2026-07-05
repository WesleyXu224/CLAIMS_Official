# VLM Feedback Module

This folder is the release-facing entry for the video-based VLM feedback stage.

Purpose:

- consume rendered motion videos
- filter videos by prompt-motion semantic alignment
- run VLM-based analysis and scoring
- produce feedback that is merged with controller metrics for the next prompt loop

This release module now contains a self-contained batch entrypoint:

- [`video_prompt_alignment.py`](video_prompt_alignment.py)
- [`run_vlm_feedback.py`](run_vlm_feedback.py)

Recommended order:

1. run `video_prompt_alignment.py` on rendered `.mp4` videos
2. remove clips whose verdict is `mismatch`
3. run `run_vlm_feedback.py` on the accepted subset

The alignment stage extracts 60 frames per video, stitches them into a single image, and evaluates prompt-motion semantic consistency. The feedback stage then extracts 15 frames per accepted video, runs two provider passes (`gpt4o` and `qwen`), and exports merged JSON and CSV outputs.

Required runtime setup:

- install an environment with `openai`, `PyYAML`, `opencv-python`, `Pillow`, and `tqdm`
- export `GPT4O_API_KEY`
- export `QWEN_API_KEY`

Current default API configuration:

- `gpt4o`:
  - base URL: `https://api2.aigcbest.top/v1`
  - model: `gpt-4o`
- `qwen`:
  - base URL: `https://api2.aigcbest.top/v1`
  - model: `qwen-vl-max`

Default config:

- [`config_difficulty.yaml`](config_difficulty.yaml)

Example batch command:

```bash
cd CLAIMS_release
export QWEN_API_KEY=YOUR_QWEN_KEY
python vlm-feedback/video_prompt_alignment.py \
  --config vlm-feedback/config_difficulty.yaml \
  --videos-root outputs/videos/loop0/mp4 \
  --manifest-path outputs/prompts/loop0/loop0_prompt_manifest.csv \
  --build-input-csv \
  --provider qwen \
  --alignment-threshold 50 \
  --output-dir outputs/vlm_alignment/loop0
```

Then run feedback on the retained videos. For the public release, the recommended retained set is the non-`mismatch` subset produced by `scripts/filter_phc_training_set.py`:

```bash
cd CLAIMS_release
export GPT4O_API_KEY=YOUR_GPT4O_KEY
export QWEN_API_KEY=YOUR_QWEN_KEY
python vlm-feedback/run_vlm_feedback.py \
  --batch \
  --input outputs/videos/loop0/mp4 \
  --allowed-videos-csv outputs/phc/loop0/kept_videos.csv \
  --output outputs/vlm_feedback/loop0
```

Outputs:

- `alignment_summary.csv`
- `alignment_stats.csv`
- `accepted_videos.csv`
- `accepted_alignment_inputs.csv`
- `gpt4o/{video_name}/{video_name}_analysis.json`
- `qwen/{video_name}/{video_name}_analysis.json`
- `gpt_analysis_merged.json`
- `qwen_analysis_merged.json`
- `claims_loop0_vlm_feedback.csv`
- one `batch_analysis_report.json` for the batch

The merged CSV keeps the legacy column schema used by the original internal pipeline:

- old-style difficulty outputs:
  - `gpt4o_difficulty_score`
  - `gpt4o_action_sequence`
  - `gpt4o_technical_complexity`
  - `gpt4o_movement_intensity`
  - `gpt4o_balance_requirement`
  - `gpt4o_continuity`
  - `gpt4o_scoring_reason`
  - and the matching `qwen_*` fields
- appended PHC metrics:
  - `sample_index`
  - `name`
  - `num_frames`
  - `mpjpe_g`
  - `mpjpe_l`
  - `mpjpe_pa`
  - `accel_dist`
  - `vel_dist`
  - `success`

To merge VLM feedback with PHC per-sample metrics:

```bash
cd CLAIMS_release
python vlm-feedback/merge_feedback_with_metrics.py \
  outputs/vlm_feedback/loop0/claims_loop0_vlm_feedback.csv \
  PHC/outputs/eval/phc/claims/loop0_prompts_hik_filtered/latest/per_sample_metrics.csv \
  outputs/vlm_feedback/loop0/merged_feedback_with_metrics.csv
```
