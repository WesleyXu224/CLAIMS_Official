# CLAIMS Release Workspace

This directory is the release-facing entrypoint for the CLAIMS pipeline.

CLAIMS is an iterative loop:

```text
loop0 prompts
  -> text-to-motion generation
  -> HIK / PHC source conversion
  -> AMASS conversion / video rendering
  -> prompt-motion alignment filtering
  -> keep non-mismatch motions only
  -> filtered PHC training set / training / evaluation
  -> VLM feedback on retained videos
  -> PHC metrics + VLM feedback
  -> next-loop prompts
  -> repeat
```

## What This Folder Is For

`CLAIMS_release/` provides the cleanest public workflow in this repository:

- prompt generation
- motion generation
- HIK / PHC source conversion
- AMASS export
- video rendering
- prompt-motion alignment filtering
- filtered PHC training-set construction
- PHC training and evaluation
- VLM feedback
- next-loop prompt synthesis

The wrapper scripts in `scripts/` point to the maintained code under the repository root.

## Layout

```text
CLAIMS_release/
|-- README.md
|-- docs/
|   |-- pipeline.md
|   |-- environments.md
|   `-- data_assets.md
|-- scripts/
|   |-- generate_initial_prompts.py
|   |-- filter_phc_training_set.py
|   `-- generate_next_loop_prompts.py
|-- outputs/
|-- prompt-generate/
|-- motion-generate/
|-- motion-convert/
|-- PHC/
`-- vlm-feedback/
```

## Prerequisites

You will typically need two Python environments:

- `mdm`: for prompt generation and text-to-motion generation
- `phc_420`: for PHC training, evaluation, AMASS export, and rendering

Additional assets you must prepare yourself:

- MDM checkpoint for text-to-motion generation
- SMPL / SMPL-X assets
- Isaac Gym runtime required by PHC

Details are documented in [docs/environments.md](docs/environments.md) and [docs/data_assets.md](docs/data_assets.md).

## Quick Start With Released PHC Checkpoints

If you want to directly run inference with released CLAIMS controllers, use the Hugging Face asset release:

- `https://huggingface.co/Jimmy061/claims_phc`

The asset release includes:

- CLAIMS PHC checkpoints from `L0` to `L6`
- the baseline PHC checkpoint used for comparison
- four released evaluation datasets in `pkl` format

Minimal evaluation command:

```bash
cd "$RELEASE_ROOT/PHC"
CUDA_VISIBLE_DEVICES=0 \
"$PHC_PYTHON" phc/run_hydra.py \
  learning=im_big \
  exp_name=claims_eval \
  env=env_im \
  robot=smpl_humanoid \
  env.motion_file=<eval_dataset.pkl> \
  env.num_envs=<num_motions> \
  test=True \
  epoch=-1 \
  im_eval=True \
  headless=True \
  no_log=True \
  checkpoint=<checkpoint.pth>
```

Use this path when you only need controller inference and evaluation. Use the step-by-step pipeline below when you want to reproduce the full CLAIMS closed-loop data generation process.

## Recommended Repository Variables

Run commands from the repository root unless stated otherwise:

```bash
export REPO_ROOT=$(pwd)
export RELEASE_ROOT="$REPO_ROOT/CLAIMS_release"
```

Optional convenience variables:

```bash
export MDM_PYTHON=python
export PHC_PYTHON=python
export SMPL_DATA_DIR=/path/to/smpl_assets
```

Replace these with the correct interpreter paths if you do not rely on the currently activated environment.

## Step 1: Generate `loop0` Prompts

Run:

```bash
cd "$RELEASE_ROOT"
python scripts/generate_initial_prompts.py --count 40 --seed 7
```

Outputs:

```text
outputs/prompts/loop0/
  combat_prompt.txt
  dance_prompt.txt
  gymnastics_prompt.txt
  martial_arts_prompt.txt
  sport_prompt.txt
  loop0_prompts.txt
  loop0_prompt_manifest.csv
```

Notes:

- one prompt file is produced per category
- `loop0_prompts.txt` concatenates all prompts
- `loop0_prompt_manifest.csv` stores the canonical `prompt -> category` mapping for later loops

## Step 2: Generate Motions From Prompts

Run inside the `motion-generate/` module with the `mdm` environment:

```bash
cd "$RELEASE_ROOT/motion-generate"
"$MDM_PYTHON" language_to_pose_server_generate_example.py \
  --model_path ./save/humanml_trans_dec_512_bert/models_to_upload/humanml_trans_dec_512_bert/model000200000.pt \
  --text_encoder_type bert \
  --prompts_path "$RELEASE_ROOT/outputs/prompts/loop0" \
  --prompts_file_names loop0_prompts \
  --motion_save_path "$RELEASE_ROOT/outputs/motions/loop0"
```

Main input:

- `outputs/prompts/loop0/loop0_prompts.txt`

Main output:

- `outputs/motions/loop0/loop0_prompts/`

## Step 3: Convert Generated Motions To PHC Source Format

This has two sub-steps.

### 3.1 Convert MDM Motions To HIK

Use the release wrapper instead of calling the internal `sample/edit.py` module directly.

```bash
cd "$RELEASE_ROOT"
python scripts/convert_motions_to_hik.py \
  --input-dir outputs/motions/loop0/loop0_prompts \
  --output-dir outputs/motions/loop0/loop0_prompts_hik
```

Output:

- `outputs/motions/loop0/loop0_prompts_hik/`

### 3.2 Convert HIK Motions To PHC `pkl`

```bash
cd "$RELEASE_ROOT/PHC/scripts/data_process"
"$PHC_PYTHON" convert_data_mdm_hik.py \
  --base_path "$RELEASE_ROOT/outputs/motions/loop0" \
  --folders loop0_prompts_hik \
  --single_output_root "$RELEASE_ROOT/outputs/phc/loop0/single_dict" \
  --full_output_dir "$RELEASE_ROOT/outputs/phc/loop0/full" \
  --smpl_data_dir "$SMPL_DATA_DIR"
```

Outputs:

- `outputs/phc/loop0/single_dict/`
- `outputs/phc/loop0/full/loop0_prompts_hik.pkl`

At this point, you have the motion assets needed for both branches:

- the PHC-readable motion source: `outputs/phc/loop0/full/loop0_prompts_hik.pkl`
- the per-sample HIK files: `outputs/motions/loop0/loop0_prompts_hik/`

Do not train PHC yet. First render videos and run prompt-motion alignment filtering.

## Step 4: Convert PHC Motions To AMASS `npz`

```bash
cd "$REPO_ROOT"
"$PHC_PYTHON" CLAIMS_release/PHC/scripts/data_process/convert_phc_amass.py \
  --pkl_path "$RELEASE_ROOT/outputs/phc/loop0/full/loop0_prompts_hik.pkl" \
  --output_dir "$RELEASE_ROOT/outputs/amass/loop0/npz"
```

Output:

- `outputs/amass/loop0/npz/`

## Step 5: Render Videos From AMASS `npz`

```bash
cd "$REPO_ROOT"
"$PHC_PYTHON" CLAIMS_release/vlm-feedback/Smpl_Visualization.py \
  --input_dir "$RELEASE_ROOT/outputs/amass/loop0/npz" \
  --output_dir "$RELEASE_ROOT/outputs/videos/loop0/mp4" \
  --model_dir "$SMPL_DATA_DIR"
```

Output:

- `outputs/videos/loop0/mp4/`

## Step 6: Filter Videos By Prompt-Motion Alignment

Before PHC training, filter rendered videos by semantic alignment between the original prompt and the rendered motion clip.

Run:

```bash
cd "$RELEASE_ROOT"
"$PHC_PYTHON" vlm-feedback/video_prompt_alignment.py \
  --config "$RELEASE_ROOT/vlm-feedback/config_difficulty.yaml" \
  --videos-root "$RELEASE_ROOT/outputs/videos/loop0/mp4" \
  --manifest-path "$RELEASE_ROOT/outputs/prompts/loop0/loop0_prompt_manifest.csv" \
  --build-input-csv \
  --provider qwen \
  --alignment-threshold 50 \
  --output-dir "$RELEASE_ROOT/outputs/vlm_alignment/loop0"
```

Outputs:

- `outputs/vlm_alignment/loop0/alignment_input.csv`
- `outputs/vlm_alignment/loop0/alignment_summary.csv`
- `outputs/vlm_alignment/loop0/alignment_stats.csv`
- `outputs/vlm_alignment/loop0/accepted_videos.csv`
- `outputs/vlm_alignment/loop0/accepted_alignment_inputs.csv`

Interpretation:

- `alignment_summary.csv` is the per-sample alignment result table
- `accepted_videos.csv` and `accepted_alignment_inputs.csv` are threshold-based exports from the alignment script
- `alignment_stats.csv` is only a running acceptance log written during batch processing

Filtering rule used in this release:

- samples with verdict `mismatch` are filtered out
- all remaining samples, including `partial` and `aligned`, are retained
- only the retained subset should be used for downstream PHC training and VLM feedback

## Step 7: Build The Filtered PHC Training Set

Use the alignment result table to create the retained training subset. This step keeps every sample whose verdict is not `mismatch`.

```bash
cd "$RELEASE_ROOT"
"$PHC_PYTHON" scripts/filter_phc_training_set.py \
  --alignment-summary-csv "$RELEASE_ROOT/outputs/vlm_alignment/loop0/alignment_summary.csv" \
  --input-pkl "$RELEASE_ROOT/outputs/phc/loop0/full/loop0_prompts_hik.pkl" \
  --output-pkl "$RELEASE_ROOT/outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl" \
  --input-single-dir "$RELEASE_ROOT/outputs/phc/loop0/single_dict" \
  --output-single-dir "$RELEASE_ROOT/outputs/phc/loop0/single_dict_filtered" \
  --kept-videos-csv "$RELEASE_ROOT/outputs/phc/loop0/kept_videos.csv" \
  --summary-json "$RELEASE_ROOT/outputs/phc/loop0/filter_summary.json"
```

Outputs:

- `outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl`
- `outputs/phc/loop0/single_dict_filtered/`
- `outputs/phc/loop0/kept_videos.csv`
- `outputs/phc/loop0/filter_summary.json`

This filtered `pkl` is the motion file that should be passed to PHC training.

## Step 8: Train PHC

Run from the `PHC/` module with the `phc_420` environment:

```bash
cd "$RELEASE_ROOT/PHC"
CUDA_VISIBLE_DEVICES=0 \
"$PHC_PYTHON" phc/run_hydra.py \
  learning=im_big \
  exp_name=claims \
  env=env_im \
  robot=smpl_humanoid \
  env.motion_file="$RELEASE_ROOT/outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl" \
  env.num_envs=1024 \
  headless=True \
  no_log=True
```

Checkpoint output directory:

- `PHC/output/HumanoidIm/claims/`

## Step 9: Evaluate PHC And Save Tracking Metrics

```bash
cd "$RELEASE_ROOT/PHC"
CUDA_VISIBLE_DEVICES=0 \
"$PHC_PYTHON" phc/run_hydra.py \
  learning=im_big \
  exp_name=claims \
  env=env_im \
  robot=smpl_humanoid \
  env.motion_file="$RELEASE_ROOT/outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl" \
  env.num_envs=5 \
  test=True \
  epoch=-1 \
  im_eval=True \
  headless=True \
  no_log=True
```

Public-facing evaluation output:

- `PHC/outputs/eval/phc/claims/loop0_prompts_hik_filtered/latest/summary.json`
- `PHC/outputs/eval/phc/claims/loop0_prompts_hik_filtered/latest/metrics.csv`
- `PHC/outputs/eval/phc/claims/loop0_prompts_hik_filtered/latest/per_sample_metrics.csv`

`per_sample_metrics.csv` is the file used by the next-loop VLM merge stage.

## Step 10: Run VLM Feedback On Retained Videos Only

Set your API credentials through environment variables:

```bash
export GPT4O_API_KEY=YOUR_GPT4O_API_KEY
export QWEN_API_KEY=YOUR_QWEN_API_KEY
```

Then run:

```bash
cd "$REPO_ROOT"
python CLAIMS_release/vlm-feedback/run_vlm_feedback.py \
  --batch \
  --input "$RELEASE_ROOT/outputs/videos/loop0/mp4" \
  --allowed-videos-csv "$RELEASE_ROOT/outputs/phc/loop0/kept_videos.csv" \
  --output "$RELEASE_ROOT/outputs/vlm_feedback/loop0"
```

Main outputs:

- `outputs/vlm_feedback/loop0/gpt_analysis_merged.json`
- `outputs/vlm_feedback/loop0/qwen_analysis_merged.json`
- `outputs/vlm_feedback/loop0/merged_feedback.csv`
- `outputs/vlm_feedback/loop0/claims_loop0_vlm_feedback.csv`

`claims_loop0_vlm_feedback.csv` is produced only from the non-mismatch subset.

## Step 11: Generate Next-Loop Prompts

Set the prompt-generation API key:

```bash
export PROMPT_NEXT_API_KEY=YOUR_PROMPT_NEXT_API_KEY
export PROMPT_NEXT_API_MODEL=gpt-4o
```

Then run:

```bash
cd "$RELEASE_ROOT"
python scripts/generate_next_loop_prompts.py \
  --input-csv "$RELEASE_ROOT/outputs/vlm_feedback/loop0/claims_loop0_vlm_feedback.csv" \
  --output-dir "$RELEASE_ROOT/outputs/prompts/loop1" \
  --prompt-manifest "$RELEASE_ROOT/outputs/prompts/loop0/loop0_prompt_manifest.csv" \
  --group-size 5 \
  --target-count 40 \
  --seed 7
```

Outputs:

- `outputs/prompts/loop1/category_csv/`
- `outputs/prompts/loop1/category_prompts/`
- `outputs/prompts/loop1/loop1_prompts.txt`
- `outputs/prompts/loop1/loop1_prompt_manifest.csv`

Step 11 rule:

- each of the five categories is expanded to `40` rows
- duplication stays within the same category only
- duplication order is:
  - `NO` samples from worst metrics to best metrics
  - then `YES` samples from worst metrics to best metrics
- the final target is `200` prompts total

Important constraint:

- if one category has `0` rows in the step 10 CSV, step 11 stops with an error

## Public Release Notes

This repository does not ship:

- private API keys
- generated outputs under `outputs/`
- large checkpoints in git history
- SMPL assets

For reproducibility assets such as PHC checkpoints or evaluation data, use the external asset release referenced from the repository root README.

## More Documentation

- [docs/pipeline.md](docs/pipeline.md)
- [docs/environments.md](docs/environments.md)
- [docs/data_assets.md](docs/data_assets.md)
