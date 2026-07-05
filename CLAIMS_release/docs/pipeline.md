# CLAIMS Pipeline

This document describes the public release workflow for CLAIMS.

## Overview

The CLAIMS loop is:

```text
loop0 prompts
  -> motion generation
  -> HIK / PHC source conversion
  -> AMASS export
  -> video rendering
  -> prompt-motion alignment filtering
  -> keep non-mismatch motions only
  -> build filtered PHC training set
  -> PHC training
  -> PHC evaluation
  -> VLM feedback on retained videos
  -> PHC metrics + VLM feedback
  -> next-loop prompts
  -> repeat
```

The public release uses prompt-motion alignment as a filtering stage before PHC training. Samples with verdict `mismatch` are removed. Samples with verdict `partial` or `aligned` are retained for both PHC training and downstream VLM feedback.

## Stage 1: Generate Initial `loop0` Prompts

Release entrypoint:

- `scripts/generate_initial_prompts.py`

Purpose:

- create the first prompt pool
- keep prompts grouped by the five motion categories
- save a prompt manifest for category tracing

Typical command:

```bash
cd CLAIMS_release
python scripts/generate_initial_prompts.py --count 40 --seed 7
```

Main outputs:

- `outputs/prompts/loop0/loop0_prompts.txt`
- `outputs/prompts/loop0/loop0_prompt_manifest.csv`

## Stage 2: Generate Motions From Prompts

Primary module:

- `motion-generate/`

Purpose:

- convert prompt text into motion samples with the text-to-motion model

Typical command:

```bash
cd CLAIMS_release/motion-generate
python language_to_pose_server_generate_example.py \
  --model_path <mdm_checkpoint> \
  --text_encoder_type bert \
  --prompts_path ../outputs/prompts/loop0 \
  --prompts_file_names loop0_prompts \
  --motion_save_path ../outputs/motions/loop0
```

Main output:

- `outputs/motions/loop0/loop0_prompts/`

## Stage 3: Convert Generated Motions To PHC Source Assets

### 3.1 Convert MDM Motions To HIK

Release entrypoint:

- `scripts/convert_motions_to_hik.py`

Internal implementation:

- `motion-generate/sample/edit.py`

Public release note:

- `sample/edit.py` remains the underlying implementation
- external users should call the wrapper script instead of using the internal module directly

Typical command:

```bash
cd CLAIMS_release
python scripts/convert_motions_to_hik.py \
  --input-dir outputs/motions/loop0/loop0_prompts \
  --output-dir outputs/motions/loop0/loop0_prompts_hik
```

Output:

- `outputs/motions/loop0/loop0_prompts_hik/`

### 3.2 Convert HIK Motions To PHC `pkl`

Primary script:

- `PHC/scripts/data_process/convert_data_mdm_hik.py`

Purpose:

- convert HIK motions into PHC-readable motion dictionaries

Typical command:

```bash
cd CLAIMS_release/PHC/scripts/data_process
python convert_data_mdm_hik.py \
  --base_path ../../outputs/motions/loop0 \
  --folders loop0_prompts_hik \
  --single_output_root ../../outputs/phc/loop0/single_dict \
  --full_output_dir ../../outputs/phc/loop0/full \
  --smpl_data_dir <smpl_asset_dir>
```

Main outputs:

- `outputs/phc/loop0/single_dict/`
- `outputs/phc/loop0/full/loop0_prompts_hik.pkl`

At this point, do not start PHC training yet. First render videos, evaluate prompt-motion alignment, and filter out `mismatch` samples.

## Stage 4: Convert PHC Motions To AMASS `npz`

Primary script:

- `PHC/scripts/data_process/convert_phc_amass.py`

Purpose:

- convert PHC-side motion dictionaries into AMASS-style `npz`

Typical command:

```bash
cd CLAIMS_release
python PHC/scripts/data_process/convert_phc_amass.py \
  --pkl_path outputs/phc/loop0/full/loop0_prompts_hik.pkl \
  --output_dir outputs/amass/loop0/npz
```

Main output:

- `outputs/amass/loop0/npz/`

## Stage 5: Render Videos From `npz`

Primary script:

- `vlm-feedback/Smpl_Visualization.py`

Purpose:

- render AMASS-style motions into videos for semantic filtering and VLM analysis

Typical command:

```bash
cd CLAIMS_release
python vlm-feedback/Smpl_Visualization.py \
  --input_dir outputs/amass/loop0/npz \
  --output_dir outputs/videos/loop0/mp4 \
  --model_dir <smpl_asset_dir>
```

Main output:

- `outputs/videos/loop0/mp4/`

## Stage 6: Filter Rendered Videos By Prompt-Motion Alignment

Primary script:

- `vlm-feedback/video_prompt_alignment.py`

Purpose:

- compare each rendered video against its source prompt
- assign `aligned`, `partial`, or `mismatch`
- export alignment summaries and threshold-based helper CSVs

Typical command:

```bash
cd CLAIMS_release
python vlm-feedback/video_prompt_alignment.py \
  --config vlm-feedback/config_difficulty.yaml \
  --videos-root outputs/videos/loop0/mp4 \
  --manifest-path outputs/prompts/loop0/loop0_prompt_manifest.csv \
  --build-input-csv \
  --provider qwen \
  --alignment-threshold 50 \
  --output-dir outputs/vlm_alignment/loop0
```

Main outputs:

- `outputs/vlm_alignment/loop0/alignment_summary.csv`
- `outputs/vlm_alignment/loop0/alignment_stats.csv`
- `outputs/vlm_alignment/loop0/accepted_videos.csv`
- `outputs/vlm_alignment/loop0/accepted_alignment_inputs.csv`

Important interpretation:

- `alignment_summary.csv` is the per-sample alignment table
- `alignment_stats.csv` is only a running acceptance log written during batch processing
- `accepted_videos.csv` and `accepted_alignment_inputs.csv` are threshold-based helper exports

Public release filtering rule:

- remove samples whose verdict is `mismatch`
- keep samples whose verdict is `partial` or `aligned`

## Stage 7: Build The Filtered PHC Training Set

Release entrypoint:

- `scripts/filter_phc_training_set.py`

Purpose:

- convert the alignment result table into the retained PHC training subset
- write a filtered merged `pkl`
- optionally write a filtered `single_dict/` directory

Typical command:

```bash
cd CLAIMS_release
python scripts/filter_phc_training_set.py \
  --alignment-summary-csv outputs/vlm_alignment/loop0/alignment_summary.csv \
  --input-pkl outputs/phc/loop0/full/loop0_prompts_hik.pkl \
  --output-pkl outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl \
  --input-single-dir outputs/phc/loop0/single_dict \
  --output-single-dir outputs/phc/loop0/single_dict_filtered \
  --kept-videos-csv outputs/phc/loop0/kept_videos.csv \
  --summary-json outputs/phc/loop0/filter_summary.json
```

Main outputs:

- `outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl`
- `outputs/phc/loop0/single_dict_filtered/`
- `outputs/phc/loop0/kept_videos.csv`

This filtered `pkl` is the motion file that should be passed to PHC training and PHC evaluation.

## Stage 8: Train PHC

Primary script:

- `PHC/phc/run_hydra.py`

Purpose:

- train the humanoid controller on the non-mismatch subset

Typical command:

```bash
cd CLAIMS_release/PHC
CUDA_VISIBLE_DEVICES=<gpu_id> python phc/run_hydra.py \
  learning=im_big \
  exp_name=claims \
  env=env_im \
  robot=smpl_humanoid \
  env.motion_file=../outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl \
  env.num_envs=<num_envs> \
  headless=True \
  no_log=True
```

## Stage 9: Evaluate PHC And Collect Tracking Metrics

Primary script:

- `PHC/phc/run_hydra.py`

Relevant evaluation code:

- `PHC/phc/learning/im_amp.py`
- `PHC/phc/learning/im_amp_players.py`

Purpose:

- evaluate PHC on the filtered motion set
- produce tracking metrics such as `mpjpe_g`, `mpjpe_l`, `mpjpe_pa`, `accel_dist`, `vel_dist`, and success labels

Typical command:

```bash
cd CLAIMS_release/PHC
CUDA_VISIBLE_DEVICES=<gpu_id> python phc/run_hydra.py \
  learning=im_big \
  exp_name=claims \
  env=env_im \
  robot=smpl_humanoid \
  env.motion_file=../outputs/phc/loop0/full/loop0_prompts_hik_filtered.pkl \
  env.num_envs=<num_motions> \
  test=True \
  epoch=-1 \
  headless=True \
  im_eval=True \
  no_log=True
```

Main public-facing output:

- `PHC/outputs/eval/phc/claims/loop0_prompts_hik_filtered/latest/per_sample_metrics.csv`

## Stage 10: Run VLM Feedback On Retained Videos

Primary script:

- `vlm-feedback/run_vlm_feedback.py`

Purpose:

- analyze rendered motion videos with VLM providers
- restrict processing to the same non-mismatch subset used by PHC
- merge the VLM outputs with PHC per-sample metrics

Typical command:

```bash
cd CLAIMS_release
python vlm-feedback/run_vlm_feedback.py \
  --batch \
  --input outputs/videos/loop0/mp4 \
  --allowed-videos-csv outputs/phc/loop0/kept_videos.csv \
  --output outputs/vlm_feedback/loop0
```

Main output:

- `outputs/vlm_feedback/loop0/claims_loop0_vlm_feedback.csv`

## Stage 11: Generate The Next Prompt Loop

Release entrypoint:

- `scripts/generate_next_loop_prompts.py`

Implementation module:

- `prompt-generate/prompt_next.py`

Purpose:

- merge PHC metrics and VLM feedback
- split by motion category
- expand each category to the target count
- generate the next loop of harder prompts

Typical command:

```bash
cd CLAIMS_release
PROMPT_NEXT_API_KEY=YOUR_API_KEY \
PROMPT_NEXT_API_MODEL=gpt-4o \
python scripts/generate_next_loop_prompts.py \
  --input-csv outputs/vlm_feedback/loop0/claims_loop0_vlm_feedback.csv \
  --output-dir outputs/prompts/loop1 \
  --prompt-manifest outputs/prompts/loop0/loop0_prompt_manifest.csv \
  --group-size 5 \
  --target-count 40 \
  --seed 7
```

Main outputs:

- `outputs/prompts/loop1/category_csv/`
- `outputs/prompts/loop1/category_prompts/`
- `outputs/prompts/loop1/loop1_prompts.txt`
- `outputs/prompts/loop1/loop1_prompt_manifest.csv`

## Public Release Recommendation

For a clean public release:

- document `scripts/` as the official entrypoints
- treat `motion-generate/sample/edit.py` as an internal implementation detail
- keep machine-specific commands out of the public pipeline docs
- use environment variables for API keys and asset directories
