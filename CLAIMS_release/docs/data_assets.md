# Data, Assets, And Outputs

This document describes the main artifacts in the CLAIMS pipeline.

## Inputs Not Shipped In Git

- text-to-motion model checkpoints
- PHC controller checkpoints
- SMPL / SMPL-X assets
- Isaac Gym runtime dependencies
- external datasets
- private API keys for VLM or LLM services

## Main Intermediate Artifacts

### Prompt Files

Produced by:

- [`scripts/generate_initial_prompts.py`](../scripts/generate_initial_prompts.py)
- [`scripts/generate_next_loop_prompts.py`](../scripts/generate_next_loop_prompts.py)

Typical forms:

- text files
- CSV files

### Generated Motions

Produced by:

- [`motion-generate/language_to_pose_server_generate_example.py`](../motion-generate/language_to_pose_server_generate_example.py)

### HIK Motions

Produced by:

- [`motion-generate/sample/edit.py`](../motion-generate/sample/edit.py)

### PHC Motion Dictionaries

Produced by:

- [`PHC/scripts/data_process/convert_data_mdm_hik.py`](../PHC/scripts/data_process/convert_data_mdm_hik.py)

Typical form:

- `pkl`

### PHC Evaluation Metrics

Produced by:

- [`PHC/phc/learning/im_amp.py`](../PHC/phc/learning/im_amp.py)
- [`PHC/phc/learning/im_amp_players.py`](../PHC/phc/learning/im_amp_players.py)

Typical fields:

- `mpjpe_g`
- `mpjpe_l`
- `mpjpe_pa`
- `accel_dist`
- `vel_dist`
- `success`

### AMASS-Style Motions

Produced by:

- [`PHC/scripts/data_process/convert_phc_amass.py`](../PHC/scripts/data_process/convert_phc_amass.py)

Typical form:

- `npz`

### Rendered Videos

Produced by:

- the AMASS-to-video rendering path used before VLM analysis

Typical use:

- input to VLM feedback

### VLM Feedback

Produced by:

- [`vlm-feedback/`](../vlm-feedback/)

Typical forms:

- score CSV
- analysis JSON
- merged evaluation tables

## Recommended Output Organization

For the public release, keep generated artifacts outside tracked source code when possible. A simple convention is:

```text
outputs/
|-- prompts/
|-- motions/
|-- hik/
|-- pkl/
|-- npz/
|-- videos/
|-- metrics/
`-- vlm/
```

This keeps the code tree clean and makes the iterative loop easier to reproduce.
