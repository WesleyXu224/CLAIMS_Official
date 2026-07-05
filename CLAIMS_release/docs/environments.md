# Environments

The public CLAIMS release should be presented with only two conda environments:

- `mdm`
- `phc_420`

This is the simplest and most reproducible story for users.

## 1. `mdm`

Use `mdm` for:

- loop0 prompt generation
- next-loop prompt generation
- text-to-motion generation
- motion-to-HIK conversion on the MDM side

Relevant code in this repository:

- `prompt-generate/`
- `motion-generate/`

Recommended upstream reference:

- MDM official repository: `https://github.com/GuyTevet/motion-diffusion-model`

Public release guidance:

- users should first follow the environment setup from the upstream MDM repository
- then install any small additional CLAIMS-specific dependencies if needed
- the CLAIMS code here assumes an MDM-compatible runtime rather than introducing a separate third environment

Typical workflow:

```bash
conda activate mdm
cd CLAIMS_release/motion-generate
python language_to_pose_server_generate_example.py ...
```

## 2. `phc_420`

Use `phc_420` for:

- HIK to PHC conversion
- PHC training
- PHC evaluation
- PHC to AMASS conversion
- SMPL-based rendering used in the VLM branch

Relevant code in this repository:

- `PHC/`
- `vlm-feedback/`

Recommended upstream reference:

- PHC official repository: `https://github.com/ZhengyiLuo/PHC`

Public release guidance:

- users should first follow the environment setup from the upstream PHC repository
- then use the CLAIMS scripts on top of that environment
- this environment is also the one used for the PHC-side conversion and rendering utilities in this repository

Typical workflow:

```bash
conda activate phc_420
cd CLAIMS_release/PHC
python phc/run_hydra.py ...
```

## Why Only Two Environments

For the public release, we do not recommend presenting:

- extra historical local environments
- machine-specific environment names
- a separate heavyweight VLM environment unless strictly necessary

Instead:

- `mdm` covers prompt and motion generation
- `phc_420` covers controller training, evaluation, conversion, export, and rendering

The VLM feedback scripts should be documented as a lightweight module that runs in the PHC-side environment unless users have a specific dependency conflict.

## Secrets And Machine-Specific Paths

Public release rules:

- do not hard-code API keys in source files
- do not hard-code private absolute paths in documentation
- use environment variables for credentials and local asset locations

Examples:

```bash
export GPT4O_API_KEY=YOUR_GPT4O_API_KEY
export QWEN_API_KEY=YOUR_QWEN_API_KEY
export PROMPT_NEXT_API_KEY=YOUR_PROMPT_NEXT_API_KEY
export SMPL_DATA_DIR=/path/to/smpl_assets
```

## Summary

If you want the shortest public explanation, use exactly this wording:

- `mdm` environment: follow the upstream MDM repository and use it for prompt and motion generation
- `phc_420` environment: follow the upstream PHC repository and use it for conversion, training, evaluation, AMASS export, rendering, and VLM feedback
