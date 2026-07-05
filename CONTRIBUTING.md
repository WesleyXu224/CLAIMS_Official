# Contributing

## Scope

This repository is primarily a research code release for the CLAIMS project. Contributions that improve reproducibility, documentation quality, environment setup, or bug fixes are welcome.

## Before Opening A Pull Request

Please make sure your change:

- is directly related to the CLAIMS release pipeline
- does not introduce private paths, private checkpoints, or API keys
- does not commit generated outputs, cached assets, or large binaries
- updates the corresponding documentation when commands or paths change

## Environment And Reproducibility

When reporting a bug, include:

- operating system
- Python version
- environment name
- GPU / CUDA version if relevant
- exact command used
- full traceback or error message

## Code Style

- prefer clear, minimal changes
- keep scripts runnable from documented entrypoints
- use environment variables for credentials and machine-specific paths
- avoid hard-coded absolute local paths in public-facing code

## Large Files And Assets

Do not commit:

- model checkpoints
- generated motion outputs
- rendered videos
- private datasets
- SMPL / SMPL-X assets unless redistribution is explicitly allowed

## Security

If you find a private credential, remove it from the patch and report it immediately instead of committing it.
