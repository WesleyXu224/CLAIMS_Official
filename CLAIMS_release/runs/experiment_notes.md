# Experiment Notes

This file summarizes the practical experiment flow in release-facing language.

## Core Loop

1. Generate initial prompts for the first loop.
2. Generate motions from prompts.
3. Convert motions into controller-readable training format.
4. Train the controller on the converted motions.
5. Evaluate the controller to obtain tracking metrics.
6. Convert motions into AMASS-style `npz`.
7. Render videos from the converted motions.
8. Run VLM feedback on the rendered videos.
9. Merge controller metrics and VLM feedback to build the next prompt loop.

## Typical Experiment Branches

- standard iterative loop
- ablations without VLM feedback
- ablations without tracking metrics
- third-party test-set evaluation

## Release Guidance

- keep exact private machine paths out of the public-facing README
- keep API keys out of the release tree
- keep experiment-specific checkpoints and outputs outside tracked source code
