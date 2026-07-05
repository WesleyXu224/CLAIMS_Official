# Motion Generate Module

This directory contains the CLAIMS text-to-motion generation code.

## Notes On External Assets

Some large runtime assets used during local development are not committed to the public repository, including:

- pretrained checkpoints
- body model assets
- glove embeddings
- dataset symlinks

Public release rule:

- use the upstream MDM environment and data layout described in `CLAIMS_release/docs/environments.md`
- place required assets locally after cloning the repository
- do not expect these large external resources to be bundled in git
