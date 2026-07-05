# Release Script Entrypoints

These wrappers provide release-facing functional entrypoints.

- `generate_initial_prompts.py`: runs the initial prompt generation stage.
- `convert_motions_to_hik.py`: converts generated motions to HIK format through a release-facing wrapper.
- `generate_next_loop_prompts.py`: runs the next-loop prompt optimization stage.

These wrappers dispatch to the maintained source files in the repository modules and hide older internal script names from public-facing usage where appropriate.
