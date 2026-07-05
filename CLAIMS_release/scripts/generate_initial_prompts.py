import subprocess
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    release_root = Path(__file__).resolve().parents[1]
    target = repo_root / "prompt-generate" / "prompt_loop0.py"
    if not target.exists():
        raise FileNotFoundError("Could not locate the initial prompt generation script.")
    release_prompt_dir = release_root / "outputs" / "prompts" / "loop0"
    cmd = [sys.executable, str(target), "--output_dir", str(release_prompt_dir)]
    cmd.extend(sys.argv[1:])
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
