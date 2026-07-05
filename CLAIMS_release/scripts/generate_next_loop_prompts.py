import runpy
import sys
from pathlib import Path


def main():
    repo_root = Path(__file__).resolve().parents[2]
    target = repo_root / "prompt-generate" / "prompt_next.py"
    if not target.exists():
        raise FileNotFoundError("Could not locate the next-loop prompt generation script.")
    sys.argv[0] = str(target)
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
