import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert generated CLAIMS motions into HIK format."
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing generated motion files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where HIK-format outputs will be written.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    module_root = repo_root / "motion-generate"
    command = [
        sys.executable,
        "-m",
        "sample.edit",
        args.input_dir,
        args.output_dir,
    ]
    subprocess.run(command, check=True, cwd=module_root)


if __name__ == "__main__":
    main()
