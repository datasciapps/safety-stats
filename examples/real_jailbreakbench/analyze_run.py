"""Re-run the statistics and static visualizer for a completed JBB-100 run."""

from __future__ import annotations

import argparse
from pathlib import Path

from run_jailbreakbench import analyze_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Directory containing analysis-plan.yaml and logs/",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    destination = analyze_run(args.run_dir.resolve(), overwrite=args.overwrite)
    print(f"Statistical report: {destination / 'report.md'}")
    print(f"Static visualizer: {destination / 'report.html'}")


if __name__ == "__main__":
    main()
