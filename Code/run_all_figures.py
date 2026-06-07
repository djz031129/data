"""Run the complete manuscript figure-generation script.

Usage from repository root:
    python Code/run_all_figures.py

This runner generates only the full manuscript figures Fig. 1-Fig. 4.
Individual single-panel figure generation has intentionally been removed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    py = sys.executable
    run([py, "Code/full_figures/generate_figures.py"])
    print("Full manuscript figures generated under Code/output/full_figures/.")


if __name__ == "__main__":
    main()
