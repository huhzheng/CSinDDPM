"""Write reproducibility metadata without starting a training run."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from demo_diffusion.reproducibility import collect_environment, current_git_commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="revision_outputs/environment.json")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = collect_environment()
    payload["git_commit"] = current_git_commit(ROOT)
    payload["dependencies"] = {
        name: _version(name)
        for name in [
            "blobfile",
            "numpy",
            "pandas",
            "pytest",
            "scipy",
            "tifffile",
            "torch",
            "tqdm",
        ]
    }
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(output)


def _version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "NOT_INSTALLED"


if __name__ == "__main__":
    main()
