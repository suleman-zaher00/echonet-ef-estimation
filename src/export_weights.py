"""Create a small release file from a training checkpoint.

The training checkpoint also stores the optimizer and scheduler state, which is
not needed for inference and triples the file size. This script keeps only the
model weights and prints the size and SHA-256 hash of the result.

Usage (from the project root):

    python -m src.export_weights
    python -m src.export_weights --input weights/best_model.pth --output weights/best_model_release.pth
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from . import config
from .model import load_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input", type=Path, default=config.WEIGHTS_PATH)
    parser.add_argument("--output", type=Path, default=config.WEIGHTS_PATH.with_name("best_model_release.pth"))
    args = parser.parse_args()

    # Loading through the model class also checks that all keys match.
    model = load_model(args.input, device=torch.device("cpu"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)

    print(f"Input:  {args.input} ({args.input.stat().st_size / 1e6:.1f} MB)")
    print(f"Output: {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")
    print(f"SHA-256: {sha256(args.output)}")


if __name__ == "__main__":
    main()
