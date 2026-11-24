"""Estimate the ejection fraction (EF) of a single echocardiogram video.

Usage (from the project root):

    python -m src.predict path/to/video.avi
    python -m src.predict path/to/video.avi --gradcam   # also save an overlay GIF

The model was trained on EchoNet-Dynamic apical four-chamber videos
(112x112). Other video sources have not been validated.

Research use only. This is not a medical device and must not be used for
clinical decisions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torchvision.transforms as T

from . import config
from .dataset import read_video, sample_indices
from .gradcam import GradCAM3D, save_gif
from .model import get_device, load_model

TEST_MAE = 4.40          # mean absolute error on the TEST split (see README)
SCREENING_CUTOFF = 50.0  # predicted EF <= this raises an alert (cut-off chosen on VAL)


def preprocess(path: Path) -> torch.Tensor:
    """Video file -> normalised clip of shape (1, 3, T, H, W)."""
    transform = T.Compose([T.ToPILImage(), T.ToTensor(), T.Normalize(config.MEAN, config.STD)])
    frames = read_video(path, config.IMG_SIZE)
    indices = sample_indices(len(frames), config.NUM_FRAMES)
    clip = torch.stack([transform(frames[i]) for i in indices]).permute(1, 0, 2, 3)
    return clip.unsqueeze(0)


def describe(ef: float) -> str:
    if ef >= 50:
        return "preserved range (>= 50%)"
    if ef > 40:
        return "mildly reduced range (41-49%)"
    if ef > 30:
        return "reduced range (31-40%)"
    return "severely reduced range (<= 30%)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("video", type=Path, help="path to an .avi echocardiogram video")
    parser.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    parser.add_argument("--gradcam", action="store_true", help="also save a Grad-CAM overlay GIF")
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    device = get_device()
    clip = preprocess(args.video).to(device)
    model = load_model(args.weights, device)

    cam = None
    if args.gradcam:
        with GradCAM3D(model, model.backbone.layer4[-1]) as grad_cam:
            cam, ef = grad_cam.generate(clip)
    else:
        with torch.inference_mode():
            ef = float(model(clip).item())

    alert = ef <= SCREENING_CUTOFF
    print("=" * 56)
    print(f"Video:         {args.video.name}")
    print(f"Predicted EF:  {ef:.1f}%  (typical error about +/- {TEST_MAE:.1f} points)")
    print(f"Range:         {describe(ef)}")
    print(f"Screening:     {'ALERT' if alert else 'no alert'} (cut-off: predicted EF <= {SCREENING_CUTOFF:.0f}%)")
    print("Research use only. Not for clinical decisions.")
    print("=" * 56)

    if cam is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        gif_path = args.out_dir / f"gradcam_{args.video.stem}.gif"
        save_gif(clip, cam, None, ef, gif_path)
        print(f"Saved: {gif_path}")


if __name__ == "__main__":
    main()
