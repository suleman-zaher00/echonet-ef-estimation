"""Grad-CAM visualisation for the EF regression model.

Usage (from the project root):

    python -m src.gradcam                     # random TEST video (seed 0)
    python -m src.gradcam --seed 5            # another random video
    python -m src.gradcam --file 0X1A2B3C     # a specific video (name without .avi)

Writes a 3-panel PNG (frame / heatmap / overlay) and an animated GIF of the
overlay to the output folder.

Grad-CAM shows which regions most influenced the prediction. It is a
visual aid, not proof that the model reasons like a clinician.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .dataset import EchoNetDataset, load_split
from .model import get_device, load_model


class GradCAM3D:
    """Grad-CAM for a 3D CNN, hooked on one convolutional layer."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module) -> None:
        self.model = model
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._handles = [
            target_layer.register_forward_hook(self._save_activations),
            target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_activations(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()

    def __enter__(self) -> "GradCAM3D":
        return self

    def __exit__(self, *exc) -> None:
        self.remove()

    def generate(self, clip: torch.Tensor) -> tuple[np.ndarray, float]:
        """Return (cam, prediction) for a single clip of shape (1, 3, T, H, W).

        cam has shape (T, H, W), values in [0, 1], resized to the input clip.
        """
        self.model.zero_grad(set_to_none=True)
        output = self.model(clip)
        output.backward(torch.ones_like(output))

        weights = self.gradients[0].mean(dim=(1, 2, 3))  # one weight per channel
        cam = (weights[:, None, None, None] * self.activations[0]).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        size = tuple(clip.shape[2:])  # (T, H, W)
        cam = F.interpolate(cam[None, None], size=size, mode="trilinear", align_corners=False)[0, 0]
        return np.clip(cam.cpu().numpy(), 0.0, 1.0), float(output.item())


def denormalize(frame: torch.Tensor) -> np.ndarray:
    """(3, H, W) normalised tensor -> (H, W, 3) float image in [0, 1]."""
    mean = torch.tensor(config.MEAN)[:, None, None]
    std = torch.tensor(config.STD)[:, None, None]
    return np.clip((frame.cpu() * std + mean).permute(1, 2, 0).numpy(), 0.0, 1.0)


def overlay(image: np.ndarray, cam_frame: np.ndarray, alpha: float = 0.4):
    """Blend a JET heatmap onto an image. Returns (overlay, heatmap)."""
    heat = cv2.applyColorMap(np.uint8(255 * cam_frame), cv2.COLORMAP_JET)[:, :, ::-1]
    heat = heat.astype(np.float32) / 255.0
    return np.clip((1 - alpha) * image + alpha * heat, 0.0, 1.0), heat


def save_panel(frame, heat, blended, true_ef, pred_ef, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(frame)
    axes[0].set_title(f"Original frame\nTrue EF: {true_ef:.1f}%")
    axes[1].imshow(heat)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(blended)
    axes[2].set_title(f"Overlay\nPredicted EF: {pred_ef:.1f}%")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def save_gif(clip: torch.Tensor, cam: np.ndarray, true_ef, pred_ef, path: Path,
             scale: int = 3, duration_ms: int = 100) -> None:
    label = f"Pred {pred_ef:.1f}%"
    if true_ef is not None:
        label += f" | True {true_ef:.1f}%"
    frames = []
    for t in range(clip.shape[2]):
        blended, _ = overlay(denormalize(clip[0, :, t]), cam[t])
        big = cv2.resize(blended, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        big = np.ascontiguousarray((big * 255).astype(np.uint8))
        cv2.putText(big, label, (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        frames.append(Image.fromarray(big))
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--file", default=None, help="video name without .avi")
    parser.add_argument("--index", type=int, default=None, help="row index inside the split")
    parser.add_argument("--seed", type=int, default=0, help="seed for the random choice")
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    device = get_device()
    df = load_split(config.FILELIST_CSV, args.split)
    if args.file:
        matches = df.index[df["FileName"] == args.file]
        if len(matches) == 0:
            raise SystemExit(f"{args.file} not found in the {args.split} split")
        idx = int(matches[0])
    elif args.index is not None:
        idx = args.index
    else:
        idx = int(np.random.default_rng(args.seed).integers(len(df)))

    dataset = EchoNetDataset(df.iloc[[idx]], config.VIDEOS_DIR, return_name=True)
    clip, true_ef, name = dataset[0]
    clip = clip.unsqueeze(0).to(device)
    true_ef = float(true_ef)

    model = load_model(args.weights, device)
    with GradCAM3D(model, model.backbone.layer4[-1]) as grad_cam:
        cam, pred_ef = grad_cam.generate(clip)

    mid = clip.shape[2] // 2
    frame = denormalize(clip[0, :, mid])
    blended, heat = overlay(frame, cam[mid])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.out_dir / f"gradcam_{name}.png"
    gif_path = args.out_dir / f"gradcam_{name}.gif"
    save_panel(frame, heat, blended, true_ef, pred_ef, png_path)
    save_gif(clip, cam, true_ef, pred_ef, gif_path)

    print(f"Video: {name} | true EF: {true_ef:.1f}% | predicted EF: {pred_ef:.1f}% "
          f"| abs. error: {abs(pred_ef - true_ef):.1f}%")
    print(f"Saved: {png_path}\nSaved: {gif_path}")


if __name__ == "__main__":
    main()
