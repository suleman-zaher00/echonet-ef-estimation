"""Model definition and weight loading."""
from __future__ import annotations

import pickle
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models.video import R2Plus1D_18_Weights, r2plus1d_18


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EFRegressor(nn.Module):
    """R(2+1)D-18 video backbone with a small regression head.

    Input:  (B, 3, T, H, W) normalised clip.
    Output: (B, 1) ejection fraction in percent.
    """

    def __init__(self, pretrained: bool = False, dropout: float = 0.3, hidden: int = 256):
        super().__init__()
        weights = R2Plus1D_18_Weights.DEFAULT if pretrained else None
        self.backbone = r2plus1d_18(weights=weights)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def load_model(
    weights_path: str | Path,
    device: torch.device | None = None,
    trusted: bool = False,
) -> EFRegressor:
    """Build the model and load weights, ready for inference (eval mode).

    Accepts either a bare state dict or a training checkpoint containing
    "model_state_dict". A "module." prefix left by nn.DataParallel is removed.

    Loading uses torch's safe mode (weights_only=True). Pass trusted=True only
    for checkpoint files you created yourself and that fail in safe mode.
    """
    device = device or get_device()
    path = Path(weights_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Weights not found at {path}. Set EF_WEIGHTS or see the README "
            "for download instructions."
        )

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        raise RuntimeError(
            "Safe loading failed. If you created this file yourself, retry "
            "with load_model(..., trusted=True)."
        ) from exc

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}

    model = EFRegressor(pretrained=False)
    model.load_state_dict(state_dict)
    return model.to(device).eval()
