"""Evaluate the trained model on an EchoNet-Dynamic split.

Usage (from the project root):

    python -m src.evaluate                 # full TEST split
    python -m src.evaluate --limit 20      # quick smoke test on 20 videos

Outputs are written to the output folder (default: outputs/):
    predictions.csv, metrics.json, scatter.png, bland_altman.png
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write figures to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from torch.utils.data import DataLoader

from . import config
from .dataset import EchoNetDataset, load_split
from .model import get_device, load_model


def predict(model, loader, device):
    names, y_true, y_pred = [], [], []
    total = len(loader)
    start = time.time()
    with torch.inference_mode():
        for i, (clips, targets, file_names) in enumerate(loader, start=1):
            outputs = model(clips.to(device)).view(-1).cpu().numpy()
            names.extend(file_names)
            y_true.extend(targets.numpy().tolist())
            y_pred.extend(outputs.tolist())
            if i % 10 == 0 or i == total:
                print(f"  batch {i}/{total} | {time.time() - start:.0f}s elapsed")
    return names, np.array(y_true), np.array(y_pred)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> dict:
    diffs = y_pred - y_true
    bias = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else float("nan")

    metrics = {
        "n_videos": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else float("nan"),
        "pearson_r": float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 2 else float("nan"),
        "bland_altman": {
            "mean_difference": bias,
            "lower_limit": bias - 1.96 * sd,
            "upper_limit": bias + 1.96 * sd,
        },
    }

    # Reduced EF detection: positive class = true EF <= threshold.
    positive = y_true <= threshold
    predicted_positive = y_pred <= threshold
    tp = int(np.sum(positive & predicted_positive))
    fn = int(np.sum(positive & ~predicted_positive))
    tn = int(np.sum(~positive & ~predicted_positive))
    fp = int(np.sum(~positive & predicted_positive))
    detection = {
        "threshold": threshold,
        "n_positive": int(positive.sum()),
        "n_negative": int((~positive).sum()),
        "sensitivity": tp / (tp + fn) if (tp + fn) else float("nan"),
        "specificity": tn / (tn + fp) if (tn + fp) else float("nan"),
        "auc": float("nan"),
    }
    if positive.any() and (~positive).any():
        # A lower predicted EF means a higher risk, so use -prediction as the score.
        detection["auc"] = float(roc_auc_score(positive, -y_pred))
    metrics["reduced_ef_detection"] = detection
    return metrics


def plot_scatter(y_true, y_pred, metrics, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=12, alpha=0.5, edgecolors="none")
    lo, hi = 0, 100
    ax.plot([lo, hi], [lo, hi], "r--", label="Perfect prediction")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Ground-truth EF (%)")
    ax.set_ylabel("Predicted EF (%)")
    ax.set_title(f"Predicted vs. true EF (R² = {metrics['r2']:.2f}, r = {metrics['pearson_r']:.2f})")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_bland_altman(y_true, y_pred, metrics, path: Path) -> None:
    ba = metrics["bland_altman"]
    means = (y_true + y_pred) / 2
    diffs = y_pred - y_true
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(means, diffs, s=12, alpha=0.5, edgecolors="none")
    ax.axhline(ba["mean_difference"], color="red", linestyle="--",
               label=f"Mean difference ({ba['mean_difference']:.2f})")
    ax.axhline(ba["upper_limit"], color="gray", linestyle="--",
               label=f"+1.96 SD ({ba['upper_limit']:.2f})")
    ax.axhline(ba["lower_limit"], color="gray", linestyle="--",
               label=f"-1.96 SD ({ba['lower_limit']:.2f})")
    ax.set_xlabel("Mean of ground truth and prediction (EF %)")
    ax.set_ylabel("Prediction - ground truth (EF %)")
    ax.set_title("Bland-Altman plot")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--weights", type=Path, default=config.WEIGHTS_PATH)
    parser.add_argument("--split", default="TEST", choices=["TRAIN", "VAL", "TEST"])
    parser.add_argument("--limit", type=int, default=None, help="evaluate only the first N videos")
    parser.add_argument("--batch-size", type=int, default=config.EVAL_BATCH_SIZE)
    parser.add_argument("--workers", type=int, default=config.NUM_WORKERS)
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    df = load_split(config.FILELIST_CSV, args.split)
    if args.limit:
        df = df.head(args.limit)
    print(f"Evaluating {len(df)} videos from the {args.split} split")

    dataset = EchoNetDataset(df, config.VIDEOS_DIR, return_name=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == "cuda")

    model = load_model(args.weights, device)
    names, y_true, y_pred = predict(model, loader, device)

    metrics = compute_metrics(y_true, y_pred, config.REDUCED_EF_THRESHOLD)
    metrics["split"] = args.split

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"FileName": names, "EF_true": y_true, "EF_pred": y_pred,
                  "abs_error": np.abs(y_pred - y_true)}).to_csv(args.out_dir / "predictions.csv", index=False)
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_scatter(y_true, y_pred, metrics, args.out_dir / "scatter.png")
    plot_bland_altman(y_true, y_pred, metrics, args.out_dir / "bland_altman.png")

    det = metrics["reduced_ef_detection"]
    print("\n" + "=" * 50)
    print(f"Split: {args.split} | videos: {metrics['n_videos']}")
    print(f"MAE: {metrics['mae']:.2f} | RMSE: {metrics['rmse']:.2f} | "
          f"R2: {metrics['r2']:.3f} | Pearson r: {metrics['pearson_r']:.3f}")
    print(f"Reduced EF (<= {det['threshold']:.0f}%): sensitivity {det['sensitivity']:.3f} | "
          f"specificity {det['specificity']:.3f} | AUC {det['auc']:.3f}")
    print(f"Results saved to: {args.out_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
