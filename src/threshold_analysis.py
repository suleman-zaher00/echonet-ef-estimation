"""Reduced-EF detection analysis from saved predictions (no model run needed).

Usage (from the project root):

    python -m src.threshold_analysis

Optional, for a fair cut-off choice: first run the evaluation on the validation
split, then pass its predictions so the cut-off is chosen on VAL and only
*reported* on TEST:

    python -m src.evaluate --split VAL --out-dir outputs_val
    python -m src.threshold_analysis --select-on outputs_val/predictions.csv

Outputs (in the output folder): roc_curve.png, threshold_table.csv.
It also prints a few videos suggested for Grad-CAM showcase figures.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from . import config


def _ratio(num: int, den: int) -> float:
    return num / den if den else float("nan")


def metrics_at(df: pd.DataFrame, cutoff: float, positive_ef: float) -> dict:
    """Classify 'predicted EF <= cutoff' as an alert; truth is 'EF <= positive_ef'."""
    positive = (df["EF_true"] <= positive_ef).to_numpy()
    alert = (df["EF_pred"] <= cutoff).to_numpy()
    tp = int((positive & alert).sum())
    fn = int((positive & ~alert).sum())
    tn = int((~positive & ~alert).sum())
    fp = int((~positive & alert).sum())
    return {
        "cutoff": cutoff,
        "sensitivity": _ratio(tp, tp + fn),
        "specificity": _ratio(tn, tn + fp),
        "ppv": _ratio(tp, tp + fp),
        "npv": _ratio(tn, tn + fn),
        "accuracy": _ratio(tp + tn, len(df)),
    }


def youden_cutoff(df: pd.DataFrame, positive_ef: float) -> float:
    """Cut-off on predicted EF that maximises sensitivity + specificity - 1."""
    positive = df["EF_true"] <= positive_ef
    fpr, tpr, thresholds = roc_curve(positive, -df["EF_pred"])
    return float(-thresholds[np.argmax(tpr - fpr)])


def plot_roc(df: pd.DataFrame, positive_ef: float, marks: dict[str, float], path: Path) -> float:
    positive = df["EF_true"] <= positive_ef
    fpr, tpr, _ = roc_curve(positive, -df["EF_pred"])
    auc = roc_auc_score(positive, -df["EF_pred"])

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, lw=2, label=f"Model (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance")
    for label, cutoff in marks.items():
        m = metrics_at(df, cutoff, positive_ef)
        ax.scatter(1 - m["specificity"], m["sensitivity"], s=60, zorder=3,
                   label=f"{label}: predicted EF <= {cutoff:.0f}%")
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(f"Detection of reduced EF (true EF <= {positive_ef:.0f}%)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return float(auc)


def suggest_examples(df: pd.DataFrame, positive_ef: float) -> None:
    err = df["abs_error"]
    normal = df[df["EF_true"] >= 55]
    reduced = df[df["EF_true"] <= positive_ef]
    picks = {
        "accurate, normal EF": normal.loc[normal["abs_error"].idxmin()],
        "accurate, reduced EF": reduced.loc[reduced["abs_error"].idxmin()],
        "largest error (failure case)": df.loc[err.idxmax()],
    }
    print("\nSuggested Grad-CAM showcase videos (best/worst cases, label them honestly):")
    for title, row in picks.items():
        print(f"  {title}: true {row['EF_true']:.1f}% | pred {row['EF_pred']:.1f}% "
              f"\n    python -m src.gradcam --file {row['FileName']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--predictions", type=Path, default=config.OUTPUT_DIR / "predictions.csv")
    parser.add_argument("--select-on", type=Path, default=None,
                        help="VAL predictions.csv used to choose the cut-off (recommended)")
    parser.add_argument("--out-dir", type=Path, default=config.OUTPUT_DIR)
    args = parser.parse_args()

    positive_ef = config.REDUCED_EF_THRESHOLD
    df = pd.read_csv(args.predictions)

    marks = {"Default": positive_ef}
    if args.select_on:
        chosen = youden_cutoff(pd.read_csv(args.select_on), positive_ef)
        marks["Chosen on VAL"] = chosen
        print(f"Cut-off chosen on the validation split (Youden): predicted EF <= {chosen:.1f}%")
    else:
        chosen = youden_cutoff(df, positive_ef)
        marks["Best on TEST (optimistic)"] = chosen
        print(f"Best cut-off on this same data: {chosen:.1f}% - optimistic, because it was "
              "tuned on the data it is reported on. Use --select-on for a fair estimate.")

    cutoffs = sorted({35.0, 40.0, 45.0, 50.0, round(chosen, 1)})
    table = pd.DataFrame([metrics_at(df, c, positive_ef) for c in cutoffs])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "threshold_table.csv", index=False)
    auc = plot_roc(df, positive_ef, marks, args.out_dir / "roc_curve.png")

    print(f"\nAUC for true EF <= {positive_ef:.0f}%: {auc:.3f}")
    print("Alert when predicted EF <= cutoff:")
    print(table.round(3).to_string(index=False))
    suggest_examples(df, positive_ef)
    print(f"\nSaved: {args.out_dir / 'roc_curve.png'}\nSaved: {args.out_dir / 'threshold_table.csv'}")


if __name__ == "__main__":
    main()
