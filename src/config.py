"""Central configuration for the EchoNet-Dynamic EF estimation project.

All paths can be overridden with environment variables, so no code edits are
needed when moving between machines:

    ECHONET_ROOT    folder that contains FileList.csv and the Videos/ directory
    EF_WEIGHTS      path to the trained model weights (.pth)
    EF_OUTPUT_DIR   folder where figures, GIFs and reports are written
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, default)).expanduser()


# ---------------------------------------------------------------- paths
DATASET_ROOT = _env_path("ECHONET_ROOT", PROJECT_ROOT / "data" / "EchoNet-Dynamic")
VIDEOS_DIR = DATASET_ROOT / "Videos"
FILELIST_CSV = DATASET_ROOT / "FileList.csv"

WEIGHTS_PATH = _env_path("EF_WEIGHTS", PROJECT_ROOT / "weights" / "best_model.pth")
OUTPUT_DIR = _env_path("EF_OUTPUT_DIR", PROJECT_ROOT / "outputs")

# ---------------------------------------------------------------- input
NUM_FRAMES = 32  # frames sampled uniformly from each video
IMG_SIZE = 112   # EchoNet-Dynamic videos are 112x112

# Kinetics-400 statistics, matching the pretrained R(2+1)D-18 backbone.
MEAN = [0.43216, 0.394666, 0.37645]
STD = [0.22803, 0.22145, 0.216989]

# ------------------------------------------- training setup (reference)
# These are the values used for the released weights (see
# notebooks/training_kaggle.ipynb). Training ran on 2x T4 GPUs.
TRAIN_BATCH_SIZE = 16
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5

# ------------------------------------------------------- evaluation
EVAL_BATCH_SIZE = 8
NUM_WORKERS = 2

# Clinical cut-off: EF <= 40% is commonly used to define reduced EF
# (heart failure with reduced ejection fraction).
REDUCED_EF_THRESHOLD = 40.0
