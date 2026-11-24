"""EchoNet-Dynamic video dataset.

Preprocessing is intentionally identical to the one used during training so
that the released weights behave exactly as they did on Kaggle:

    decode all frames -> BGR to RGB -> resize to IMG_SIZE -> sample NUM_FRAMES
    uniformly -> ToTensor -> Normalize -> tensor of shape (C, T, H, W)
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset

from .config import IMG_SIZE, MEAN, NUM_FRAMES, STD


def load_split(filelist_csv: str | Path, split: str) -> pd.DataFrame:
    """Return the rows of FileList.csv that belong to TRAIN, VAL or TEST."""
    manifest = pd.read_csv(filelist_csv)
    df = manifest[manifest["Split"].str.upper() == split.upper()]
    if df.empty:
        raise ValueError(f"No rows found for split '{split}' in {filelist_csv}")
    return df.reset_index(drop=True)


def read_video(path: str | Path, image_size: int = IMG_SIZE) -> np.ndarray:
    """Decode a video into a uint8 array of shape (frames, H, W, 3), RGB."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(cv2.resize(frame, (image_size, image_size)))
    finally:
        cap.release()
    if not frames:
        raise RuntimeError(f"Video contains no frames: {path}")
    return np.stack(frames).astype(np.uint8)


def sample_indices(total_frames: int, num_frames: int = NUM_FRAMES) -> np.ndarray:
    """Pick `num_frames` indices spread uniformly across the whole video."""
    return np.linspace(0, max(total_frames - 1, 0), num_frames).astype(int)


class EchoNetDataset(Dataset):
    """Returns (clip, ef) pairs, or (clip, ef, file_name) if return_name=True.

    clip has shape (3, num_frames, image_size, image_size); ef is a float tensor
    holding the ejection fraction in percent.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        video_dir: str | Path,
        num_frames: int = NUM_FRAMES,
        image_size: int = IMG_SIZE,
        is_train: bool = False,
        return_name: bool = False,
    ) -> None:
        self.df = dataframe.reset_index(drop=True)
        self.video_dir = Path(video_dir)
        self.num_frames = num_frames
        self.image_size = image_size
        self.is_train = is_train
        self.return_name = return_name

        normalize = T.Normalize(mean=MEAN, std=STD)
        if is_train:
            # Same augmentation as the training run. Note that it is applied
            # independently to every frame.
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomRotation(15),
                T.ColorJitter(brightness=0.2, contrast=0.2),
                T.ToTensor(),
                normalize,
            ])
        else:
            self.transform = T.Compose([T.ToPILImage(), T.ToTensor(), normalize])

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        video_path = self.video_dir / f"{row['FileName']}.avi"
        frames = read_video(video_path, self.image_size)
        indices = sample_indices(len(frames), self.num_frames)
        clip = torch.stack([self.transform(frames[i]) for i in indices])
        clip = clip.permute(1, 0, 2, 3)  # (T, C, H, W) -> (C, T, H, W)
        ef = torch.tensor(row["EF"], dtype=torch.float32)
        if self.return_name:
            return clip, ef, str(row["FileName"])
        return clip, ef
