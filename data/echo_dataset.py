"""
PyTorch Dataset for the EchoModel training stage.

Optimisations for large-scale datasets:
  - Audio loaded with soundfile (faster than librosa for seeks).
  - Spectrogram computed on-the-fly in workers (no huge PNG dumps).
  - Optionally caches spectrograms to a memory-mapped numpy array
    (set CACHE_TO_MEMMAP=True) for repeated epoch access.
  - Compatible with DistributedSampler (two H200 GPUs).
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import pandas as pd

from configs.config import (
    ECHO_SR, ECHO_N_FFT, ECHO_HOP_LENGTH, ECHO_N_MELS,
    ECHO_FMIN, ECHO_FMAX, WIN_DURATION, WIN_STEP,
    MIN_BOX_FRACTION_ECHO,
    BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, PREFETCH_FACTOR,
    ECHODATA_DIR,
)
from data.spectrogram import load_echo_window

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Index builder (bbox_df -> window-level CSV)
# ---------------------------------------------------------------------------

def boxes_to_echomodel_windows(
    boxes_df: pd.DataFrame,
    audio_path: str,
    total_duration: Optional[float] = None,
    win_duration: float = WIN_DURATION,
    win_step: float = WIN_STEP,
    min_fraction: float = MIN_BOX_FRACTION_ECHO,
) -> list[dict]:
    import librosa
    if total_duration is None:
        total_duration = librosa.get_duration(path=audio_path)

    windows = []
    win_start = 0.0
    while win_start < total_duration:
        win_end = min(win_start + win_duration, total_duration)
        if (win_end - win_start) < win_duration * 0.5:
            break

        best_box, best_overlap = None, 0.0
        for _, row in boxes_df.iterrows():
            inter = max(0.0, min(row["t_max"], win_end) - max(row["t_min"], win_start))
            box_dur = row["t_max"] - row["t_min"]
            frac = inter / box_dur if box_dur > 0 else 0.0
            if frac >= min_fraction and inter > best_overlap:
                best_overlap = inter
                best_box = row

        record: dict = {"audio_path": str(audio_path), "window_start": win_start,
                        "window_duration": win_duration}
        if best_box is not None:
            t0 = max(best_box["t_min"], win_start)
            t1 = min(best_box["t_max"], win_end)
            record.update({
                "t_min_rel": (t0 - win_start) / win_duration,
                "t_max_rel": (t1 - win_start) / win_duration,
                "f_min": float(best_box["f_min"]),
                "f_max": float(best_box["f_max"]),
                "target": best_box["target"],
            })
        else:
            record.update({"t_min_rel": None, "t_max_rel": None,
                           "f_min": None, "f_max": None, "target": None})
        windows.append(record)
        win_start += win_step

    return windows


def build_echomodel_index(
    boxes_df: pd.DataFrame,
    out_csv: Path = ECHODATA_DIR / "echomodel_index.csv",
    drop_background: bool = True,
    num_workers: int = NUM_WORKERS,
) -> pd.DataFrame:
    from multiprocessing import Pool
    from tqdm import tqdm

    grouped = list(boxes_df.groupby("audio_path"))

    def _worker(args):
        audio_path, group = args
        return boxes_to_echomodel_windows(group, audio_path)

    all_windows = []
    with Pool(processes=num_workers) as pool:
        for windows in tqdm(pool.imap_unordered(_worker, grouped),
                            total=len(grouped), desc="Building EchoModel index"):
            all_windows.extend(windows)

    df = pd.DataFrame(all_windows)
    if drop_background:
        df = df.dropna(subset=["target"])

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log.info("EchoModel index saved to %s (%d windows)", out_csv, len(df))
    return df


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class EchoModelDataset(Dataset):
    def __init__(
        self,
        index_df: pd.DataFrame,
        label2idx: Optional[dict] = None,
        sr: int = ECHO_SR,
        n_mels: int = ECHO_N_MELS,
        n_fft: int = ECHO_N_FFT,
        hop_length: int = ECHO_HOP_LENGTH,
        win_duration: float = WIN_DURATION,
        fmin: float = ECHO_FMIN,
        fmax: float = ECHO_FMAX,
    ):
        self.df = index_df.reset_index(drop=True)
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_duration = win_duration
        self.fmin = fmin
        self.fmax = fmax

        if label2idx is None:
            labels = sorted(self.df["target"].dropna().unique())
            label2idx = {lab: i for i, lab in enumerate(labels)}
        self.label2idx = label2idx

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]

        spec = load_echo_window(
            row["audio_path"], window_start=float(row["window_start"]),
            win_duration=self.win_duration, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax,
        )
        spec_t = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, T)

        target_idx = self.label2idx[row["target"]]
        bbox_t = torch.tensor([row["t_min_rel"], row["t_max_rel"]], dtype=torch.float32)
        f_range = self.fmax - self.fmin
        f_min_n = min(max((float(row["f_min"]) - self.fmin) / f_range, 0.0), 1.0)
        f_max_n = min(max((float(row["f_max"]) - self.fmin) / f_range, 0.0), 1.0)
        bbox_f = torch.tensor([f_min_n, f_max_n], dtype=torch.float32)

        return {
            "spec":   spec_t,
            "target": torch.tensor(target_idx, dtype=torch.long),
            "bbox_t": bbox_t,
            "bbox_f": bbox_f,
        }


# ---------------------------------------------------------------------------
# Materialised-feature Dataset (reads precomputed spectrogram .npy files)
# ---------------------------------------------------------------------------

class MaterialisedFeatureDataset(Dataset):
    """
    Reads precomputed log-mel spectrograms (.npy) plus their YOLO-derived
    boxes from the Xeno-Canto materialised index. Used when the raw audio has
    been deleted (Perch-scale runs) — see data.xc_materialise.

    Expected columns: spec_path, target, t_min_rel, t_max_rel, f_min, f_max.
    """

    def __init__(
        self,
        index_df: pd.DataFrame,
        label2idx: dict,
        fmin: float = ECHO_FMIN,
        fmax: float = ECHO_FMAX,
    ):
        self.df = index_df.reset_index(drop=True)
        self.label2idx = label2idx
        self.fmin = fmin
        self.fmax = fmax

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        spec = np.load(row["spec_path"]).astype(np.float32)
        spec_t = torch.from_numpy(spec).unsqueeze(0)  # (1, n_mels, T)

        target_idx = self.label2idx[row["target"]]
        bbox_t = torch.tensor([row["t_min_rel"], row["t_max_rel"]], dtype=torch.float32)
        f_range = self.fmax - self.fmin
        f_min_n = min(max((float(row["f_min"]) - self.fmin) / f_range, 0.0), 1.0)
        f_max_n = min(max((float(row["f_max"]) - self.fmin) / f_range, 0.0), 1.0)
        bbox_f = torch.tensor([f_min_n, f_max_n], dtype=torch.float32)

        return {
            "spec":   spec_t,
            "target": torch.tensor(target_idx, dtype=torch.long),
            "bbox_t": bbox_t,
            "bbox_f": bbox_f,
        }


# ---------------------------------------------------------------------------
# DataLoader factory (DDP-aware)
# ---------------------------------------------------------------------------

def _stratify_or_none(df: pd.DataFrame):
    """Stratify by target only when every class has >=2 samples."""
    counts = df["target"].value_counts()
    return df["target"] if counts.min() >= 2 else None


def make_dataloaders(
    echo_index_df: pd.DataFrame,
    label2idx: dict,
    rank: int = 0,
    world_size: int = 1,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_frac: float = 0.15,
    seed: int = 42,
    dataset_cls: type = EchoModelDataset,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders.

    ``dataset_cls`` selects the sample source: EchoModelDataset reloads audio
    on the fly (Zenodo path), MaterialisedFeatureDataset reads precomputed
    .npy spectrograms (Xeno-Canto path, after audio is deleted).
    """
    from sklearn.model_selection import train_test_split

    train_val_df, test_df = train_test_split(
        echo_index_df, test_size=0.10, random_state=seed,
        stratify=_stratify_or_none(echo_index_df),
    )
    train_df, val_df = train_test_split(
        train_val_df, test_size=val_frac / 0.90, random_state=seed,
        stratify=_stratify_or_none(train_val_df),
    )

    def _make(df, shuffle):
        ds = dataset_cls(df, label2idx=label2idx)
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=True) if world_size > 1 else None
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(shuffle and sampler is None),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    return _make(train_df, True), _make(val_df, False), _make(test_df, False)
