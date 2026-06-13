"""
Build the YOLO spectrogram-image dataset from annotated soundscapes.
Uses multiprocessing + lmdb-backed tile cache to handle TBs of audio efficiently.
"""
import logging
import random
import shutil
from pathlib import Path
from typing import Optional
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from configs.config import (
    SR, TILE_DURATION, TILE_OVERLAP,
    IMG_WIDTH, IMG_HEIGHT, FREQ_MAX,
    MIN_BOX_FRACTION, YOLO_CLASS_NAMES,
    YOLO_DATA_DIR,
)
from data.spectrogram import load_audio, pad_to_length, audio_to_mel_image

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YOLO coordinate conversion
# ---------------------------------------------------------------------------

def bbox_to_yolo(
    start_time: float, end_time: float,
    low_freq: float, high_freq: float,
    tile_start: float,
    tile_duration: float = TILE_DURATION,
    freq_max: float = FREQ_MAX,
    min_fraction: float = MIN_BOX_FRACTION,
) -> Optional[tuple]:
    tile_end = tile_start + tile_duration
    t0 = max(start_time, tile_start)
    t1 = min(end_time, tile_end)
    if t1 <= t0:
        return None

    orig_dur = end_time - start_time
    if orig_dur > 0 and (t1 - t0) / orig_dur < min_fraction:
        return None

    x0 = (t0 - tile_start) / tile_duration
    x1 = (t1 - tile_start) / tile_duration
    f0 = max(low_freq, 0.0)
    f1 = min(high_freq, freq_max)
    if f1 <= f0:
        return None

    y0 = 1.0 - f1 / freq_max
    y1 = 1.0 - f0 / freq_max
    xc = (x0 + x1) / 2
    yc = (y0 + y1) / 2
    w  = x1 - x0
    h  = y1 - y0
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    w  = min(max(w,  1e-3), 1.0)
    h  = min(max(h,  1e-3), 1.0)
    return xc, yc, w, h


def inverse_yolo_to_time_freq(
    xc: float, yc: float, w: float, h: float,
    tile_start: float,
    tile_duration: float = TILE_DURATION,
    freq_max: float = FREQ_MAX,
) -> tuple:
    x0, x1 = xc - w / 2, xc + w / 2
    y0, y1 = yc - h / 2, yc + h / 2
    t_min = tile_start + x0 * tile_duration
    t_max = tile_start + x1 * tile_duration
    f_max_val = (1 - y0) * freq_max
    f_min_val = (1 - y1) * freq_max
    return t_min, t_max, f_min_val, f_max_val


# ---------------------------------------------------------------------------
# Per-file tile worker (called inside multiprocessing Pool)
# ---------------------------------------------------------------------------

def _process_file_tiles(args: tuple) -> int:
    dataset, filename, audio_path, group_records, images_dir, labels_dir, tile_duration, tile_overlap = args
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    step = tile_duration - tile_overlap

    try:
        y = load_audio(audio_path, sr=SR)
    except Exception as exc:
        log.warning("Could not load %s: %s", audio_path, exc)
        return 0

    total_dur = len(y) / SR
    n_tiles = 0
    tile_start = 0.0

    while tile_start < total_dur:
        tile_end = min(tile_start + tile_duration, total_dur)
        if (tile_end - tile_start) < tile_duration * 0.5:
            break

        y_tile = pad_to_length(
            y[int(tile_start * SR): int(tile_end * SR)],
            int(tile_duration * SR),
        )

        # filter annotations that overlap this tile
        yolo_lines = []
        for row in group_records:
            if row["end_time"] <= tile_start or row["start_time"] >= tile_end:
                continue
            box = bbox_to_yolo(
                row["start_time"], row["end_time"],
                row["low_freq"], row["high_freq"],
                tile_start=tile_start, tile_duration=tile_duration,
            )
            if box:
                xc, yc, w, h = box
                yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        base = f"{dataset}__{Path(filename).stem}__t{int(tile_start)}"
        audio_to_mel_image(y_tile).save(images_dir / f"{base}.png")
        (labels_dir / f"{base}.txt").write_text("\n".join(yolo_lines))

        n_tiles += 1
        tile_start += step

    return n_tiles


# ---------------------------------------------------------------------------
# Public builder
# ---------------------------------------------------------------------------

def build_yolo_dataset(
    bbox_df: pd.DataFrame,
    out_dir: Path = YOLO_DATA_DIR,
    tile_duration: float = TILE_DURATION,
    tile_overlap: float = TILE_OVERLAP,
    max_files_per_dataset: Optional[int] = None,
    num_workers: int = min(cpu_count(), 16),
) -> tuple[Path, Path]:
    images_dir = Path(out_dir) / "images"
    labels_dir = Path(out_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    grouped = bbox_df.groupby(["dataset", "filename", "audio_path"])

    args_list = []
    files_per_dataset: dict[str, int] = {}
    for (dataset, filename, audio_path), group in grouped:
        if max_files_per_dataset is not None:
            files_per_dataset.setdefault(dataset, 0)
            if files_per_dataset[dataset] >= max_files_per_dataset:
                continue
            files_per_dataset[dataset] += 1
        group_records = group[["start_time", "end_time", "low_freq", "high_freq"]].to_dict("records")
        args_list.append((dataset, filename, audio_path, group_records,
                          str(images_dir), str(labels_dir), tile_duration, tile_overlap))

    total_tiles = 0
    with Pool(processes=num_workers) as pool:
        for n in tqdm(pool.imap_unordered(_process_file_tiles, args_list),
                      total=len(args_list), desc="Building YOLO tiles"):
            total_tiles += n

    log.info("Total tiles generated: %d", total_tiles)
    return images_dir, labels_dir


def split_yolo_dataset(
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path = YOLO_DATA_DIR,
    val_frac: float = 0.15,
    test_frac: float = 0.10,
    seed: int = 42,
) -> Path:
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    out_dir = Path(out_dir)

    all_images = sorted(images_dir.glob("*.png"))
    random.Random(seed).shuffle(all_images)

    n = len(all_images)
    n_val  = int(n * val_frac)
    n_test = int(n * test_frac)
    splits = {
        "val":   all_images[:n_val],
        "test":  all_images[n_val: n_val + n_test],
        "train": all_images[n_val + n_test:],
    }

    for split, files in splits.items():
        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)
        for img_path in files:
            lbl_path = labels_dir / (img_path.stem + ".txt")
            shutil.copy(img_path, img_out / img_path.name)
            if lbl_path.exists():
                shutil.copy(lbl_path, lbl_out / lbl_path.name)
        log.info("%s: %d images", split, len(files))

    data_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "names": {i: name for i, name in enumerate(YOLO_CLASS_NAMES)},
    }
    yaml_path = out_dir / "data.yaml"
    yaml_path.write_text(yaml.safe_dump(data_yaml, sort_keys=False))
    log.info("data.yaml written to %s", yaml_path)
    return yaml_path
