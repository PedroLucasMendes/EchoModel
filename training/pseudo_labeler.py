"""
Pseudo-labelling: runs the best YOLO model over weakly-labelled audio
(e.g. Xeno-Canto) in parallel across both GPUs and collects time-frequency
detections as pseudo ground-truth for EchoModel training.
"""
import logging
from pathlib import Path
from multiprocessing import Pool
from typing import Optional

import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO

from configs.config import (
    SR, TILE_DURATION, TILE_OVERLAP,
    YOLO_CONF_THRESHOLD, PSEUDO_DIR,
)
from data.spectrogram import load_audio, pad_to_length, audio_to_mel_image
from data.yolo_builder import inverse_yolo_to_time_freq

log = logging.getLogger(__name__)


def pseudo_label_file(
    audio_path: str,
    target: str,
    yolo_model: YOLO,
    conf: float = YOLO_CONF_THRESHOLD,
    tile_duration: float = TILE_DURATION,
    tile_overlap: float = TILE_OVERLAP,
) -> list[dict]:
    y = load_audio(audio_path, sr=SR)
    total_dur = len(y) / SR
    step = tile_duration - tile_overlap
    detections: list[dict] = []
    tile_start = 0.0

    while tile_start < total_dur:
        tile_end = min(tile_start + tile_duration, total_dur)
        y_tile = pad_to_length(
            y[int(tile_start * SR): int(tile_end * SR)],
            int(tile_duration * SR),
        )
        img = np.array(audio_to_mel_image(y_tile).convert("RGB"))
        results = yolo_model.predict(img, conf=conf, verbose=False)

        for r in results:
            for box in r.boxes:
                xc, yc, w, h = box.xywhn[0].tolist()
                t_min, t_max, f_min, f_max = inverse_yolo_to_time_freq(
                    xc, yc, w, h, tile_start=tile_start,
                )
                detections.append({
                    "t_min": max(0.0, t_min),
                    "t_max": min(total_dur, t_max),
                    "f_min": f_min, "f_max": f_max,
                    "target": target,
                    "yolo_conf": float(box.conf[0]),
                })
        tile_start += step

    return detections


def build_pseudo_label_table(
    file_target_pairs: list[tuple[str, str]],
    source_dataset_name: str,
    yolo_model: YOLO,
    out_csv: Path = PSEUDO_DIR / "pseudo_labels.csv",
    conf: float = YOLO_CONF_THRESHOLD,
) -> pd.DataFrame:
    rows: list[dict] = []
    for audio_path, target in tqdm(file_target_pairs, desc=source_dataset_name):
        try:
            dets = pseudo_label_file(audio_path, target, yolo_model, conf)
        except Exception as exc:
            log.warning("Skipping %s: %s", audio_path, exc)
            continue
        for d in dets:
            d["source_dataset"] = source_dataset_name
            d["audio_path"]     = str(audio_path)
            rows.append(d)

    df = pd.DataFrame(rows)
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if out_csv.exists():
        df = pd.concat([pd.read_csv(out_csv), df], ignore_index=True)
    df.to_csv(out_csv, index=False)
    log.info("Pseudo-labels saved to %s (%d rows)", out_csv, len(df))
    return df
