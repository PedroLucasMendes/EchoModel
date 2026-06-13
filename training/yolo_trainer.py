"""
YOLO screening: trains multiple YOLO variants on the spectrogram tiles,
evaluates each on the test split, and returns the best-performing weights.
Multi-GPU is handled by Ultralytics' built-in DDP via the `device` argument.
"""
import logging
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from configs.config import (
    YOLO_VARIANTS, EPOCHS_YOLO_SCREEN, IMG_WIDTH, IMG_HEIGHT,
    YOLO_RUNS_DIR, YOLO_DATA_DIR,
)

log = logging.getLogger(__name__)


def train_yolo_variants(
    data_yaml_path: Path,
    variants: list[str] = YOLO_VARIANTS,
    epochs: int = EPOCHS_YOLO_SCREEN,
    img_size: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
    runs_dir: Path = YOLO_RUNS_DIR,
    device: str = "0,1",   # both H200 GPUs
) -> pd.DataFrame:
    results_summary = []

    for variant in variants:
        run_name = variant.replace(".pt", "")
        log.info("=== Training %s ===", run_name)

        model = YOLO(variant)
        model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=max(img_size),
            rect=True,
            project=str(runs_dir),
            name=run_name,
            exist_ok=True,
            device=device,
        )

        metrics = model.val(data=str(data_yaml_path), split="test")
        results_summary.append({
            "model":     run_name,
            "mAP50":     float(metrics.box.map50),
            "mAP50-95":  float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
        })

    results_df = pd.DataFrame(results_summary).sort_values("mAP50-95", ascending=False)
    log.info("\n%s", results_df.to_string(index=False))
    return results_df


def load_best_yolo(results_df: pd.DataFrame, runs_dir: Path = YOLO_RUNS_DIR) -> YOLO:
    best_name = results_df.iloc[0]["model"]
    weights   = runs_dir / best_name / "weights" / "best.pt"
    log.info("Best YOLO: %s  weights: %s", best_name, weights)
    return YOLO(str(weights))
