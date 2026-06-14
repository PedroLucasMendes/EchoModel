"""
YOLO screening: trains multiple YOLO variants on the spectrogram tiles,
evaluates each on the test split, and returns the best-performing weights.

Parallel training: variants are grouped in pairs and each pair runs
simultaneously, one variant per GPU, using multiprocessing.
"""
import logging
import multiprocessing as mp
from pathlib import Path

import pandas as pd
from ultralytics import YOLO

from configs.config import (
    YOLO_VARIANTS, EPOCHS_YOLO_SCREEN, IMG_WIDTH, IMG_HEIGHT,
    YOLO_RUNS_DIR, YOLO_DATA_DIR,
)

log = logging.getLogger(__name__)


def _train_single(
    variant: str,
    data_yaml_path: str,
    epochs: int,
    img_size: int,
    runs_dir: str,
    gpu_id: int,
    resume: bool,
    result_queue: mp.Queue,
) -> None:
    """Runs in a separate process — trains one variant on one GPU."""
    run_name = variant.replace(".pt", "")
    last_ckpt = Path(runs_dir) / run_name / "weights" / "last.pt"

    if resume and last_ckpt.exists():
        model = YOLO(str(last_ckpt))
        model.train(resume=True)
    else:
        model = YOLO(variant)
        model.train(
            data=data_yaml_path,
            epochs=epochs,
            imgsz=img_size,
            rect=True,
            project=runs_dir,
            name=run_name,
            exist_ok=True,
            device=gpu_id,
        )

    metrics = model.val(data=data_yaml_path, split="test")
    result_queue.put({
        "model":     run_name,
        "mAP50":     float(metrics.box.map50),
        "mAP50-95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
    })


def train_yolo_variants(
    data_yaml_path: Path,
    variants: list[str] = YOLO_VARIANTS,
    epochs: int = EPOCHS_YOLO_SCREEN,
    img_size: tuple[int, int] = (IMG_WIDTH, IMG_HEIGHT),
    runs_dir: Path = YOLO_RUNS_DIR,
    device: str = "0,1",
    resume: bool = False,
) -> pd.DataFrame:
    gpus = [int(g.strip()) for g in device.split(",")]
    n_gpus = len(gpus)
    results_summary = []

    # Split variants into rounds of n_gpus each
    rounds = [variants[i:i + n_gpus] for i in range(0, len(variants), n_gpus)]

    for round_idx, round_variants in enumerate(rounds):
        log.info("=== Round %d/%d: %s ===", round_idx + 1, len(rounds), round_variants)
        result_queue: mp.Queue = mp.Queue()
        processes = []

        for gpu_id, variant in zip(gpus, round_variants):
            log.info("  Launching %s on GPU %d", variant, gpu_id)
            p = mp.Process(
                target=_train_single,
                args=(
                    variant,
                    str(data_yaml_path),
                    epochs,
                    max(img_size),
                    str(runs_dir),
                    gpu_id,
                    resume,
                    result_queue,
                ),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            if p.exitcode != 0:
                raise RuntimeError(f"Training process exited with code {p.exitcode}")

        for _ in round_variants:
            results_summary.append(result_queue.get())

    results_df = pd.DataFrame(results_summary).sort_values("mAP50-95", ascending=False)
    log.info("\n%s", results_df.to_string(index=False))
    return results_df


def load_best_yolo(results_df: pd.DataFrame, runs_dir: Path = YOLO_RUNS_DIR) -> YOLO:
    best_name = results_df.iloc[0]["model"]
    weights   = runs_dir / best_name / "weights" / "best.pt"
    log.info("Best YOLO: %s  weights: %s", best_name, weights)
    return YOLO(str(weights))
