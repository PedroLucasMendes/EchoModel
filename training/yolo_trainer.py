"""
YOLO screening: trains multiple YOLO variants on the spectrogram tiles,
evaluates each on the test split, and returns the best-performing weights.

Parallel training: variants are grouped in pairs and each pair runs
simultaneously, one variant per GPU, using multiprocessing.
"""
import logging
import multiprocessing as mp
from multiprocessing import get_context
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
    try:
        # Make runs_dir absolute: Ultralytics treats a relative `project` as a
        # path inside its default runs/detect/ dir, which scatters weights under
        # runs/detect/<project>/... instead of where we expect.
        runs_dir = str(Path(runs_dir).resolve())
        last_ckpt = Path(runs_dir) / run_name / "weights" / "last.pt"

        if resume and last_ckpt.exists():
            model = YOLO(str(last_ckpt))
            # A run whose last.pt already reached the final epoch cannot be
            # resumed ("training to N epochs is finished, nothing to resume").
            # In that case just load the checkpoint and skip straight to val.
            ckpt = model.ckpt or {}
            trained_epochs = ckpt.get("epoch", -1)
            if trained_epochs is not None and 0 <= trained_epochs < epochs - 1:
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

        # Resolve the actual best.pt path. Prefer the trainer's save_dir when a
        # training run happened; otherwise fall back to the expected location.
        if getattr(model, "trainer", None) is not None:
            save_dir = Path(model.trainer.save_dir)
        else:
            save_dir = Path(runs_dir) / run_name
        best_weights = save_dir / "weights" / "best.pt"
        if not best_weights.exists():
            best_weights = save_dir / "weights" / "last.pt"

        metrics = model.val(data=data_yaml_path, split="test", device=gpu_id)
        result_queue.put({
            "model":     run_name,
            "weights":   str(best_weights.resolve()),
            "mAP50":     float(metrics.box.map50),
            "mAP50-95":  float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall":    float(metrics.box.mr),
        })
    except Exception:
        import traceback
        # Surface the real error in the parent instead of a bare exit code.
        result_queue.put({"model": run_name, "error": traceback.format_exc()})
        raise


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
        ctx = get_context("spawn")
        result_queue = ctx.Queue()
        processes = []

        for gpu_id, variant in zip(gpus, round_variants):
            log.info("  Launching %s on GPU %d", variant, gpu_id)
            p = ctx.Process(
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

        # Drain the queue before joining: a full pipe would otherwise deadlock
        # the child on put() while the parent blocks on join().
        round_results = [result_queue.get() for _ in round_variants]

        for p in processes:
            p.join()

        errors = [r for r in round_results if "error" in r]
        if errors:
            msgs = "\n\n".join(f"[{r['model']}]\n{r['error']}" for r in errors)
            raise RuntimeError(f"YOLO training subprocess(es) failed:\n\n{msgs}")
        if any(p.exitcode != 0 for p in processes):
            raise RuntimeError(
                "A training process exited with a non-zero code but produced no "
                "error message (likely killed — e.g. OOM). Check the log above."
            )

        results_summary.extend(round_results)

    results_df = pd.DataFrame(results_summary).sort_values("mAP50-95", ascending=False)
    log.info("\n%s", results_df.to_string(index=False))
    return results_df


def load_best_yolo(results_df: pd.DataFrame, runs_dir: Path = YOLO_RUNS_DIR) -> YOLO:
    best_row  = results_df.iloc[0]
    best_name = best_row["model"]

    # Prefer the weights path captured during training; fall back to rebuilding
    # it from runs_dir for backward compatibility with older result frames.
    weights = best_row.get("weights") if "weights" in results_df.columns else None
    if not weights or not Path(weights).exists():
        weights = runs_dir / best_name / "weights" / "best.pt"

    log.info("Best YOLO: %s  weights: %s", best_name, weights)
    return YOLO(str(weights))
