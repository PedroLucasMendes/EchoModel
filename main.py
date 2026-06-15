"""
EchoModel — entry point.

Pipeline stages (controlled via --stage):
  download       : Download all Zenodo datasets
  build_yolo     : Build spectrogram-tile YOLO dataset
  train_yolo     : Screen YOLO variants; select best
  pseudo_label   : Run best YOLO to pseudo-label weakly-labelled audio
  build_echo     : Build EchoModel window index CSV
  train_echo     : Train EchoModel on two H200 GPUs via DDP
  evaluate       : Evaluate best checkpoint on test split
  all            : Run the complete pipeline end-to-end

Single-GPU / CPU:
  python main.py --stage <stage>

Two H200 GPUs (DDP via torchrun):
  torchrun --nproc_per_node=2 main.py --stage train_echo

For stages other than train_echo the script always runs single-process;
torchrun is only needed (and only beneficial) for the training stage.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Make project root importable regardless of cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))

from utils.logging_setup import setup_logging
from configs.config import (
    PROJECT_ROOT, RAW_DIR, YOLO_DATA_DIR, YOLO_RUNS_DIR,
    PSEUDO_DIR, ECHODATA_DIR, ECHOMODEL_DIR,
    DATASETS_TO_DOWNLOAD, BATCH_SIZE, NUM_WORKERS, NUM_EPOCHS,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_dirs() -> None:
    for d in [PROJECT_ROOT, RAW_DIR, YOLO_DATA_DIR, YOLO_RUNS_DIR,
              PSEUDO_DIR, ECHODATA_DIR, ECHOMODEL_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)


def _ddp_rank() -> tuple[int, int]:
    """Return (rank, world_size) from torchrun env vars, defaulting to (0, 1)."""
    rank       = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


# ---------------------------------------------------------------------------
# Stage: download
# ---------------------------------------------------------------------------

def stage_download(args) -> None:
    from data.download import download_all_datasets
    datasets = args.datasets if args.datasets else DATASETS_TO_DOWNLOAD
    log.info("Downloading datasets: %s", datasets)
    download_all_datasets(datasets=datasets, raw_dir=RAW_DIR)


# ---------------------------------------------------------------------------
# Stage: build_yolo
# ---------------------------------------------------------------------------

def stage_build_yolo(args) -> None:
    from data.annotations import load_all_annotations
    from data.yolo_builder import build_yolo_dataset, split_yolo_dataset

    datasets = args.datasets if args.datasets else DATASETS_TO_DOWNLOAD
    bbox_df  = load_all_annotations(datasets=datasets, raw_dir=RAW_DIR)

    images_dir, labels_dir = build_yolo_dataset(
        bbox_df,
        out_dir=YOLO_DATA_DIR,
        max_files_per_dataset=args.max_files,
        num_workers=args.workers,
    )
    split_yolo_dataset(images_dir, labels_dir, out_dir=YOLO_DATA_DIR)
    log.info("YOLO dataset ready at %s", YOLO_DATA_DIR)


# ---------------------------------------------------------------------------
# Stage: train_yolo
# ---------------------------------------------------------------------------

def stage_train_yolo(args) -> None:
    from training.yolo_trainer import train_yolo_variants, load_best_yolo

    data_yaml = Path(YOLO_DATA_DIR) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found — run build_yolo first.")

    results_df = train_yolo_variants(
        data_yaml_path=data_yaml,
        device=args.device,
        resume=args.resume,
    )
    results_df.to_csv(YOLO_RUNS_DIR / "yolo_screening.csv", index=False)

    best = load_best_yolo(results_df)
    log.info("Best YOLO model: %s", results_df.iloc[0]["model"])


# ---------------------------------------------------------------------------
# Stage: pseudo_label
# ---------------------------------------------------------------------------

def stage_pseudo_label(args) -> None:
    import pandas as pd
    from ultralytics import YOLO
    from training.pseudo_labeler import build_pseudo_label_table

    screening_csv = YOLO_RUNS_DIR / "yolo_screening.csv"
    if not screening_csv.exists():
        raise FileNotFoundError("Run train_yolo first.")

    results_df = pd.read_csv(screening_csv)
    best_row   = results_df.iloc[0]
    best_name  = best_row["model"]
    # Prefer the weights path recorded during screening; fall back to rebuilding.
    weights = best_row["weights"] if "weights" in results_df.columns else None
    if not weights or not Path(weights).exists():
        weights = YOLO_RUNS_DIR / best_name / "weights" / "best.pt"
    yolo_model = YOLO(str(weights))

    if not args.xc_metadata:
        log.warning("No --xc_metadata provided; skipping pseudo-labelling.")
        return

    xc_meta = pd.read_csv(args.xc_metadata)
    pairs   = list(zip(xc_meta["audio_path"], xc_meta["target"]))
    build_pseudo_label_table(pairs, "xeno_canto", yolo_model)


# ---------------------------------------------------------------------------
# Stage: build_echo
# ---------------------------------------------------------------------------

def stage_build_echo(args) -> None:
    import pandas as pd
    from data.echo_dataset import build_echomodel_index
    from data.annotations import load_all_annotations

    datasets = args.datasets if args.datasets else DATASETS_TO_DOWNLOAD
    bbox_df  = load_all_annotations(datasets=datasets, raw_dir=RAW_DIR)

    gt_boxes = bbox_df.rename(columns={
        "start_time": "t_min", "end_time": "t_max",
        "low_freq": "f_min", "high_freq": "f_max", "label": "target",
    })[["audio_path", "t_min", "t_max", "f_min", "f_max", "target"]]

    pseudo_csv = PSEUDO_DIR / "pseudo_labels.csv"
    if pseudo_csv.exists():
        pseudo = pd.read_csv(pseudo_csv)[
            ["audio_path", "t_min", "t_max", "f_min", "f_max", "target"]
        ]
        combined = pd.concat([gt_boxes, pseudo], ignore_index=True)
        log.info("Combined GT + pseudo: %d boxes", len(combined))
    else:
        combined = gt_boxes
        log.info("No pseudo-labels found; using GT only (%d boxes)", len(combined))

    build_echomodel_index(
        combined,
        out_csv=ECHODATA_DIR / "echomodel_index.csv",
        num_workers=args.workers,
    )


# ---------------------------------------------------------------------------
# Stage: train_echo
# ---------------------------------------------------------------------------

def stage_train_echo(args) -> None:
    import pandas as pd
    from data.echo_dataset import make_dataloaders
    from training.echo_trainer import train_echomodel, setup_ddp, cleanup_ddp

    rank, world_size = _ddp_rank()

    if world_size > 1:
        setup_ddp(rank, world_size)

    setup_logging(rank=rank)

    index_csv = ECHODATA_DIR / "echomodel_index.csv"
    if not index_csv.exists():
        raise FileNotFoundError("Run build_echo first.")

    echo_index_df = pd.read_csv(index_csv)
    labels        = sorted(echo_index_df["target"].unique())
    label2idx     = {lab: i for i, lab in enumerate(labels)}
    num_classes   = len(label2idx)
    log.info("Number of species classes: %d", num_classes)

    train_loader, val_loader, test_loader = make_dataloaders(
        echo_index_df, label2idx,
        rank=rank, world_size=world_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = train_echomodel(
        train_loader, val_loader,
        num_classes=num_classes,
        rank=rank, world_size=world_size,
        num_epochs=args.epochs,
        runs_dir=ECHOMODEL_DIR,
        compile_model=not args.no_compile,
    )

    if rank == 0:
        log.info("Training complete. Evaluating on test split …")
        from evaluation.metrics import evaluate
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)
        results = evaluate(model, test_loader, num_classes, device)
        log.info("Test results: %s", results)

    if world_size > 1:
        cleanup_ddp()


# ---------------------------------------------------------------------------
# Stage: evaluate
# ---------------------------------------------------------------------------

def stage_evaluate(args) -> None:
    import pandas as pd
    from data.echo_dataset import make_dataloaders
    from evaluation.metrics import evaluate
    from models.echomodel import EchoModel

    index_csv = ECHODATA_DIR / "echomodel_index.csv"
    echo_index_df = pd.read_csv(index_csv)
    labels    = sorted(echo_index_df["target"].unique())
    label2idx = {lab: i for i, lab in enumerate(labels)}
    num_classes = len(label2idx)

    _, _, test_loader = make_dataloaders(
        echo_index_df, label2idx,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    checkpoint = args.checkpoint or str(ECHOMODEL_DIR / "echomodel_best.pt")
    device     = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = EchoModel(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    log.info("Loaded checkpoint from %s", checkpoint)

    results = evaluate(model, test_loader, num_classes, device)
    log.info("Evaluation results: %s", results)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="EchoModel pipeline — bird vocalisation detection & classification"
    )
    parser.add_argument(
        "--stage", required=True,
        choices=["download", "build_yolo", "train_yolo", "pseudo_label",
                 "build_echo", "train_echo", "evaluate", "all"],
        help="Pipeline stage to run",
    )
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Subset of dataset keys to process (default: all)")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Max audio files per dataset (quick smoke test)")
    parser.add_argument("--workers", type=int, default=NUM_WORKERS,
                        help="DataLoader / Pool workers")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Per-GPU batch size")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS,
                        help="Training epochs")
    parser.add_argument("--device", type=str, default="0,1",
                        help="CUDA device(s) for YOLO training (e.g. '0,1')")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint for evaluation")
    parser.add_argument("--resume", action="store_true",
                        help="Resume YOLO training from last checkpoint")
    parser.add_argument("--xc_metadata", type=str, default=None,
                        help="CSV with columns [audio_path, target] for pseudo-labelling")
    parser.add_argument("--no_compile", action="store_true",
                        help="Disable torch.compile (useful for debugging)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    rank, _ = _ddp_rank()
    setup_logging(rank=rank)
    _make_dirs()

    log.info("=== EchoModel pipeline | stage=%s ===", args.stage)
    log.info("PyTorch %s | CUDA available: %s | GPUs: %d",
             torch.__version__,
             torch.cuda.is_available(),
             torch.cuda.device_count())

    stages = {
        "download":     stage_download,
        "build_yolo":   stage_build_yolo,
        "train_yolo":   stage_train_yolo,
        "pseudo_label": stage_pseudo_label,
        "build_echo":   stage_build_echo,
        "train_echo":   stage_train_echo,
        "evaluate":     stage_evaluate,
    }

    if args.stage == "all":
        for name, fn in stages.items():
            if name == "pseudo_label" and not args.xc_metadata:
                log.info("Skipping pseudo_label (no --xc_metadata provided)")
                continue
            log.info("--- Running stage: %s ---", name)
            fn(args)
    else:
        stages[args.stage](args)

    log.info("=== Stage '%s' finished ===", args.stage)


if __name__ == "__main__":
    main()
