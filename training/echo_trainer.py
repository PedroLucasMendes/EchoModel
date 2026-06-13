"""
EchoModel training loop with full DDP support for two H200 GPUs.

Launched via torchrun inside main.py:
  torchrun --nproc_per_node=2 main.py --stage train_echo

Key optimisations:
  - torch.compile (PyTorch ≥ 2.0) for fused CUDA kernels on H200.
  - Mixed precision (bf16) via torch.amp — H200 has native bf16 throughput.
  - Gradient accumulation to decouple effective batch size from per-GPU VRAM.
  - GradScaler disabled for bf16 (not needed; uses GradScaler only for fp16).
  - Cosine annealing LR scheduler with linear warmup.
  - Checkpoint saves best val-loss model and every N epochs.
"""
import logging
import os
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler
import pandas as pd

from configs.config import (
    NUM_EPOCHS, LR, WEIGHT_DECAY, ECHOMODEL_DIR,
    BATCH_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS,
)
from models.echomodel import EchoModel, echomodel_loss

log = logging.getLogger(__name__)

_AMP_DTYPE = torch.bfloat16   # H200 has native bf16


def _warmup_cosine_scheduler(optimizer, warmup_epochs: int, total_epochs: int):
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
    warmup = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0,
                      total_iters=warmup_epochs)
    cosine = CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
    return SequentialLR(optimizer, schedulers=[warmup, cosine],
                        milestones=[warmup_epochs])


def setup_ddp(rank: int, world_size: int) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def run_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: GradScaler | None = None,
    accum_steps: int = 1,
) -> tuple[float, dict]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    comp_losses = {"cls": 0.0, "time": 0.0, "freq": 0.0}
    n_batches = 0
    ctx = torch.no_grad() if not is_train else nullcontext()

    with ctx:
        for step, batch in enumerate(loader):
            spec         = batch["spec"].to(device, non_blocking=True)
            target       = batch["target"].to(device, non_blocking=True)
            bbox_t_true  = batch["bbox_t"].to(device, non_blocking=True)
            bbox_f_true  = batch["bbox_f"].to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=_AMP_DTYPE):
                class_logits, bbox_t_pred, bbox_f_pred = model(spec)
                loss, comps = echomodel_loss(
                    class_logits, bbox_t_pred, bbox_f_pred,
                    target, bbox_t_true, bbox_f_true,
                )
                loss = loss / accum_steps

            if is_train:
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if (step + 1) % accum_steps == 0:
                    if scaler:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item() * accum_steps
            for k in comp_losses:
                comp_losses[k] += comps[k]
            n_batches += 1

    avg_loss  = total_loss / max(n_batches, 1)
    avg_comps = {k: v / max(n_batches, 1) for k, v in comp_losses.items()}
    return avg_loss, avg_comps


def train_echomodel(
    train_loader,
    val_loader,
    num_classes: int,
    rank: int = 0,
    world_size: int = 1,
    num_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    runs_dir: Path = ECHOMODEL_DIR,
    accum_steps: int = 2,
    compile_model: bool = True,
    warmup_epochs: int = 5,
) -> EchoModel:
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    model = EchoModel(
        num_classes=num_classes,
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
    ).to(device)

    if compile_model and hasattr(torch, "compile"):
        log.info("Compiling model with torch.compile …")
        model = torch.compile(model)

    if world_size > 1:
        model = DDP(model, device_ids=[rank], find_unused_parameters=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                                  fused=True)   # fused AdamW — faster on H200
    scheduler = _warmup_cosine_scheduler(optimizer, warmup_epochs, num_epochs)
    # bf16 doesn't need GradScaler; use it only for fp16 fallback
    scaler = None

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, num_epochs + 1):
        if world_size > 1 and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)

        train_loss, train_c = run_epoch(
            model, train_loader, device, optimizer, scaler, accum_steps)
        val_loss,   val_c   = run_epoch(model, val_loader, device)
        scheduler.step()

        if rank == 0:
            log.info(
                "[%02d/%02d] train=%.4f (cls=%.3f t=%.3f f=%.3f) | "
                "val=%.4f (cls=%.3f t=%.3f f=%.3f)",
                epoch, num_epochs,
                train_loss, train_c["cls"], train_c["time"], train_c["freq"],
                val_loss,   val_c["cls"],   val_c["time"],   val_c["freq"],
            )
            history.append({
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                **{f"train_{k}": v for k, v in train_c.items()},
                **{f"val_{k}":   v for k, v in val_c.items()},
            })

            raw_model = model.module if isinstance(model, DDP) else model
            # strip torch.compile wrapper if present
            if hasattr(raw_model, "_orig_mod"):
                raw_model = raw_model._orig_mod

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(raw_model.state_dict(), runs_dir / "echomodel_best.pt")
                log.info("  -> saved best checkpoint (val_loss=%.4f)", best_val_loss)

            if epoch % 10 == 0:
                torch.save(raw_model.state_dict(),
                           runs_dir / f"echomodel_epoch{epoch:03d}.pt")

    if rank == 0:
        pd.DataFrame(history).to_csv(runs_dir / "training_history.csv", index=False)

    raw_model = model.module if isinstance(model, DDP) else model
    if hasattr(raw_model, "_orig_mod"):
        raw_model = raw_model._orig_mod
    return raw_model
