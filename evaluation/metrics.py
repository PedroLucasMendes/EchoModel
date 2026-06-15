"""
Evaluation metrics for EchoModel vs Perch v2 / BirdNET v3.

  - cmAP  : class-mean Average Precision (BirdCLEF standard)
  - tf_iou: 2-D time-frequency bounding-box IoU
  - top-k : top-1 / top-5 accuracy
"""
import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(__name__)


def class_mean_average_precision(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
) -> float:
    aps = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0:
            continue
        aps.append(average_precision_score(y_true_c, y_score[:, c]))
    return float(np.mean(aps)) if aps else float("nan")


def macro_roc_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    num_classes: int,
) -> float:
    """Class-mean ROC-AUC — the most stable metric reported for Perch v2.

    Averaged only over classes present in y_true (a class with no positive or
    no negative samples has an undefined ROC-AUC and is skipped).
    """
    aucs = []
    for c in range(num_classes):
        y_true_c = (y_true == c).astype(int)
        if y_true_c.sum() == 0 or y_true_c.sum() == len(y_true_c):
            continue
        aucs.append(roc_auc_score(y_true_c, y_score[:, c]))
    return float(np.mean(aucs)) if aucs else float("nan")


def time_freq_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    t_min_a, t_max_a, f_min_a, f_max_a = box_a
    t_min_b, t_max_b, f_min_b, f_max_b = box_b

    inter_t = max(0.0, min(t_max_a, t_max_b) - max(t_min_a, t_min_b))
    inter_f = max(0.0, min(f_max_a, f_max_b) - max(f_min_a, f_min_b))
    inter   = inter_t * inter_f
    area_a  = (t_max_a - t_min_a) * (f_max_a - f_min_a)
    area_b  = (t_max_b - t_min_b) * (f_max_b - f_min_b)
    union   = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    device: torch.device,
    amp_dtype: torch.dtype = torch.bfloat16,
) -> dict:
    model.eval()
    all_targets, all_logits = [], []
    all_t_pred,  all_t_true = [], []
    all_f_pred,  all_f_true = [], []

    for batch in loader:
        spec        = batch["spec"].to(device, non_blocking=True)
        target      = batch["target"]
        bbox_t_true = batch["bbox_t"]
        bbox_f_true = batch["bbox_f"]

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            logits, bt_pred, bf_pred = model(spec)

        all_targets.append(target.numpy())
        all_logits.append(logits.float().cpu().numpy())
        all_t_pred.append(bt_pred.float().cpu().numpy())
        all_t_true.append(bbox_t_true.numpy())
        all_f_pred.append(bf_pred.float().cpu().numpy())
        all_f_true.append(bbox_f_true.numpy())

    targets = np.concatenate(all_targets)
    logits  = np.concatenate(all_logits)
    t_pred  = np.concatenate(all_t_pred)
    t_true  = np.concatenate(all_t_true)
    f_pred  = np.concatenate(all_f_pred)
    f_true  = np.concatenate(all_f_true)

    # cmAP and macro ROC-AUC (Perch v2 reports both; ROC-AUC is most stable)
    probs   = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    cmap    = class_mean_average_precision(targets, probs, num_classes)
    roc_auc = macro_roc_auc(targets, probs, num_classes)

    # top-1 / top-5
    preds = logits.argmax(axis=1)
    top1  = float((preds == targets).mean())
    top5  = float(np.mean([
        targets[i] in logits[i].argsort()[-5:]
        for i in range(len(targets))
    ]))

    # mean time-freq IoU
    ious = [
        time_freq_iou(
            (t_pred[i, 0], t_pred[i, 1], f_pred[i, 0], f_pred[i, 1]),
            (t_true[i, 0], t_true[i, 1], f_true[i, 0], f_true[i, 1]),
        )
        for i in range(len(targets))
    ]
    mean_iou = float(np.mean(ious))

    results = {
        "cmAP":     cmap,
        "ROC_AUC":  roc_auc,
        "top1":     top1,
        "top5":     top5,
        "mean_iou": mean_iou,
    }
    log.info("Evaluation: %s", results)
    return results
