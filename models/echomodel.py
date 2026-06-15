"""
EchoModel — full model combining EchoConvBackbone + EchoFormer with
classification, temporal localisation, and frequency localisation heads.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.backbone import EchoConvBackbone
from models.transformer import EchoFormer
from configs.config import EMBED_DIM, NUM_HEADS, NUM_LAYERS, W_CLS, W_T, W_F


class EchoModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        embed_dim: int = EMBED_DIM,
        num_heads: int = NUM_HEADS,
        num_layers: int = NUM_LAYERS,
    ):
        super().__init__()
        self.backbone    = EchoConvBackbone(embed_dim=embed_dim)
        # 128 mels / 2^5 = 4 freq tokens; 500 frames / 2^5 = 16 time tokens
        self.transformer = EchoFormer(
            embed_dim=embed_dim, freq_tokens=4, time_tokens=16,
            num_heads=num_heads, num_layers=num_layers,
        )
        self.cls_head  = nn.Linear(embed_dim, num_classes)
        self.time_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.SiLU(),
            nn.Linear(embed_dim // 2, 2),
        )
        self.freq_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2), nn.SiLU(),
            nn.Linear(embed_dim // 2, 2),
        )

    def forward(
        self, x: torch.Tensor, return_attention: bool = False
    ) -> tuple:
        feat_map = self.backbone(x)

        if return_attention:
            cls_out, loc_out, loc_attn = self.transformer(feat_map, return_attention=True)
        else:
            cls_out, loc_out = self.transformer(feat_map)
            loc_attn = None

        class_logits = self.cls_head(cls_out)
        bbox_t = torch.sigmoid(self.time_head(loc_out))
        bbox_f = torch.sigmoid(self.freq_head(loc_out))

        if return_attention:
            return class_logits, bbox_t, bbox_f, loc_attn
        return class_logits, bbox_t, bbox_f


def echomodel_loss(
    class_logits: torch.Tensor,
    bbox_t_pred:  torch.Tensor,
    bbox_f_pred:  torch.Tensor,
    target:       torch.Tensor,
    bbox_t_true:  torch.Tensor,
    bbox_f_true:  torch.Tensor,
    has_box:      torch.Tensor | None = None,
    w_cls: float = W_CLS,
    w_t:   float = W_T,
    w_f:   float = W_F,
) -> tuple[torch.Tensor, dict]:
    """Classification + time/frequency localisation loss.

    ``target`` may be either hard integer labels (shape [B]) or a soft/multi-hot
    target distribution (shape [B, C], e.g. from mixup). The latter follows
    Perch 2.0, which uses soft cross-entropy so every vocalisation in a mixed
    window is recognised regardless of loudness.

    ``has_box`` (shape [B], 1/0) masks the localisation loss: boxless windows
    (full-window fallback) contribute only to the classification loss, so the
    localisation head is never trained on a fake box.
    """
    if target.dim() == 1:
        loss_cls = F.cross_entropy(class_logits, target)
    else:
        loss_cls = soft_cross_entropy(class_logits, target)

    loss_t_ps = F.smooth_l1_loss(bbox_t_pred, bbox_t_true, reduction="none").mean(dim=1)
    loss_f_ps = F.smooth_l1_loss(bbox_f_pred, bbox_f_true, reduction="none").mean(dim=1)
    if has_box is not None:
        denom = has_box.sum().clamp_min(1.0)
        loss_t = (loss_t_ps * has_box).sum() / denom
        loss_f = (loss_f_ps * has_box).sum() / denom
    else:
        loss_t = loss_t_ps.mean()
        loss_f = loss_f_ps.mean()

    total = w_cls * loss_cls + w_t * loss_t + w_f * loss_f
    return total, {
        "cls":  loss_cls.item(),
        "time": loss_t.item(),
        "freq": loss_f.item(),
    }


def soft_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Cross entropy against a (possibly multi-hot) soft target distribution."""
    log_probs = F.log_softmax(logits, dim=1)
    return -(target * log_probs).sum(dim=1).mean()


def mixup_batch(
    spec: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    n: int = 4,
    alpha: float = 2.0,
    beta: float = 2.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Perch-style multi-component mixup.

    Mixes ``N ~ BetaBin(n, alpha, beta) + 1`` shuffled copies of the batch with
    Dirichlet weights, rescaling by sqrt(sum w_i^2) to preserve gain, and builds
    a multi-hot target (max over components) rather than a weighted average —
    so every species present should be predicted with high confidence.

    Returns (mixed_spec, multi_hot_target [B, C]). The bounding-box targets are
    left to the caller; mixed windows keep the *primary* (first-component) box.
    """
    B = spec.size(0)
    device = spec.device

    # Number of components for this batch.
    k = int(torch.distributions.Beta(alpha, beta).sample().item() * n)
    k = max(1, min(n, k)) + 1  # BetaBin(n,..)+1, clamped to [2, n+1]

    # Dirichlet weights, gain-preserving normalisation.
    w = torch.distributions.Dirichlet(torch.ones(k, device=device)).sample()
    w = w / torch.sqrt((w ** 2).sum())

    mixed = torch.zeros_like(spec)
    multi_hot = torch.zeros(B, num_classes, device=device)
    base_onehot = F.one_hot(target, num_classes).float()

    for i in range(k):
        # First component keeps the original order so the retained bounding box
        # (left unchanged by the caller) matches a real component in the mix.
        perm = torch.arange(B, device=device) if i == 0 else torch.randperm(B, device=device)
        mixed = mixed + w[i] * spec[perm]
        multi_hot = torch.maximum(multi_hot, base_onehot[perm])

    return mixed, multi_hot
