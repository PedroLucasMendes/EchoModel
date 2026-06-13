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
    w_cls: float = W_CLS,
    w_t:   float = W_T,
    w_f:   float = W_F,
) -> tuple[torch.Tensor, dict]:
    loss_cls = F.cross_entropy(class_logits, target)
    loss_t   = F.smooth_l1_loss(bbox_t_pred, bbox_t_true)
    loss_f   = F.smooth_l1_loss(bbox_f_pred, bbox_f_true)
    total    = w_cls * loss_cls + w_t * loss_t + w_f * loss_f
    return total, {
        "cls":  loss_cls.item(),
        "time": loss_t.item(),
        "freq": loss_f.item(),
    }
