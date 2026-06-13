"""
EchoConvBackbone — EfficientNet-style MBConv backbone.
Reduces (1, 128, 500) log-mel input to (embed_dim, 4, 16) spatial tokens.
"""
import torch
import torch.nn as nn


class MBConvBlock(nn.Module):
    """Inverted-residual block with depthwise-separable conv + Squeeze-Excitation."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 expand_ratio: int = 4, se_ratio: float = 0.25):
        super().__init__()
        mid_ch = in_ch * expand_ratio

        self.expand = (
            nn.Sequential(
                nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                nn.BatchNorm2d(mid_ch),
                nn.SiLU(),
            ) if expand_ratio != 1 else nn.Identity()
        )
        if expand_ratio == 1:
            mid_ch = in_ch

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(),
        )

        se_ch = max(1, int(in_ch * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, se_ch, 1), nn.SiLU(),
            nn.Conv2d(se_ch, mid_ch, 1), nn.Sigmoid(),
        )

        self.project = nn.Sequential(
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.use_residual = (stride == 1 and in_ch == out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.expand(x)
        h = self.depthwise(h)
        h = h * self.se(h)
        h = self.project(h)
        return h + x if self.use_residual else h


class EchoConvBackbone(nn.Module):
    """
    Five stride-2 stages: (1, 128, 500) -> (embed_dim, 4, 16).
    Mirrors the EfficientNet-B3 spatial downsampling used in Perch v2.
    """

    def __init__(self, embed_dim: int = 192):
        super().__init__()
        self.stem   = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.SiLU(),
        )
        self.stage1 = MBConvBlock(32,  48,  stride=2)
        self.stage2 = MBConvBlock(48,  96,  stride=2)
        self.stage3 = MBConvBlock(96,  144, stride=2)
        self.stage4 = MBConvBlock(144, embed_dim, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)      # (B, 32,  64, 250)
        x = self.stage1(x)    # (B, 48,  32, 125)
        x = self.stage2(x)    # (B, 96,  16,  63)
        x = self.stage3(x)    # (B, 144,  8,  32)
        x = self.stage4(x)    # (B, embed_dim, 4, 16)
        return x
