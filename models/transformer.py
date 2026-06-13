"""
EchoFormer — Transformer encoder operating on the 2-D token grid produced
by EchoConvBackbone, with [CLS] (classification) and [LOC] (localisation)
special tokens and separable 2-D positional embeddings.
"""
import torch
import torch.nn as nn


class _TransformerLayerWithAttn(nn.Module):
    """Pre-norm Transformer encoder layer that also returns attention weights."""

    def __init__(self, embed_dim: int, num_heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.norm1(x)
        attn_out, attn_w = self.attn(h, h, h, need_weights=True,
                                     average_attn_weights=True)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x, attn_w


class EchoFormer(nn.Module):
    """
    Flattens (B, C, F', T') -> (B, F'*T', C), adds separable 2-D positional
    embeddings, prepends [CLS] and [LOC] tokens, and runs self-attention.
    """

    def __init__(self, embed_dim: int, freq_tokens: int, time_tokens: int,
                 num_heads: int = 4, num_layers: int = 3,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.freq_tokens = freq_tokens
        self.time_tokens = time_tokens

        self.freq_pos = nn.Parameter(torch.zeros(freq_tokens, embed_dim))
        self.time_pos = nn.Parameter(torch.zeros(time_tokens, embed_dim))
        nn.init.trunc_normal_(self.freq_pos, std=0.02)
        nn.init.trunc_normal_(self.time_pos, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.loc_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.loc_token, std=0.02)

        self.layers = nn.ModuleList([
            _TransformerLayerWithAttn(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, feat_map: torch.Tensor, return_attention: bool = False
    ) -> tuple:
        B, C, Fp, Tp = feat_map.shape
        tokens = feat_map.flatten(2).transpose(1, 2)        # (B, F'*T', C)

        pos = (self.freq_pos[:, None, :] + self.time_pos[None, :, :])  # (F', T', C)
        tokens = tokens + pos.reshape(Fp * Tp, C).unsqueeze(0)

        cls = self.cls_token.expand(B, -1, -1)
        loc = self.loc_token.expand(B, -1, -1)
        x = torch.cat([cls, loc, tokens], dim=1)            # (B, 2+F'*T', C)

        last_attn = None
        for layer in self.layers:
            x, last_attn = layer(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        loc_out = x[:, 1]

        if return_attention:
            loc_attn = last_attn[:, 1, 2:].reshape(B, Fp, Tp)
            return cls_out, loc_out, loc_attn
        return cls_out, loc_out
