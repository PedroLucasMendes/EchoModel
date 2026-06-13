"""
Visualisation helpers: attention heatmaps over spectrograms.
"""
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image


def plot_loc_attention(
    model: nn.Module,
    spec: torch.Tensor,
    device: torch.device,
    ax=None,
):
    """
    spec: (1, n_mels, T) tensor (already normalised, as returned by EchoModelDataset).
    Overlays the [LOC] token's attention map on the spectrogram.
    """
    model.eval()
    with torch.no_grad():
        x = spec.unsqueeze(0).to(device)
        _, _, _, loc_attn = model(x, return_attention=True)

    attn = loc_attn[0].cpu().numpy()                  # (F'=4, T'=16)
    attn_img = Image.fromarray((attn / (attn.max() + 1e-8) * 255).astype(np.uint8))
    attn_rs   = np.array(
        attn_img.resize((spec.shape[-1], spec.shape[-2]), Image.BILINEAR)
    ) / 255.0

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 4))

    ax.imshow(spec[0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
    ax.imshow(attn_rs, origin="lower", aspect="auto", cmap="viridis", alpha=0.40)
    ax.set_title("Log-mel spectrogram + [LOC] attention heatmap")
    ax.set_xlabel("Time frames")
    ax.set_ylabel("Mel bands")
    return ax
