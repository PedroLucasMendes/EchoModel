"""
Audio-to-spectrogram utilities.
All heavy lifting is done in numpy/librosa so it can run across many workers.
For the YOLO stage spectrograms are saved as PNG images; for the EchoModel
stage the mel filterbank tensor is returned directly.
"""
import io
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import librosa
import soundfile as sf
from PIL import Image

from configs.config import (
    SR, N_FFT, HOP_LENGTH, N_MELS, FREQ_MAX,
    IMG_WIDTH, IMG_HEIGHT,
    ECHO_SR, ECHO_N_FFT, ECHO_HOP_LENGTH, ECHO_N_MELS,
    ECHO_FMIN, ECHO_FMAX, WIN_DURATION,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# YOLO-stage helpers
# ---------------------------------------------------------------------------

def load_audio(
    path: str | Path,
    sr: int = SR,
    offset: float = 0.0,
    duration: Optional[float] = None,
    mono: bool = True,
) -> np.ndarray:
    y, _ = librosa.load(str(path), sr=sr, mono=mono, offset=offset, duration=duration)
    return y


def pad_to_length(y: np.ndarray, target_samples: int) -> np.ndarray:
    if len(y) < target_samples:
        return np.pad(y, (0, target_samples - len(y)))
    return y[:target_samples]


def audio_to_mel_image(
    y: np.ndarray,
    sr: int = SR,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    n_mels: int = N_MELS,
    img_w: int = IMG_WIDTH,
    img_h: int = IMG_HEIGHT,
    freq_max: int = FREQ_MAX,
) -> Image.Image:
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmax=freq_max,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-8)
    S_uint8 = (np.flipud(S_norm) * 255).astype(np.uint8)
    return Image.fromarray(S_uint8).resize((img_w, img_h), Image.BILINEAR)


# ---------------------------------------------------------------------------
# EchoModel-stage helpers
# ---------------------------------------------------------------------------

def audio_to_echo_mel(
    y: np.ndarray,
    sr: int = ECHO_SR,
    n_fft: int = ECHO_N_FFT,
    hop_length: int = ECHO_HOP_LENGTH,
    n_mels: int = ECHO_N_MELS,
    fmin: float = ECHO_FMIN,
    fmax: float = ECHO_FMAX,
) -> np.ndarray:
    """Returns a normalised log-mel spectrogram as float32 (n_mels, T)."""
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_norm = (S_db - S_db.mean()) / (S_db.std() + 1e-8)
    return S_norm.astype(np.float32)


def load_echo_window(
    audio_path: str | Path,
    window_start: float,
    win_duration: float = WIN_DURATION,
    sr: int = ECHO_SR,
    n_fft: int = ECHO_N_FFT,
    hop_length: int = ECHO_HOP_LENGTH,
    n_mels: int = ECHO_N_MELS,
    fmin: float = ECHO_FMIN,
    fmax: float = ECHO_FMAX,
) -> np.ndarray:
    """Load one 5-second window and return its log-mel spectrogram."""
    y, _ = librosa.load(
        str(audio_path), sr=sr, mono=True,
        offset=window_start, duration=win_duration,
    )
    y = pad_to_length(y, int(win_duration * sr))
    return audio_to_echo_mel(y, sr, n_fft, hop_length, n_mels, fmin, fmax)
