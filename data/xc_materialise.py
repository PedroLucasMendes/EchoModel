"""
Materialise Xeno-Canto recordings into EchoModel training features.

For each downloaded recording we, in a single pass:
  1. load the audio,
  2. select one or more 5 s windows (Perch-style, §2.1),
  3. compute the log-mel spectrogram of each window,
  4. run the best YOLO model on the window to obtain the time-frequency
     bounding box (the species *target* comes from Xeno-Canto; the *box* comes
     from YOLO — YOLO is an offline teacher only),
  5. save the spectrogram as a small .npy and append an index row.

The raw audio is NOT kept — the caller deletes it per batch. Only the light
spectrogram tensors and the index CSV persist, which is what makes a full
Perch-scale run fit on disk.

The materialised index has columns:
    spec_path, target, t_min_rel, t_max_rel, f_min, f_max, yolo_conf, source
matching what the feature Dataset (data.echo_dataset) consumes.
"""
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from ultralytics import YOLO

from configs.config import (
    ECHO_SR, WIN_DURATION, ECHO_FMIN, ECHO_FMAX,
    XC_WINDOWS_PER_REC, XC_WINDOW_SELECT, XC_YOLO_CONF, XC_KEEP_BOXLESS,
    XC_FEATURES_DIR,
)
from data.spectrogram import (
    load_audio, select_windows, audio_to_echo_mel, audio_to_mel_image,
)
from data.yolo_builder import inverse_yolo_to_time_freq

log = logging.getLogger(__name__)

_INDEX_COLS = [
    "spec_path", "target", "t_min_rel", "t_max_rel",
    "f_min", "f_max", "yolo_conf", "has_box", "source",
]


def _best_box_for_window(
    window_samples: np.ndarray,
    yolo_model: YOLO,
    conf: float,
    win_duration: float,
) -> Optional[dict]:
    """Run YOLO on one window; return the highest-confidence box or None."""
    img = np.array(audio_to_mel_image(window_samples).convert("RGB"))
    results = yolo_model.predict(img, conf=conf, verbose=False)

    best, best_conf = None, -1.0
    for r in results:
        for box in r.boxes:
            c = float(box.conf[0])
            if c > best_conf:
                xc, yc, w, h = box.xywhn[0].tolist()
                # tile_start=0 and tile_duration=win_duration: the image spans
                # exactly this window, so coordinates are relative to it.
                t_min, t_max, f_min, f_max = inverse_yolo_to_time_freq(
                    xc, yc, w, h, tile_start=0.0, tile_duration=win_duration,
                )
                best_conf = c
                best = {
                    "t_min": t_min, "t_max": t_max,
                    "f_min": f_min, "f_max": f_max, "yolo_conf": c,
                }
    return best


def materialise_batch(
    pairs: list[tuple[str, str]],
    yolo_model: YOLO,
    features_dir: Path = XC_FEATURES_DIR,
    win_duration: float = WIN_DURATION,
    sr: int = ECHO_SR,
    windows_per_rec: int = XC_WINDOWS_PER_REC,
    window_select: str = XC_WINDOW_SELECT,
    conf: float = XC_YOLO_CONF,
    keep_boxless: bool = XC_KEEP_BOXLESS,
) -> list[dict]:
    """
    Process one batch of (audio_path, target) pairs into feature rows.

    Returns a list of index-row dicts (one per materialised window). When YOLO
    finds no box and ``keep_boxless`` is True, the window is still kept with a
    full-window fallback box (yolo_conf=0) — the species label is valid for the
    classifier and the bird is assumed present somewhere in this focal recording.
    """
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for audio_path, target in pairs:
        try:
            y = load_audio(audio_path, sr=sr)
        except Exception as exc:
            log.warning("Skipping %s (load failed): %s", audio_path, exc)
            continue
        if y is None or len(y) == 0:
            continue

        windows = select_windows(
            y, sr=sr, win_duration=win_duration,
            n_windows=windows_per_rec, method=window_select,
        )

        stem = Path(audio_path).stem
        for wi, (win_start, win_samples) in enumerate(windows):
            spec_path = features_dir / f"{stem}_w{wi}.npy"
            # Resume safety: if this window was already materialised (crash after
            # save, before the batch was checkpointed), don't redo it or emit a
            # duplicate index row.
            if spec_path.exists():
                continue
            box = _best_box_for_window(win_samples, yolo_model, conf, win_duration)
            has_box = box is not None
            if box is None:
                if not keep_boxless:
                    continue
                # Full-window fallback box: whole time span, full freq range.
                # has_box=0 tells the trainer to skip the bbox loss for this
                # window so the localisation head isn't taught a fake box.
                box = {
                    "t_min": 0.0, "t_max": win_duration,
                    "f_min": ECHO_FMIN, "f_max": ECHO_FMAX, "yolo_conf": 0.0,
                }

            spec = audio_to_echo_mel(win_samples, sr=sr)
            np.save(spec_path, spec.astype(np.float16))

            # Box times are absolute within the 5 s window -> make relative.
            t_min_rel = max(0.0, box["t_min"]) / win_duration
            t_max_rel = min(win_duration, box["t_max"]) / win_duration
            rows.append({
                "spec_path": str(spec_path),
                "target":    target,
                "t_min_rel": float(t_min_rel),
                "t_max_rel": float(t_max_rel),
                "f_min":     float(box["f_min"]),
                "f_max":     float(box["f_max"]),
                "yolo_conf": float(box["yolo_conf"]),
                "has_box":   int(has_box),
                "source":    "xeno_canto",
            })

    return rows


def append_index(rows: list[dict], index_csv: Path) -> int:
    """Append rows to the materialised index CSV; return total row count."""
    index_csv = Path(index_csv)
    index_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=_INDEX_COLS)
    header = not index_csv.exists()
    df.to_csv(index_csv, mode="a", header=header, index=False)
    # Cheap row count without loading the whole file into memory.
    with open(index_csv) as f:
        total = sum(1 for _ in f) - 1
    return max(0, total)
