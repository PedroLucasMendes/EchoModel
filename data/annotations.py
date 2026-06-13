"""
Load and normalise bounding-box annotation CSVs from all Zenodo datasets.
Returns a single merged DataFrame with canonical column names.
"""
import logging
from pathlib import Path

import pandas as pd

from configs.config import RAW_DIR, DATASETS_TO_DOWNLOAD

log = logging.getLogger(__name__)

_RENAME = {
    # Zenodo Bioacoustics datasets use title-case headers
    "Filename": "filename",
    "Begin Time (s)": "start_time",
    "End Time (s)": "end_time",
    "Low Freq (Hz)": "low_freq",
    "High Freq (Hz)": "high_freq",
    "Species eBird Code": "label",
}


def _build_audio_index(ds_dir: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for ext in ("*.flac", "*.wav", "*.ogg"):
        for p in ds_dir.rglob(ext):
            index[p.name] = p
    return index


def load_annotations(dataset_key: str, raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    ds_dir = Path(raw_dir) / dataset_key
    ann_path = ds_dir / "annotations.csv"
    if not ann_path.exists():
        log.warning("[skip] %s not found — skipping %s", ann_path, dataset_key)
        return pd.DataFrame()

    df = pd.read_csv(ann_path)
    df.columns = df.columns.str.strip()
    df = df.rename(columns=_RENAME)
    df["dataset"] = dataset_key

    if "filename" not in df.columns:
        log.error("%s: 'filename' column missing. Available columns: %s", dataset_key, list(df.columns))
        return pd.DataFrame()

    audio_index = _build_audio_index(ds_dir)
    df["audio_path"] = df["filename"].map(lambda f: str(audio_index.get(f, "")))
    return df


def load_all_annotations(
    datasets: list[str] = DATASETS_TO_DOWNLOAD,
    raw_dir: Path = RAW_DIR,
) -> pd.DataFrame:
    frames = []
    for key in datasets:
        df = load_annotations(key, raw_dir)
        if not df.empty:
            log.info("%s: %d bounding boxes loaded", key, len(df))
            frames.append(df)

    if not frames:
        raise RuntimeError("No annotation files found. Run the download step first.")

    bbox_df = pd.concat(frames, ignore_index=True)
    bbox_df = bbox_df[bbox_df["audio_path"].notna() & (bbox_df["audio_path"] != "")]
    log.info("Total bounding boxes: %d | unique species: %d",
             len(bbox_df), bbox_df["label"].nunique())
    return bbox_df
