"""
Load and normalise bounding-box annotation CSVs from all Zenodo datasets.
Returns a single merged DataFrame with canonical column names.
"""
import logging
import re
from pathlib import Path

import pandas as pd

from configs.config import RAW_DIR, DATASETS_TO_DOWNLOAD

log = logging.getLogger(__name__)

# Patterns that identify each canonical column, matched against the
# normalised (lowercase, no punctuation) raw column name.
_PATTERNS: list[tuple[str, list[str]]] = [
    ("filename",   ["filename", "file name", "file"]),
    ("start_time", ["begin time", "start time", "tstart", "start"]),
    ("end_time",   ["end time", "tend", "end"]),
    ("low_freq",   ["low freq", "low frequency", "fmin", "freq low"]),
    ("high_freq",  ["high freq", "high frequency", "fmax", "freq high"]),
    ("label",      ["species ebird code", "ebird code", "species", "label", "class"]),
]

_REQUIRED = {"filename", "start_time", "end_time", "low_freq", "high_freq", "label"}


def _normalise(s: str) -> str:
    """Lowercase and strip punctuation/units for fuzzy column matching."""
    return re.sub(r"[^a-z0-9 ]", " ", s.lower()).strip()


def _map_columns(raw_cols: list[str]) -> dict[str, str]:
    """Return a rename dict mapping raw column names → canonical names."""
    rename: dict[str, str] = {}
    for raw in raw_cols:
        norm = _normalise(raw)
        for canonical, patterns in _PATTERNS:
            if any(p in norm for p in patterns):
                rename[raw] = canonical
                break
    return rename


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

    rename = _map_columns(list(df.columns))
    df = df.rename(columns=rename)
    df["dataset"] = dataset_key

    missing = _REQUIRED - set(df.columns)
    if missing:
        log.error(
            "%s: could not map columns %s. Raw headers: %s",
            dataset_key, sorted(missing), list(df.columns),
        )
        return pd.DataFrame()

    audio_index = _build_audio_index(ds_dir)
    df["audio_path"] = df["filename"].map(lambda f: str(audio_index.get(f, "")))
    return df


def load_species_map(
    datasets: list[str] = DATASETS_TO_DOWNLOAD,
    raw_dir: Path = RAW_DIR,
) -> dict[str, str]:
    """
    Build a mapping {label -> scientific name} from each dataset's species.csv.

    The annotation `label` is usually an eBird code (e.g. "amerob"); querying
    Xeno-Canto needs the scientific name (e.g. "Turdus migratorius"). The
    species.csv shipped with each Zenodo record holds that correspondence.
    Column names vary, so we match them fuzzily.
    """
    code_keys = ["ebird code", "ebird", "code", "species code", "label"]
    sci_keys  = ["scientific name", "scientific", "sci name", "latin", "species"]

    mapping: dict[str, str] = {}
    for key in datasets:
        sp_path = Path(raw_dir) / key / "species.csv"
        if not sp_path.exists():
            log.warning("[skip] %s not found — no species map for %s", sp_path, key)
            continue
        df = pd.read_csv(sp_path)
        df.columns = df.columns.str.strip()
        norm = {c: _normalise(c) for c in df.columns}

        code_col = next((c for c, n in norm.items() if any(k in n for k in code_keys)), None)
        sci_col  = next((c for c, n in norm.items()
                         if any(k in n for k in sci_keys) and c != code_col), None)
        if not code_col or not sci_col:
            log.warning("%s: could not find code/scientific columns in %s",
                        key, list(df.columns))
            continue

        for code, sci in zip(df[code_col], df[sci_col]):
            if pd.notna(code) and pd.notna(sci):
                mapping.setdefault(str(code).strip(), str(sci).strip())

    log.info("Species map: %d label -> scientific-name entries", len(mapping))
    return mapping


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
