"""
Parallel dataset download from Zenodo with resume support.
Uses concurrent.futures for parallel file downloads and aiohttp-style chunking.
"""
import zipfile
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

from configs.config import (
    ZENODO_API, DOWNLOAD_CHUNK_SIZE,
    BBOX_DATASETS, RAW_DIR, DATASETS_TO_DOWNLOAD,
)

log = logging.getLogger(__name__)


def _download_file(url: str, dest_path: Path, chunk_size: int = DOWNLOAD_CHUNK_SIZE) -> Path:
    dest_path = Path(dest_path)
    if dest_path.exists():
        log.info("[skip] %s already exists", dest_path.name)
        return dest_path

    tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
    resume_pos = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {"Range": f"bytes={resume_pos}-"} if resume_pos else {}
    resp = requests.get(url, stream=True, timeout=120, headers=headers)
    if resume_pos and resp.status_code == 416:
        # server doesn't support range — restart
        resume_pos = 0
        resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0)) + resume_pos
    mode = "ab" if resume_pos else "wb"
    with open(tmp_path, mode) as f, tqdm(
        total=total, initial=resume_pos, unit="B",
        unit_scale=True, desc=dest_path.name, leave=False,
    ) as pbar:
        for chunk in resp.iter_content(chunk_size):
            f.write(chunk)
            pbar.update(len(chunk))

    tmp_path.rename(dest_path)
    return dest_path


def _extract_zip(zip_path: Path, dest_dir: Path) -> None:
    log.info("Extracting %s …", zip_path.name)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest_dir)
    log.info("Extracted %s", zip_path.name)


def download_zenodo_record(
    record_id: str,
    dest_dir: Path,
    only_files: list[str] | None = None,
    unzip: bool = True,
    max_workers: int = 4,
) -> Path:
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    meta = requests.get(ZENODO_API.format(record_id=record_id), timeout=30).json()
    files = meta.get("files", [])

    targets = [
        (f["key"], f["links"]["self"])
        for f in files
        if only_files is None or f["key"] in only_files
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_download_file, url, dest_dir / fname): fname
            for fname, url in targets
        }
        for fut in as_completed(futures):
            fname = futures[fut]
            out_path = fut.result()
            if unzip and out_path.suffix == ".zip":
                _extract_zip(out_path, dest_dir)

    return dest_dir


def download_all_datasets(
    datasets: list[str] = DATASETS_TO_DOWNLOAD,
    raw_dir: Path = RAW_DIR,
    file_filter: list[str] | None = None,
    max_workers_per_record: int = 4,
) -> None:
    if file_filter is None:
        file_filter = ["annotations.csv", "species.csv", "soundscape_data.zip"]

    for key in datasets:
        info = BBOX_DATASETS[key]
        out_dir = raw_dir / key
        log.info("=== %s (Zenodo %s) ===", info["name"], info["zenodo_id"])
        download_zenodo_record(
            record_id=info["zenodo_id"],
            dest_dir=out_dir,
            only_files=file_filter,
            unzip=True,
            max_workers=max_workers_per_record,
        )
