"""
Xeno-Canto downloader for pseudo-labelling.

Streams weakly-labelled recordings (one species label per file, no
time-frequency boxes) from the Xeno-Canto API *in batches*: a batch is
downloaded, handed to the caller for pseudo-labelling, then deleted before the
next batch is fetched. This keeps disk usage bounded even when pulling every
available recording for a large set of species (Perch-style), since the full
Xeno-Canto archive is far too large to hold at once.

Requires a free API key (https://xeno-canto.org/account) exported as the
XENO_CANTO_API_KEY environment variable — the v3 API rejects unauthenticated
requests.
"""
import os
import logging
from pathlib import Path
from typing import Iterator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from configs.config import (
    XC_API_URL, XC_DOWNLOAD_DIR, XC_BATCH_SIZE,
    XC_MIN_QUALITY, XC_MAX_PER_SPECIES, XC_DOWNLOAD_WORKERS,
    DOWNLOAD_CHUNK_SIZE,
)

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["A", "B", "C", "D", "E"]


def _api_key() -> str:
    key = os.environ.get("XENO_CANTO_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "XENO_CANTO_API_KEY is not set. Get a free key at "
            "https://xeno-canto.org/account and export it before running "
            "the pseudo_label stage."
        )
    return key


def _build_query(scientific_name: str, min_quality: Optional[str]) -> str:
    """Build a Xeno-Canto query string for one species, birds only."""
    # sp: matches the scientific name; grp:birds restricts to avian recordings.
    query = f'sp:"{scientific_name}" grp:birds'
    if min_quality:
        # q:>:C means "quality C or better"; build the OR set explicitly so we
        # only keep ratings at least as good as min_quality.
        cutoff = _QUALITY_ORDER.index(min_quality.upper())
        allowed = _QUALITY_ORDER[: cutoff + 1]
        query += " " + " ".join(f"q:{r}" for r in allowed)
    return query


def _get(query: str, page: int) -> dict:
    resp = requests.get(
        XC_API_URL,
        params={"query": query, "key": _api_key(), "page": page},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def list_recordings(
    scientific_name: str,
    min_quality: Optional[str] = XC_MIN_QUALITY,
    max_per_species: Optional[int] = XC_MAX_PER_SPECIES,
) -> list[dict]:
    """Return recording metadata dicts for one species across all pages."""
    query = _build_query(scientific_name, min_quality)
    recordings: list[dict] = []

    try:
        first = _get(query, page=1)
    except Exception as exc:
        log.warning("Xeno-Canto query failed for %s: %s", scientific_name, exc)
        return []

    num_pages = int(first.get("numPages", 1))
    recordings.extend(first.get("recordings", []))

    for page in range(2, num_pages + 1):
        if max_per_species and len(recordings) >= max_per_species:
            break
        try:
            recordings.extend(_get(query, page=page).get("recordings", []))
        except Exception as exc:
            log.warning("Page %d failed for %s: %s", page, scientific_name, exc)
            break

    if max_per_species:
        recordings = recordings[:max_per_species]
    return recordings


def _file_url(rec: dict) -> Optional[str]:
    """Extract the audio download URL, tolerating v2/v3 field variations."""
    url = rec.get("file") or rec.get("file-url") or rec.get("url")
    if url and url.startswith("//"):
        url = "https:" + url
    return url


def _download_one(rec: dict, target: str, dest_dir: Path) -> Optional[tuple[str, str]]:
    """Download a single recording. Returns (local_path, target) or None."""
    url = _file_url(rec)
    rec_id = rec.get("id", "unknown")
    if not url:
        log.warning("No file URL for recording %s (%s)", rec_id, target)
        return None

    # Keep a flat, predictable filename; extension is best-effort.
    fname = rec.get("file-name") or f"XC{rec_id}.mp3"
    dest = dest_dir / f"XC{rec_id}_{Path(fname).name}"
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(DOWNLOAD_CHUNK_SIZE):
                f.write(chunk)
    except Exception as exc:
        log.warning("Download failed for XC%s: %s", rec_id, exc)
        dest.unlink(missing_ok=True)
        return None
    return (str(dest), target)


def iter_species_batches(
    species_to_name: dict[str, str],
    batch_size: int = XC_BATCH_SIZE,
    min_quality: Optional[str] = XC_MIN_QUALITY,
    max_per_species: Optional[int] = XC_MAX_PER_SPECIES,
    dest_dir: Path = XC_DOWNLOAD_DIR,
    workers: int = XC_DOWNLOAD_WORKERS,
) -> Iterator[list[tuple[str, str]]]:
    """
    Yield batches of (local_audio_path, target) pairs, downloading on demand.

    ``species_to_name`` maps the dataset target label (e.g. an eBird code) to
    the scientific name used to query Xeno-Canto. After the caller finishes a
    batch it must delete the audio files (see delete_batch) before the next
    batch is yielded so disk usage stays bounded.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    # Flatten all (recording, target) jobs across species, then batch them.
    jobs: list[tuple[dict, str]] = []
    for target, sci_name in species_to_name.items():
        if not sci_name:
            log.warning("No scientific name for target %s — skipping", target)
            continue
        recs = list_recordings(sci_name, min_quality, max_per_species)
        log.info("%s (%s): %d recordings", target, sci_name, len(recs))
        jobs.extend((rec, target) for rec in recs)

    log.info("Total recordings to download: %d (batch size %d)", len(jobs), batch_size)

    for start in range(0, len(jobs), batch_size):
        batch_jobs = jobs[start: start + batch_size]
        downloaded: list[tuple[str, str]] = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(_download_one, rec, target, dest_dir)
                for rec, target in batch_jobs
            ]
            for fut in as_completed(futures):
                res = fut.result()
                if res is not None:
                    downloaded.append(res)
        log.info(
            "Batch %d–%d: downloaded %d/%d files",
            start, start + len(batch_jobs), len(downloaded), len(batch_jobs),
        )
        yield downloaded


def delete_batch(pairs: list[tuple[str, str]]) -> None:
    """Delete the audio files of a processed batch to free disk space."""
    for path, _ in pairs:
        Path(path).unlink(missing_ok=True)
