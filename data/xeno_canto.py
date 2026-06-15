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
import json
import logging
from pathlib import Path
from typing import Iterator, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from configs.config import (
    XC_API_URL, XC_DOWNLOAD_DIR, XC_BATCH_SIZE,
    XC_MIN_QUALITY, XC_MAX_PER_SPECIES, XC_DOWNLOAD_WORKERS,
    XC_PROGRESS_FILE, DOWNLOAD_CHUNK_SIZE,
)

log = logging.getLogger(__name__)

_QUALITY_ORDER = ["A", "B", "C", "D", "E"]


# ---------------------------------------------------------------------------
# Resume support: checkpoint progress at both species and batch granularity so
# a crash mid-species only loses the current batch, not the whole species.
# ---------------------------------------------------------------------------

def _read_progress(progress_file: Path) -> dict:
    p = Path(progress_file)
    if p.exists():
        try:
            data = json.loads(p.read_text())
            data.setdefault("done_species", [])
            data.setdefault("done_batches", [])
            return data
        except Exception as exc:
            log.warning("Could not read progress file %s: %s", p, exc)
    return {"done_species": [], "done_batches": []}


def _write_progress(data: dict, progress_file: Path) -> None:
    p = Path(progress_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({
        "done_species": sorted(data.get("done_species", [])),
        "done_batches": sorted(data.get("done_batches", [])),
    }))
    tmp.replace(p)  # atomic — never leaves a half-written progress file


def load_progress(progress_file: Path = XC_PROGRESS_FILE) -> set[str]:
    """Return the set of species labels already fully processed (for resume)."""
    return set(_read_progress(progress_file)["done_species"])


def load_done_batches(progress_file: Path = XC_PROGRESS_FILE) -> set[str]:
    """Return the set of '<label>#<batch_idx>' keys already materialised."""
    return set(_read_progress(progress_file)["done_batches"])


def mark_batch_done(label: str, batch_idx: int,
                    progress_file: Path = XC_PROGRESS_FILE) -> None:
    """Checkpoint a single completed batch of a species."""
    data = _read_progress(progress_file)
    data["done_batches"] = list(set(data["done_batches"]) | {f"{label}#{batch_idx}"})
    _write_progress(data, progress_file)


def mark_species_done(label: str, progress_file: Path = XC_PROGRESS_FILE) -> None:
    """Mark a species fully done and drop its now-redundant per-batch keys."""
    data = _read_progress(progress_file)
    data["done_species"] = list(set(data["done_species"]) | {label})
    data["done_batches"] = [
        b for b in data["done_batches"] if not b.startswith(f"{label}#")
    ]
    _write_progress(data, progress_file)


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


def _scientific_name(rec: dict) -> Optional[str]:
    """Build 'Genus species' from a recording, tolerating field variations."""
    gen = (rec.get("gen") or "").strip()
    sp = (rec.get("sp") or "").strip()
    if gen and sp:
        return f"{gen} {sp}"
    return None


def list_all_bird_species(
    cache_file: Path,
    countries: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Enumerate the Xeno-Canto avian catalogue as {scientific_name: scientific_name}.

    Walks ``grp:birds`` per country (keeps each query's page count tractable) and
    collects unique 'Genus species' pairs. The result is cached to ``cache_file``
    so the (slow) enumeration runs only once; delete the file to refresh.

    NOTE: this is the Perch-scale species list. It is large (~10k species) and
    the walk itself makes many API calls — it is meant to be run once.
    """
    cache_file = Path(cache_file)
    if cache_file.exists():
        species = json.loads(cache_file.read_text())
        log.info("Loaded %d bird species from cache %s", len(species), cache_file)
        return species

    # A broad spread of countries covers essentially the whole avian catalogue;
    # querying per country keeps each query under the API's page ceiling.
    countries = countries or _DEFAULT_XC_COUNTRIES

    found: set[str] = set()
    for i, country in enumerate(countries):
        query = f'grp:birds cnt:"{country}"'
        try:
            first = _get(query, page=1)
        except Exception as exc:
            log.warning("Species enumeration failed for %s: %s", country, exc)
            continue
        num_pages = int(first.get("numPages", 1))
        for rec in first.get("recordings", []):
            name = _scientific_name(rec)
            if name:
                found.add(name)
        for page in range(2, num_pages + 1):
            try:
                data = _get(query, page=page)
            except Exception as exc:
                log.warning("  %s page %d failed: %s", country, page, exc)
                break
            for rec in data.get("recordings", []):
                name = _scientific_name(rec)
                if name:
                    found.add(name)
        log.info("[species %d/%d] %s -> %d unique species so far",
                 i + 1, len(countries), country, len(found))

    species = {name: name for name in sorted(found)}
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(species, indent=0))
    log.info("Enumerated %d bird species -> cached at %s", len(species), cache_file)
    return species


# A representative spread of high-diversity countries. The XC archive's species
# are overwhelmingly covered by querying these; extend as needed.
_DEFAULT_XC_COUNTRIES = [
    "Brazil", "Colombia", "Peru", "Ecuador", "Bolivia", "Venezuela",
    "United States", "Mexico", "Canada", "Argentina", "Panama", "Costa Rica",
    "India", "China", "Indonesia", "Australia", "Malaysia", "Thailand",
    "Kenya", "Tanzania", "South Africa", "Uganda", "Cameroon", "Ethiopia",
    "Russia", "Germany", "United Kingdom", "Spain", "France", "Sweden",
    "Papua New Guinea", "Philippines", "Vietnam", "Myanmar", "Madagascar",
    "Japan", "Turkey", "Iran", "Kazakhstan", "Mongolia",
]


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


def _download_batch(
    batch_jobs: list[tuple[dict, str]],
    dest_dir: Path,
    workers: int,
) -> list[tuple[str, str]]:
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
    return downloaded


def iter_species_batches(
    species_to_name: dict[str, str],
    batch_size: int = XC_BATCH_SIZE,
    min_quality: Optional[str] = XC_MIN_QUALITY,
    max_per_species: Optional[int] = XC_MAX_PER_SPECIES,
    dest_dir: Path = XC_DOWNLOAD_DIR,
    workers: int = XC_DOWNLOAD_WORKERS,
    skip_species: Optional[set[str]] = None,
    skip_batches: Optional[set[str]] = None,
) -> Iterator[tuple[str, int, bool, list[tuple[str, str]]]]:
    """
    Yield ``(label, batch_idx, is_last_batch_of_species, [(audio_path, target)])``.

    Iterates species by species and batches the recordings within each species.
    ``species_to_name`` maps the target label to the scientific name used to
    query XC. The caller must delete each batch's audio (delete_batch) before
    the next, keeping disk usage bounded.

    Resume: species in ``skip_species`` are skipped entirely; individual
    '<label>#<batch_idx>' keys in ``skip_batches`` are skipped so a crash
    mid-species only re-downloads the batch that was in flight.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    skip_species = skip_species or set()
    skip_batches = skip_batches or set()

    for target, sci_name in species_to_name.items():
        if target in skip_species:
            log.info("[resume] skipping already-done species %s", target)
            continue
        if not sci_name:
            log.warning("No scientific name for target %s — skipping", target)
            continue

        recs = list_recordings(sci_name, min_quality, max_per_species)
        log.info("%s (%s): %d recordings", target, sci_name, len(recs))
        if not recs:
            # Still yield an (empty, last) signal so the caller can mark it done.
            yield (target, 0, True, [])
            continue

        jobs = [(rec, target) for rec in recs]
        n_batches = (len(jobs) + batch_size - 1) // batch_size
        for bi in range(n_batches):
            is_last = bi == n_batches - 1
            if f"{target}#{bi}" in skip_batches:
                log.info("  [resume] skipping %s batch %d/%d", target, bi + 1, n_batches)
                # Still yield an empty batch so the caller advances/marks state.
                yield (target, bi, is_last, [])
                continue
            batch_jobs = jobs[bi * batch_size: (bi + 1) * batch_size]
            downloaded = _download_batch(batch_jobs, dest_dir, workers)
            log.info("  %s batch %d/%d: %d/%d files",
                     target, bi + 1, n_batches, len(downloaded), len(batch_jobs))
            yield (target, bi, is_last, downloaded)


def delete_batch(pairs: list[tuple[str, str]]) -> None:
    """Delete the audio files of a processed batch to free disk space."""
    for path, _ in pairs:
        Path(path).unlink(missing_ok=True)
