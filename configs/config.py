"""
Central configuration for EchoModel.
All hyperparameters and paths are defined here.
"""
import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load KEY=VALUE pairs from a .env at the repo root into os.environ.

    Zero-dependency (no python-dotenv). Existing environment variables win, so
    an explicitly exported value always overrides the .env file.
    """
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT    = Path("./echomodel_project")
RAW_DIR         = PROJECT_ROOT / "01_raw_bbox_datasets"
YOLO_DATA_DIR   = PROJECT_ROOT / "02_yolo_dataset"
YOLO_RUNS_DIR   = PROJECT_ROOT / "03_yolo_runs"
PSEUDO_DIR      = PROJECT_ROOT / "04_pseudo_labels"
ECHODATA_DIR    = PROJECT_ROOT / "05_echomodel_dataset"
ECHOMODEL_DIR   = PROJECT_ROOT / "06_echomodel_runs"

# ---------------------------------------------------------------------------
# Zenodo datasets (bounding-box annotations)
# ---------------------------------------------------------------------------
BBOX_DATASETS = {
    "western_us": {
        "zenodo_id": "7050014",
        "name": "Fully-annotated soundscapes - Western United States",
        "n_hours": 33, "n_boxes": 20147, "n_species": 56,
    },
    "northeastern_us": {
        "zenodo_id": "7079380",
        "name": "Fully-annotated soundscapes - Northeastern United States",
        "n_hours": 285, "n_boxes": 50760, "n_species": 81,
    },
    "amazon_basin": {
        "zenodo_id": "7079124",
        "name": "Fully-annotated soundscapes - Southwestern Amazon Basin",
        "n_hours": 21, "n_boxes": 14798, "n_species": 132,
    },
    "hawaii": {
        "zenodo_id": "7078499",
        "name": "Fully-annotated soundscapes - Island of Hawai'i",
        "n_hours": None, "n_boxes": None, "n_species": None,
    },
    "coffee_farms": {
        "zenodo_id": "7525349",
        "name": "Fully-annotated soundscapes - Neotropical Coffee Farms (CO/CR)",
        "n_hours": 34, "n_boxes": 6952, "n_species": 89,
    },
    "sierra_nevada_south": {
        "zenodo_id": "7525805",
        "name": "Fully-annotated soundscapes - Southern Sierra Nevada",
        "n_hours": None, "n_boxes": None, "n_species": None,
    },
}

# Datasets to download on first run (use all keys to download everything)
DATASETS_TO_DOWNLOAD = list(BBOX_DATASETS.keys())

# ---------------------------------------------------------------------------
# Audio / spectrogram — YOLO stage
# ---------------------------------------------------------------------------
SR              = 32000
N_FFT           = 1024
HOP_LENGTH      = 320       # ~10 ms per frame at 32 kHz
N_MELS          = 128
FREQ_MAX        = SR // 2   # 16 000 Hz

TILE_DURATION   = 10.0      # seconds per YOLO tile
TILE_OVERLAP    = 2.0       # tile overlap
IMG_WIDTH       = 640
IMG_HEIGHT      = 320
MIN_BOX_FRACTION = 0.30

YOLO_CLASS_NAMES = ["bird_call"]

# ---------------------------------------------------------------------------
# Audio / spectrogram — EchoModel stage
# ---------------------------------------------------------------------------
ECHO_SR         = 32000
ECHO_WIN_MS     = 20
ECHO_HOP_MS     = 10
ECHO_N_FFT      = int(ECHO_SR * ECHO_WIN_MS / 1000)       # 640
ECHO_HOP_LENGTH = int(ECHO_SR * ECHO_HOP_MS / 1000)       # 320
ECHO_N_MELS     = 128
ECHO_FMIN       = 60.0
ECHO_FMAX       = 16000.0

WIN_DURATION    = 5.0
WIN_OVERLAP     = 2.5
WIN_STEP        = WIN_DURATION - WIN_OVERLAP
MIN_BOX_FRACTION_ECHO = 0.30

# ---------------------------------------------------------------------------
# EchoModel architecture
# ---------------------------------------------------------------------------
EMBED_DIM       = 192
NUM_HEADS       = 4
NUM_LAYERS      = 3

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
BATCH_SIZE      = 128       # per GPU (effective batch = BATCH_SIZE * num_gpus)
NUM_EPOCHS      = 60
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
NUM_WORKERS     = 16        # DataLoader workers per GPU
PIN_MEMORY      = True
PREFETCH_FACTOR = 4

# YOLO screening
YOLO_VARIANTS       = ["yolov8n.pt", "yolov8s.pt", "yolo11n.pt", "yolo11s.pt"]
EPOCHS_YOLO_SCREEN  = 20

# Loss weights
W_CLS = 1.0
W_T   = 0.5
W_F   = 0.5

# Perch-style training augmentation
USE_MIXUP        = True
# Number of components ~ BetaBin(n, a, b) + 1 (Perch preferred N in 2..5).
MIXUP_N          = 4
MIXUP_ALPHA      = 2.0
MIXUP_BETA       = 2.0
MIXUP_PROB       = 0.5    # fraction of batches to apply mixup to

# ---------------------------------------------------------------------------
# Pseudo-labelling
# ---------------------------------------------------------------------------
YOLO_CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Xeno-Canto download (weakly-labelled audio for pseudo-labelling)
# ---------------------------------------------------------------------------
# API v3 endpoint. Requires a free API key from https://xeno-canto.org/account
# exported as the XENO_CANTO_API_KEY environment variable.
XC_API_URL          = "https://xeno-canto.org/api/3/recordings"
XC_DOWNLOAD_DIR     = PROJECT_ROOT / "04_pseudo_labels" / "xc_audio"
# Process recordings in batches: download a batch, pseudo-label it, delete the
# audio, then move on — keeps disk usage bounded over the whole dataset.
XC_BATCH_SIZE       = 50
# Like Perch v2, do not filter by recording rating (A–E). Set e.g. "A" or "B"
# to restrict to higher-quality recordings. None = all ratings.
XC_MIN_QUALITY      = None
# Safety cap on recordings per species (None = all available, Perch-style).
XC_MAX_PER_SPECIES  = None
XC_DOWNLOAD_WORKERS = 8

# Pilot mode: restrict the run to a handful of species / few recordings each so
# the whole pipeline (download -> spectrogram -> YOLO boxes -> materialise ->
# train) can be validated end-to-end before scaling to the full archive.
XC_PILOT            = True
XC_PILOT_SPECIES    = 10     # how many species to sample in pilot mode
XC_PILOT_PER_SPECIES = 5     # recordings per species in pilot mode

# "All species" mode (full Perch-scale run). When True the species list is the
# entire Xeno-Canto avian catalogue rather than just the Zenodo ground-truth
# species. Ignored while XC_PILOT is True.
XC_ALL_SPECIES      = False

# Windows materialised per recording (Perch selects 5 s windows from each file).
XC_WINDOWS_PER_REC  = 1
# Window-selection strategy: "random" or "energy_peak" (Perch uses both).
XC_WINDOW_SELECT    = "energy_peak"

# Directory holding the materialised spectrogram features (.npy). The raw audio
# is deleted after each batch; only these light tensors persist on disk.
XC_FEATURES_DIR     = ECHODATA_DIR / "xc_features"
# File recording which XC batches are already materialised (for resume).
XC_PROGRESS_FILE    = ECHODATA_DIR / "xc_progress.json"

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
ZENODO_API   = "https://zenodo.org/api/records/{record_id}"
DOWNLOAD_CHUNK_SIZE = 1 << 20   # 1 MiB
