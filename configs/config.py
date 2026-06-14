"""
Central configuration for EchoModel.
All hyperparameters and paths are defined here.
"""
from pathlib import Path

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

# ---------------------------------------------------------------------------
# Pseudo-labelling
# ---------------------------------------------------------------------------
YOLO_CONF_THRESHOLD = 0.25

# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------
ZENODO_API   = "https://zenodo.org/api/records/{record_id}"
DOWNLOAD_CHUNK_SIZE = 1 << 20   # 1 MiB
