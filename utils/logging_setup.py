"""
Configures Python logging: console + rotating file, rank-aware for DDP.
"""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def setup_logging(log_dir: Path = Path("logs"), rank: int = 0) -> None:
    log_dir.mkdir(parents=True, exist_ok=True)
    level = logging.INFO

    fmt = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(level)

    if rank == 0:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        fh = RotatingFileHandler(
            log_dir / f"echomodel_rank{rank}.log",
            maxBytes=50 * 1024 * 1024,
            backupCount=5,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    else:
        # non-rank-0 processes write only to their own file
        fh = RotatingFileHandler(
            log_dir / f"echomodel_rank{rank}.log",
            maxBytes=10 * 1024 * 1024,
            backupCount=2,
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
