"""Console + file logging for pipeline scripts.

The log directory is supplied by the caller (resolved from configuration). Log
records include timestamp, level, logger name, an optional run id and the
message. This module does not decide where logs live and never touches data.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]


def make_formatter(run_id: Optional[str] = None) -> logging.Formatter:
    run_tag = f"run={run_id} " if run_id else ""
    fmt = (
        "%(asctime)s | %(levelname)-8s | %(name)s | " + run_tag + "%(message)s"
    )
    return logging.Formatter(fmt, datefmt="%Y-%m-%dT%H:%M:%S")


def get_logger(
    name: str,
    log_dir: Optional[PathLike] = None,
    run_id: Optional[str] = None,
    level: int = logging.INFO,
    file_name: Optional[str] = None,
) -> logging.Logger:
    """Return a configured logger with a console handler and optional file handler.

    Parameters
    ----------
    name:
        Logger name (usually ``__name__`` of the calling script/module).
    log_dir:
        If provided, a file handler writing into this directory is attached.
        The directory is created if missing (logs are project products, not
        data). If ``None`` only console logging is configured.
    run_id:
        Optional run id embedded in each log line and used in the default log
        file name.
    file_name:
        Override for the log file name. Defaults to ``<run_id>.log`` (or
        ``run.log``).
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Avoid duplicating handlers if the logger already exists.
    if logger.handlers:
        return logger

    formatter = make_formatter(run_id=run_id)

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir is not None:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        fname = file_name or f"{run_id or 'run'}.log"
        file_handler = logging.FileHandler(log_path / fname, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger


__all__ = ["get_logger", "make_formatter"]
