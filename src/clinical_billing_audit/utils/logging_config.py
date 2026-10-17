"""Logging configuration for the clinical-billing-audit pipeline."""

import logging
import sys
from pathlib import Path


def setup_logging(
    level: str = "INFO",
    log_file: str | None = None,
    fmt: str | None = None,
) -> logging.Logger:
    """Configure and return the root logger for the pipeline.

    Parameters
    ----------
    level : str
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : str | None
        Optional path to a log file. Creates parent directories if needed.
    fmt : str | None
        Optional format string. Uses a sensible default if not provided.

    Returns
    -------
    logging.Logger
        Configured root logger.
    """
    if fmt is None:
        fmt = "%(asctime)s | %(name)-30s | %(levelname)-8s | %(message)s"

    root_logger = logging.getLogger("clinical_billing_audit")
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Avoid duplicate handlers on repeated calls
    root_logger.handlers.clear()

    # Console handler
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))
    console_handler.setFormatter(logging.Formatter(fmt))
    root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)  # Log everything to file
        file_handler.setFormatter(logging.Formatter(fmt))
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger under the pipeline namespace.

    Parameters
    ----------
    name : str
        Logger name (usually ``__name__``).

    Returns
    -------
    logging.Logger
        Child logger.
    """
    return logging.getLogger(f"clinical_billing_audit.{name}")
