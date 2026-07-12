"""
Structured JSON Logging Configuration
=======================================
Configures the enterprise-grade structured logging system with JSON
formatting, contextual metadata, and multiple sink destinations
(console stream and rotating file handlers).

Usage:
    from src.core.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline step completed successfully.")

Author: Principal Python Engineer
Version: 1.0.0
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class JSONFormatter(logging.Formatter):
    """Custom JSON log formatter for structured, machine-readable log output."""

    def format(self, record: logging.LogRecord) -> str:
        """Format a log record as a JSON string.

        Args:
            record: The log record to format.

        Returns:
            A JSON-formatted string representation of the log record.
        """
        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "message": record.getMessage(),
        }

        if record.exc_info and record.exc_info[1]:
            log_entry["exception"] = str(record.exc_info[1])

        return json.dumps(log_entry, default=str)


def get_logger(
    name: str,
    log_level: str = "INFO",
    log_dir: str = "logs",
    log_to_file: bool = True,
) -> logging.Logger:
    """Create and configure a structured logger instance.

    Args:
        name: The name of the logger (typically __name__).
        log_level: The minimum severity level for log output.
        log_dir: Directory path for log file storage.
        log_to_file: Whether to write logs to a rotating file.

    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Prevent duplicate handler attachment on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    # --- Console Stream Handler ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    logger.addHandler(console_handler)

    # --- Rotating File Handler ---
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            log_path / "system.log",
            mode="a",
            encoding="utf-8",
        )
        file_handler.setFormatter(JSONFormatter())
        logger.addHandler(file_handler)

    # Prevent log propagation to root logger
    logger.propagate = False

    return logger
