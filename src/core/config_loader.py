"""
YAML & Environment Configuration Loader
==========================================
Provides utility functions to load and parse YAML configuration files
and environment variables into structured Python dictionaries.

Usage:
    from src.core.config_loader import load_model_config

    config = load_model_config()
    n_estimators = config["model"]["xgboost"]["n_estimators"]

Author: Principal Python Engineer
Version: 1.0.0
"""

from pathlib import Path
from typing import Any

import yaml

from src.core.exceptions import Customer360BaseError
from src.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Default Configuration File Paths
# ---------------------------------------------------------------------------
_CONFIG_DIR: Path = Path(__file__).resolve().parent.parent.parent / "config"
_MODEL_CONFIG_PATH: Path = _CONFIG_DIR / "model_config.yaml"


def load_yaml(filepath: Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file.

    Args:
        filepath: Absolute or relative path to the YAML file.

    Returns:
        A dictionary containing the parsed YAML content.

    Raises:
        Customer360BaseError: If the file is not found or cannot be parsed.
    """
    if not filepath.exists():
        error_msg = f"Configuration file not found: {filepath}"
        logger.error(error_msg)
        raise Customer360BaseError(error_msg)

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            config: dict[str, Any] = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from: {filepath.name}")
            return config
    except yaml.YAMLError as e:
        error_msg = f"Failed to parse YAML configuration: {e}"
        logger.error(error_msg)
        raise Customer360BaseError(error_msg) from e


def load_model_config() -> dict[str, Any]:
    """Load the machine learning and domain configuration manifest.

    Returns:
        A dictionary containing model hyperparameters, feature engineering
        rules, RFM scoring parameters, and KPI benchmark targets.
    """
    return load_yaml(_MODEL_CONFIG_PATH)
