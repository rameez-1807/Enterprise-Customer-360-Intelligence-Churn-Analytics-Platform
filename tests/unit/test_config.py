"""
Unit Test Suite for Global Configuration & Settings Loader
===========================================================
Verifies that YAML parsing and domain configuration manifests load
successfully with required schema structures.

Author: Principal Python Test Architect
Version: 1.0.0
"""

from pathlib import Path

import pytest

from src.core.config_loader import load_model_config, load_yaml
from src.core.exceptions import Customer360BaseError


def test_load_model_config() -> None:
    """Verifies that the model hyperparameter and metrics config can be loaded."""
    config = load_model_config()
    assert isinstance(config, dict)
    assert "model" in config
    assert "target" in config
    assert "kpi_benchmarks" in config


def test_load_yaml_nonexistent_file() -> None:
    """Verifies that attempting to load a nonexistent file raises Customer360BaseError."""
    fake_path = Path("nonexistent_file_path.yaml")
    with pytest.raises(Customer360BaseError):
        load_yaml(fake_path)
