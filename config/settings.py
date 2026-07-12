"""
Enterprise Customer 360 Intelligence & Churn Analytics Platform
================================================================
Application Settings & Path Configuration Module.

Centralizes all static configuration values including file paths,
directory schemas, environment profile settings, and system metadata.
This module is the single source of truth for structural configuration.

Author: Principal Python Engineer
Version: 1.0.0
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load environment variables from .env file
# ---------------------------------------------------------------------------
load_dotenv()

# ---------------------------------------------------------------------------
# Project Root Directory
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Environment Profile
# ---------------------------------------------------------------------------
APP_ENV: str = os.getenv("APP_ENV", "DEV")

# ---------------------------------------------------------------------------
# Data Directory Paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR: Path = PROJECT_ROOT / os.getenv("RAW_DATA_DIR", "data/raw")
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / os.getenv("PROCESSED_DATA_DIR", "data/processed")
OUTPUT_DATA_DIR: Path = PROJECT_ROOT / os.getenv("OUTPUT_DATA_DIR", "data/outputs")

# ---------------------------------------------------------------------------
# Source Dataset Configuration
# ---------------------------------------------------------------------------
RAW_DATASET_FILENAME: str = os.getenv("RAW_DATASET_FILENAME", "E Commerce Dataset.xlsx")
RAW_DATASET_PATH: Path = RAW_DATA_DIR / RAW_DATASET_FILENAME

# ---------------------------------------------------------------------------
# Model Artifact Paths
# ---------------------------------------------------------------------------
MODEL_ARTIFACTS_DIR: Path = PROJECT_ROOT / os.getenv("MODEL_ARTIFACTS_DIR", "data/outputs/models")
MODEL_VERSION: str = os.getenv("MODEL_VERSION", "1.0.0")

# ---------------------------------------------------------------------------
# Logging Configuration
# ---------------------------------------------------------------------------
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT: str = os.getenv("LOG_FORMAT", "json")
LOG_DIR: Path = PROJECT_ROOT / os.getenv("LOG_DIR", "logs")

# ---------------------------------------------------------------------------
# Dashboard Configuration
# ---------------------------------------------------------------------------
DASHBOARD_HOST: str = os.getenv("DASHBOARD_HOST", "localhost")
DASHBOARD_PORT: int = int(os.getenv("DASHBOARD_PORT", "8501"))
DASHBOARD_THEME: str = os.getenv("DASHBOARD_THEME", "dark")
DASHBOARD_DEBUG: bool = os.getenv("DASHBOARD_DEBUG", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Performance Tuning
# ---------------------------------------------------------------------------
ENABLE_DATA_CACHING: bool = os.getenv("ENABLE_DATA_CACHING", "true").lower() == "true"
MAX_MEMORY_MB: int = int(os.getenv("MAX_MEMORY_MB", "4096"))
