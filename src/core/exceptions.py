"""
Custom Enterprise Domain Exception Hierarchy
==============================================
Defines a structured exception hierarchy for the Customer 360 platform.
Each exception class maps to a specific failure boundary within the
data processing, model inference, and dashboard rendering pipelines.

Usage:
    from src.core.exceptions import IngestionError, SchemaValidationError

    raise IngestionError("Failed to read source file: file not found.")

Author: Principal Python Engineer
Version: 1.0.0
"""


class Customer360BaseError(Exception):
    """Base exception class for all Customer 360 platform errors."""

    def __init__(self, message: str = "An unexpected platform error occurred.") -> None:
        self.message = message
        super().__init__(self.message)


# ---------------------------------------------------------------------------
# Data Ingestion Layer Exceptions
# ---------------------------------------------------------------------------


class IngestionError(Customer360BaseError):
    """Raised when raw data file reading or extraction fails."""

    def __init__(self, message: str = "Data ingestion failed.") -> None:
        super().__init__(message)


class SchemaValidationError(Customer360BaseError):
    """Raised when incoming data does not conform to expected schema rules."""

    def __init__(self, message: str = "Schema validation failed.") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Data Cleaning & Feature Engineering Layer Exceptions
# ---------------------------------------------------------------------------


class ImputationError(Customer360BaseError):
    """Raised when statistical imputation of missing values fails."""

    def __init__(self, message: str = "Data imputation failed.") -> None:
        super().__init__(message)


class FeatureEngineeringError(Customer360BaseError):
    """Raised when feature construction or transformation encounters an error."""

    def __init__(self, message: str = "Feature engineering failed.") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Machine Learning Layer Exceptions
# ---------------------------------------------------------------------------


class ModelTrainingError(Customer360BaseError):
    """Raised when model training or cross-validation encounters a failure."""

    def __init__(self, message: str = "Model training failed.") -> None:
        super().__init__(message)


class ModelInferenceError(Customer360BaseError):
    """Raised when batch inference or model deserialization fails."""

    def __init__(self, message: str = "Model inference failed.") -> None:
        super().__init__(message)


class ExplainabilityError(Customer360BaseError):
    """Raised when SHAP explainability computation encounters a failure."""

    def __init__(self, message: str = "SHAP explainability computation failed.") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Analytics Layer Exceptions
# ---------------------------------------------------------------------------


class KPICalculationError(Customer360BaseError):
    """Raised when a business KPI formula computation fails."""

    def __init__(self, message: str = "KPI calculation failed.") -> None:
        super().__init__(message)


# ---------------------------------------------------------------------------
# Dashboard Presentation Layer Exceptions
# ---------------------------------------------------------------------------


class DashboardRenderError(Customer360BaseError):
    """Raised when a dashboard page or component fails to render."""

    def __init__(self, message: str = "Dashboard rendering failed.") -> None:
        super().__init__(message)
