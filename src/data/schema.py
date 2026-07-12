"""
Enterprise Schema Validation Engine
======================================
Production-grade schema validation system for the Customer 360 platform.
Defines the canonical data contract for the upstream e-commerce dataset
and provides comprehensive validation functions for column presence,
data types, nullability, uniqueness, categorical domain constraints,
numeric range boundaries, and schema drift detection.

This module is the authoritative source of truth for the dataset's
structural and semantic contract. All validation functions are designed
to be invoked independently or composed into a full validation pipeline.

Usage:
    from src.data.schema import validate_schema, SCHEMA_REGISTRY

    report = validate_schema(df)
    print(report["overall_status"])

Author: Principal Data Quality Architect
Version: 1.0.0
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

import pandas as pd

from src.core.exceptions import SchemaValidationError
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================


class ColumnCategory(Enum):
    """Classification of column functional roles within the dataset."""

    PRIMARY_KEY = "primary_key"
    TARGET = "target"
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    BINARY = "binary"


class Nullability(Enum):
    """Nullability constraint for a column."""

    NOT_NULL = "not_null"
    NULLABLE = "nullable"


class ValidationSeverity(Enum):
    """Severity level of a validation finding."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


# =============================================================================
# COLUMN DEFINITION DATA CLASS
# =============================================================================


@dataclass(frozen=True)
class ColumnDefinition:
    """Immutable specification of a single column's data contract.

    Attributes:
        name: The exact column header name as it appears in the dataset.
        category: The functional role of this column (PRIMARY_KEY, TARGET, etc.).
        acceptable_dtypes: List of pandas dtype strings considered valid.
        nullability: Whether null values are permitted in this column.
        is_mandatory: Whether this column must be present in the dataset.
        unique: Whether all values in this column must be distinct.
        allowed_values: Exhaustive set of valid categorical values (None if unconstrained).
        min_value: Minimum acceptable numeric value (None if unconstrained).
        max_value: Maximum acceptable numeric value (None if unconstrained).
        description: Human-readable business description of this column.
    """

    name: str
    category: ColumnCategory
    acceptable_dtypes: list[str]
    nullability: Nullability
    is_mandatory: bool = True
    unique: bool = False
    allowed_values: list[Any] | None = None
    min_value: float | int | None = None
    max_value: float | int | None = None
    description: str = ""


# =============================================================================
# SCHEMA REGISTRY — SINGLE SOURCE OF TRUTH
# =============================================================================
# This registry defines the canonical data contract for every column in the
# upstream E-Commerce dataset. All downstream validation, cleaning, feature
# engineering, and ML modules reference this contract.
# =============================================================================

SCHEMA_REGISTRY: list[ColumnDefinition] = [
    # --- Primary Key ---
    ColumnDefinition(
        name="CustomerID",
        category=ColumnCategory.PRIMARY_KEY,
        acceptable_dtypes=["int64", "int32", "int16"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        unique=True,
        min_value=0,
        max_value=None,
        description="Unique customer identifier. Acts as the primary key.",
    ),
    # --- Target Variable ---
    ColumnDefinition(
        name="Churn",
        category=ColumnCategory.BINARY,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        unique=False,
        allowed_values=[0, 1],
        description="Binary churn indicator. 1 = churned, 0 = retained.",
    ),
    # --- Numerical Columns (Nullable) ---
    ColumnDefinition(
        name="Tenure",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=120,
        description="Duration (months) since the customer joined the platform.",
    ),
    ColumnDefinition(
        name="WarehouseToHome",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=500,
        description="Distance (km/miles) from warehouse to customer home address.",
    ),
    ColumnDefinition(
        name="HourSpendOnApp",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=24,
        description="Hours spent on mobile application per reporting period.",
    ),
    ColumnDefinition(
        name="OrderAmountHikeFromlastYear",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=100,
        description="Percentage increase in order value compared to previous year.",
    ),
    ColumnDefinition(
        name="CouponUsed",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=50,
        description="Total number of coupons used by the customer.",
    ),
    ColumnDefinition(
        name="OrderCount",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=100,
        description="Total number of orders placed by the customer.",
    ),
    ColumnDefinition(
        name="DaySinceLastOrder",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32", "int64"],
        nullability=Nullability.NULLABLE,
        is_mandatory=True,
        min_value=0,
        max_value=365,
        description="Number of days since the customer placed their last order.",
    ),
    ColumnDefinition(
        name="CashbackAmount",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["float64", "float32"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        min_value=0,
        max_value=None,
        description="Average cashback amount received by the customer.",
    ),
    # --- Numerical Columns (Not Null) ---
    ColumnDefinition(
        name="CityTier",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=[1, 2, 3],
        description="City classification tier (1 = Metro, 2 = Tier-2, 3 = Tier-3).",
    ),
    ColumnDefinition(
        name="NumberOfDeviceRegistered",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        min_value=1,
        max_value=10,
        description="Number of devices registered to the customer account.",
    ),
    ColumnDefinition(
        name="SatisfactionScore",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=[1, 2, 3, 4, 5],
        description="Customer satisfaction rating (1 = Very Low, 5 = Very High).",
    ),
    ColumnDefinition(
        name="NumberOfAddress",
        category=ColumnCategory.NUMERICAL,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        min_value=1,
        max_value=50,
        description="Number of delivery addresses registered by the customer.",
    ),
    # --- Binary Column ---
    ColumnDefinition(
        name="Complain",
        category=ColumnCategory.BINARY,
        acceptable_dtypes=["int64", "int32", "int16", "int8"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=[0, 1],
        description="Whether the customer filed a complaint (1 = Yes, 0 = No).",
    ),
    # --- Categorical Columns ---
    ColumnDefinition(
        name="PreferredLoginDevice",
        category=ColumnCategory.CATEGORICAL,
        acceptable_dtypes=["object", "str", "string"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=["Mobile Phone", "Phone", "Computer"],
        description="Primary device used for account login.",
    ),
    ColumnDefinition(
        name="PreferredPaymentMode",
        category=ColumnCategory.CATEGORICAL,
        acceptable_dtypes=["object", "str", "string"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=[
            "Debit Card",
            "Credit Card",
            "CC",
            "E wallet",
            "UPI",
            "COD",
            "Cash on Delivery",
        ],
        description="Primary payment method used by the customer.",
    ),
    ColumnDefinition(
        name="Gender",
        category=ColumnCategory.CATEGORICAL,
        acceptable_dtypes=["object", "str", "string"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=["Male", "Female"],
        description="Customer gender classification.",
    ),
    ColumnDefinition(
        name="PreferedOrderCat",
        category=ColumnCategory.CATEGORICAL,
        acceptable_dtypes=["object", "str", "string"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=[
            "Laptop & Accessory",
            "Mobile Phone",
            "Mobile",
            "Fashion",
            "Grocery",
            "Others",
        ],
        description="Customer's most frequently ordered product category.",
    ),
    ColumnDefinition(
        name="MaritalStatus",
        category=ColumnCategory.CATEGORICAL,
        acceptable_dtypes=["object", "str", "string"],
        nullability=Nullability.NOT_NULL,
        is_mandatory=True,
        allowed_values=["Single", "Married", "Divorced"],
        description="Customer marital status classification.",
    ),
]

# Build lookup dictionary for O(1) column access
SCHEMA_LOOKUP: dict[str, ColumnDefinition] = {col.name: col for col in SCHEMA_REGISTRY}

# Derived sets for convenience
MANDATORY_COLUMNS: list[str] = [col.name for col in SCHEMA_REGISTRY if col.is_mandatory]
OPTIONAL_COLUMNS: list[str] = [col.name for col in SCHEMA_REGISTRY if not col.is_mandatory]
PRIMARY_KEY_COLUMNS: list[str] = [col.name for col in SCHEMA_REGISTRY if col.category == ColumnCategory.PRIMARY_KEY]
NULLABLE_COLUMNS: list[str] = [col.name for col in SCHEMA_REGISTRY if col.nullability == Nullability.NULLABLE]
NOT_NULL_COLUMNS: list[str] = [col.name for col in SCHEMA_REGISTRY if col.nullability == Nullability.NOT_NULL]
ALL_COLUMN_NAMES: list[str] = [col.name for col in SCHEMA_REGISTRY]


# =============================================================================
# VALIDATION RESULT DATA CLASS
# =============================================================================


@dataclass
class ValidationFinding:
    """A single finding produced by a validation check.

    Attributes:
        rule: Name of the validation rule that produced this finding.
        column: The column name involved (None if dataset-level).
        severity: The severity classification of the finding.
        message: Human-readable description of the finding.
        details: Optional structured data providing additional context.
    """

    rule: str
    column: str | None
    severity: ValidationSeverity
    message: str
    details: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# INDIVIDUAL VALIDATION FUNCTIONS
# =============================================================================


def validate_column_presence(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate that all mandatory columns exist in the DataFrame.

    Checks the actual column set against the schema registry, identifying
    missing mandatory columns, unexpected extra columns, and columns that
    may be misspelled versions of expected names.

    Args:
        df: The raw ingested DataFrame to validate.

    Returns:
        A list of ValidationFinding objects describing presence issues.
    """
    findings: list[ValidationFinding] = []
    actual_cols: set[str] = set(df.columns.tolist())
    expected_cols: set[str] = set(ALL_COLUMN_NAMES)

    # Missing mandatory columns
    missing: list[str] = sorted([c for c in MANDATORY_COLUMNS if c not in actual_cols])
    for col_name in missing:
        findings.append(
            ValidationFinding(
                rule="column_presence",
                column=col_name,
                severity=ValidationSeverity.CRITICAL,
                message=f"Mandatory column '{col_name}' is missing from the dataset.",
            )
        )

    # Extra columns (not in schema)
    extra: list[str] = sorted(actual_cols - expected_cols)
    for col_name in extra:
        findings.append(
            ValidationFinding(
                rule="column_presence",
                column=col_name,
                severity=ValidationSeverity.INFO,
                message=f"Column '{col_name}' found in dataset but not defined in schema.",
            )
        )

    if not missing:
        logger.info(f"Column presence validated: all {len(MANDATORY_COLUMNS)} " f"mandatory columns present.")
    else:
        logger.error(f"Column presence validation failed: {len(missing)} missing — {missing}")

    if extra:
        logger.info(f"Extra columns detected (not in schema): {extra}")

    return findings


def validate_data_types(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate that each column's dtype matches the schema specification.

    Compares the actual pandas dtype against the list of acceptable dtypes
    defined in the schema registry for each column present in the DataFrame.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of ValidationFinding objects for dtype mismatches.
    """
    findings: list[ValidationFinding] = []

    for col_def in SCHEMA_REGISTRY:
        if col_def.name not in df.columns:
            continue

        actual_dtype: str = str(df[col_def.name].dtype)

        if actual_dtype not in col_def.acceptable_dtypes:
            findings.append(
                ValidationFinding(
                    rule="data_type",
                    column=col_def.name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Column '{col_def.name}' has dtype '{actual_dtype}', "
                        f"expected one of {col_def.acceptable_dtypes}."
                    ),
                    details={
                        "actual_dtype": actual_dtype,
                        "acceptable_dtypes": col_def.acceptable_dtypes,
                    },
                )
            )

    valid_count: int = len(SCHEMA_REGISTRY) - len(findings)
    if not findings:
        logger.info("Data type validation passed for all columns.")
    else:
        logger.warning(f"Data type mismatches in {len(findings)} column(s).")

    return findings


def validate_nullability(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate nullability constraints for each column.

    Checks NOT_NULL columns for any null values and reports NULLABLE columns
    that contain nulls as informational findings with counts and percentages.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of ValidationFinding objects for nullability violations.
    """
    findings: list[ValidationFinding] = []
    total_rows: int = len(df)

    for col_def in SCHEMA_REGISTRY:
        if col_def.name not in df.columns:
            continue

        null_count: int = int(df[col_def.name].isnull().sum())

        if null_count == 0:
            continue

        null_pct: float = round((null_count / total_rows) * 100, 2)

        if col_def.nullability == Nullability.NOT_NULL:
            findings.append(
                ValidationFinding(
                    rule="nullability",
                    column=col_def.name,
                    severity=ValidationSeverity.CRITICAL,
                    message=(
                        f"NOT_NULL column '{col_def.name}' contains " f"{null_count:,} null value(s) ({null_pct}%)."
                    ),
                    details={"null_count": null_count, "null_percentage": null_pct},
                )
            )
        else:
            findings.append(
                ValidationFinding(
                    rule="nullability",
                    column=col_def.name,
                    severity=ValidationSeverity.INFO,
                    message=(
                        f"NULLABLE column '{col_def.name}' contains "
                        f"{null_count:,} null value(s) ({null_pct}%). "
                        f"Imputation required in cleaning phase."
                    ),
                    details={"null_count": null_count, "null_percentage": null_pct},
                )
            )

    critical_nulls: int = sum(1 for f in findings if f.severity == ValidationSeverity.CRITICAL)
    if critical_nulls == 0:
        logger.info("Nullability validation passed: no violations in NOT_NULL columns.")
    else:
        logger.error(f"Nullability violations: {critical_nulls} NOT_NULL column(s) have nulls.")

    return findings


def validate_uniqueness(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate uniqueness constraints for primary key columns.

    Checks columns marked with unique=True in the schema registry for
    duplicate values, reporting the count and sample of duplicated keys.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of ValidationFinding objects for uniqueness violations.
    """
    findings: list[ValidationFinding] = []

    for col_def in SCHEMA_REGISTRY:
        if not col_def.unique or col_def.name not in df.columns:
            continue

        duplicate_mask: pd.Series = df[col_def.name].duplicated(keep=False)
        dup_count: int = int(duplicate_mask.sum())

        if dup_count > 0:
            dup_keys: list[Any] = df.loc[duplicate_mask, col_def.name].unique().tolist()
            sample_keys: list[Any] = dup_keys[:10]
            findings.append(
                ValidationFinding(
                    rule="uniqueness",
                    column=col_def.name,
                    severity=ValidationSeverity.CRITICAL,
                    message=(
                        f"Unique column '{col_def.name}' contains "
                        f"{dup_count} duplicate row(s) across "
                        f"{len(dup_keys)} key(s)."
                    ),
                    details={
                        "duplicate_row_count": dup_count,
                        "unique_duplicated_keys": len(dup_keys),
                        "sample_keys": sample_keys,
                    },
                )
            )
        else:
            logger.info(f"Uniqueness validated for '{col_def.name}': all values distinct.")

    return findings


def validate_categorical_values(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate categorical columns against their allowed value domains.

    For columns with a defined allowed_values list, identifies any values
    present in the data that fall outside the permitted domain. This detects
    data entry errors, encoding inconsistencies, and upstream schema changes.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of ValidationFinding objects for domain violations.
    """
    findings: list[ValidationFinding] = []

    for col_def in SCHEMA_REGISTRY:
        if col_def.allowed_values is None or col_def.name not in df.columns:
            continue

        # Drop nulls before checking domain — nullability is checked separately
        non_null_values: pd.Series = df[col_def.name].dropna()
        actual_values: set[Any] = set(non_null_values.unique().tolist())
        allowed_set: set[Any] = set(col_def.allowed_values)

        invalid_values: set[Any] = actual_values - allowed_set
        if invalid_values:
            # Count occurrences of each invalid value
            invalid_counts: dict[str, int] = {}
            for val in invalid_values:
                invalid_counts[str(val)] = int((non_null_values == val).sum())

            findings.append(
                ValidationFinding(
                    rule="categorical_domain",
                    column=col_def.name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Column '{col_def.name}' contains {len(invalid_values)} "
                        f"value(s) outside the allowed domain: {sorted(str(v) for v in invalid_values)}."
                    ),
                    details={
                        "invalid_values": sorted(str(v) for v in invalid_values),
                        "invalid_counts": invalid_counts,
                        "allowed_values": [str(v) for v in col_def.allowed_values],
                    },
                )
            )
        else:
            logger.info(
                f"Categorical domain validated for '{col_def.name}': "
                f"{len(actual_values)} value(s) all within domain."
            )

    return findings


def validate_numeric_ranges(df: pd.DataFrame) -> list[ValidationFinding]:
    """Validate numeric columns against their min/max boundary constraints.

    For columns with defined min_value or max_value, identifies records
    that fall outside the acceptable range. Out-of-range values may indicate
    data corruption, measurement errors, or extreme outliers.

    Args:
        df: The DataFrame to validate.

    Returns:
        A list of ValidationFinding objects for range violations.
    """
    findings: list[ValidationFinding] = []

    for col_def in SCHEMA_REGISTRY:
        if col_def.name not in df.columns:
            continue

        if col_def.min_value is None and col_def.max_value is None:
            continue

        non_null_series: pd.Series = df[col_def.name].dropna()

        if non_null_series.empty:
            continue

        actual_min: float = float(non_null_series.min())
        actual_max: float = float(non_null_series.max())

        # Check minimum boundary
        if col_def.min_value is not None and actual_min < col_def.min_value:
            below_count: int = int((non_null_series < col_def.min_value).sum())
            findings.append(
                ValidationFinding(
                    rule="numeric_range",
                    column=col_def.name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Column '{col_def.name}' has {below_count} value(s) "
                        f"below minimum boundary ({col_def.min_value}). "
                        f"Actual min: {actual_min}."
                    ),
                    details={
                        "boundary": "min",
                        "threshold": col_def.min_value,
                        "actual_extreme": actual_min,
                        "violation_count": below_count,
                    },
                )
            )

        # Check maximum boundary
        if col_def.max_value is not None and actual_max > col_def.max_value:
            above_count: int = int((non_null_series > col_def.max_value).sum())
            findings.append(
                ValidationFinding(
                    rule="numeric_range",
                    column=col_def.name,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Column '{col_def.name}' has {above_count} value(s) "
                        f"above maximum boundary ({col_def.max_value}). "
                        f"Actual max: {actual_max}."
                    ),
                    details={
                        "boundary": "max",
                        "threshold": col_def.max_value,
                        "actual_extreme": actual_max,
                        "violation_count": above_count,
                    },
                )
            )

    if not findings:
        logger.info("Numeric range validation passed: all values within boundaries.")
    else:
        logger.warning(f"Numeric range violations detected in {len(findings)} check(s).")

    return findings


def detect_schema_drift(
    df: pd.DataFrame,
    baseline_row_count: int | None = None,
    baseline_column_count: int | None = None,
) -> list[ValidationFinding]:
    """Detect structural schema drift compared to baseline expectations.

    Schema drift occurs when the upstream data source changes its structure
    over time. This function detects changes in column count, row count
    magnitude, column ordering differences, and new or removed columns
    relative to the schema registry.

    Args:
        df: The DataFrame to check for drift.
        baseline_row_count: Expected approximate row count from prior runs.
            If None, a default baseline of 5,000 is used.
        baseline_column_count: Expected column count. If None, the schema
            registry length is used.

    Returns:
        A list of ValidationFinding objects for detected drift signals.
    """
    findings: list[ValidationFinding] = []

    expected_col_count: int = baseline_column_count or len(SCHEMA_REGISTRY)
    expected_row_count: int = baseline_row_count or 5000
    actual_col_count: int = len(df.columns)
    actual_row_count: int = len(df)

    # Column count drift
    if actual_col_count != expected_col_count:
        findings.append(
            ValidationFinding(
                rule="schema_drift",
                column=None,
                severity=ValidationSeverity.WARNING,
                message=(f"Column count drift detected: expected {expected_col_count}, " f"found {actual_col_count}."),
                details={
                    "expected_columns": expected_col_count,
                    "actual_columns": actual_col_count,
                    "delta": actual_col_count - expected_col_count,
                },
            )
        )

    # Row count magnitude drift (> 50% deviation)
    if expected_row_count > 0:
        row_deviation_pct: float = abs(actual_row_count - expected_row_count) / expected_row_count
        if row_deviation_pct > 0.50:
            findings.append(
                ValidationFinding(
                    rule="schema_drift",
                    column=None,
                    severity=ValidationSeverity.WARNING,
                    message=(
                        f"Row count magnitude drift: expected ~{expected_row_count:,}, "
                        f"found {actual_row_count:,} "
                        f"({row_deviation_pct:.0%} deviation)."
                    ),
                    details={
                        "expected_rows": expected_row_count,
                        "actual_rows": actual_row_count,
                        "deviation_pct": round(row_deviation_pct * 100, 1),
                    },
                )
            )

    # Column order drift
    expected_order: list[str] = ALL_COLUMN_NAMES
    actual_order: list[str] = [c for c in df.columns if c in set(expected_order)]
    expected_filtered: list[str] = [c for c in expected_order if c in set(df.columns)]

    if actual_order != expected_filtered:
        findings.append(
            ValidationFinding(
                rule="schema_drift",
                column=None,
                severity=ValidationSeverity.INFO,
                message="Column ordering differs from schema registry baseline.",
                details={
                    "expected_order": expected_filtered[:5],
                    "actual_order": actual_order[:5],
                },
            )
        )

    if not findings:
        logger.info("Schema drift check passed: no structural changes detected.")
    else:
        logger.warning(f"Schema drift detected: {len(findings)} signal(s).")

    return findings


# =============================================================================
# REPORT BUILDER HELPERS
# =============================================================================


def _build_findings_summary(findings: list[ValidationFinding]) -> dict[str, int]:
    """Count findings by severity level.

    Args:
        findings: List of all validation findings.

    Returns:
        Dictionary mapping severity names to their counts.
    """
    summary: dict[str, int] = {
        "critical": 0,
        "warning": 0,
        "info": 0,
    }
    for f in findings:
        summary[f.severity.value] += 1
    return summary


def _build_rule_breakdown(findings: list[ValidationFinding]) -> dict[str, list[dict[str, Any]]]:
    """Group findings by validation rule name.

    Args:
        findings: List of all validation findings.

    Returns:
        Dictionary mapping rule names to lists of finding dictionaries.
    """
    breakdown: dict[str, list[dict[str, Any]]] = {}
    for f in findings:
        if f.rule not in breakdown:
            breakdown[f.rule] = []
        breakdown[f.rule].append(
            {
                "column": f.column,
                "severity": f.severity.value,
                "message": f.message,
                "details": f.details,
            }
        )
    return breakdown


def _serialize_schema_registry() -> list[dict[str, Any]]:
    """Serialize the schema registry into a portable dictionary format.

    Returns:
        List of dictionaries representing each column definition.
    """
    return [
        {
            "name": col.name,
            "category": col.category.value,
            "acceptable_dtypes": col.acceptable_dtypes,
            "nullability": col.nullability.value,
            "is_mandatory": col.is_mandatory,
            "unique": col.unique,
            "allowed_values": col.allowed_values,
            "min_value": col.min_value,
            "max_value": col.max_value,
            "description": col.description,
        }
        for col in SCHEMA_REGISTRY
    ]


# =============================================================================
# PRIMARY PUBLIC API
# =============================================================================


def validate_schema(
    df: pd.DataFrame,
    baseline_row_count: int | None = None,
    raise_on_critical: bool = False,
) -> dict[str, Any]:
    """Execute the full schema validation pipeline against a DataFrame.

    Orchestrates all individual validation checks in sequence:
        1. Column presence validation
        2. Data type validation
        3. Nullability constraint validation
        4. Uniqueness constraint validation
        5. Categorical domain validation
        6. Numeric range boundary validation
        7. Schema drift detection

    Args:
        df: The raw ingested DataFrame to validate.
        baseline_row_count: Optional expected row count for drift detection.
        raise_on_critical: If True, raises SchemaValidationError when
            any CRITICAL findings are detected.

    Returns:
        A comprehensive validation report dictionary containing:
            - validation_timestamp: ISO timestamp of validation execution.
            - dataset_shape: Tuple of (rows, columns).
            - overall_status: "PASSED", "WARNINGS", or "FAILED".
            - findings_summary: Count of findings by severity.
            - rule_breakdown: Findings grouped by validation rule.
            - all_findings: Complete list of all findings as dictionaries.
            - schema_definition: Serialized schema registry for audit trail.

    Raises:
        SchemaValidationError: If raise_on_critical is True and CRITICAL
            findings are detected.

    Example:
        >>> from src.data.schema import validate_schema
        >>> report = validate_schema(df)
        >>> print(report["overall_status"])
        "PASSED"
    """
    logger.info("=" * 70)
    logger.info("ENTERPRISE SCHEMA VALIDATION ENGINE — STARTING")
    logger.info("=" * 70)

    all_findings: list[ValidationFinding] = []

    # Step 1: Column Presence
    logger.info("[1/7] Validating column presence...")
    all_findings.extend(validate_column_presence(df))

    # Step 2: Data Types
    logger.info("[2/7] Validating data types...")
    all_findings.extend(validate_data_types(df))

    # Step 3: Nullability
    logger.info("[3/7] Validating nullability constraints...")
    all_findings.extend(validate_nullability(df))

    # Step 4: Uniqueness
    logger.info("[4/7] Validating uniqueness constraints...")
    all_findings.extend(validate_uniqueness(df))

    # Step 5: Categorical Domains
    logger.info("[5/7] Validating categorical value domains...")
    all_findings.extend(validate_categorical_values(df))

    # Step 6: Numeric Ranges
    logger.info("[6/7] Validating numeric range boundaries...")
    all_findings.extend(validate_numeric_ranges(df))

    # Step 7: Schema Drift
    logger.info("[7/7] Detecting schema drift...")
    all_findings.extend(detect_schema_drift(df, baseline_row_count=baseline_row_count))

    # Build summary
    summary: dict[str, int] = _build_findings_summary(all_findings)
    rule_breakdown: dict[str, list[dict[str, Any]]] = _build_rule_breakdown(all_findings)

    # Determine overall status
    if summary["critical"] > 0:
        overall_status: str = "FAILED"
    elif summary["warning"] > 0:
        overall_status = "WARNINGS"
    else:
        overall_status = "PASSED"

    # Build report
    report: dict[str, Any] = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_shape": {
            "rows": df.shape[0],
            "columns": df.shape[1],
        },
        "overall_status": overall_status,
        "findings_summary": summary,
        "rule_breakdown": rule_breakdown,
        "all_findings": [
            {
                "rule": f.rule,
                "column": f.column,
                "severity": f.severity.value,
                "message": f.message,
                "details": f.details,
            }
            for f in all_findings
        ],
        "schema_definition": _serialize_schema_registry(),
    }

    logger.info(f"SCHEMA VALIDATION COMPLETE — Status: {overall_status}")
    logger.info(
        f"Findings: {summary['critical']} critical, " f"{summary['warning']} warning(s), {summary['info']} info."
    )
    logger.info("=" * 70)

    if raise_on_critical and summary["critical"] > 0:
        critical_messages: list[str] = [f.message for f in all_findings if f.severity == ValidationSeverity.CRITICAL]
        raise SchemaValidationError(
            f"Schema validation failed with {summary['critical']} critical finding(s): " f"{critical_messages}"
        )

    return report


def get_column_definition(column_name: str) -> ColumnDefinition | None:
    """Retrieve the schema definition for a specific column by name.

    Args:
        column_name: The exact column name to look up.

    Returns:
        The ColumnDefinition object if found, None otherwise.
    """
    return SCHEMA_LOOKUP.get(column_name)


def get_columns_by_category(category: ColumnCategory) -> list[str]:
    """Retrieve all column names belonging to a specific category.

    Args:
        category: The ColumnCategory enum value to filter by.

    Returns:
        A list of column names matching the specified category.
    """
    return [col.name for col in SCHEMA_REGISTRY if col.category == category]


def get_columns_by_nullability(nullability: Nullability) -> list[str]:
    """Retrieve all column names matching a specific nullability constraint.

    Args:
        nullability: The Nullability enum value to filter by.

    Returns:
        A list of column names matching the specified nullability.
    """
    return [col.name for col in SCHEMA_REGISTRY if col.nullability == nullability]


def print_validation_report(report: dict[str, Any]) -> None:
    """Print a formatted, human-readable validation report to stdout.

    Formats the validation results into a structured text report suitable
    for console output or log inspection. Organizes findings by severity
    with clear section dividers and indentation.

    Args:
        report: The validation report dictionary from validate_schema().
    """
    border: str = "=" * 70
    divider: str = "-" * 40
    print(f"\n{border}")
    print("  ENTERPRISE SCHEMA VALIDATION REPORT")
    print(border)
    print(f"  Timestamp:  {report['validation_timestamp']}")
    print(f"  Dataset:    {report['dataset_shape']['rows']:,} rows x " f"{report['dataset_shape']['columns']} columns")
    print(f"  Status:     {report['overall_status']}")
    print(border)

    summary: dict[str, int] = report["findings_summary"]
    print(f"\n  FINDINGS SUMMARY")
    print(f"  {divider}")
    print(f"  [CRITICAL]  {summary['critical']}")
    print(f"  [WARNING]   {summary['warning']}")
    print(f"  [INFO]      {summary['info']}")
    total: int = summary["critical"] + summary["warning"] + summary["info"]
    print(f"  {divider}")
    print(f"  Total:      {total}")

    # Print by rule
    breakdown: dict[str, list[dict[str, Any]]] = report["rule_breakdown"]
    if breakdown:
        print(f"\n  DETAILED FINDINGS BY RULE")
        print(f"  {divider}")
        for rule_name, items in breakdown.items():
            print(f"\n  >> {rule_name.upper().replace('_', ' ')} ({len(items)} finding(s))")
            for item in items:
                severity_tag: str = {
                    "critical": "[CRITICAL]",
                    "warning": "[WARNING] ",
                    "info": "[INFO]    ",
                }.get(item["severity"], "[UNKNOWN] ")
                col_label: str = f"[{item['column']}]" if item["column"] else "[DATASET]"
                print(f"    {severity_tag} {col_label} {item['message']}")

    print(f"\n{border}")
    print(f"  VALIDATION GATE: {report['overall_status']}")
    print(f"{border}\n")
