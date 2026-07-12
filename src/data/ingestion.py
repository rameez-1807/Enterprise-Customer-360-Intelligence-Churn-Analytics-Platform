"""
Enterprise Data Ingestion Engine
==================================
Production-grade data ingestion system for the Customer 360 platform.
Handles reading upstream Excel datasets, automatic sheet detection,
schema validation, column verification, data type enforcement, duplicate
detection, and structured ingestion report generation.

This module serves as the single entry point for all raw data flowing
into the analytical pipeline. It enforces strict quality gates before
any downstream processing occurs.

Usage:
    from src.data.ingestion import ingest_dataset

    df, metadata = ingest_dataset()
    print(metadata)

Author: Principal Data Engineer
Version: 1.0.0
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.core.exceptions import IngestionError, SchemaValidationError
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# EXPECTED SCHEMA REGISTRY
# =============================================================================
# Defines the canonical column contract for the upstream e-commerce dataset.
# Any deviation from this registry triggers a validation error or warning.
# =============================================================================

EXPECTED_COLUMNS: list[str] = [
    "CustomerID",
    "Churn",
    "Tenure",
    "PreferredLoginDevice",
    "CityTier",
    "WarehouseToHome",
    "PreferredPaymentMode",
    "Gender",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "PreferedOrderCat",
    "SatisfactionScore",
    "MaritalStatus",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
]

EXPECTED_DTYPES: dict[str, list[str]] = {
    "CustomerID": ["int64"],
    "Churn": ["int64"],
    "Tenure": ["float64"],
    "PreferredLoginDevice": ["object", "str", "string"],
    "CityTier": ["int64"],
    "WarehouseToHome": ["float64"],
    "PreferredPaymentMode": ["object", "str", "string"],
    "Gender": ["object", "str", "string"],
    "HourSpendOnApp": ["float64"],
    "NumberOfDeviceRegistered": ["int64"],
    "PreferedOrderCat": ["object", "str", "string"],
    "SatisfactionScore": ["int64"],
    "MaritalStatus": ["object", "str", "string"],
    "NumberOfAddress": ["int64"],
    "Complain": ["int64"],
    "OrderAmountHikeFromlastYear": ["float64"],
    "CouponUsed": ["float64"],
    "OrderCount": ["float64"],
    "DaySinceLastOrder": ["float64"],
    "CashbackAmount": ["float64"],
}

PRIMARY_KEY_COLUMN: str = "CustomerID"
TARGET_SHEET_NAME: str = "E Comm"


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _resolve_file_path(filepath: Path | str | None = None) -> Path:
    """Resolve and validate the absolute path to the source dataset file.

    If no filepath is provided, falls back to the default path defined
    in config/settings.py. Validates that the file physically exists
    on disk before returning.

    Args:
        filepath: Optional explicit path to the Excel dataset file.
            If None, the default path from settings is used.

    Returns:
        A resolved Path object pointing to the dataset file.

    Raises:
        IngestionError: If the resolved file does not exist on disk.
    """
    if filepath is None:
        from config.settings import PROJECT_ROOT, RAW_DATASET_FILENAME

        resolved: Path = PROJECT_ROOT / RAW_DATASET_FILENAME
    else:
        resolved = Path(filepath).resolve()

    if not resolved.exists():
        error_msg = f"Source dataset file not found at: {resolved}"
        logger.error(error_msg)
        raise IngestionError(error_msg)

    if not resolved.is_file():
        error_msg = f"Path exists but is not a file: {resolved}"
        logger.error(error_msg)
        raise IngestionError(error_msg)

    logger.info(f"Source file resolved: {resolved.name} ({resolved.stat().st_size:,} bytes)")
    return resolved


def _discover_sheets(filepath: Path) -> list[str]:
    """Discover all sheet names within an Excel workbook.

    Uses openpyxl in read-only mode for memory-efficient sheet
    enumeration without loading full cell data.

    Args:
        filepath: Path to the Excel workbook file.

    Returns:
        A list of sheet name strings found in the workbook.

    Raises:
        IngestionError: If the workbook cannot be opened or parsed.
    """
    try:
        xlsx_file = pd.ExcelFile(filepath, engine="openpyxl")
        sheet_names: list[str] = xlsx_file.sheet_names
        xlsx_file.close()
        logger.info(f"Discovered {len(sheet_names)} sheet(s): {sheet_names}")
        return sheet_names
    except Exception as e:
        error_msg = f"Failed to read Excel workbook structure: {e}"
        logger.error(error_msg)
        raise IngestionError(error_msg) from e


def _validate_sheet_exists(available_sheets: list[str], target_sheet: str) -> str:
    """Validate that the required data sheet exists in the workbook.

    Performs case-insensitive matching as a fallback if exact match fails,
    logging a warning when a fuzzy match is used.

    Args:
        available_sheets: List of sheet names discovered in the workbook.
        target_sheet: The expected sheet name to locate.

    Returns:
        The confirmed sheet name string (exact or fuzzy-matched).

    Raises:
        IngestionError: If no matching sheet is found in the workbook.
    """
    # Exact match
    if target_sheet in available_sheets:
        logger.info(f"Target sheet '{target_sheet}' found (exact match).")
        return target_sheet

    # Case-insensitive fallback
    sheet_map: dict[str, str] = {s.lower().strip(): s for s in available_sheets}
    normalized_target: str = target_sheet.lower().strip()

    if normalized_target in sheet_map:
        matched: str = sheet_map[normalized_target]
        logger.warning(f"Exact sheet name '{target_sheet}' not found. " f"Using case-insensitive match: '{matched}'.")
        return matched

    error_msg = f"Required sheet '{target_sheet}' not found in workbook. " f"Available sheets: {available_sheets}"
    logger.error(error_msg)
    raise IngestionError(error_msg)


def _read_excel_sheet(filepath: Path, sheet_name: str) -> pd.DataFrame:
    """Read a specific sheet from an Excel workbook into a DataFrame.

    Uses the openpyxl engine for .xlsx format compatibility. Strips
    leading and trailing whitespace from all string column headers
    to prevent downstream key-matching failures.

    Args:
        filepath: Path to the Excel workbook file.
        sheet_name: Name of the sheet to read.

    Returns:
        A pandas DataFrame containing the raw sheet data.

    Raises:
        IngestionError: If the sheet cannot be read or is empty.
    """
    try:
        df: pd.DataFrame = pd.read_excel(
            filepath,
            sheet_name=sheet_name,
            engine="openpyxl",
        )
    except Exception as e:
        error_msg = f"Failed to read sheet '{sheet_name}' from '{filepath.name}': {e}"
        logger.error(error_msg)
        raise IngestionError(error_msg) from e

    # Sanitize column headers
    df.columns = df.columns.str.strip()

    if df.empty:
        error_msg = f"Sheet '{sheet_name}' is empty (0 records)."
        logger.error(error_msg)
        raise IngestionError(error_msg)

    logger.info(f"Successfully read sheet '{sheet_name}': " f"{df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df


def _validate_required_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
) -> dict[str, Any]:
    """Validate that all required columns are present in the DataFrame.

    Identifies missing columns (expected but not found), extra columns
    (found but not expected), and potentially misspelled column names
    using Levenshtein-style similarity detection.

    Args:
        df: The raw ingested DataFrame to validate.
        expected_columns: The canonical list of required column names.

    Returns:
        A dictionary containing validation results:
            - missing_columns: Columns expected but not found.
            - extra_columns: Columns found but not expected.
            - matched_columns: Columns successfully matched.
            - possible_typos: Suspected misspellings with suggestions.
            - is_valid: Boolean indicating if all required columns are present.

    Raises:
        SchemaValidationError: If critical required columns are missing
            and no fallback match can be determined.
    """
    actual_columns: set[str] = set(df.columns.tolist())
    expected_set: set[str] = set(expected_columns)

    missing: list[str] = sorted(expected_set - actual_columns)
    extra: list[str] = sorted(actual_columns - expected_set)
    matched: list[str] = sorted(expected_set & actual_columns)

    # Detect possible typos by finding close matches
    possible_typos: list[dict[str, str]] = []
    if missing and extra:
        for miss in missing:
            for ext in extra:
                similarity: float = _calculate_similarity(miss, ext)
                if similarity >= 0.75:
                    possible_typos.append(
                        {
                            "expected": miss,
                            "found": ext,
                            "similarity": f"{similarity:.0%}",
                        }
                    )

    validation_result: dict[str, Any] = {
        "missing_columns": missing,
        "extra_columns": extra,
        "matched_columns": matched,
        "possible_typos": possible_typos,
        "is_valid": len(missing) == 0,
    }

    if missing:
        logger.warning(f"Missing columns detected: {missing}")
    if extra:
        logger.info(f"Extra columns detected (not in schema): {extra}")
    if possible_typos:
        logger.warning(f"Possible column name typos detected: {possible_typos}")
    if not missing:
        logger.info(f"All {len(expected_columns)} required columns validated successfully.")

    return validation_result


def _calculate_similarity(str_a: str, str_b: str) -> float:
    """Calculate normalized character-level similarity between two strings.

    Uses a simplified sequence matching algorithm (longest common
    subsequence ratio) to detect potential column name misspellings
    without requiring external libraries.

    Args:
        str_a: First string to compare.
        str_b: Second string to compare.

    Returns:
        A float between 0.0 and 1.0 representing similarity,
        where 1.0 means identical strings.
    """
    a_lower: str = str_a.lower()
    b_lower: str = str_b.lower()

    if a_lower == b_lower:
        return 1.0

    max_len: int = max(len(a_lower), len(b_lower))
    if max_len == 0:
        return 1.0

    # Levenshtein distance calculation
    m, n = len(a_lower), len(b_lower)
    dp: list[list[int]] = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost: int = 0 if a_lower[i - 1] == b_lower[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

    distance: int = dp[m][n]
    return 1.0 - (distance / max_len)


def _validate_data_types(
    df: pd.DataFrame,
    expected_dtypes: dict[str, list[str]],
) -> dict[str, Any]:
    """Validate that DataFrame column data types match expected specifications.

    Compares the actual pandas dtype of each column against a list of
    acceptable dtype strings. Columns with nulls may have been promoted
    from int to float by pandas; these are flagged as warnings rather
    than errors. Pandas 3.x may report string columns as 'str' instead
    of 'object', so multiple acceptable types are supported per column.

    Args:
        df: The ingested DataFrame to validate.
        expected_dtypes: Mapping of column names to lists of acceptable
            dtype strings (e.g., {"Gender": ["object", "str", "string"]}).

    Returns:
        A dictionary containing:
            - matched_types: Columns with correct data types.
            - mismatched_types: Columns with unexpected data types.
            - is_valid: Boolean indicating all types match.
    """
    matched_types: list[str] = []
    mismatched_types: list[dict[str, str]] = []

    for column, acceptable_types in expected_dtypes.items():
        if column not in df.columns:
            continue

        actual_type: str = str(df[column].dtype)

        if actual_type in acceptable_types:
            matched_types.append(column)
        else:
            mismatched_types.append(
                {
                    "column": column,
                    "expected": ", ".join(acceptable_types),
                    "actual": actual_type,
                }
            )

    if mismatched_types:
        logger.warning(f"Data type mismatches detected in {len(mismatched_types)} column(s).")
        for mismatch in mismatched_types:
            logger.warning(
                f"  Column '{mismatch['column']}': "
                f"expected '{mismatch['expected']}', "
                f"got '{mismatch['actual']}'."
            )
    else:
        logger.info("All column data types validated successfully.")

    return {
        "matched_types": matched_types,
        "mismatched_types": mismatched_types,
        "is_valid": len(mismatched_types) == 0,
    }


def _detect_duplicates(df: pd.DataFrame, key_column: str) -> dict[str, Any]:
    """Detect duplicate records based on a primary key column.

    Identifies rows where the primary key appears more than once,
    which indicates data quality issues in the upstream source.

    Args:
        df: The ingested DataFrame to check.
        key_column: The column name acting as the primary key.

    Returns:
        A dictionary containing:
            - total_duplicates: Count of duplicate rows.
            - duplicate_keys: List of duplicated key values.
            - is_clean: Boolean indicating no duplicates exist.
    """
    if key_column not in df.columns:
        logger.warning(f"Primary key column '{key_column}' not found. Skipping duplicate check.")
        return {
            "total_duplicates": 0,
            "duplicate_keys": [],
            "is_clean": True,
        }

    duplicate_mask: pd.Series = df[key_column].duplicated(keep=False)
    duplicate_count: int = int(duplicate_mask.sum())

    duplicate_keys: list[Any] = []
    if duplicate_count > 0:
        duplicate_keys = df.loc[duplicate_mask, key_column].unique().tolist()
        logger.warning(
            f"Detected {duplicate_count} duplicate rows across "
            f"{len(duplicate_keys)} key value(s): {duplicate_keys[:10]}"
            f"{'...' if len(duplicate_keys) > 10 else ''}"
        )
    else:
        logger.info(f"No duplicate records found on key '{key_column}'.")

    return {
        "total_duplicates": duplicate_count,
        "duplicate_keys": duplicate_keys,
        "is_clean": duplicate_count == 0,
    }


def _profile_missing_values(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a missing value profile for the entire DataFrame.

    Calculates null counts and percentages per column, identifying
    columns that exceed critical thresholds for data quality auditing.

    Args:
        df: The ingested DataFrame to profile.

    Returns:
        A dictionary containing:
            - total_nulls: Total null values across all columns.
            - columns_with_nulls: List of columns with null values
              and their respective counts and percentages.
            - columns_without_nulls: List of columns with no nulls.
            - null_percentage_overall: Overall dataset null percentage.
    """
    null_counts: pd.Series = df.isnull().sum()
    total_rows: int = len(df)
    total_cells: int = total_rows * len(df.columns)
    total_nulls: int = int(null_counts.sum())

    columns_with_nulls: list[dict[str, Any]] = []
    columns_without_nulls: list[str] = []

    for column in df.columns:
        count: int = int(null_counts[column])
        if count > 0:
            columns_with_nulls.append(
                {
                    "column": column,
                    "null_count": count,
                    "null_percentage": round((count / total_rows) * 100, 2),
                }
            )
        else:
            columns_without_nulls.append(column)

    # Sort by null percentage descending for prioritization
    columns_with_nulls.sort(key=lambda x: x["null_percentage"], reverse=True)

    overall_pct: float = round((total_nulls / total_cells) * 100, 2) if total_cells > 0 else 0.0

    if columns_with_nulls:
        logger.warning(
            f"Missing values detected in {len(columns_with_nulls)} column(s) "
            f"({total_nulls:,} total nulls, {overall_pct}% of all cells)."
        )
    else:
        logger.info("No missing values detected in any column.")

    return {
        "total_nulls": total_nulls,
        "columns_with_nulls": columns_with_nulls,
        "columns_without_nulls": columns_without_nulls,
        "null_percentage_overall": overall_pct,
    }


def _build_ingestion_metadata(
    filepath: Path,
    sheet_name: str,
    df: pd.DataFrame,
    available_sheets: list[str],
    column_validation: dict[str, Any],
    dtype_validation: dict[str, Any],
    duplicate_report: dict[str, Any],
    missing_profile: dict[str, Any],
) -> dict[str, Any]:
    """Build a structured ingestion metadata report.

    Consolidates all validation results, profiling outputs, and
    file-level metadata into a single auditable dictionary that
    serves as the quality gate evidence for downstream consumers.

    Args:
        filepath: Path to the source file that was ingested.
        sheet_name: Name of the sheet that was read.
        df: The ingested DataFrame.
        available_sheets: All sheet names in the workbook.
        column_validation: Results from column validation.
        dtype_validation: Results from data type validation.
        duplicate_report: Results from duplicate detection.
        missing_profile: Results from missing value profiling.

    Returns:
        A comprehensive metadata dictionary documenting the
        full ingestion audit trail.
    """
    metadata: dict[str, Any] = {
        "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
        "source_file": {
            "filename": filepath.name,
            "filepath": str(filepath),
            "file_size_bytes": filepath.stat().st_size,
            "available_sheets": available_sheets,
            "ingested_sheet": sheet_name,
        },
        "dataset_shape": {
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "column_names": df.columns.tolist(),
        },
        "column_validation": column_validation,
        "dtype_validation": dtype_validation,
        "duplicate_report": duplicate_report,
        "missing_value_profile": missing_profile,
        "quality_gate": {
            "columns_valid": column_validation["is_valid"],
            "dtypes_valid": dtype_validation["is_valid"],
            "duplicates_clean": duplicate_report["is_clean"],
            "overall_pass": (column_validation["is_valid"] and duplicate_report["is_clean"]),
        },
    }

    gate_status: str = "PASSED" if metadata["quality_gate"]["overall_pass"] else "FAILED"
    logger.info(f"Ingestion quality gate: {gate_status}")

    return metadata


def _downcast_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to reduce memory footprint.

    Converts int64 columns to int32 and float64 columns to float32
    where the value ranges allow, reducing in-memory size by
    approximately 50% without precision loss for analytical workloads.

    Args:
        df: The DataFrame to optimize.

    Returns:
        A new DataFrame with downcasted numeric types.
    """
    df_optimized: pd.DataFrame = df.copy()

    int_cols: list[str] = df_optimized.select_dtypes(include=["int64"]).columns.tolist()
    float_cols: list[str] = df_optimized.select_dtypes(include=["float64"]).columns.tolist()

    for col in int_cols:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="integer")

    for col in float_cols:
        df_optimized[col] = pd.to_numeric(df_optimized[col], downcast="float")

    original_mb: float = df.memory_usage(deep=True).sum() / (1024 * 1024)
    optimized_mb: float = df_optimized.memory_usage(deep=True).sum() / (1024 * 1024)
    savings_pct: float = ((original_mb - optimized_mb) / original_mb) * 100 if original_mb > 0 else 0.0

    logger.info(f"Memory optimization: {original_mb:.2f} MB → {optimized_mb:.2f} MB " f"(saved {savings_pct:.1f}%).")

    return df_optimized


# =============================================================================
# PRIMARY PUBLIC API
# =============================================================================


def ingest_dataset(
    filepath: Path | str | None = None,
    sheet_name: str = TARGET_SHEET_NAME,
    expected_columns: list[str] | None = None,
    expected_dtypes: dict[str, list[str]] | None = None,
    primary_key: str = PRIMARY_KEY_COLUMN,
    optimize_memory: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Ingest, validate, and profile a raw e-commerce dataset from Excel.

    This is the primary public entry point for the data ingestion pipeline.
    It executes the following sequential quality gates:
        1. File path resolution and existence verification.
        2. Workbook sheet discovery and target sheet validation.
        3. Excel sheet reading with header sanitization.
        4. Required column presence validation with typo detection.
        5. Data type conformity validation.
        6. Primary key duplicate detection.
        7. Missing value profiling.
        8. Optional memory optimization via numeric downcasting.
        9. Structured metadata report generation.

    Args:
        filepath: Optional explicit path to the Excel dataset file.
            If None, the default path from config/settings.py is used.
        sheet_name: Name of the target sheet to read from the workbook.
            Defaults to "E Comm".
        expected_columns: Optional list of required column names to validate.
            If None, the module-level EXPECTED_COLUMNS registry is used.
        expected_dtypes: Optional mapping of column names to expected dtypes.
            If None, the module-level EXPECTED_DTYPES registry is used.
        primary_key: Column name to use for duplicate detection.
            Defaults to "CustomerID".
        optimize_memory: Whether to downcast numeric columns for memory
            optimization. Defaults to True.

    Returns:
        A tuple containing:
            - df: A pandas DataFrame with the raw ingested data.
            - metadata: A comprehensive dictionary documenting the
              full ingestion audit trail, validation results, and
              data quality profile.

    Raises:
        IngestionError: If the source file cannot be found, opened,
            or the target sheet does not exist or is empty.
        SchemaValidationError: If critical required columns are missing
            and no recovery is possible.

    Example:
        >>> from src.data.ingestion import ingest_dataset
        >>> df, metadata = ingest_dataset()
        >>> print(f"Ingested {metadata['dataset_shape']['total_rows']} records.")
        >>> print(f"Quality gate: {metadata['quality_gate']['overall_pass']}")
    """
    logger.info("=" * 70)
    logger.info("ENTERPRISE DATA INGESTION ENGINE — STARTING")
    logger.info("=" * 70)

    # Step 1: Resolve and validate file path
    resolved_path: Path = _resolve_file_path(filepath)

    # Step 2: Discover available sheets
    available_sheets: list[str] = _discover_sheets(resolved_path)

    # Step 3: Validate target sheet exists
    confirmed_sheet: str = _validate_sheet_exists(available_sheets, sheet_name)

    # Step 4: Read Excel sheet into DataFrame
    df: pd.DataFrame = _read_excel_sheet(resolved_path, confirmed_sheet)

    # Step 5: Validate required columns
    cols_to_validate: list[str] = expected_columns if expected_columns is not None else EXPECTED_COLUMNS
    column_validation: dict[str, Any] = _validate_required_columns(df, cols_to_validate)

    if not column_validation["is_valid"]:
        missing: list[str] = column_validation["missing_columns"]
        logger.error(f"Schema validation failed: {len(missing)} required column(s) missing: {missing}")
        raise SchemaValidationError(
            f"Missing required columns: {missing}. " f"Possible typos: {column_validation['possible_typos']}"
        )

    # Step 6: Validate data types
    dtypes_to_validate: dict[str, list[str]] = (
        expected_dtypes if expected_dtypes is not None else EXPECTED_DTYPES
    )
    dtype_validation: dict[str, Any] = _validate_data_types(df, dtypes_to_validate)

    # Step 7: Detect duplicates
    duplicate_report: dict[str, Any] = _detect_duplicates(df, primary_key)

    # Step 8: Profile missing values
    missing_profile: dict[str, Any] = _profile_missing_values(df)

    # Step 9: Optimize memory (optional)
    if optimize_memory:
        df = _downcast_numeric_columns(df)

    # Step 10: Build comprehensive metadata report
    metadata: dict[str, Any] = _build_ingestion_metadata(
        filepath=resolved_path,
        sheet_name=confirmed_sheet,
        df=df,
        available_sheets=available_sheets,
        column_validation=column_validation,
        dtype_validation=dtype_validation,
        duplicate_report=duplicate_report,
        missing_profile=missing_profile,
    )

    logger.info(
        f"INGESTION COMPLETE — {df.shape[0]:,} records × {df.shape[1]} columns "
        f"| Quality Gate: {'PASSED' if metadata['quality_gate']['overall_pass'] else 'FAILED'}"
    )
    logger.info("=" * 70)

    return df, metadata


def read_sheet_as_dataframe(
    filepath: Path | str,
    sheet_name: str,
) -> pd.DataFrame:
    """Read a specific sheet from an Excel file as a raw DataFrame.

    A lightweight utility function for reading arbitrary sheets without
    running the full validation pipeline. Useful for reading supplementary
    sheets like data dictionaries or metadata tabs.

    Args:
        filepath: Path to the Excel workbook file.
        sheet_name: Name of the sheet to read.

    Returns:
        A pandas DataFrame containing the sheet data.

    Raises:
        IngestionError: If the file or sheet cannot be read.
    """
    resolved: Path = _resolve_file_path(filepath)
    available: list[str] = _discover_sheets(resolved)
    confirmed: str = _validate_sheet_exists(available, sheet_name)
    return _read_excel_sheet(resolved, confirmed)


def get_dataset_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Generate a high-level statistical summary of an ingested DataFrame.

    Produces quick descriptive statistics useful for pipeline health
    monitoring and pre-cleaning diagnostic checks.

    Args:
        df: A pandas DataFrame to summarize.

    Returns:
        A dictionary containing:
            - shape: Tuple of (rows, columns).
            - memory_usage_mb: Approximate memory usage in megabytes.
            - numeric_columns: List of numeric column names.
            - categorical_columns: List of categorical column names.
            - null_total: Total number of null values.
            - duplicate_rows: Number of fully duplicated rows.
    """
    memory_mb: float = round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    numeric_cols: list[str] = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols: list[str] = df.select_dtypes(include=["object", "category"]).columns.tolist()

    summary: dict[str, Any] = {
        "shape": df.shape,
        "memory_usage_mb": memory_mb,
        "numeric_columns": numeric_cols,
        "categorical_columns": categorical_cols,
        "null_total": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    logger.info(
        f"Dataset summary: {summary['shape'][0]:,} rows, "
        f"{len(numeric_cols)} numeric / {len(categorical_cols)} categorical columns, "
        f"{memory_mb:.2f} MB in memory."
    )

    return summary
