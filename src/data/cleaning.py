"""
Enterprise Data Cleaning & Imputation Engine
=============================================
Production-grade data cleaning and imputation system for the Customer 360 platform.
Performs missing value imputation, duplicate removal, primary key integrity
checks, categorical normalization, whitespace scrubbing, invalid value filtering,
and outlier clipping.

This module provides data cleaning steps that can be run independently or orchestrated
into a full cleaning pipeline. All transformations are configurable.

Author: Principal Data Quality Engineer
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from src.core.config_loader import load_model_config
from src.core.exceptions import FeatureEngineeringError, ImputationError
from src.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DEFAULT CATEGORICAL NORMALIZATION MAPS
# ---------------------------------------------------------------------------
# Derived from statistical analysis of the E-Commerce dataset.
# Consolidates redundant categories into standardized labels.
# ---------------------------------------------------------------------------
DEFAULT_NORMALIZATION_MAPS: Dict[str, Dict[str, str]] = {
    "PreferredLoginDevice": {"Phone": "Mobile Phone"},
    "PreferredPaymentMode": {"CC": "Credit Card", "Cash on Delivery": "COD"},
    "PreferedOrderCat": {"Mobile": "Mobile Phone"},
}


def clean_whitespaces(df: pd.DataFrame) -> pd.DataFrame:
    """Scrubs leading/trailing whitespace from string columns and index headers.

    Args:
        df: Pandas DataFrame to clean.

    Returns:
        A new DataFrame with sanitized string columns.
    """
    df_clean = df.copy()
    df_clean.columns = df_clean.columns.str.strip()

    # Clean cell values
    str_cols = df_clean.select_dtypes(include=["object", "string"]).columns
    for col in str_cols:
        df_clean[col] = df_clean[col].apply(lambda x: x.strip() if isinstance(x, str) else x)

    logger.info(f"Stripped whitespace from {len(str_cols)} string columns.")
    return df_clean


def handle_duplicates(
    df: pd.DataFrame, primary_key: Optional[str] = "CustomerID", keep_strategy: str = "first"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Identifies and removes fully duplicated rows and primary key duplicates.

    Args:
        df: Pandas DataFrame.
        primary_key: Name of the unique identifier column.
        keep_strategy: How to handle duplicates ('first' to keep first occurrence,
                       'last' to keep last occurrence, or 'drop' to remove all).

    Returns:
        A tuple containing:
            - The deduplicated DataFrame.
            - A metadata dictionary summarizing duplicate row statistics.
    """
    df_clean = df.copy()
    initial_rows = len(df_clean)

    # 1. Remove exact row duplicates
    exact_duplicates = df_clean.duplicated()
    exact_dup_count = int(exact_duplicates.sum())
    if exact_dup_count > 0:
        df_clean = df_clean.drop_duplicates(keep=keep_strategy)
        logger.info(f"Removed {exact_dup_count} exact row duplicate(s).")

    # 2. Handle Primary Key duplicate rows (if key exists)
    pk_dup_count = 0
    if primary_key and primary_key in df_clean.columns:
        pk_duplicates = df_clean.duplicated(subset=[primary_key])
        pk_dup_count = int(pk_duplicates.sum())
        if pk_dup_count > 0:
            df_clean = df_clean.drop_duplicates(subset=[primary_key], keep=keep_strategy)
            logger.warning(f"Removed {pk_dup_count} duplicate primary key record(s) on '{primary_key}'.")

    final_rows = len(df_clean)
    removed_rows = initial_rows - final_rows

    metadata = {
        "initial_rows": initial_rows,
        "final_rows": final_rows,
        "exact_duplicates_removed": exact_dup_count,
        "pk_duplicates_removed": pk_dup_count,
        "total_removed": removed_rows,
    }

    return df_clean, metadata


def normalize_categories(
    df: pd.DataFrame, normalization_maps: Optional[Dict[str, Dict[str, str]]] = None
) -> pd.DataFrame:
    """Normalizes spelling inconsistencies and merges redundant categories.

    Normalizes categories (e.g. 'CC' -> 'Credit Card', 'Mobile' -> 'Mobile Phone')
    using mapping definitions to ensure consistent grouping representation.

    Args:
        df: Pandas DataFrame.
        normalization_maps: Mapping of column names to mapping dictionaries.
                            Defaults to DEFAULT_NORMALIZATION_MAPS.

    Returns:
        DataFrame with normalized categorical values.
    """
    df_clean = df.copy()
    maps = normalization_maps or DEFAULT_NORMALIZATION_MAPS

    for col, mapping in maps.items():
        if col in df_clean.columns:
            before_unique = set(df_clean[col].dropna().unique())
            df_clean[col] = df_clean[col].replace(mapping)
            after_unique = set(df_clean[col].dropna().unique())
            merged = before_unique - after_unique
            if merged:
                logger.info(f"Normalized column '{col}': merged categories {merged} -> {after_unique}")

    return df_clean


def impute_missing_values(
    df: pd.DataFrame,
    numerical_strategy: str = "median",
    categorical_strategy: str = "mode",
    group_by_col: Optional[str] = "PreferedOrderCat",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Imputes missing values for numerical and categorical columns.

    Imputes numerical nulls using median (either global or grouped by a column
    like PreferedOrderCat to capture category-level behavior). Imputes categorical
    nulls using column mode.

    Args:
        df: Pandas DataFrame to impute.
        numerical_strategy: 'median' or 'mean' for numerical columns.
        categorical_strategy: 'mode' or a custom string for categorical columns.
        group_by_col: Column to group by for conditional numerical imputation
                      (e.g. impute Tenure based on median Tenure of customer's
                      preferred order category). Falls back to global median if group key is null.

    Returns:
        A tuple containing:
            - Imputed DataFrame.
            - A dictionary detailing imputed columns and statistics.

    Raises:
        ImputationError: If an invalid strategy is specified.
    """
    df_clean = df.copy()
    config = load_model_config()

    # Extract imputation targets from config
    numerical_cols = config.get("data_processing", {}).get("imputation_columns", {}).get("numerical", [])
    categorical_cols = config.get("data_processing", {}).get("imputation_columns", {}).get("categorical", [])

    imputation_details = {}

    # 1. Numerical Imputation
    for col in numerical_cols:
        if col not in df_clean.columns:
            continue

        null_count = int(df_clean[col].isnull().sum())
        if null_count == 0:
            continue

        # Strategy implementation
        if numerical_strategy == "median":
            if group_by_col and group_by_col in df_clean.columns:
                # Grouped median imputation
                medians = df_clean.groupby(group_by_col)[col].transform("median")
                # Fill remaining nulls (if entire group is null) with global median
                global_val = float(df_clean[col].median())
                df_clean[col] = df_clean[col].fillna(medians).fillna(global_val)
                fill_desc = f"grouped median by {group_by_col} (fallback global {global_val})"
            else:
                global_val = float(df_clean[col].median())
                df_clean[col] = df_clean[col].fillna(global_val)
                fill_desc = f"global median ({global_val})"
        elif numerical_strategy == "mean":
            global_val = float(df_clean[col].mean())
            df_clean[col] = df_clean[col].fillna(global_val)
            fill_desc = f"global mean ({global_val})"
        else:
            raise ImputationError(f"Unsupported numerical imputation strategy: {numerical_strategy}")

        imputation_details[col] = {
            "imputed_count": null_count,
            "strategy": numerical_strategy,
            "description": fill_desc,
        }
        logger.info(f"Imputed {null_count} nulls in numeric '{col}' using {fill_desc}.")

    # 2. Categorical Imputation
    for col in categorical_cols:
        if col not in df_clean.columns:
            continue

        null_count = int(df_clean[col].isnull().sum())
        if null_count == 0:
            continue

        if categorical_strategy == "mode":
            mode_val = df_clean[col].mode().iloc[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            fill_desc = f"mode ({mode_val})"
        else:
            df_clean[col] = df_clean[col].fillna(categorical_strategy)
            fill_desc = f"constant value '{categorical_strategy}'"

        imputation_details[col] = {
            "imputed_count": null_count,
            "strategy": categorical_strategy,
            "description": fill_desc,
        }
        logger.info(f"Imputed {null_count} nulls in categorical '{col}' using {fill_desc}.")

    return df_clean, imputation_details


def clip_outliers(
    df: pd.DataFrame, iqr_multiplier: Optional[float] = None, target_columns: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Caps numerical outliers using the IQR (Interquartile Range) boundary method.

    Rather than dropping records containing outliers (which destroys valid business
    interactions), this caps extreme values at Q1 - (multiplier * IQR) and
    Q3 + (multiplier * IQR).

    Args:
        df: Pandas DataFrame.
        iqr_multiplier: IQR multiplier coefficient. Defaults to config file multiplier (1.5).
        target_columns: Columns to apply outlier capping. Defaults to config file target columns.

    Returns:
        A tuple containing:
            - DataFrame with clipped numerical columns.
            - A dictionary detailing lower/upper bounds and count of capped rows per column.
    """
    df_clean = df.copy()
    config = load_model_config()

    multiplier = iqr_multiplier or config.get("data_processing", {}).get("outlier_iqr_multiplier", 1.5)
    cols = target_columns or config.get("data_processing", {}).get("outlier_columns", [])

    outlier_details = {}

    for col in cols:
        if col not in df_clean.columns or not pd.api.types.is_numeric_dtype(df_clean[col]):
            continue

        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1

        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr

        # Count violations
        below_count = int((df_clean[col] < lower_bound).sum())
        above_count = int((df_clean[col] > upper_bound).sum())
        total_capped = below_count + above_count

        if total_capped > 0:
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
            logger.info(
                f"Clipped {total_capped} outliers in '{col}' "
                f"to range [{lower_bound:.2f}, {upper_bound:.2f}] (Below: {below_count}, Above: {above_count})."
            )

        outlier_details[col] = {
            "lower_bound": float(lower_bound),
            "upper_bound": float(upper_bound),
            "below_capped": below_count,
            "above_capped": above_count,
            "total_capped": total_capped,
        }

    return df_clean, outlier_details


def enforce_datatypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcasts numerical fields and enforces schema datatypes to optimize memory.

    Converts objects to clean string dtypes, and ensures correct numeric sizes.

    Args:
        df: Pandas DataFrame.

    Returns:
        Clean DataFrame with enforced datatypes.
    """
    df_clean = df.copy()

    # Float downcasting to float32
    float_cols = df_clean.select_dtypes(include=["float64"]).columns
    for col in float_cols:
        df_clean[col] = df_clean[col].astype(np.float32)

    # Int downcasting
    int_cols = df_clean.select_dtypes(include=["int64"]).columns
    for col in int_cols:
        df_clean[col] = df_clean[col].astype(np.int32)

    # String dtypes
    obj_cols = [
        col
        for col in df_clean.columns
        if df_clean[col].dtype in ["O", "object", "string", "str"] or isinstance(df_clean[col].dtype, pd.StringDtype)
    ]
    for col in obj_cols:
        df_clean[col] = df_clean[col].astype(str)

    logger.info(f"Enforced data types: {len(float_cols)} floats, {len(int_cols)} ints, {len(obj_cols)} strings.")
    return df_clean


def clean_dataset(
    df: pd.DataFrame,
    primary_key: str = "CustomerID",
    numerical_impute_strategy: str = "median",
    categorical_impute_strategy: str = "mode",
    group_by_col: Optional[str] = "PreferedOrderCat",
    iqr_multiplier: Optional[float] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Runs the complete data cleaning and imputation pipeline.

    Orchestrates:
        1. Whitespace scrubbing
        2. Row/ID deduplication
        3. Categorical value mapping and normalization
        4. Conditional missing value imputation
        5. Numeric outlier clipping (IQR Capping)
        6. Schema datatype enforcement

    Args:
        df: Raw DataFrame.
        primary_key: Column name of the primary key.
        numerical_impute_strategy: Strategy for numerical imputation ('median'/'mean').
        categorical_impute_strategy: Strategy for categorical imputation ('mode').
        group_by_col: Column to group by for conditional numerical imputation.
        iqr_multiplier: IQR multiplier for outlier treatment.

    Returns:
        A tuple containing:
            - The fully cleaned and imputed DataFrame.
            - A cleaning audit metadata dictionary.
    """
    logger.info("=" * 80)
    logger.info("ENTERPRISE DATA CLEANING PIPELINE — STARTING")
    logger.info("=" * 80)

    # Step 1: Clean whitespaces
    df_clean = clean_whitespaces(df)

    # Step 2: Handle duplicate rows and customer keys
    df_clean, duplicate_meta = handle_duplicates(df_clean, primary_key=primary_key)

    # Step 3: Normalize categorical spelling/redundancies
    df_clean = normalize_categories(df_clean)

    # Step 4: Impute missing values
    df_clean, imputation_meta = impute_missing_values(
        df_clean,
        numerical_strategy=numerical_impute_strategy,
        categorical_strategy=categorical_impute_strategy,
        group_by_col=group_by_col,
    )

    # Step 5: Clip numeric outliers (IQR Capping)
    df_clean, outlier_meta = clip_outliers(df_clean, iqr_multiplier=iqr_multiplier)

    # Step 6: Enforce datatypes & downcast
    df_clean = enforce_datatypes(df_clean)

    # Compile audit summary
    metadata = {
        "cleaning_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": df.shape,
        "output_shape": df_clean.shape,
        "deduplication": duplicate_meta,
        "imputation": imputation_meta,
        "outlier_capping": outlier_meta,
    }

    logger.info(f"Cleaning complete: cleaned dataset shape {df_clean.shape}.")
    logger.info("=" * 80)

    return df_clean, metadata


def print_cleaning_report(metadata: Dict[str, Any]) -> None:
    """Prints a professional, human-readable summary of cleaning statistics.

    Args:
        metadata: Dict containing cleaning metadata generated by clean_dataset().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE DATA CLEANING & AUDIT REPORT")
    print(border)
    print(f"Timestamp (UTC):  {metadata['cleaning_timestamp']}")
    print(f"Initial Shape:    {metadata['input_shape'][0]:,} rows x {metadata['input_shape'][1]} columns")
    print(f"Cleaned Shape:    {metadata['output_shape'][0]:,} rows x {metadata['output_shape'][1]} columns")
    print(border)

    print("\n1. DEDUPLICATION SUMMARY")
    print(section_divider)
    dedup = metadata["deduplication"]
    print(f"Exact Duplicate Rows Removed:     {dedup['exact_duplicates_removed']:,}")
    print(f"Duplicate Primary Keys Removed:   {dedup['pk_duplicates_removed']:,}")
    print(f"Total Rows Pruned:               {dedup['total_removed']:,}")

    print("\n2. MISSING VALUE IMPUTATION")
    print(section_divider)
    impute = metadata["imputation"]
    if not impute:
        print("No missing values required imputation.")
    else:
        for col, details in impute.items():
            print(f"Column '{col}': Imputed {details['imputed_count']:,} null(s) using {details['description']}")

    print("\n3. OUTLIER TREATMENT (IQR CAPPING)")
    print(section_divider)
    outliers = metadata["outlier_capping"]
    if not outliers:
        print("No outlier treatment was applied.")
    else:
        for col, details in outliers.items():
            if details["total_capped"] > 0:
                print(
                    f"Column '{col}': Capped {details['total_capped']:,} outlier(s) "
                    f"[Below: {details['below_capped']:,}, Above: {details['above_capped']:,}] "
                    f"into range [{details['lower_bound']:.2f}, {details['upper_bound']:.2f}]"
                )

    print(f"\n{border}")
    print("                 END OF ENTERPRISE DATA CLEANING REPORT")
    print(f"{border}\n")
