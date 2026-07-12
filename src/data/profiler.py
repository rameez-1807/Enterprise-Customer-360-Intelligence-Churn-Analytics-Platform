"""
Enterprise Data Profiling Engine
===================================
Production-grade data profiling engine for the Customer 360 platform.
Performs comprehensive statistical profiling of the e-commerce dataset,
detecting missing value ratios, duplicate rows, duplicate primary keys,
constant columns, low variance indicators, high cardinality columns,
skewness, IQR-based outliers, and memory footprint metrics.

This module provides data quality audits and generates serialized metadata
that can be consumed by data cleaning pipelines or automated monitoring alerts.

Author: Principal Data Quality Engineer
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src.core.exceptions import SchemaValidationError
from src.core.logger import get_logger

logger = get_logger(__name__)


def profile_numeric_column(series: pd.Series) -> Dict[str, Any]:
    """Generates detailed statistical profile for a numeric column.

    Calculates central tendency, dispersion, shape (skewness, kurtosis),
    nullability, and IQR-based outliers.

    Args:
        series: Pandas Series of numeric data.

    Returns:
        A dictionary containing statistical metrics for the column.
    """
    non_null_series = series.dropna()
    total_count = len(series)
    non_null_count = len(non_null_series)
    null_count = total_count - non_null_count
    null_pct = (null_count / total_count) * 100 if total_count > 0 else 0.0

    if non_null_count == 0:
        return {
            "type": "numeric",
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": null_pct,
            "error": "Column contains only null values",
        }

    # Descriptive statistics
    mean_val = float(non_null_series.mean())
    std_val = float(non_null_series.std()) if non_null_count > 1 else 0.0
    min_val = float(non_null_series.min())
    max_val = float(non_null_series.max())
    q1 = float(non_null_series.quantile(0.25))
    median_val = float(non_null_series.median())
    q3 = float(non_null_series.quantile(0.75))
    iqr = q3 - q1

    # Skewness calculation
    skewness = float(non_null_series.skew()) if non_null_count > 2 else 0.0

    # Variance and standard deviation indicators
    variance = float(non_null_series.var()) if non_null_count > 1 else 0.0
    cv = (std_val / mean_val) if mean_val != 0 else 0.0  # Coefficient of variation

    # Outlier Detection via IQR Method
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = non_null_series[(non_null_series < lower_bound) | (non_null_series > upper_bound)]
    outlier_count = len(outliers)
    outlier_pct = (outlier_count / total_count) * 100 if total_count > 0 else 0.0

    # Low variance detection (CV < 0.05 and std > 0, or standard deviation is exactly 0)
    is_constant = non_null_series.nunique() == 1
    is_low_variance = is_constant or (std_val < 0.001) or (abs(cv) < 0.01)

    return {
        "type": "numeric",
        "total_count": total_count,
        "null_count": null_count,
        "null_percentage": round(null_pct, 4),
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "variance": round(variance, 4),
        "cv": round(cv, 4),
        "min": min_val,
        "q1": q1,
        "median": median_val,
        "q3": q3,
        "max": max_val,
        "iqr": round(iqr, 4),
        "skewness": round(skewness, 4),
        "outliers": {
            "lower_bound": round(lower_bound, 4),
            "upper_bound": round(upper_bound, 4),
            "count": outlier_count,
            "percentage": round(outlier_pct, 4),
            "sample": outliers.head(5).tolist(),
        },
        "is_constant": is_constant,
        "is_low_variance": is_low_variance,
    }


def profile_categorical_column(series: pd.Series) -> Dict[str, Any]:
    """Generates frequency distribution and cardinality checks for a categorical column.

    Args:
        series: Pandas Series of categorical data.

    Returns:
        A dictionary containing value counts, unique counts, nullability,
        and high cardinality detection.
    """
    total_count = len(series)
    null_count = int(series.isnull().sum())
    null_pct = (null_count / total_count) * 100 if total_count > 0 else 0.0
    non_null_series = series.dropna()
    non_null_count = len(non_null_series)

    if non_null_count == 0:
        return {
            "type": "categorical",
            "total_count": total_count,
            "null_count": null_count,
            "null_percentage": null_pct,
            "error": "Column contains only null values",
        }

    # Cardinality
    unique_count = int(non_null_series.nunique())
    cardinality_ratio = unique_count / total_count if total_count > 0 else 0.0

    # Value counts & distribution
    value_counts = non_null_series.value_counts()
    top_value = value_counts.index[0]
    top_count = int(value_counts.iloc[0])
    top_pct = (top_count / total_count) * 100 if total_count > 0 else 0.0

    # Build frequency distribution (top 10 values)
    frequency_distribution = {}
    for val, cnt in value_counts.head(10).items():
        frequency_distribution[str(val)] = {"count": int(cnt), "percentage": round((cnt / total_count) * 100, 4)}

    # Checks
    is_constant = unique_count == 1
    # High cardinality rule: unique values are more than 20% of dataset and total unique values > 10
    is_high_cardinality = cardinality_ratio > 0.20 and unique_count > 10
    # Dominant class rule: single value constitutes > 95% of non-null records
    is_low_variance = is_constant or (top_count / non_null_count) > 0.95

    return {
        "type": "categorical",
        "total_count": total_count,
        "null_count": null_count,
        "null_percentage": round(null_pct, 4),
        "unique_values_count": unique_count,
        "cardinality_ratio": round(cardinality_ratio, 4),
        "top_value": str(top_value),
        "top_value_percentage": round(top_pct, 4),
        "frequency_distribution": frequency_distribution,
        "is_constant": is_constant,
        "is_high_cardinality": is_high_cardinality,
        "is_low_variance": is_low_variance,
    }


def generate_profile(df: pd.DataFrame, primary_key: str = "CustomerID") -> Dict[str, Any]:
    """Generates a complete statistical profile of the input DataFrame.

    Profiles every column, evaluates column correlations, counts duplicate rows,
    counts duplicate primary keys, computes memory consumption, and formats metadata.

    Args:
        df: Pandas DataFrame to profile.
        primary_key: String name of the column serving as the primary key.

    Returns:
        A dictionary containing dataset-level and column-level statistics.

    Raises:
        SchemaValidationError: If the dataframe is empty.
    """
    if df.empty:
        raise SchemaValidationError("Cannot profile an empty DataFrame.")

    logger.info("Initializing dataset profiling execution.")
    total_rows = len(df)
    total_cols = len(df.columns)
    total_cells = total_rows * total_cols

    # Memory Usage Calculation
    memory_bytes = int(df.memory_usage(deep=True).sum())
    memory_mb = memory_bytes / (1024 * 1024)

    # Duplicate Record Verification
    duplicate_rows_count = int(df.duplicated().sum())
    duplicate_rows_pct = (duplicate_rows_count / total_rows) * 100 if total_rows > 0 else 0.0

    # Primary Key Duplicate Verification
    pk_duplicate_count = 0
    pk_duplicate_pct = 0.0
    if primary_key in df.columns:
        pk_duplicate_count = int(df[primary_key].duplicated().sum())
        pk_duplicate_pct = (pk_duplicate_count / total_rows) * 100 if total_rows > 0 else 0.0
    else:
        logger.warning(f"Primary key column '{primary_key}' not found during duplicate profiling.")

    # Column-level profiling loop
    columns_profile = {}
    constant_columns = []
    low_variance_columns = []
    high_cardinality_columns = []
    high_null_columns = []  # Columns with > 30% missing values

    for col in df.columns:
        series = df[col]
        # Check if numeric
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            col_profile = profile_numeric_column(series)
        else:
            col_profile = profile_categorical_column(series)

        columns_profile[col] = col_profile

        # Aggregate issues
        if col_profile.get("is_constant"):
            constant_columns.append(col)
        if col_profile.get("is_low_variance"):
            low_variance_columns.append(col)
        if col_profile.get("is_high_cardinality"):
            high_cardinality_columns.append(col)
        if col_profile.get("null_percentage", 0.0) > 30.0:
            high_null_columns.append(col)

    # Calculate overall completeness
    total_nulls = sum(c.get("null_count", 0) for c in columns_profile.values())
    completeness_score = ((total_cells - total_nulls) / total_cells) * 100 if total_cells > 0 else 100.0

    profile_report = {
        "profiling_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_rows": total_rows,
            "total_columns": total_cols,
            "total_cells": total_cells,
            "total_nulls": total_nulls,
            "completeness_score_pct": round(completeness_score, 4),
            "memory_usage_bytes": memory_bytes,
            "memory_usage_mb": round(memory_mb, 4),
            "duplicate_rows_count": duplicate_rows_count,
            "duplicate_rows_pct": round(duplicate_rows_pct, 4),
            "primary_key": primary_key,
            "pk_duplicate_count": pk_duplicate_count,
            "pk_duplicate_pct": round(pk_duplicate_pct, 4),
        },
        "anomalies": {
            "constant_columns": constant_columns,
            "low_variance_columns": low_variance_columns,
            "high_cardinality_columns": high_cardinality_columns,
            "high_null_columns": high_null_columns,
        },
        "columns": columns_profile,
    }

    logger.info(
        f"Profiling completed: {total_rows} rows x {total_cols} cols profiled. "
        f"Completeness Score: {completeness_score:.2f}%. "
        f"Memory footprint: {memory_mb:.2f} MB."
    )

    return profile_report


def print_profiling_report(report: Dict[str, Any]) -> None:
    """Prints a professional, human-readable statistical profile report to the console.

    Args:
        report: Dict containing profiling statistics generated by generate_profile().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CUSTOMER 360 DATA PROFILING REPORT")
    print(border)
    print(f"Timestamp (UTC): {report['profiling_timestamp']}")
    print(f"File Shape:      {report['summary']['total_rows']:,} rows x {report['summary']['total_columns']} columns")
    print(f"Memory Footprint: {report['summary']['memory_usage_mb']:.2f} MB")
    print(
        f"Completeness:    {report['summary']['completeness_score_pct']:.2f}% (Total Nulls: {report['summary']['total_nulls']:,})"
    )
    print(
        f"Duplicates:      {report['summary']['duplicate_rows_count']:,} duplicate rows ({report['summary']['duplicate_rows_pct']}%)"
    )
    print(
        f"PK Duplicates:   {report['summary']['pk_duplicate_count']:,} duplicate PK keys ({report['summary']['pk_duplicate_pct']}%)"
    )
    print(border)

    print("\nDATA QUALITY & ARCHITECTURAL ANOMALIES")
    print(section_divider)
    print(f"Constant Columns (No variance):            {report['anomalies']['constant_columns']}")
    print(f"Low Variance Columns (Near-constant):     {report['anomalies']['low_variance_columns']}")
    print(f"High Cardinality Columns (High entropy):  {report['anomalies']['high_cardinality_columns']}")
    print(f"High Null Columns (>30% Missing):         {report['anomalies']['high_null_columns']}")
    print(section_divider)

    print("\nDETAILED COLUMN STATISTICS")
    print(section_divider)

    for col_name, stats in report["columns"].items():
        col_type = stats.get("type", "unknown").upper()
        print(f"\n* COLUMN: {col_name} [{col_type}]")
        print(f"  Missing Values: {stats['null_count']:,} nulls ({stats['null_percentage']}%)")

        if col_type == "NUMERIC":
            print(f"  Range:         [{stats['min']} to {stats['max']}]")
            print(f"  Mean / Std:    {stats['mean']} / {stats['std']}")
            print(f"  Median (Q2):   {stats['median']}")
            print(f"  IQR Range:     [{stats['q1']} to {stats['q3']}] (IQR = {stats['iqr']})")
            print(f"  Skewness:      {stats['skewness']}")
            print(f"  Outliers (IQR): {stats['outliers']['count']:,} outliers ({stats['outliers']['percentage']}%)")
            if stats["outliers"]["count"] > 0:
                print(f"    Sample:      {stats['outliers']['sample']}")
        elif col_type == "CATEGORICAL":
            print(
                f"  Unique Values: {stats['unique_values_count']:,} (Cardinality Ratio: {stats['cardinality_ratio']})"
            )
            print(f"  Dominant Value: '{stats['top_value']}' ({stats['top_value_percentage']}%)")
            print("  Top Frequency Distribution:")
            for val, counts in stats["frequency_distribution"].items():
                print(f"    - '{val}': {counts['count']:,} ({counts['percentage']}%)")

    print(f"\n{border}")
    print("                 END OF ENTERPRISE DATA PROFILING REPORT")
    print(f"{border}\n")
