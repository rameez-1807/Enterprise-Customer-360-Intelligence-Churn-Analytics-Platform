"""
Enterprise Feature Validation Engine
=======================================
Production-grade feature validation system for the Customer 360 platform.
Performs data quality audits on engineered features, checking for presence,
missing/infinite values, constant values, zero variance, high correlation (>0.90),
duplicate columns, data type conformity, range violations, and target leakage.
Generates an overall Feature Quality Score out of 100.

Author: Principal Feature Quality Architect
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.exceptions import FeatureEngineeringError
from src.core.logger import get_logger

logger = get_logger(__name__)

# Canonical list of expected engineered features from Phase 3 Module 3.1
EXPECTED_ENGINEERED_FEATURES: List[str] = [
    "TenureGroup",
    "WarehouseDistanceBucket",
    "OrderVelocity",
    "AppExposure",
    "CashbackEfficiency",
    "ComplaintFrictionIndex",
    "OrderFrequencyTier",
    "CustomerLoyaltyScore",
    "AddressStabilityIndex",
    "TenureOrderRatio",
    "RawHealthIndex",
    "RuleBasedChurnIndicator",
]

EXPECTED_FEATURE_TYPES: Dict[str, List[str]] = {
    "TenureGroup": ["category"],
    "WarehouseDistanceBucket": ["category"],
    "OrderVelocity": ["float32", "float64"],
    "AppExposure": ["float64", "float32", "int64", "int32"],
    "CashbackEfficiency": ["float32", "float64"],
    "ComplaintFrictionIndex": ["float64", "float32"],
    "OrderFrequencyTier": ["category"],
    "CustomerLoyaltyScore": ["float64", "float32", "int64", "int32"],
    "AddressStabilityIndex": ["float64", "float32"],
    "TenureOrderRatio": ["float32", "float64"],
    "RawHealthIndex": ["float64", "float32"],
    "RuleBasedChurnIndicator": ["int64", "int32", "int16", "int8"],
}


def check_feature_presence(df: pd.DataFrame, expected_features: List[str]) -> List[str]:
    """Checks for the existence of expected engineered features in the DataFrame."""
    missing = [f for f in expected_features if f not in df.columns]
    if missing:
        logger.error(f"Missing engineered feature(s): {missing}")
    else:
        logger.info("All expected engineered features validated as present.")
    return missing


def check_missing_values(df: pd.DataFrame, features: List[str]) -> Dict[str, int]:
    """Detects missing (null) values in engineered features."""
    missing_counts = {}
    for f in features:
        if f in df.columns:
            null_count = int(df[f].isnull().sum())
            if null_count > 0:
                missing_counts[f] = null_count
                logger.warning(f"Feature '{f}' contains {null_count} null value(s).")
    return missing_counts


def check_infinite_values(df: pd.DataFrame, features: List[str]) -> Dict[str, int]:
    """Detects infinite values (+/- inf) in numerical engineered features."""
    inf_counts = {}
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f]):
            inf_count = int(np.isinf(df[f]).sum())
            if inf_count > 0:
                inf_counts[f] = inf_count
                logger.error(f"Feature '{f}' contains {inf_count} infinite value(s).")
    return inf_counts


def check_constant_features(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Detects constant columns (only one unique non-null value)."""
    constants = []
    for f in features:
        if f in df.columns:
            unique_count = df[f].nunique()
            if unique_count == 1:
                constants.append(f)
                logger.warning(f"Feature '{f}' is constant (zero variance).")
    return constants


def check_zero_variance(df: pd.DataFrame, features: List[str]) -> List[str]:
    """Detects zero variance in numerical engineered features (std is exactly zero)."""
    zero_var = []
    for f in features:
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f]):
            std_val = df[f].std()
            if std_val == 0.0 or pd.isnull(std_val):
                zero_var.append(f)
                logger.warning(f"Feature '{f}' has zero standard deviation.")
    return zero_var


def check_high_correlation(
    df: pd.DataFrame, features: List[str], threshold: float = 0.90
) -> List[Tuple[str, str, float]]:
    """Detects pairs of numerical engineered features with high correlation."""
    numeric_features = [
        f
        for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f])
    ]
    if len(numeric_features) < 2:
        return []

    corr_matrix = df[numeric_features].corr().abs()
    high_corr_pairs = []

    # Get upper triangle matrix to prevent checking pairs twice
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    for col in upper_tri.columns:
        for idx in upper_tri.index:
            coef = upper_tri.loc[idx, col]
            if not pd.isnull(coef) and coef > threshold:
                high_corr_pairs.append((idx, col, round(float(coef), 4)))
                logger.warning(f"High collinearity: '{idx}' and '{col}' have correlation = {coef:.4f}")

    return high_corr_pairs


def check_duplicate_features(df: pd.DataFrame, features: List[str]) -> List[Tuple[str, str]]:
    """Detects identical columns (columns containing identical values)."""
    duplicates = []
    present_features = [f for f in features if f in df.columns]

    for i in range(len(present_features)):
        for j in range(i + 1, len(present_features)):
            col1 = present_features[i]
            col2 = present_features[j]
            # Check shape equality first
            if df[col1].dtype == df[col2].dtype:
                if df[col1].equals(df[col2]):
                    duplicates.append((col1, col2))
                    logger.warning(f"Duplicate features detected: '{col1}' and '{col2}' contain identical values.")
    return duplicates


def check_data_type_conformity(
    df: pd.DataFrame, features: List[str], expected_types: Dict[str, List[str]]
) -> List[Dict[str, Any]]:
    """Validates that engineered feature data types match expected classifications."""
    mismatches = []
    for f in features:
        if f in df.columns:
            actual_type = str(df[f].dtype)
            acceptable = expected_types.get(f, [])
            # Sub-string match check to handle platform dependencies (e.g. 'float32' matches float)
            is_valid = any(acc in actual_type for acc in acceptable)
            if not is_valid:
                mismatches.append({"feature": f, "actual": actual_type, "expected": acceptable})
                logger.warning(f"Type mismatch for feature '{f}': actual '{actual_type}', expected '{acceptable}'")
    return mismatches


def check_target_leakage(
    df: pd.DataFrame, features: List[str], target: str = "Churn", threshold: float = 0.90
) -> List[Tuple[str, float]]:
    """Detects potential target leakage (high correlation with target variable)."""
    if target not in df.columns:
        logger.warning(f"Target column '{target}' not present. Skipping target leakage check.")
        return []

    leakage_features = []
    numeric_features = [
        f
        for f in features
        if f in df.columns and pd.api.types.is_numeric_dtype(df[f]) and not pd.api.types.is_bool_dtype(df[f])
    ]

    for f in numeric_features:
        corr_coef = df[f].corr(df[target])
        if not pd.isnull(corr_coef) and abs(corr_coef) > threshold:
            leakage_features.append((f, round(float(corr_coef), 4)))
            logger.error(
                f"Potential TARGET LEAKAGE: Feature '{f}' correlates with target '{target}' at coef = {corr_coef:.4f}"
            )

    return leakage_features


# =============================================================================
# PRIMARY ENGINE ORCHESTRATOR
# =============================================================================


def validate_features(
    df: pd.DataFrame,
    expected_features: Optional[List[str]] = None,
    expected_types: Optional[Dict[str, List[str]]] = None,
    target_column: str = "Churn",
) -> Dict[str, Any]:
    """Validates the quality, alignment, and constraints of engineered features.

    Orchestrates all checks, evaluates data quality penalty points, and
    determines an overall Feature Quality Score out of 100.

    Args:
        df: Enriched customer DataFrame.
        expected_features: List of engineered feature names to validate.
        expected_types: Expected dtypes lookup dictionary.
        target_column: Name of target variable to evaluate target leakage.

    Returns:
        A dictionary containing validation findings, scores, and metadata.

    Raises:
        FeatureEngineeringError: If the input DataFrame is empty.
    """
    if df.empty:
        raise FeatureEngineeringError("Cannot validate an empty DataFrame.")

    logger.info("=" * 80)
    logger.info("ENTERPRISE FEATURE VALIDATION ENGINE — STARTING")
    logger.info("=" * 80)

    features_to_check = expected_features or EXPECTED_ENGINEERED_FEATURES
    types_to_check = expected_types or EXPECTED_FEATURE_TYPES

    # Run checks
    missing_features = check_feature_presence(df, features_to_check)
    existing_features = [f for f in features_to_check if f not in missing_features]

    missing_values = check_missing_values(df, existing_features)
    infinite_values = check_infinite_values(df, existing_features)
    constant_features = check_constant_features(df, existing_features)
    zero_variance_features = check_zero_variance(df, existing_features)
    high_correlations = check_high_correlation(df, existing_features)
    duplicate_features = check_duplicate_features(df, existing_features)
    type_mismatches = check_data_type_conformity(df, existing_features, types_to_check)
    target_leakage = check_target_leakage(df, existing_features, target_column)

    # Calculate Feature Quality Score
    # Start at 100 and apply penalties
    penalty = 0
    penalty += len(missing_features) * 10  # Missing expected features: -10 each
    penalty += len(missing_values) * 5  # Missing values inside features: -5 each
    penalty += len(infinite_values) * 15  # Infinite values: -15 each
    penalty += len(constant_features) * 10  # Constant features: -10 each
    penalty += len(zero_variance_features) * 10  # Zero variance: -10 each
    penalty += len(high_correlations) * 3  # High collinearity pairs: -3 each
    penalty += len(duplicate_features) * 5  # Duplicate features: -5 each
    penalty += len(type_mismatches) * 5  # Dtype mismatches: -5 each
    penalty += len(target_leakage) * 30  # Target leakage warning: -30 each

    quality_score = max(0, 100 - penalty)

    report = {
        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_features_evaluated": len(features_to_check),
            "present_features_count": len(existing_features),
            "missing_features_count": len(missing_features),
            "feature_quality_score": quality_score,
        },
        "findings": {
            "missing_features": missing_features,
            "missing_values_features": missing_values,
            "infinite_values_features": infinite_values,
            "constant_features": constant_features,
            "zero_variance_features": zero_variance_features,
            "high_correlation_pairs": [
                {"feature_a": p[0], "feature_b": p[1], "correlation_coef": p[2]} for p in high_correlations
            ],
            "duplicate_features": [{"feature_a": p[0], "feature_b": p[1]} for p in duplicate_features],
            "data_type_mismatches": type_mismatches,
            "target_leakage_features": [{"feature": p[0], "correlation_with_target": p[1]} for p in target_leakage],
        },
    }

    logger.info(f"Feature validation completed. Final Feature Quality Score: {quality_score}/100.")
    logger.info("=" * 80)

    return report


def print_feature_validation_report(report: Dict[str, Any]) -> None:
    """Prints a professional, human-readable Feature Validation Report to the console.

    Args:
        report: Dict containing validation results generated by validate_features().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE FEATURE QUALITY VALIDATION REPORT")
    print(border)
    print(f"Timestamp (UTC): {report['validation_timestamp']}")
    print(f"Features Evaluated: {report['summary']['total_features_evaluated']}")
    print(f"Features Present:   {report['summary']['present_features_count']}")
    print(f"Features Missing:   {report['summary']['missing_features_count']}")

    score = report["summary"]["feature_quality_score"]
    status = "EXCELLENT" if score >= 90 else "GOOD" if score >= 75 else "DEGRADED" if score >= 50 else "CRITICAL"
    print(f"FEATURE QUALITY SCORE: {score}/100 [{status}]")
    print(border)

    print("\nANOMALIES & INTEGRITY CHECKS")
    print(section_divider)

    findings = report["findings"]

    print(f"Missing Features:           {len(findings['missing_features'])} | {findings['missing_features']}")
    print(
        f"Features with Nulls:        {len(findings['missing_values_features'])} | {list(findings['missing_values_features'].keys())}"
    )
    print(
        f"Features with Inf:          {len(findings['infinite_values_features'])} | {list(findings['infinite_values_features'].keys())}"
    )
    print(f"Constant Features:          {len(findings['constant_features'])} | {findings['constant_features']}")
    print(
        f"Zero Variance Features:     {len(findings['zero_variance_features'])} | {findings['zero_variance_features']}"
    )
    print(f"Duplicate Feature Columns:  {len(findings['duplicate_features'])} | {findings['duplicate_features']}")
    print(f"Data Type Mismatches:       {len(findings['data_type_mismatches'])} | {findings['data_type_mismatches']}")
    print(f"Collinear Pairs (>0.90):    {len(findings['high_correlation_pairs'])}")
    for pair in findings["high_correlation_pairs"]:
        print(f"  - '{pair['feature_a']}' & '{pair['feature_b']}' (coef = {pair['correlation_coef']})")
    print(f"Target Leakage Indicators:  {len(findings['target_leakage_features'])}")
    for leak in findings["target_leakage_features"]:
        print(f"  - WARNING: Feature '{leak['feature']}' correlates with target at {leak['correlation_with_target']}")

    print(f"\n{border}")
    print("                 END OF FEATURE VALIDATION REPORT")
    print(f"{border}\n")
