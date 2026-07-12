"""
Enterprise Feature Engineering Engine
========================================
Production-grade domain feature construction module for the Customer 360 platform.
Builds business-ready derived features from the cleaned customer dataset,
such as tenure groups, distance buckets, app exposure, cashback efficiency,
complaint friction indices, order velocity, address stability indices, and
rule-based churn risk indicators.

All original columns are preserved. This module strictly constructs derived
analytical variables and outputs a structured feature engineering metadata report.

Author: Principal Feature Engineering Architect
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.core.config_loader import load_model_config
from src.core.exceptions import FeatureEngineeringError
from src.core.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# FEATURE GENERATION FUNCTIONS (INDEPENDENT)
# =============================================================================


def generate_tenure_groups(df: pd.DataFrame) -> pd.Series:
    """Classifies customer tenure into defined cohort bins.

    Tenure groups are defined by standard lifecycle boundaries:
    0-6m (new users), 6-12m (onboarding users), 12-24m (retained users),
    24-48m (loyal users), 48m+ (VIP veterans).

    Args:
        df: Cleaned customer DataFrame containing 'Tenure'.

    Returns:
        Pandas Series containing categorical tenure labels.
    """
    if "Tenure" not in df.columns:
        raise FeatureEngineeringError("Required column 'Tenure' is missing from DataFrame.")

    config = load_model_config()
    bins = config.get("feature_engineering", {}).get("tenure_bins", [0, 6, 12, 24, 48, 120])
    labels = config.get("feature_engineering", {}).get("tenure_labels", ["0-6m", "6-12m", "12-24m", "24-48m", "48m+"])

    # Cut operation with left edge excluded, right edge included.
    # Clip Tenure max to handle edge values above the max bin threshold.
    capped_tenure = df["Tenure"].clip(upper=bins[-1] - 0.1)

    return pd.cut(capped_tenure, bins=bins, labels=labels, include_lowest=True)


def generate_warehouse_distance_buckets(df: pd.DataFrame) -> pd.Series:
    """Bins warehouse-to-home distances into logical geographic tiers.

    Tiers:
    - 'Near' (<= 10): Fast delivery, lower logistics costs.
    - 'Moderate' (10-30): Average shipping times.
    - 'Far' (> 30): Long shipping times, higher probability of shipping friction.

    Args:
        df: Cleaned customer DataFrame containing 'WarehouseToHome'.

    Returns:
        Pandas Series containing categorical distance tier labels.
    """
    if "WarehouseToHome" not in df.columns:
        raise FeatureEngineeringError("Required column 'WarehouseToHome' is missing.")

    bins = [0, 10, 30, float("inf")]
    labels = ["Near", "Moderate", "Far"]

    return pd.cut(df["WarehouseToHome"], bins=bins, labels=labels, include_lowest=True)


def generate_order_velocity(df: pd.DataFrame) -> pd.Series:
    """Calculates order momentum or velocity.

    Represents order count normalized by the number of days since last purchase,
    capturing purchase frequency velocity.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing order velocity scores.
    """
    if "OrderCount" not in df.columns or "DaySinceLastOrder" not in df.columns:
        raise FeatureEngineeringError("Missing 'OrderCount' or 'DaySinceLastOrder' columns.")

    # Avoid zero division
    return df["OrderCount"] / (df["DaySinceLastOrder"] + 1.0)


def generate_app_exposure(df: pd.DataFrame) -> pd.Series:
    """Calculates overall mobile app interaction exposure.

    exposure score is a product of average app hours spent and the number of devices
    registered on the account.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing app exposure scores.
    """
    if "HourSpendOnApp" not in df.columns or "NumberOfDeviceRegistered" not in df.columns:
        raise FeatureEngineeringError("Missing 'HourSpendOnApp' or 'NumberOfDeviceRegistered' columns.")

    return df["HourSpendOnApp"] * df["NumberOfDeviceRegistered"]


def generate_cashback_efficiency(df: pd.DataFrame) -> pd.Series:
    """Calculates cashback incentive return per placed order.

    Identifies customers who maximize cashback returns relative to total
    order counts (potential high incentive dependency).

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing cashback efficiency scores.
    """
    if "CashbackAmount" not in df.columns or "OrderCount" not in df.columns:
        raise FeatureEngineeringError("Missing 'CashbackAmount' or 'OrderCount' columns.")

    return df["CashbackAmount"] / (df["OrderCount"] + 1.0)


def generate_complaint_friction(df: pd.DataFrame) -> pd.Series:
    """Computes complaint friction index.

    Calculated as Complain / SatisfactionScore. Combines complaint logging
    and low satisfaction to highlight high-risk customers experiencing operational friction.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing complaint friction index.
    """
    if "Complain" not in df.columns or "SatisfactionScore" not in df.columns:
        raise FeatureEngineeringError("Missing 'Complain' or 'SatisfactionScore' columns.")

    # Higher scores represent higher friction (e.g. 1 / 1.0 = 1.0 vs 1 / 5.0 = 0.2)
    # Cast SatisfactionScore to float to prevent integer division
    return df["Complain"] / df["SatisfactionScore"].astype(float)


def generate_order_frequency_tier(df: pd.DataFrame) -> pd.Series:
    """Bins order count into purchase frequency tiers.

    Tiers:
    - 'Low' (<= 2 orders)
    - 'Medium' (3-5 orders)
    - 'High' (> 5 orders)

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing categorical frequency labels.
    """
    if "OrderCount" not in df.columns:
        raise FeatureEngineeringError("Required column 'OrderCount' is missing.")

    bins = [0, 2, 5, float("inf")]
    labels = ["Low", "Medium", "High"]

    return pd.cut(df["OrderCount"], bins=bins, labels=labels, include_lowest=True)


def generate_customer_loyalty_score(df: pd.DataFrame) -> pd.Series:
    """Computes tenure-weighted satisfaction index.

    A product of customer tenure and their satisfaction rating, indicating
    brand attachment.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing loyalty score values.
    """
    if "Tenure" not in df.columns or "SatisfactionScore" not in df.columns:
        raise FeatureEngineeringError("Missing 'Tenure' or 'SatisfactionScore' columns.")

    return df["Tenure"] * df["SatisfactionScore"]


def generate_address_churn_index(df: pd.DataFrame) -> pd.Series:
    """Computes address stability index.

    The ratio of distinct delivery addresses relative to account tenure.
    High ratios suggest transient usage patterns, or shipping address rotation.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing stability index values.
    """
    if "NumberOfAddress" not in df.columns or "Tenure" not in df.columns:
        raise FeatureEngineeringError("Missing 'NumberOfAddress' or 'Tenure' columns.")

    return df["NumberOfAddress"] / (df["Tenure"] + 1.0)


def generate_tenure_order_ratio(df: pd.DataFrame) -> pd.Series:
    """Computes monthly order velocity.

    Represents average number of orders placed per month of customer tenure.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing monthly order ratio.
    """
    if "OrderCount" not in df.columns or "Tenure" not in df.columns:
        raise FeatureEngineeringError("Missing 'OrderCount' or 'Tenure' columns.")

    return df["OrderCount"] / (df["Tenure"] + 1.0)


def generate_raw_health_index(df: pd.DataFrame) -> pd.Series:
    """Computes raw customer account health index.

    A weighted index mapping positive sentiment (SatisfactionScore), application
    exposure (HourSpendOnApp), and severe penalties for logged complaints (Complain).

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing raw health index values.
    """
    if "SatisfactionScore" not in df.columns or "Complain" not in df.columns or "HourSpendOnApp" not in df.columns:
        raise FeatureEngineeringError("Required columns for Health Index are missing.")

    # Formula: SatisfactionScore - 2.0 * Complain + (HourSpendOnApp / 5.0)
    return df["SatisfactionScore"] - (2.0 * df["Complain"]) + (df["HourSpendOnApp"] / 5.0)


def generate_rule_based_churn_indicator(df: pd.DataFrame) -> pd.Series:
    """Computes rule-based high Churn Risk binary indicator flags.

    Identifies customer profiles showing severe signs of churn risk based on
    heuristic logic (e.g. registered complaint combined with very low satisfaction score).

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        Pandas Series containing binary indicators (1 for high-risk, 0 otherwise).
    """
    if "Complain" not in df.columns or "SatisfactionScore" not in df.columns:
        raise FeatureEngineeringError("Required columns for churn heuristics are missing.")

    # High Risk rule: customer filed a complaint AND satisfaction score is <= 2
    high_risk_condition = (df["Complain"] == 1) & (df["SatisfactionScore"] <= 2)
    return high_risk_condition.astype(int)


# =============================================================================
# PRIMARY ENGINE ORCHESTRATOR
# =============================================================================


def build_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Orchestrates the construction of all domain analytical features.

    Executes all independent feature generator modules, appends them as new
    columns while leaving original input fields untouched, and compiles
    a structured metadata engineering manifest.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        A tuple containing:
            - The enriched DataFrame containing original and engineered features.
            - A dictionary listing all constructed features and their descriptions.

    Raises:
        FeatureEngineeringError: If the input DataFrame is empty or missing columns.
    """
    if df.empty:
        raise FeatureEngineeringError("Cannot build features from an empty DataFrame.")

    logger.info("=" * 80)
    logger.info("ENTERPRISE FEATURE ENGINEERING ENGINE — STARTING")
    logger.info("=" * 80)

    df_features = df.copy()

    # Feature definitions mapping for metadata construction
    feature_registry = {
        "TenureGroup": {
            "generator": generate_tenure_groups,
            "description": "Categorical cohort grouping based on account lifespan (months).",
        },
        "WarehouseDistanceBucket": {
            "generator": generate_warehouse_distance_buckets,
            "description": "Categorical distance bucket classifying warehouse proximity (Near/Moderate/Far).",
        },
        "OrderVelocity": {
            "generator": generate_order_velocity,
            "description": "Velocity ratio of order counts normalized by days since last purchase.",
        },
        "AppExposure": {
            "generator": generate_app_exposure,
            "description": "Composite index representing hours spent on app multiplied by registered devices.",
        },
        "CashbackEfficiency": {
            "generator": generate_cashback_efficiency,
            "description": "Cashback yield value received per placed order.",
        },
        "ComplaintFrictionIndex": {
            "generator": generate_complaint_friction,
            "description": "Ratio of complaint presence to satisfaction score (higher shows friction).",
        },
        "OrderFrequencyTier": {
            "generator": generate_order_frequency_tier,
            "description": "Categorical classification of total purchase count (Low/Medium/High).",
        },
        "CustomerLoyaltyScore": {
            "generator": generate_customer_loyalty_score,
            "description": "Composite value of account lifespan tenure multiplied by satisfaction score.",
        },
        "AddressStabilityIndex": {
            "generator": generate_address_churn_index,
            "description": "Address change frequency ratio calculated as registered addresses divided by tenure.",
        },
        "TenureOrderRatio": {
            "generator": generate_tenure_order_ratio,
            "description": "Order placement velocity calculated as order counts divided by tenure.",
        },
        "RawHealthIndex": {
            "generator": generate_raw_health_index,
            "description": "Composite scale combining satisfaction score, app utilization, and complaint penalties.",
        },
        "RuleBasedChurnIndicator": {
            "generator": generate_rule_based_churn_indicator,
            "description": "Heuristic binary indicator flag identifying high risk customers (Complain=1 & CSAT<=2).",
        },
    }

    metadata_features = {}

    for feature_name, config in feature_registry.items():
        try:
            logger.info(f"Generating feature: {feature_name}...")
            generator = config["generator"]
            if callable(generator):
                df_features[feature_name] = (
                    generator(df_clean=df_features)
                    if hasattr(generator, "__code__") and "df_clean" in generator.__code__.co_varnames
                    else generator(df_features)
                )

            # Record metadata
            metadata_features[feature_name] = {
                "dtype": str(df_features[feature_name].dtype),
                "description": config["description"],
            }
        except Exception as e:
            error_msg = f"Failed to construct feature '{feature_name}': {e}"
            logger.error(error_msg)
            raise FeatureEngineeringError(error_msg) from e

    metadata = {
        "engineering_timestamp": datetime.now(timezone.utc).isoformat(),
        "input_shape": df.shape,
        "output_shape": df_features.shape,
        "features_created_count": len(metadata_features),
        "feature_registry": metadata_features,
    }

    logger.info(f"Feature engineering pipeline completed. Created {len(metadata_features)} features.")
    logger.info("=" * 80)

    return df_features, metadata


def get_feature_list(metadata: Dict[str, Any]) -> List[str]:
    """Helper function to retrieve list of generated feature names from metadata.

    Args:
        metadata: Metadata dictionary produced by build_features().

    Returns:
        List of engineered feature names.
    """
    return list(metadata.get("feature_registry", {}).keys())
