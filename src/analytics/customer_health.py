"""
Enterprise Customer Health Scoring Engine
============================================
Calculates a standardized customer health score on a 0–100 scale by
evaluating service friction, platform engagement, purchase momentum,
and relationship tenure. It provides customer health grading, risk tiering,
distribution metrics, and customer success playbook recommendations.

Author: Principal Customer Success Architect
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.core.exceptions import KPICalculationError
from src.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# DEFAULT CONFIGURABLE WEIGHTS
# ---------------------------------------------------------------------------
# Balance weights across four independent customer dimensions:
# - Service Quality: 35% (heavy indicator of immediate churn/retention risk)
# - Purchase Momentum: 25% (tracks recent buying frequency vs. inactivity)
# - App Engagement: 20% (browsing intensity and touchpoint metrics)
# - Tenure/Loyalty: 20% (historical customer lifetime value stability)
# ---------------------------------------------------------------------------
DEFAULT_HEALTH_WEIGHTS: Dict[str, float] = {
    "service_quality": 0.35,
    "purchase_momentum": 0.25,
    "app_engagement": 0.20,
    "tenure_loyalty": 0.20,
}

# ---------------------------------------------------------------------------
# CUSTOMER SUCCESS PLAYBOOK & RECOMMENDATIONS
# ---------------------------------------------------------------------------
# Strategic recommendations for Customer Success teams based on health grades.
# ---------------------------------------------------------------------------
HEALTH_PLAYBOOK: Dict[str, Dict[str, str]] = {
    "Excellent": {
        "risk_level": "Low Risk",
        "description": "Premium brand advocates. Highly satisfied with consistent purchasing activity.",
        "action": "Enroll in VIP referral program. Request product review quotes and offer loyalty tier upgrades.",
    },
    "Healthy": {
        "risk_level": "Low Risk",
        "description": "Engaged and satisfied customer base with regular purchases.",
        "action": "Trigger new category recommendation ads and offer bundle discounts to increase LTV.",
    },
    "Stable": {
        "risk_level": "Medium Risk",
        "description": "Customers showing flat transactional activity. Average satisfaction.",
        "action": "Send engagement surveys, recommend popular trending products, and offer small cashback incentives.",
    },
    "Warning": {
        "risk_level": "High Risk",
        "description": "Customers showing signs of friction or dropping purchase frequencies.",
        "action": "Deliver high-value win-back coupons, target with reactive push notifications, and gather feedback.",
    },
    "Critical": {
        "risk_level": "Critical Risk",
        "description": "Severe customer friction (Complained and low CSAT) or prolonged inactivity.",
        "action": "Activate emergency recovery protocol. Route directly to senior support teams for resolution.",
    },
}


# =============================================================================
# INDEPENDENT COMPONENT SCORING & NORMALIZATION
# =============================================================================


def calculate_service_quality_score(df: pd.DataFrame) -> pd.Series:
    """Calculates the normalized Service Quality Score (SQS) on a 0-100 scale.

    Combines self-reported SatisfactionScore and Complain status.
    - Base CSAT: SatisfactionScore mapped to a scale of 20 to 100.
    - Complaint Penalty: If Complain == 1, applies a penalty (-40 points).
    - Result is clipped between 0 and 100.

    Args:
        df: Customer DataFrame containing 'SatisfactionScore' and 'Complain'.

    Returns:
        Pandas Series containing SQS values.
    """
    if "SatisfactionScore" not in df.columns or "Complain" not in df.columns:
        raise KPICalculationError("Missing required columns for Service Quality Score calculation.")

    # Base CSAT score normalized: 1 -> 20, 2 -> 40, 3 -> 60, 4 -> 80, 5 -> 100
    base_csat = (df["SatisfactionScore"] / 5.0) * 100.0

    # Apply -40 penalty for active complaints
    complaint_penalty = df["Complain"] * 40.0

    sqs = base_csat - complaint_penalty
    return sqs.clip(lower=0.0, upper=100.0)


def calculate_purchase_momentum_score(df: pd.DataFrame) -> pd.Series:
    """Calculates the normalized Purchase Momentum Score (PMS) on a 0-100 scale.

    Uses DaySinceLastOrder as an inverse metric of purchasing recency.
    - An interval of 0 days since order yields a score of 100.
    - An interval of 30 or more days since last order yields a score of 0.

    Args:
        df: Customer DataFrame containing 'DaySinceLastOrder'.

    Returns:
        Pandas Series containing PMS values.
    """
    if "DaySinceLastOrder" not in df.columns:
        raise KPICalculationError("Missing required column 'DaySinceLastOrder' for Purchase Momentum.")

    # Linear decay score: 100 at 0 days, decaying to 0 at 30 days or more
    pms = (1.0 - (df["DaySinceLastOrder"] / 30.0)) * 100.0
    return pms.clip(lower=0.0, upper=100.0)


def calculate_app_engagement_score(df: pd.DataFrame) -> pd.Series:
    """Calculates the normalized App Engagement Score (AES) on a 0-100 scale.

    Maps AppExposure (browsing hours * registered devices) into a normalized scale.
    - Exposure metrics are capped at a maximum threshold of 20.
    - Values are scaled proportionally: (AppExposure / 20) * 100.

    Args:
        df: Customer DataFrame containing 'AppExposure'.

    Returns:
        Pandas Series containing AES values.
    """
    if "AppExposure" not in df.columns:
        raise KPICalculationError("Missing required column 'AppExposure' for App Engagement.")

    # Cap exposure value at 20.0 to prevent outlier distortion, then map to 0-100
    aes = (df["AppExposure"] / 20.0) * 100.0
    return aes.clip(lower=0.0, upper=100.0)


def calculate_tenure_loyalty_score(df: pd.DataFrame) -> pd.Series:
    """Calculates the normalized Relationship Loyalty Score (RLS) on a 0-100 scale.

    Maps Customer Tenure into a normalized scale where 48 months or more
    (4 years+) represents the maximum loyalty tier score of 100.

    Args:
        df: Customer DataFrame containing 'Tenure'.

    Returns:
        Pandas Series containing RLS values.
    """
    if "Tenure" not in df.columns:
        raise KPICalculationError("Missing required column 'Tenure' for Tenure Loyalty.")

    # Map tenure to 0-100, capping at 48 months (4 years+)
    rls = (df["Tenure"] / 48.0) * 100.0
    return rls.clip(lower=0.0, upper=100.0)


# =============================================================================
# PRIMARY ENGINE ORCHESTRATOR
# =============================================================================


def compute_customer_health(
    df: pd.DataFrame, weights: Optional[Dict[str, float]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Calculates customer-level health scores, grades, and risk tiers.

    Orchestrates the calculation of normalized components, combines them using
    configured weights, classifies customers into health grades, and compiles
    a structured quality audit metadata dictionary.

    Args:
        df: Enriched customer DataFrame.
        weights: Optional dictionary containing custom weights.
                 Defaults to DEFAULT_HEALTH_WEIGHTS.

    Returns:
        A tuple containing:
            - DataFrame enriched with CustomerHealthScore, Health_Category,
              Health_Grade, and Risk_Level columns.
            - A dictionary detailing segment sizes, stats, and recommendations.

    Raises:
        KPICalculationError: If calculations or weight configurations fail.
    """
    logger.info("Initializing customer health score computation.")

    df_health = df.copy()
    w = weights or DEFAULT_HEALTH_WEIGHTS

    # Validate weight sum is equal to 1.0 (approx)
    weight_sum = sum(w.values())
    if not (0.99 <= weight_sum <= 1.01):
        raise KPICalculationError(f"Customer health weights must sum to 1.0 (currently sum to {weight_sum}).")

    # 1. Calculate Component Scores
    df_health["Service_Quality_Score"] = calculate_service_quality_score(df_health)
    df_health["Purchase_Momentum_Score"] = calculate_purchase_momentum_score(df_health)
    df_health["App_Engagement_Score"] = calculate_app_engagement_score(df_health)
    df_health["Tenure_Loyalty_Score"] = calculate_tenure_loyalty_score(df_health)

    # 2. Compute Weighted Health Score
    df_health["CustomerHealthScore"] = (
        (df_health["Service_Quality_Score"] * w["service_quality"])
        + (df_health["Purchase_Momentum_Score"] * w["purchase_momentum"])
        + (df_health["App_Engagement_Score"] * w["app_engagement"])
        + (df_health["Tenure_Loyalty_Score"] * w["tenure_loyalty"])
    )

    # Round health scores for consistency
    df_health["CustomerHealthScore"] = df_health["CustomerHealthScore"].round(2)

    # 3. Map Health Grades, Categories, and Risk Levels
    # Define thresholds
    bins = [-0.1, 39.99, 59.99, 74.99, 89.99, 100.1]
    categories = ["Critical", "Warning", "Stable", "Healthy", "Excellent"]
    grades = ["F", "D", "C", "B", "A"]
    risks = ["Critical Risk", "High Risk", "Medium Risk", "Low Risk", "Low Risk"]

    df_health["Health_Category"] = pd.cut(df_health["CustomerHealthScore"], bins=bins, labels=categories)
    df_health["Health_Grade"] = pd.cut(df_health["CustomerHealthScore"], bins=bins, labels=grades)
    df_health["Risk_Level"] = pd.cut(df_health["CustomerHealthScore"], bins=bins, labels=risks, ordered=False)

    # Convert categories to standard string types to prevent serialization issues
    df_health["Health_Category"] = df_health["Health_Category"].astype(str)
    df_health["Health_Grade"] = df_health["Health_Grade"].astype(str)
    df_health["Risk_Level"] = df_health["Risk_Level"].astype(str)

    # Compile health distribution statistics
    total_customers = len(df_health)
    avg_health_score = float(df_health["CustomerHealthScore"].mean())

    category_counts = df_health["Health_Category"].value_counts()

    # Calculate key percentage boundaries
    healthy_pct = float(((df_health["CustomerHealthScore"] >= 75.0).sum() / total_customers) * 100)
    critical_pct = float(((df_health["CustomerHealthScore"] < 40.0).sum() / total_customers) * 100)

    category_stats = {}
    for cat in ["Excellent", "Healthy", "Stable", "Warning", "Critical"]:
        count = int(category_counts.get(cat, 0))
        pct = (count / total_customers) * 100 if total_customers > 0 else 0.0

        # Calculate metric averages for this category
        cat_df = df_health[df_health["Health_Category"] == cat]
        avg_csat = float(cat_df["SatisfactionScore"].mean()) if count > 0 else 0.0
        avg_complaints = float(cat_df["Complain"].mean() * 100) if count > 0 else 0.0
        avg_recency = float(cat_df["DaySinceLastOrder"].mean()) if count > 0 else 0.0
        avg_tenure = float(cat_df["Tenure"].mean()) if count > 0 else 0.0
        churn_rate = float(cat_df["Churn"].mean() * 100) if count > 0 else 0.0

        playbook = HEALTH_PLAYBOOK.get(cat, {"risk_level": "Unknown", "description": "", "action": ""})

        category_stats[cat] = {
            "customer_count": count,
            "percentage": round(pct, 2),
            "churn_rate_pct": round(churn_rate, 2),
            "averages": {
                "csat": round(avg_csat, 2),
                "complaint_rate_pct": round(avg_complaints, 2),
                "recency_days": round(avg_recency, 1),
                "tenure_months": round(avg_tenure, 1),
            },
            "risk_level": playbook["risk_level"],
            "description": playbook["description"],
            "playbook_action": playbook["action"],
        }

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_customers_evaluated": total_customers,
        "average_health_score": round(avg_health_score, 2),
        "healthy_percentage": round(healthy_pct, 2),
        "critical_percentage": round(critical_pct, 2),
        "applied_weights": w,
        "health_categories_summary": category_stats,
    }

    logger.info(f"Customer health scoring completed. Average Health Score: {avg_health_score:.2f}/100.")
    return df_health, metadata


def print_customer_health_report(metadata: Dict[str, Any]) -> None:
    """Prints a professional, human-readable Customer Health Report to the console.

    Args:
        metadata: Dict containing health scoring metadata produced by compute_customer_health().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CUSTOMER HEALTH SUMMARY REPORT")
    print(border)
    print(f"Timestamp (UTC):       {metadata['execution_timestamp']}")
    print(f"Total Base Evaluated:  {metadata['total_customers_evaluated']:,} customers")
    print(f"AVERAGE HEALTH SCORE:  {metadata['average_health_score']}/100")
    print(f"Healthy Customer Ratio: {metadata['healthy_percentage']}% (Score >= 75)")
    print(f"Critical Customer Ratio: {metadata['critical_percentage']}% (Score < 40)")
    print(border)

    print("\nHEALTH CATEGORIES DISTRIBUTION & PLAYBOOKS")
    print(section_divider)

    summary = metadata["health_categories_summary"]
    grades_map = {
        "Excellent": "A",
        "Healthy": "B",
        "Stable": "C",
        "Warning": "D",
        "Critical": "F"
    }

    # Sort categories to print in order of health level (Excellent -> Critical)
    for cat in ["Excellent", "Healthy", "Stable", "Warning", "Critical"]:
        info = summary.get(cat, {})
        if not info:
            continue
        print(
            f"\n* CATEGORY: {cat} [Grade: {grades_map.get(cat, cat[0])}] | {info['risk_level']}"
        )
        print(f"  Size:         {info['customer_count']:,} customers ({info['percentage']}%)")
        print(f"  Churn Rate:   {info['churn_rate_pct']}% churn within segment")
        print(
            f"  Profile:      CSAT={info['averages']['csat']}/5.0 | "
            f"Complaints={info['averages']['complaint_rate_pct']}% | "
            f"Recency={info['averages']['recency_days']} days | "
            f"Tenure={info['averages']['tenure_months']} months"
        )
        print(f"  Playbook:     {info['playbook_action']}")

    print(f"\n{border}")
    print("                 END OF CUSTOMER HEALTH SUMMARY REPORT")
    print(f"{border}\n")
