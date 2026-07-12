"""
Enterprise Customer Segmentation Engine
==========================================
Provides a multi-dimensional customer segmentation module. Combines
raw customer characteristics, engineered features, and behavioral RFM scores
to classify the user base into standard business cohorts (e.g., Champions,
Discount Seekers, High Engagement, Complaint-Prone, At-Risk).

This module utilizes a Multi-Label Flag Tagging model, which is the standard
architecture in modern Enterprise CDPs (Customer Data Platforms). It allows
a customer to occupy multiple operational segments concurrently (e.g., a customer
can be both 'Mobile-First' and 'At-Risk').

Author: Principal Customer Analytics Consultant
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.core.exceptions import KPICalculationError
from src.core.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# COHORT STRATEGY MATRIX
# =============================================================================
# Encapsulates business definition, value, marketing and retention strategies,
# and executive recommendations for each customer segment.
# =============================================================================

SEGMENT_STRATEGY_MATRIX: Dict[str, Dict[str, str]] = {
    "Champions": {
        "definition": "Top RFM tier: high recency, high frequency, and high monetary proxy values.",
        "value": "Core revenue advocates; high repeat purchases and highest average lifetime value.",
        "marketing": "VIP tier invites, early-access campaigns, referral code rewards.",
        "retention": "Direct executive reach-outs, white-glove support, brand ambassador perks.",
        "recommendation": "Maximize customer advocacy. Deploy referral programs to acquire lookalike cohorts.",
    },
    "Loyal Customers": {
        "definition": "Active customers with high transaction frequencies and consistent loyalty index scores.",
        "value": "Stable revenue base; high brand affinity and regular transaction history.",
        "marketing": "Value cross-selling, upsell tiers, product reviews and loyalty milestone incentives.",
        "retention": "Exclusive loyalty points, tier rewards, and priority support routing.",
        "recommendation": "Transition to Champions. Offer upsell bundles and milestone progress alerts.",
    },
    "High Value Spenders": {
        "definition": "Customers in the top tier of cashback rewards (spent proxy > $220).",
        "value": "Highest margin contributions per order; high average order value (AOV).",
        "marketing": "Premium product category showcases, personalized shopping guides, high-ticket cross-selling.",
        "retention": "Free express shipping, priority customer support channels, customized packaging.",
        "recommendation": "Protect high margins. Set up automatic support overrides for any complaints logged by this cohort.",
    },
    "Growth Customers": {
        "definition": "Onboarding customers (tenure 6-12 months) who show consistent purchasing activity and no complaints.",
        "value": "Mid-tier potential; represents the next generation of loyal advocates.",
        "marketing": "Category expansion campaigns, recurring purchase reminders, educational app content.",
        "retention": "Satisfaction follow-up surveys, milestone anniversary coupons.",
        "recommendation": "Deepen category adoption. Suggest categories outside their primary purchase cluster.",
    },
    "Frequent Buyers": {
        "definition": "Customers with high purchase volumes (> 5 orders) regardless of tenure or monetary value.",
        "value": "High operational touchpoints; strong utility value for the platform.",
        "marketing": "Replenishment discounts, subscription program options, bulk purchase offers.",
        "retention": "Fast shipping guarantees, frictionless checkout flows.",
        "recommendation": "Convert to subscription services. Pitch automated replenishment programs.",
    },
    "Discount Seekers": {
        "definition": "Customers demonstrating high coupon usage (> 3 coupons) or high cashback dependency ratios.",
        "value": "Margin-sensitive; transactional transactions are highly dependent on promotions.",
        "marketing": "Flash discount sales, clearing out slow-moving inventory, targeted cashback campaigns.",
        "retention": "Affordable value-bundle promotions, low-tier discount newsletters.",
        "recommendation": "Prevent margin erosion. Limit direct discount offerings and test bundles with higher AOVs.",
    },
    "Mobile-First Users": {
        "definition": "Customers whose preferred login device is defined as Mobile Phone or Phone.",
        "value": "High digital accessibility; responsive to mobile push notifications and app alerts.",
        "marketing": "App-exclusive discounts, location-targeted alerts, mobile checkout specials.",
        "retention": "Biometric log-in support, app performance improvements, transactional SMS updates.",
        "recommendation": "Drive push-notification permissions. Deliver real-time cart recovery alerts.",
    },
    "High Engagement Users": {
        "definition": "Customers showing high application exposure metrics (Hours * Devices >= 15.0).",
        "value": "Strong brand mindshare; high browsing duration and active session history.",
        "marketing": "Social sharing incentives, product wish-list reminders, community updates.",
        "retention": "Seamless app UX, personalized homepage feeds, interactive UI rewards.",
        "recommendation": "Convert browsing duration to orders. Trigger exit-intent offers during high-browse sessions.",
    },
    "At Risk Customers": {
        "definition": "Customers flagged by predictive model scores or rule-based churn indicators.",
        "value": "Likely near-term churn; represents imminent monthly recurring revenue leakage.",
        "marketing": "High-value win-back offers, direct customer support follow-ups, product replacement benefits.",
        "retention": "Direct customer support calls, high discount values, direct escalation channels.",
        "recommendation": "Deploy defensive campaigns immediately. Prioritize high-value accounts for CRM win-back.",
    },
    "Dissatisfied Customers": {
        "definition": "Customers who gave low satisfaction ratings (CSAT score <= 2).",
        "value": "High churn risk and brand detractor potential.",
        "marketing": "Service recovery discounts, apology messages, feedback surveys.",
        "retention": "Escalation to senior customer support teams, rapid incident resolution, refund/credit offers.",
        "recommendation": "Execute service recovery campaigns. Route support cases to resolution teams immediately.",
    },
    "Complaint-Prone Customers": {
        "definition": "Customers who have logged complaints during the reporting period.",
        "value": "High customer support cost and potential brand detractor.",
        "marketing": "Frictionless return program highlights, direct support guides.",
        "retention": "Direct case resolution tracking, feedback loops, priority ticketing.",
        "recommendation": "Analyze root cause of complaints. Match complaints to logistics or product quality teams.",
    },
    "Inactive Customers": {
        "definition": "Customers with high days elapsed since last purchase (>= 10 days).",
        "value": "Dormant account; low near-term revenue generation.",
        "marketing": "Welcome back campaigns, trending products showcases, generic reactivate promotions.",
        "retention": "Low-cost email newsletters, basic reactivate discount alerts.",
        "recommendation": "Deploy automated reactivation sequences. Retain budget if no engagement within 90 days.",
    },
}


def segment_customers(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Segments the customer base using multi-label business rule flags.

    Combines cleaned attributes, engineered features, and RFM scores
    to tag every customer with boolean segment indicators.

    Args:
        df: Customer DataFrame containing clean attributes, engineered features,
            and RFM segment columns.

    Returns:
        A tuple containing:
            - DataFrame enriched with 'is_<segment>' boolean columns.
            - A dictionary containing cohort statistics and metadata.

    Raises:
        KPICalculationError: If critical validation columns are missing.
    """
    logger.info("Initializing multi-dimensional customer segmentation rules.")

    # Required columns checklist
    required_cols = [
        "CustomerID",
        "SatisfactionScore",
        "Complain",
        "DaySinceLastOrder",
        "OrderCount",
        "CashbackAmount",
        "PreferredLoginDevice",
        "CouponUsed",
        "RFM_Segment",
        "TenureGroup",
        "AppExposure",
        "RuleBasedChurnIndicator",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KPICalculationError(f"Missing required column '{col}' for segmentation.")

    df_segmented = df.copy()
    total_customers = len(df_segmented)

    # ---------------------------------------------------------------------------
    # DEFINE CORE MULTI-LABEL SEGMENTATION RULES
    # ---------------------------------------------------------------------------

    # 1. Champions
    df_segmented["is_champion"] = df_segmented["RFM_Segment"].isin(["Champions"]).astype(int)

    # 2. Loyal Customers
    df_segmented["is_loyal"] = df_segmented["RFM_Segment"].isin(["Loyal Customers"]).astype(int)

    # 3. High Value Spenders (monetary spent proxy >= 220)
    df_segmented["is_high_value"] = (df_segmented["CashbackAmount"] >= 220.0).astype(int)

    # 4. Growth Customers (tenure 6-12m AND no complaints)
    df_segmented["is_growth"] = ((df_segmented["TenureGroup"] == "6-12m") & (df_segmented["Complain"] == 0)).astype(int)

    # 5. Frequent Buyers (OrderCount >= 5)
    df_segmented["is_frequent_buyer"] = (df_segmented["OrderCount"] >= 5.0).astype(int)

    # 6. Discount Seekers (CouponUsed >= 3 OR CashbackAmount/OrderCount >= 60.0)
    # Using CashbackAmount as monetary proxy for cashback efficiency calculation
    cashback_eff = df_segmented["CashbackAmount"] / (df_segmented["OrderCount"] + 1.0)
    df_segmented["is_discount_seeker"] = ((df_segmented["CouponUsed"] >= 3.0) | (cashback_eff >= 60.0)).astype(int)

    # 7. Mobile-First Users
    df_segmented["is_mobile_first"] = df_segmented["PreferredLoginDevice"].isin(["Mobile Phone", "Phone"]).astype(int)

    # 8. High Engagement Users (AppExposure >= 15.0)
    df_segmented["is_high_engagement"] = (df_segmented["AppExposure"] >= 15.0).astype(int)

    # 9. At Risk Customers (RFM At Risk OR RuleBasedChurnIndicator == 1)
    df_segmented["is_at_risk"] = (
        df_segmented["RFM_Segment"].isin(["At Risk"]) | (df_segmented["RuleBasedChurnIndicator"] == 1)
    ).astype(int)

    # 10. Dissatisfied Customers (CSAT <= 2)
    df_segmented["is_dissatisfied"] = (df_segmented["SatisfactionScore"] <= 2).astype(int)

    # 11. Complaint-Prone Customers
    df_segmented["is_complaint_prone"] = (df_segmented["Complain"] == 1).astype(int)

    # 12. Inactive Customers (DaySinceLastOrder >= 10 days)
    df_segmented["is_inactive"] = (df_segmented["DaySinceLastOrder"] >= 10.0).astype(int)

    # ---------------------------------------------------------------------------
    # COMPILE SUMMARY STATISTICS & METADATA
    # ---------------------------------------------------------------------------

    segment_flag_mapping = {
        "Champions": "is_champion",
        "Loyal Customers": "is_loyal",
        "High Value Spenders": "is_high_value",
        "Growth Customers": "is_growth",
        "Frequent Buyers": "is_frequent_buyer",
        "Discount Seekers": "is_discount_seeker",
        "Mobile-First Users": "is_mobile_first",
        "High Engagement Users": "is_high_engagement",
        "At Risk Customers": "is_at_risk",
        "Dissatisfied Customers": "is_dissatisfied",
        "Complaint-Prone Customers": "is_complaint_prone",
        "Inactive Customers": "is_inactive",
    }

    cohort_summaries = {}

    for label, flag_col in segment_flag_mapping.items():
        count = int(df_segmented[flag_col].sum())
        percentage = (count / total_customers) * 100 if total_customers > 0 else 0.0

        # Calculate actual churn rate inside this segment to evaluate threat level
        churn_rate = float(df_segmented[df_segmented[flag_col] == 1]["Churn"].mean() * 100) if count > 0 else 0.0

        # Combine with strategic matrix
        strategy = SEGMENT_STRATEGY_MATRIX.get(label, {})

        cohort_summaries[label] = {
            "customer_count": count,
            "percentage": round(percentage, 2),
            "segment_churn_rate_pct": round(churn_rate, 2),
            "definition": strategy.get("definition", ""),
            "business_value": strategy.get("value", ""),
            "marketing_strategy": strategy.get("marketing", ""),
            "retention_strategy": strategy.get("retention", ""),
            "executive_recommendation": strategy.get("recommendation", ""),
        }

    metadata = {
        "segmentation_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_customers_evaluated": total_customers,
        "cohorts": cohort_summaries,
    }

    logger.info("Customer segmentation complete. Metadata generated.")
    return df_segmented, metadata


def print_executive_segmentation_summary(metadata: Dict[str, Any]) -> None:
    """Prints a formatted executive summary of customer segments to the console.

    Args:
        metadata: Dict containing segmentation metadata produced by segment_customers().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CUSTOMER SEGMENTATION EXECUTIVE SUMMARY")
    print(border)
    print(f"Timestamp (UTC):       {metadata['segmentation_timestamp']}")
    print(f"Total Base Evaluated:  {metadata['total_customers_evaluated']:,} customers")
    print(border)

    print("\nCOHORT PENETRATION & ACTION MATRIX")
    print(section_divider)

    cohorts = metadata["cohorts"]
    # Sort segments by customer count descending
    sorted_cohorts = sorted(cohorts.items(), key=lambda x: x[1]["customer_count"], reverse=True)

    for label, info in sorted_cohorts:
        print(f"\n* SEGMENT: {label}")
        print(f"  Penetration:  {info['customer_count']:,} customers ({info['percentage']}%)")
        print(f"  Churn Risk:   {info['segment_churn_rate_pct']}% churn rate within segment")
        print(f"  Description:  {info['definition']}")
        print(f"  Action Plan:  {info['executive_recommendation']}")

    print(f"\n{border}")
    print("                 END OF CUSTOMER SEGMENTATION SUMMARY")
    print(f"{border}\n")
