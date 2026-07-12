"""
Enterprise KPI Analytics Engine
=================================
Production-grade business intelligence and KPI engine for the Customer 360 platform.
Computes vectorized, reusable business metrics across five primary domains:
    1. Customer Base KPIs (Totals, Churn, Retention)
    2. Customer Engagement KPIs (Satisfaction, App Usage, Cashback, Order Count)
    3. Behavioral Indicators (Engagement %, Loyalty %, Mobile-First %, Discount Seekers %)
    4. Segmentation Metrics (Champions %, At Risk %, High Value %)
    5. Business Health Indices (Health Index, Loyalty Index, Satisfaction Index)

This engine compiles clean reporting dictionaries and executive recommendations,
serving as the backend for the presentation and BI dashboard layers.

Author: Principal BI Engineer
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List

import pandas as pd

from src.core.exceptions import KPICalculationError
from src.core.logger import get_logger

logger = get_logger(__name__)

# =============================================================================
# ENTERPRISE KPI DICTIONARY
# =============================================================================
# Central repository mapping KPI keys to descriptive business terminology.
# =============================================================================

KPI_DICTIONARY: Dict[str, Dict[str, str]] = {
    # --- Customer KPIs ---
    "total_customers": {
        "title": "Total Customer Base",
        "description": "Total count of unique customer IDs registered in the analytical store.",
    },
    "active_customers": {
        "title": "Active Customer Count",
        "description": "Total count of customers classified as active (Churn = 0).",
    },
    "churned_customers": {
        "title": "Churned Customer Count",
        "description": "Total count of customers classified as churned (Churn = 1).",
    },
    "churn_rate_pct": {
        "title": "Customer Churn Rate",
        "description": "The percentage of the customer base lost during the reporting period.",
    },
    "retention_rate_pct": {
        "title": "Customer Retention Rate",
        "description": "The percentage of the customer base retained during the reporting period.",
    },
    # --- Engagement KPIs ---
    "avg_satisfaction_score": {
        "title": "Average Satisfaction Score (CSAT)",
        "description": "The average satisfaction rating self-reported by customers (scale 1-5).",
    },
    "complaint_rate_pct": {
        "title": "Customer Complaint Rate",
        "description": "The percentage of customers who logged complaints during the reporting period.",
    },
    "avg_app_usage_hours": {
        "title": "Average App Usage",
        "description": "The average number of hours spent by customers browsing or ordering on the application.",
    },
    "avg_cashback_amount": {
        "title": "Average Cashback Reward",
        "description": "The average cashback amount received by customers.",
    },
    "avg_order_count": {
        "title": "Average Order Count",
        "description": "The average number of orders completed by customers.",
    },
    # --- Behavior KPIs ---
    "high_engagement_pct": {
        "title": "High Engagement User Ratio",
        "description": "The percentage of the customer base classified as highly engaged on the application.",
    },
    "high_loyalty_pct": {
        "title": "High Loyalty User Ratio",
        "description": "The percentage of the customer base demonstrating a high brand attachment score.",
    },
    "mobile_first_pct": {
        "title": "Mobile-First User Ratio",
        "description": "The percentage of the customer base utilizing mobile devices as their primary login channel.",
    },
    "discount_seeker_pct": {
        "title": "Discount Seeker Ratio",
        "description": "The percentage of the customer base with high coupon usage or high cashback reliance.",
    },
    "frequent_buyer_pct": {
        "title": "Frequent Buyer Ratio",
        "description": "The percentage of the customer base with a high total volume of purchases.",
    },
    # --- Segmentation KPIs ---
    "champions_pct": {
        "title": "Champions Cohort Ratio",
        "description": "The percentage of customers classified within the top RFM Champions tier.",
    },
    "loyal_customers_pct": {
        "title": "Loyal Customers Cohort Ratio",
        "description": "The percentage of customers classified within the Loyal Customers RFM tier.",
    },
    "at_risk_pct": {
        "title": "At Risk Cohort Ratio",
        "description": "The percentage of customers classified as having a high risk of churn.",
    },
    "high_value_pct": {
        "title": "High Value Cohort Ratio",
        "description": "The percentage of customers classified as premium high-spenders.",
    },
    "complaint_prone_pct": {
        "title": "Complaint-Prone Cohort Ratio",
        "description": "The percentage of customers who have logged operational complaints.",
    },
    # --- RFM KPIs ---
    "avg_rfm_sum": {
        "title": "Average RFM Sum",
        "description": "The average sum of the Recency, Frequency, and Monetary scores (scale 3-15).",
    },
    "top_rfm_segment": {
        "title": "Top Customer Segment",
        "description": "The RFM customer cohort containing the highest volume of unique accounts.",
    },
    "weakest_rfm_segment": {
        "title": "Weakest Customer Segment",
        "description": "The RFM customer cohort containing the lowest volume of unique accounts.",
    },
    # --- Business Health KPIs ---
    "customer_health_index": {
        "title": "Customer Health Index",
        "description": "Aggregate scale of satisfaction, app engagement, and complaint penalties.",
    },
    "engagement_index": {
        "title": "Engagement Index",
        "description": "Average monthly hours spent across registered devices.",
    },
    "satisfaction_index_pct": {
        "title": "Satisfaction Index (CSAT)",
        "description": "Average customer satisfaction rating normalized out of 100%.",
    },
    "loyalty_index": {"title": "Loyalty Index", "description": "Average tenure-weighted satisfaction rating."},
}


def calculate_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculates all descriptive and business intelligence metrics from the dataset.

    Args:
        df: Enriched customer DataFrame containing attributes, flags, and RFM metrics.

    Returns:
        A dictionary containing computed KPI values, grouped by domain categories.

    Raises:
        KPICalculationError: If expected calculation columns are missing.
    """
    logger.info("Initializing multi-domain KPI calculation.")

    required_cols = [
        "CustomerID",
        "Churn",
        "SatisfactionScore",
        "Complain",
        "HourSpendOnApp",
        "CashbackAmount",
        "OrderCount",
        "is_high_engagement",
        "is_loyal",
        "is_mobile_first",
        "is_discount_seeker",
        "is_frequent_buyer",
        "is_champion",
        "is_at_risk",
        "is_high_value",
        "is_complaint_prone",
        "RFM_Sum",
        "RFM_Segment",
        "RawHealthIndex",
        "AppExposure",
        "CustomerLoyaltyScore",
    ]

    for col in required_cols:
        if col not in df.columns:
            raise KPICalculationError(f"Missing required column '{col}' for KPI calculation.")

    try:
        total_cust = len(df)
        active_cust = int((df["Churn"] == 0).sum())
        churned_cust = int((df["Churn"] == 1).sum())

        churn_rate = (churned_cust / total_cust) * 100 if total_cust > 0 else 0.0
        retention_rate = (active_cust / total_cust) * 100 if total_cust > 0 else 0.0

        # Segment distributions
        segment_counts = df["RFM_Segment"].value_counts()
        top_segment = str(segment_counts.index[0]) if not segment_counts.empty else "None"
        weakest_segment = str(segment_counts.index[-1]) if not segment_counts.empty else "None"

        kpis = {
            "customer_base": {
                "total_customers": total_cust,
                "active_customers": active_cust,
                "churned_customers": churned_cust,
                "churn_rate_pct": round(churn_rate, 2),
                "retention_rate_pct": round(retention_rate, 2),
            },
            "engagement": {
                "avg_satisfaction_score": round(float(df["SatisfactionScore"].mean()), 2),
                "complaint_rate_pct": round(float((df["Complain"] == 1).mean() * 100), 2),
                "avg_app_usage_hours": round(float(df["HourSpendOnApp"].mean()), 2),
                "avg_cashback_amount": round(float(df["CashbackAmount"].mean()), 2),
                "avg_order_count": round(float(df["OrderCount"].mean()), 2),
            },
            "behavior": {
                "high_engagement_pct": round(float(df["is_high_engagement"].mean() * 100), 2),
                "high_loyalty_pct": round(float(df["is_loyal"].mean() * 100), 2),
                "mobile_first_pct": round(float(df["is_mobile_first"].mean() * 100), 2),
                "discount_seeker_pct": round(float(df["is_discount_seeker"].mean() * 100), 2),
                "frequent_buyer_pct": round(float(df["is_frequent_buyer"].mean() * 100), 2),
            },
            "segmentation": {
                "champions_pct": round(float(df["is_champion"].mean() * 100), 2),
                "loyal_customers_pct": round(float(df["is_loyal"].mean() * 100), 2),
                "at_risk_pct": round(float(df["is_at_risk"].mean() * 100), 2),
                "high_value_pct": round(float(df["is_high_value"].mean() * 100), 2),
                "complaint_prone_pct": round(float(df["is_complaint_prone"].mean() * 100), 2),
            },
            "rfm": {
                "avg_rfm_sum": round(float(df["RFM_Sum"].mean()), 2),
                "top_rfm_segment": top_segment,
                "weakest_rfm_segment": weakest_segment,
            },
            "health_indices": {
                "customer_health_index": round(float(df["RawHealthIndex"].mean()), 2),
                "engagement_index": round(float(df["AppExposure"].mean()), 2),
                "satisfaction_index_pct": round(float((df["SatisfactionScore"].mean() / 5.0) * 100), 2),
                "loyalty_index": round(float(df["CustomerLoyaltyScore"].mean()), 2),
            },
        }

        logger.info("KPI calculations completed successfully.")
        return kpis

    except Exception as e:
        error_msg = f"Failed to compute dataset KPIs: {e}"
        logger.error(error_msg)
        raise KPICalculationError(error_msg) from e


def generate_executive_recommendations(kpis: Dict[str, Any]) -> List[str]:
    """Generates automated strategic business recommendations based on calculated KPI values.

    Args:
        kpis: Dict of calculated KPIs from calculate_kpis().

    Returns:
        List of strings containing executive business action plans.
    """
    recommendations = []

    churn_rate = kpis["customer_base"]["churn_rate_pct"]
    complaint_rate = kpis["engagement"]["complaint_rate_pct"]
    csat = kpis["engagement"]["avg_satisfaction_score"]
    discount_seeker_pct = kpis["behavior"]["discount_seeker_pct"]
    mobile_first_pct = kpis["behavior"]["mobile_first_pct"]
    health_index = kpis["health_indices"]["customer_health_index"]

    # 1. Churn Threat Level
    if churn_rate > 15.0:
        recommendations.append(
            f"CRITICAL: Churn rate is elevated at {churn_rate}%. Immediately activate the At-Risk cohort "
            "retention campaigns and check for coupon usage anomalies."
        )
    else:
        recommendations.append(
            f"STABLE: Churn rate is within threshold at {churn_rate}%. Focus on expanding LTV "
            "across high-value segments."
        )

    # 2. Quality & CX Check
    if CSAT_alarm := (csat < 3.5 or complaint_rate > 20.0):
        recommendations.append(
            f"CX WARNING: CSAT rating is {csat}/5.0 and complaint rate is high at {complaint_rate}%. "
            "Audit warehouse operations, shipping intervals, and delivery distance bottlenecks."
        )
    else:
        recommendations.append(
            f"HEALTHY: Customer sentiment is strong (CSAT = {csat}/5.0). Maintain operational SLA targets."
        )

    # 3. Campaign & Incentive Strategy
    if discount_seeker_pct > 50.0:
        recommendations.append(
            f"MARGIN RISK: {discount_seeker_pct}% of the customer base are Discount Seekers. "
            "Transition from straight discount coupons to purchase bundles and loyalty points "
            "to protect gross margins."
        )

    # 4. Product Channel Optimization
    if mobile_first_pct > 60.0:
        recommendations.append(
            f"CHANNEL STRATEGY: Mobile-first users dominate the base ({mobile_first_pct}%). "
            "Prioritize mobile app feature updates and deploy cart recovery push notifications."
        )

    # 5. Core Health Index
    if health_index < 2.0:
        recommendations.append(
            f"HEALTH ALERT: Overall customer health index is low ({health_index}). "
            "Execute service recovery campaigns for dissatisfied customers."
        )

    return recommendations


def print_kpi_report(kpis: Dict[str, Any]) -> None:
    """Prints a formatted, professional KPI report to the console.

    Args:
        kpis: Dict of calculated KPIs from calculate_kpis().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CUSTOMER 360 BUSINESS BI REPORT")
    print(border)
    print(f"Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}")
    print(border)

    # 1. Customer Base
    print("\n[I] CUSTOMER BASE METRICS")
    print(section_divider)
    cb = kpis["customer_base"]
    print(f"  Total Registered Customers:  {cb['total_customers']:,}")
    print(f"  Active Customer Base:        {cb['active_customers']:,} ({cb['retention_rate_pct']}%)")
    print(f"  Churned Customer Base:       {cb['churned_customers']:,} ({cb['churn_rate_pct']}%)")

    # 2. Engagement
    print("\n[II] CUSTOMER ENGAGEMENT AVERAGES")
    print(section_divider)
    eg = kpis["engagement"]
    print(f"  Average Customer Satisfaction (CSAT): {eg['avg_satisfaction_score']}/5.0")
    print(f"  Operational Complaint Rate:          {eg['complaint_rate_pct']}%")
    print(f"  Average App Session Usage:           {eg['avg_app_usage_hours']} hours")
    print(f"  Average Order Count per Customer:    {eg['avg_order_count']} orders")
    print(f"  Average Cashback Reward Distributed: ${eg['avg_cashback_amount']:.2f}")

    # 3. Behavioral Penetration
    print("\n[III] BEHAVIORAL COHORT PENETRATION")
    print(section_divider)
    bh = kpis["behavior"]
    print(f"  Mobile-First User Ratio:      {bh['mobile_first_pct']}%")
    print(f"  Discount Seeker Ratio:        {bh['discount_seeker_pct']}%")
    print(f"  High Engagement User Ratio:   {bh['high_engagement_pct']}%")
    print(f"  Frequent Buyer Ratio:         {bh['frequent_buyer_pct']}%")
    print(f"  High Loyalty User Ratio:      {bh['high_loyalty_pct']}%")

    # 4. Strategic Segmentation
    print("\n[IV] CUSTOMER SEGMENT RATIOS")
    print(section_divider)
    sg = kpis["segmentation"]
    print(f"  At Risk Customer Ratio:       {sg['at_risk_pct']}%")
    print(f"  Complaint-Prone Customer Ratio:{sg['complaint_prone_pct']}%")
    print(f"  High-Value Premium Spenders:  {sg['high_value_pct']}%")
    print(f"  RFM Loyal Customer Cohort:    {sg['loyal_customers_pct']}%")
    print(f"  RFM Champions Cohort:         {sg['champions_pct']}%")

    # 5. RFM & Health Indexes
    print("\n[V] BUSINESS HEALTH & INDEX METRICS")
    print(section_divider)
    rf = kpis["rfm"]
    hi = kpis["health_indices"]
    print(f"  Average Customer RFM Score Sum: {rf['avg_rfm_sum']}/15")
    print(f"  Dominant RFM Cohort:            '{rf['top_rfm_segment']}'")
    print(f"  Least Populated RFM Cohort:     '{rf['weakest_rfm_segment']}'")
    print(f"  Customer Satisfaction Index:    {hi['satisfaction_index_pct']}%")
    print(f"  App Engagement Index:           {hi['engagement_index']}")
    print(f"  Customer Loyalty Index:         {hi['loyalty_index']}")
    print(f"  OVERALL CUSTOMER HEALTH INDEX:  {hi['customer_health_index']}")

    # 6. Recommendations
    print("\n[VI] EXECUTIVE BUSINESS RECOMMENDATIONS")
    print(section_divider)
    recs = generate_executive_recommendations(kpis)
    for idx, rec in enumerate(recs, 1):
        print(f"  {idx}. {rec}")

    print(f"\n{border}")
    print("                 END OF ENTERPRISE BUSINESS BI REPORT")
    print(f"{border}\n")
