"""
Enterprise Behavioral RFM Analytics Engine
=============================================
Provides a multidimensional Customer RFM (Recency, Frequency, Monetary)
segmentation model. Due to upstream data limitations (absence of absolute
order value columns), this engine implements a Behavioral RFM strategy,
leveraging 'DaySinceLastOrder' for Recency, 'OrderCount' for Frequency,
and 'CashbackAmount' as a proxy for Monetary spend.

Author: Principal Customer Analytics Consultant
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.core.config_loader import load_model_config
from src.core.exceptions import KPICalculationError
from src.core.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# STRATEGIC BUSINESS RECOMMENDATIONS FOR RFM SEGMENTS
# ---------------------------------------------------------------------------
# Defines action plans for each customer cohort.
# ---------------------------------------------------------------------------
SEGMENT_RECOMMENDATIONS: Dict[str, str] = {
    "Champions": (
        "Reward loyalty with premium perks. Invite to exclusive VIP events, "
        "provide early access to new product releases, and recruit as brand advocates."
    ),
    "Loyal Customers": (
        "Upsell high-value products. Offer personalized bundle discounts and "
        "introduce loyalty tier progression milestones to increase lifetime spend."
    ),
    "Promising": (
        "Increase purchase frequency. Target with recommendation-based cross-selling, "
        "provide limited-time trial vouchers, and initiate re-engagement email flows."
    ),
    "Need Attention": (
        "Prevent slide into churn. Launch time-sensitive win-back cashback offers, "
        "collect feedback on recent friction points, and provide targeted support surveys."
    ),
    "About to Sleep": (
        "Reactivate relationship. Distribute high-value reactivation discount coupons, "
        "recommend popular trending products, and highlight recent checkout improvements."
    ),
    "Can't Lose Them": (
        "VIP rescue protocol. Assign dedicated customer support handlers, "
        "offer premium cashback/coupon rewards, and contact directly to resolve complaints."
    ),
    "At Risk": (
        "Aggressive retention intervention. Target with high-priority personalized "
        "discount incentives, execute custom retargeting ads, and send feedback requests."
    ),
    "Hibernating": (
        "Low-cost reactivation. Include in periodic standard newsletters, "
        "offer basic discount coupons, or deprioritize marketing spend to preserve margins."
    ),
}


def _compute_quintile(series: pd.Series, ascending: bool = True) -> pd.Series:
    """Computes robust quintile bins (1 to 5) for a series.

    Uses ranking with the 'first' method to resolve duplicate edge boundaries
    (common in e-commerce purchase distributions) and ensures equal-sized bin splits.

    Args:
        series: Pandas Series containing numerical metric values.
        ascending: True to assign higher scores to higher values (e.g. Frequency),
                   False to assign higher scores to lower values (e.g. Recency).

    Returns:
        Pandas Series of integer scores from 1 to 5.
    """
    if series.isnull().any():
        logger.warning(
            "Null values detected in series during RFM scoring. Imputing nulls with median/mode is recommended."
        )
        series = series.fillna(series.median())

    # Rank-based binning to handle duplicate values gracefully
    ranked = series.rank(method="first", ascending=ascending)
    return pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5]).astype(int)


def compute_rfm_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Computes R, F, and M quintile scores for the customer base.

    Args:
        df: Cleaned customer DataFrame containing RFM metric columns.

    Returns:
        DataFrame containing calculated Recency_Score, Frequency_Score,
        Monetary_Score, concatenated RFM_Score string, and RFM_Sum.

    Raises:
        KPICalculationError: If expected columns are missing from the input DataFrame.
    """
    config = load_model_config()
    recency_col = config.get("rfm", {}).get("recency_column", "DaySinceLastOrder")
    frequency_col = config.get("rfm", {}).get("frequency_column", "OrderCount")
    monetary_col = config.get("rfm", {}).get("monetary_column", "CashbackAmount")

    for col in [recency_col, frequency_col, monetary_col]:
        if col not in df.columns:
            raise KPICalculationError(f"Missing required RFM column '{col}' from input DataFrame.")

    df_rfm = df.copy()

    # Recency: Smaller day counts since last purchase are better (ascending=False)
    df_rfm["Recency_Score"] = _compute_quintile(df_rfm[recency_col], ascending=False)

    # Frequency: Larger order counts are better (ascending=True)
    df_rfm["Frequency_Score"] = _compute_quintile(df_rfm[frequency_col], ascending=True)

    # Monetary: Larger cashback/spend values are better (ascending=True)
    df_rfm["Monetary_Score"] = _compute_quintile(df_rfm[monetary_col], ascending=True)

    # Composite RFM Score strings and index sums
    df_rfm["RFM_Score"] = (
        df_rfm["Recency_Score"].astype(str)
        + df_rfm["Frequency_Score"].astype(str)
        + df_rfm["Monetary_Score"].astype(str)
    )
    df_rfm["RFM_Sum"] = df_rfm["Recency_Score"] + df_rfm["Frequency_Score"] + df_rfm["Monetary_Score"]

    logger.info("RFM quintile scoring completed successfully.")
    return df_rfm


def assign_rfm_segments(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Maps composite RFM scores to strategic customer segments.

    Assigns cohort labels (e.g. Champions, At Risk) and compiles segment
    statistics, average behaviors, and marketing recommendations.

    Args:
        df: DataFrame containing computed Recency_Score, Frequency_Score,
            and Monetary_Score columns.

    Returns:
        A tuple containing:
            - DataFrame enriched with 'RFM_Segment' and 'RFM_Recommendation' columns.
            - A dictionary containing aggregate statistics and details for each segment.

    Raises:
        KPICalculationError: If 'RFM_Score' column is missing.
    """
    if "RFM_Score" not in df.columns:
        raise KPICalculationError("RFM_Score column is missing. Run compute_rfm_scores first.")

    config = load_model_config()
    segment_map = config.get("rfm", {}).get("segment_labels", {})

    df_segmented = df.copy()

    # Map the concatenated score string to the segment label.
    # Fallback to 'Hibernating' if a score is unmapped.
    df_segmented["RFM_Segment"] = df_segmented["RFM_Score"].map(segment_map).fillna("Hibernating")

    # Map recommendations
    df_segmented["RFM_Recommendation"] = (
        df_segmented["RFM_Segment"].map(SEGMENT_RECOMMENDATIONS).fillna("Standard engagement campaign.")
    )

    # Compile segment statistics
    segment_counts = df_segmented["RFM_Segment"].value_counts()
    total_customers = len(df_segmented)

    segment_stats = {}
    for segment, count in segment_counts.items():
        segment_df = df_segmented[df_segmented["RFM_Segment"] == segment]

        # Calculate averages for key drivers
        avg_recency = float(segment_df[config.get("rfm", {}).get("recency_column", "DaySinceLastOrder")].mean())
        avg_frequency = float(segment_df[config.get("rfm", {}).get("frequency_column", "OrderCount")].mean())
        avg_monetary = float(segment_df[config.get("rfm", {}).get("monetary_column", "CashbackAmount")].mean())
        churn_rate = float(segment_df["Churn"].mean() * 100) if "Churn" in segment_df.columns else 0.0

        segment_stats[str(segment)] = {
            "customer_count": int(count),
            "percentage": round((count / total_customers) * 100, 2),
            "average_recency_days": round(avg_recency, 1),
            "average_frequency_orders": round(avg_frequency, 1),
            "average_monetary_cashback": round(avg_monetary, 2),
            "churn_rate_percentage": round(churn_rate, 2),
            "recommendation": SEGMENT_RECOMMENDATIONS.get(str(segment), "Standard campaign."),
        }

    metadata = {
        "execution_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_customers_segmented": total_customers,
        "segmentation_summary": segment_stats,
    }

    logger.info(f"Segment allocation complete. Identified {len(segment_stats)} distinct customer cohorts.")
    return df_segmented, metadata


# =============================================================================
# PRIMARY ENGINE ORCHESTRATOR
# =============================================================================


def analyze_rfm(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Orchestrates the complete Behavioral RFM analysis pipeline.

    Ingests the cleaned customer DataFrame, computes quintile scores,
    maps customer cohorts, and generates CRM recommendations.

    Args:
        df: Cleaned customer DataFrame.

    Returns:
        A tuple containing:
            - DataFrame enriched with all RFM scores, segments, and recommendations.
            - Comprehensive metadata dictionary documenting execution parameters.
    """
    logger.info("=" * 80)
    logger.info("ENTERPRISE BEHAVIORAL RFM SCORING ENGINE — STARTING")
    logger.info("=" * 80)

    # Validate that we have fields to execute the behavioral proxy strategy
    config = load_model_config()
    recency_col = config.get("rfm", {}).get("recency_column", "DaySinceLastOrder")
    frequency_col = config.get("rfm", {}).get("frequency_column", "OrderCount")
    monetary_col = config.get("rfm", {}).get("monetary_column", "CashbackAmount")

    # Honest architectural validation log
    logger.info("Upstream data contract verification:")
    logger.info(f"  - Recency Column:   '{recency_col}' (DaySinceLastOrder)")
    logger.info(f"  - Frequency Column: '{frequency_col}' (OrderCount)")
    logger.info(f"  - Monetary Column:  '{monetary_col}' (CashbackAmount - Spent proxy)")

    # Execute scoring
    df_scores = compute_rfm_scores(df)

    # Execute segmentation mapping
    df_segmented, metadata = assign_rfm_segments(df_scores)

    logger.info("RFM analysis pipeline completed successfully.")
    logger.info("=" * 80)

    return df_segmented, metadata


def print_rfm_report(metadata: Dict[str, Any]) -> None:
    """Prints a professional, human-readable summary of RFM segmentation metrics.

    Args:
        metadata: Dict containing RFM metadata generated by assign_rfm_segments().
    """
    border = "=" * 80
    section_divider = "-" * 80

    print(f"\n{border}")
    print("                 ENTERPRISE CUSTOMER RFM SEGMENTATION REPORT")
    print(border)
    print(f"Timestamp (UTC): {metadata['execution_timestamp']}")
    print(f"Total Customers: {metadata['total_customers_segmented']:,}")
    print(border)

    print("\nCOHORT LEVEL SEGMENT ANALYSIS")
    print(section_divider)

    summary = metadata["segmentation_summary"]
    # Sort segments by volume descending
    sorted_segments = sorted(summary.items(), key=lambda x: x[1]["customer_count"], reverse=True)

    for segment, stats in sorted_segments:
        print(f"\n* COHORT: {segment}")
        print(f"  Size:         {stats['customer_count']:,} customers ({stats['percentage']}%)")
        print(
            f"  Averages:     Recency = {stats['average_recency_days']} days | "
            f"Frequency = {stats['average_frequency_orders']} orders | "
            f"Monetary (Cashback) = ${stats['average_monetary_cashback']:.2f}"
        )
        print(f"  Churn Rate:   {stats['churn_rate_percentage']}%")
        print(f"  CRM Action:   {stats['recommendation']}")

    print(f"\n{border}")
    print("                 END OF RFM ANALYTICS REPORT")
    print(f"{border}\n")
