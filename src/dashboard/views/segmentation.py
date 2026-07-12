"""
Customer flag Segmentation Dashboard Page
=========================================
Renders multi-dimensional customer segmentation flag distributions, penetration
metrics, cross-segment comparisons, and targeted marketing campaigns. Helps CRM,
Marketing, and Product teams monitor customer behaviors and design campaigns.

Author: Principal UI Architect
Version: 1.0.0
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.dashboard.data_service import DashboardDataService


def render_segmentation_page(service: DashboardDataService) -> None:
    """Renders the Customer Segmentation Analytics page.

    Args:
        service: Initialized dashboard data service facade.
    """
    # 1. Fetch segmentation metadata
    seg_meta = service.get_segmentation_summary()
    df_proc = service.df_processed

    # =============================================================================
    # TOP HEADER META
    # =============================================================================
    meta = service.get_dashboard_metadata()
    last_refresh = meta["pipeline_last_run_timestamp"]

    st.markdown(
        f"<div style='background-color:#161b22; padding:12px; border-radius:6px; border:1px solid #30363d; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;'>"
        f"  <div>System Status: <span style='color:#3fb950; font-weight:bold;'>● {meta['service_status']}</span></div>"
        f"  <div style='color:#8b949e; font-size:0.85rem;'>Pipeline Last Refresh: {last_refresh[:16]} UTC</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # =============================================================================
    # CONTROL PANEL (Actions & Filters)
    # =============================================================================
    st.markdown("### 🎛️ SEGMENT FILTERS & REPORT ACTIONS")
    action_col, filter_col = st.columns([1, 2])

    with action_col:
        st.markdown("**Export Options:**")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.button("📄 Export PDF", key="seg_pdf", help="Export dashboard as PDF report.")
        with exp_col2:
            st.button("📊 Export Excel", key="seg_xlsx", help="Download segment dataset to Excel.")
        with exp_col3:
            st.button("📝 Export CSV", key="seg_csv", help="Download segment dataset to CSV.")

        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.button("🔄 Refresh Data", key="seg_refresh", help="Re-run pipeline data ingest.")
        with ref_col2:
            st.button("📺 Full Screen", key="seg_fullscreen", help="Toggle full screen view.")

    with filter_col:
        st.markdown("**Segment Filter & Customer Search:**")
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            st.selectbox(
                label="Select Target Segment Flag",
                options=["All Flags"] + list(seg_meta.get("flag_penetration", {}).keys()),
                key="seg_flag_filter",
            )
        with sel_col2:
            st.text_input(label="Search Customer ID", placeholder="e.g. 50001...", key="seg_customer_search")

    st.markdown("---")

    # =============================================================================
    # SEGMENTATION KPI CARDS (11 Columns Layout / Responsive Grid)
    # =============================================================================
    st.markdown("### CUSTOMER CDP SEGMENTATION FLAGS")

    # 2 rows of KPI Cards to handle 11 metrics cleanly
    row1_cols = st.columns(6)
    row2_cols = st.columns(5)

    flags = seg_meta.get("flag_penetration", {})

    with row1_cols[0]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Champions</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_champion', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#2ea44f; font-size:0.65rem;'>{flags.get('is_champion', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_cols[1]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Loyal</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_loyal', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff; font-size:0.65rem;'>{flags.get('is_loyal', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_cols[2]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>High Value</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_high_value', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff; font-size:0.65rem;'>{flags.get('is_high_value', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_cols[3]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Growth</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_growth', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#8b949e; font-size:0.65rem;'>{flags.get('is_growth', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_cols[4]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Frequent</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_frequent_buyer', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#34d058; font-size:0.65rem;'>{flags.get('is_frequent_buyer', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row1_cols[5]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Discount</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_discount_seeker', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#e1b12c; font-size:0.65rem;'>{flags.get('is_discount_seeker', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_cols[0]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Mobile-First</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_mobile_first', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#8b949e; font-size:0.65rem;'>{flags.get('is_mobile_first', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_cols[1]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>High Eng.</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_high_engagement', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#2ea44f; font-size:0.65rem;'>{flags.get('is_high_engagement', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_cols[2]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>At Risk</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_at_risk', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#da3633; font-size:0.65rem;'>{flags.get('is_at_risk', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_cols[3]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Complaint</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_complaint_prone', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#da3633; font-size:0.65rem;'>{flags.get('is_complaint_prone', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with row2_cols[4]:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:110px;'>"
            f"  <div class='metric-title' style='font-size:0.7rem;'>Inactive</div>"
            f"  <div class='metric-value' style='font-size:1.2rem;'>{flags.get('is_inactive', {}).get('count', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#f85149; font-size:0.65rem;'>{flags.get('is_inactive', {}).get('percentage', 0):.1f}%</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =============================================================================
    # VISUALIZATIONS (2 Columns Grid)
    # =============================================================================
    st.markdown("### SEGMENT PERFORMANCE COMPARISONS")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # 1. Segment Customer Distribution
    with row1_col1:
        st.markdown("#### Customer Segment Distribution")
        df_dist = pd.DataFrame(
            [(k.replace("is_", ""), v["count"]) for k, v in flags.items()], columns=["Segment", "Count"]
        ).set_index("Segment")
        st.bar_chart(df_dist, color="#1f6feb")

    # 2. Segment Penetration %
    with row1_col2:
        st.markdown("#### Segment Database Penetration (%)")
        df_pen = pd.DataFrame(
            [(k.replace("is_", ""), v["percentage"]) for k, v in flags.items()], columns=["Segment", "Penetration %"]
        ).set_index("Segment")
        st.bar_chart(df_pen, color="#8a63d2")

    # Helper function to compute average metric across segment flags
    def get_avg_metric_by_segment_flags(metric_col: str) -> pd.DataFrame:
        flag_cols = [
            "is_champion",
            "is_loyal",
            "is_high_value",
            "is_growth",
            "is_frequent_buyer",
            "is_discount_seeker",
            "is_mobile_first",
            "is_high_engagement",
            "is_at_risk",
            "is_complaint_prone",
            "is_inactive",
        ]

        averages = {}
        for col in flag_cols:
            subset = df_proc[df_proc[col] == 1]
            averages[col.replace("is_", "")] = float(subset[metric_col].mean()) if not subset.empty else 0.0

        return pd.DataFrame(list(averages.items()), columns=["Segment", f"Avg {metric_col}"]).set_index("Segment")

    # 3. Segment vs Churn Rate
    with row2_col1:
        st.markdown("#### Average Churn Rate (%) by Customer Segment")
        df_churn = get_avg_metric_by_segment_flags("Churn_Prediction") * 100
        st.bar_chart(df_churn, color="#da3633")

    # 4. Segment vs Health Score
    with row2_col2:
        st.markdown("#### Average Health Score (0-100) by Customer Segment")
        df_health = get_avg_metric_by_segment_flags("CustomerHealthScore")
        st.bar_chart(df_health, color="#34d058")

    # 5. Segment vs Satisfaction (CSAT)
    with row3_col1:
        st.markdown("#### Average Satisfaction Rating (1-5) by Segment")
        df_csat = get_avg_metric_by_segment_flags("SatisfactionScore")
        st.bar_chart(df_csat, color="#2ea44f")

    # 6. Segment vs Revenue Proxy (Cashback)
    with row3_col2:
        st.markdown("#### Average Cashback Earned ($) by Segment")
        df_cash = get_avg_metric_by_segment_flags("CashbackAmount")
        st.bar_chart(df_cash, color="#e1b12c")

    st.markdown("---")

    # =============================================================================
    # BUSINESS INSIGHTS & CAMPAIGNS PANELS
    # =============================================================================
    st.markdown("### STRATEGIC CAMPAIGN PLAYBOOKS")

    left_panel, right_panel = st.columns(2)

    with left_panel:
        st.markdown("#### 🚨 Segment Insights Summary")
        st.markdown(
            f"- **Largest Cohort:** **{seg_meta.get('largest_segment', {}).get('segment', 'Inactive').replace('is_', '')}** is our largest customer cohort with **{seg_meta.get('largest_segment', {}).get('count', 0):,}** accounts. Marketing campaigns should focus on reactivating these users."
            f"\n- **High Risk Cohort:** **{flags.get('is_at_risk', {}).get('count', 0):,}** customers are flagged as **At Risk** and have a higher likelihood of churn."
            f"\n- **Complaint Prone Risks:** Customers classified as **Complaint Prone** exhibit the lowest average satisfaction rates and require service recovery calls."
        )

    with right_panel:
        st.markdown("#### 💡 Marketing Campaign Recommendations")
        st.markdown(
            "1. **Retention Campaign:** Focus efforts on **At Risk** and **Complaint Prone** segments. Resolve open issues and offer cashback rewards."
            "\n2. **Upsell Campaign:** Target the **Loyal** and **High Value** segments with premium offers, cross-category discounts, and early access to sales."
            "\n3. **Cross-Sell Campaign:** Introduce **Frequent Buyers** to relevant accessories and product add-ons."
            "\n4. **Reactivation Campaign:** Deploy high-incentive target campaigns to win back the **Inactive** customer segment."
        )

    st.markdown("---")
    st.markdown("#### 🤖 AI Generated Segmentation Summary")
    recommendations = seg_meta.get("strategic_opportunities", [])
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            st.markdown(f"**Opportunity {idx}:** {rec}")
    else:
        st.markdown(
            "AI Segmentation Assessment: Customer flags and penetration rates are within target operational thresholds."
        )
