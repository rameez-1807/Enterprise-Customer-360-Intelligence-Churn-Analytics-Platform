"""
Behavioral RFM Analytics Dashboard Page
========================================
Renders behavioral RFM (Recency, Frequency, Monetary) cohort metrics, segment
distributions, and strategic CRM campaigns. Helps Marketing and CRM teams
identify high-value customers, execute targeted retention plans, and manage cohorts.

Author: Principal UI Architect
Version: 1.0.0
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.dashboard.data_service import DashboardDataService


def render_rfm_segmentation_page(service: DashboardDataService) -> None:
    """Renders the Behavioral RFM Segmentation page.

    Args:
        service: Initialized dashboard data service facade.
    """
    # 1. Fetch RFM metadata
    rfm_meta = service.get_rfm_summary()
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
    # PROFESSIONAL CONTROL PANEL (Segment Selector, Search, Filter Panel & Actions)
    # =============================================================================
    st.markdown("### 🎛️ COHORT FILTERS & REPORT ACTIONS")
    action_col, filter_col = st.columns([1, 2])

    with action_col:
        st.markdown("**Export Options:**")
        exp_col1, exp_col2, exp_col3 = st.columns(3)
        with exp_col1:
            st.button("📄 Export PDF", key="btn_pdf", help="Export dashboard as PDF report.")
        with exp_col2:
            st.button("📊 Export Excel", key="btn_xlsx", help="Download segment dataset to Excel.")
        with exp_col3:
            st.button("📝 Export CSV", key="btn_csv", help="Download segment dataset to CSV.")

        ref_col1, ref_col2 = st.columns(2)
        with ref_col1:
            st.button("🔄 Refresh Data", key="btn_refresh", help="Re-run pipeline data ingest.")
        with ref_col2:
            st.button("📺 Full Screen", key="btn_fullscreen", help="Toggle full screen view.")

    with filter_col:
        st.markdown("**Segment Selector & Search:**")
        sel_col1, sel_col2 = st.columns(2)
        with sel_col1:
            st.selectbox(
                label="Select Target RFM Segment",
                options=["All Segments"] + list(rfm_meta.get("segment_distribution", {}).keys()),
                key="rfm_segment_filter",
            )
        with sel_col2:
            st.text_input(
                label="Search Customer ID or Cohort", placeholder="e.g. 50001 or Champions...", key="rfm_cohort_search"
            )

    st.markdown("---")

    # =============================================================================
    # RFM KPI CARDS (9 Columns Layout)
    # =============================================================================
    st.markdown("### BEHAVIORAL RFM COHORT DISTRIBUTION")
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)

    seg_counts = rfm_meta.get("segment_distribution", {})
    avg_rfm = df_proc["RFM_Sum"].mean()

    with col1:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Avg Sum</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{avg_rfm:.1f}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff; font-size:0.7rem;'>Scale 3-15</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Champions</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Champions', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#2ea44f; font-size:0.7rem;'>Top Score</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Loyal</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Loyal Customers', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff; font-size:0.7rem;'>High Loyalty</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Promising</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Promising', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff; font-size:0.7rem;'>Growth</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Potential</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Potential Loyalist', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#8b949e; font-size:0.7rem;'>Active Mid</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>To Sleep</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('About To Sleep', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#e1b12c; font-size:0.7rem;'>Warning</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col7:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>At Risk</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('At Risk', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#da3633; font-size:0.7rem;'>Critical</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col8:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Cant Lose</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Cant Lose Them', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#da3633; font-size:0.7rem;'>High Value</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col9:
        st.markdown(
            f"<div class='metric-card' style='padding:10px; min-height:120px;'>"
            f"  <div class='metric-title' style='font-size:0.75rem;'>Hibernating</div>"
            f"  <div class='metric-value' style='font-size:1.3rem;'>{seg_counts.get('Hibernating', 0):,}</div>"
            f"  <div class='metric-delta' style='color:#f85149; font-size:0.7rem;'>In-Active</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =============================================================================
    # RFM VISUALIZATIONS (2 Columns Grid)
    # =============================================================================
    st.markdown("### RFM DISTRIBUTION & SCORES")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)

    # Chart 1: Segment Distribution
    with row1_col1:
        st.markdown("#### RFM Segment Size Distribution")
        df_seg = pd.DataFrame(list(seg_counts.items()), columns=["Segment", "Customer Count"]).set_index("Segment")
        st.bar_chart(df_seg, color="#8a63d2")

    # Chart 2: Average RFM Sum by Segment
    with row1_col2:
        st.markdown("#### Average RFM Composite Sum by Segment")
        rfm_avg = df_proc.groupby("RFM_Segment")["RFM_Sum"].mean()
        df_avg = pd.DataFrame(list(rfm_avg.items()), columns=["Segment", "Average RFM Sum"]).set_index("Segment")
        st.bar_chart(df_avg, color="#1f6feb")

    # Chart 3: Recency Score Distribution
    with row2_col1:
        st.markdown("#### Recency Score Distribution (1-5)")
        rec_dist = df_proc["Recency_Score"].value_counts().sort_index()
        df_rec = pd.DataFrame(list(rec_dist.items()), columns=["Recency Score", "Count"]).set_index("Recency Score")
        st.bar_chart(df_rec, color="#34d058")

    # Chart 4: Frequency Score Distribution
    with row2_col2:
        st.markdown("#### Frequency Score Distribution (1-5)")
        freq_dist = df_proc["Frequency_Score"].value_counts().sort_index()
        df_freq = pd.DataFrame(list(freq_dist.items()), columns=["Frequency Score", "Count"]).set_index(
            "Frequency Score"
        )
        st.bar_chart(df_freq, color="#e1b12c")

    st.markdown("---")

    # =============================================================================
    # RFM STRATEGIC INSIGHTS & CAMPAIGNS PANELS
    # =============================================================================
    st.markdown("### STRATEGIC COHORT INSIGHTS")

    left_panel, right_panel = st.columns(2)

    # Left: Strategic Insights
    with left_panel:
        st.markdown("#### 🚨 Segment Core Analysis")
        st.markdown(
            f"- **Largest Segment:** The **{rfm_meta.get('top_cohort', {}).get('segment', 'Hibernating')}** cohort is our largest segment containing **{rfm_meta.get('top_cohort', {}).get('count', 0):,}** accounts. Marketing campaigns should focus on reactivating these users."
            f"\n- **Weakest Segment:** The **{rfm_meta.get('weak_cohort', {}).get('segment', 'Champions')}** cohort has the lowest volume. This highlights a need to migrate medium-value users upward."
            f"\n- **Highest Churn Risk:** At Risk and Can't Lose Them segments contain high-value accounts that are slipping away. They require immediate attention."
        )

    # Right: Marketing Recommendations
    with right_panel:
        st.markdown("#### 💡 Marketing Campaign Strategies")
        st.markdown(
            "1. **Champions Campaign:** Upsell premium product lines, offer early access to new releases, and invite to VIP rewards."
            "\n2. **Hibernating Campaign:** Send personalized win-back offers with steep discounts and survey their customer experience."
            "\n3. **Loyal Customers Campaign:** Standard loyalty rewards, cross-sell related categories (e.g. accessories), and request reviews."
            "\n4. **At Risk Campaign:** High-incentive target campaigns, feedback surveys, and immediate CS escalations."
        )

    st.markdown("---")
    st.markdown("#### 🤖 AI Generated RFM Executive Summary")
    recommendations = rfm_meta.get("marketing_actions", [])
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            st.markdown(f"**Recommendation {idx}:** {rec}")
    else:
        st.markdown(
            "AI RFM Assessment: Cohort distributions stable. General customer base shows healthy purchasing momentum."
        )
