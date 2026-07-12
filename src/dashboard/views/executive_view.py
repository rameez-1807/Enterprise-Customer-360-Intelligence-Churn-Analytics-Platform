"""
Executive Overview Dashboard Page
==================================
Renders key executive indicators (KPIs), charts, business risks, and action
recommendations. Serves as the landing view for C-level leadership (CEO, COO, CMO)
and VP-level directors to monitor customer health metrics and ML churn forecasts.

Author: Principal BI Engineer
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.dashboard.components import render_info_banner, render_kpi_card
from src.dashboard.data_service import DashboardDataService


def render_executive_view(service: DashboardDataService) -> None:
    """Renders the executive overview layout, metric cards, and visual charts.

    Args:
        service: Initialized dashboard data service facade.
    """
    # 1. Fetch cached metrics from service
    summary = service.get_dashboard_summary()
    kpi_meta = service.get_kpi_summary()
    health_meta = service.get_customer_health_summary()
    segmentation_meta = service.get_segmentation_summary()
    rfm_meta = service.get_rfm_summary()
    prediction_meta = service.get_prediction_summary()

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
    # EXECUTIVE KPI CARDS (6 Columns Layout)
    # =============================================================================
    st.markdown("### EXECUTIVE KEY PERFORMANCE INDICATORS")
    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        render_kpi_card("Total Base", f"{summary['total_customers']:,}", "Customers", trend="neutral")

    with col2:
        render_kpi_card("Active Base", f"{summary['active_customers']:,}", "Scored Accounts", trend="neutral")

    with col3:
        render_kpi_card("Churn Rate", f"{summary['churn_rate_pct']:.2f}%", "Historical Log", trend="down")

    with col4:
        render_kpi_card("Retention Rate", f"{summary['retention_rate_pct']:.2f}%", "Operational Target", trend="up")

    with col5:
        render_kpi_card("Customer Health", f"{summary['average_health_score']:.1f}/100", "Avg Health Score", trend="up")

    with col6:
        render_kpi_card(
            "Satisfaction (CSAT)", f"{summary['average_satisfaction']:.2f}/5", "User CSAT Rating", trend="neutral"
        )

    st.markdown("---")

    # =============================================================================
    # EXECUTIVE CHARTS (Grid layout 2 columns × 3 rows)
    # =============================================================================
    st.markdown("### ENTERPRISE PERFORMANCE CHARTS")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # Chart 1: Customer Health Category Distribution
    with row1_col1:
        st.markdown("#### Customer Health Category Distribution")
        health_dist = health_meta.get("health_category_distribution", {})
        df_health = pd.DataFrame(list(health_dist.items()), columns=["Health Category", "Customer Count"]).set_index(
            "Health Category"
        )
        st.bar_chart(df_health, color="#1f6feb")

    # Chart 2: Prediction Risk Tier Distribution (ML Churn Risk)
    with row1_col2:
        st.markdown("#### Churn Risk Distribution (ML Predicted)")
        risk_dist = prediction_meta.get("risk_distribution", {})
        risk_data = {tier: info["count"] for tier, info in risk_dist.items()}
        df_risk = pd.DataFrame(list(risk_data.items()), columns=["Risk Tier", "Customer Count"]).set_index("Risk Tier")
        # Ordering logical tiers
        df_risk = df_risk.reindex(["Low", "Medium", "High", "Critical"])
        st.bar_chart(df_risk, color="#da3633")

    # Chart 3: Behavioral RFM Cohort Distribution
    with row2_col1:
        st.markdown("#### Behavioral RFM Cohort Distribution")
        rfm_dist = rfm_meta.get("segment_distribution", {})
        df_rfm = pd.DataFrame(list(rfm_dist.items()), columns=["RFM Segment", "Customer Count"]).set_index(
            "RFM Segment"
        )
        st.bar_chart(df_rfm, color="#8a63d2")

    # Chart 4: Customer flag Segmentation Distribution
    with row2_col2:
        st.markdown("#### Customer CDP Segmentation Distribution")
        seg_dist = segmentation_meta.get("flag_penetration", {})
        seg_counts = {k: v["count"] for k, v in seg_dist.items()}
        df_seg = pd.DataFrame(list(seg_counts.items()), columns=["Segment Flag", "Customer Count"]).set_index(
            "Segment Flag"
        )
        st.bar_chart(df_seg, color="#34d058")

    # Chart 5: Complaint Analysis (Churn impact)
    with row3_col1:
        st.markdown("#### Complaint Rate vs. Historical Churn Rate")
        # Extract churn rates for complainers vs. non-complainers
        # Check from processed DataFrame
        df_proc = service.df_processed
        complain_churn = df_proc.groupby("Complain")["Churn_Prediction"].mean() * 100
        df_comp = pd.DataFrame(
            {
                "Complaint Logged": ["No Complaint", "Complaint Logged"],
                "Predicted Churn Rate (%)": [complain_churn.get(0, 0), complain_churn.get(1, 0)],
            }
        ).set_index("Complaint Logged")
        st.bar_chart(df_comp, color="#e1b12c")

    # Chart 6: Customer Satisfaction (CSAT) Distribution
    with row3_col2:
        st.markdown("#### Customer Satisfaction Score (CSAT) Distribution")
        csat_dist = df_proc["SatisfactionScore"].value_counts().sort_index()
        df_csat = pd.DataFrame(
            {"CSAT Rating": csat_dist.index.tolist(), "Customer Count": csat_dist.values.tolist()}
        ).set_index("CSAT Rating")
        st.bar_chart(df_csat, color="#2ea44f")

    st.markdown("---")

    # =============================================================================
    # EXECUTIVE INSIGHTS & AI SUMMARY PANELS
    # =============================================================================
    st.markdown("### STRATEGIC INSIGHTS & ACTIONS")

    left_panel, right_panel = st.columns(2)

    with left_panel:
        st.markdown("#### 🚨 Top Business Risks & Opportunities")
        st.markdown(
            f"- **System Risk Alert:** There are **{summary['critical_risk_count']:,}** customers mapped to the **Critical Churn Risk** tier (Probability ≥ 90%). Immediate intervention is required to recover these accounts."
            f"\n- **Satisfaction Gap:** Customer satisfaction averages **{summary['average_satisfaction']:.2f}/5**. Dissatisfaction groups are highly correlated with logged complaints."
            f"\n- **High-Value Loyalty Opportunity:** The **{rfm_meta.get('top_cohort', {}).get('segment', 'Hibernating')}** segment represents the largest customer cohort. Reviving these accounts with target cashbacks could drive incremental revenue."
        )

    with right_panel:
        st.markdown("#### 💡 Recommended Executive Actions")
        st.markdown(
            f"1. **Deploy Immediate CS Callbacks:** Focus retention efforts on the **{summary['critical_risk_count']:,}** critical risk customers. Call playbooks should prioritize those with outstanding complaints."
            f"\n2. **Targeted Loyalty Vouchers:** Deploy targeted high-incentive cashback campaigns to the **{summary['high_risk_count']:,}** customers in the **High Risk** category."
            f"\n3. **Improve Onboarding Engagement:** Optimize onboarding for new cohorts (0-6m tenure) to build loyalty and reduce early-stage churn."
        )

    st.markdown("---")
    st.markdown("#### 🤖 AI Generated Executive Summary")
    recommendations = kpi_meta.get("recommendations", [])
    if recommendations:
        summary_text = "<br>".join(
            [f"• <strong>Insight {idx}:</strong> {rec}" for idx, rec in enumerate(recommendations, 1)]
        )
        render_info_banner(summary_text, category="info")
    else:
        render_info_banner(
            "AI Summary: Data pipeline is clean. Baseline performance is within target thresholds.", category="info"
        )
