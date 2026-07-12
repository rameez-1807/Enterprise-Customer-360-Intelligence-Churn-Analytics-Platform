"""
Customer Health Analytics Dashboard Page
=========================================
Renders KPI dashboards, distribution charts, and cohort tables describing the
health indices of the customer base. Helps Customer Success and Retention teams
monitor indicators (Service Quality, Purchase Momentum, App Engagement, and Tenure Loyalty),
track high-risk accounts, and run playbook plays.

Author: Principal UI Architect
Version: 1.0.0
"""

from typing import Any, Dict

import pandas as pd
import streamlit as st

from src.dashboard.data_service import DashboardDataService


def render_customer_health_page(service: DashboardDataService) -> None:
    """Renders the Customer Health Analytics page.

    Args:
        service: Initialized dashboard data service facade.
    """
    # 1. Fetch data profiles
    health_meta = service.get_customer_health_summary()
    kpi_meta = service.get_kpi_summary()
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
    # HEALTH KPI CARDS (7 Columns Layout)
    # =============================================================================
    st.markdown("### CUSTOMER HEALTH KEY METRICS")
    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

    # Calculate health counts
    cat_counts = df_proc["Health_Category"].value_counts()
    good_count = int(cat_counts.get("Good", 0))
    stable_count = int(cat_counts.get("Stable", 0))
    warning_count = int(cat_counts.get("Warning", 0))
    critical_count = int(cat_counts.get("Critical", 0))

    with col1:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Avg Health</div>"
            f"  <div class='metric-value'>{health_meta['average_health_score']:.1f}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff;'>Out of 100</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Healthy</div>"
            f"  <div class='metric-value'>{good_count:,}</div>"
            f"  <div class='metric-delta' style='color:#2ea44f;'>Good Cat</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Stable</div>"
            f"  <div class='metric-value'>{stable_count:,}</div>"
            f"  <div class='metric-delta' style='color:#58a6ff;'>Stable Cat</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Warning</div>"
            f"  <div class='metric-value'>{warning_count:,}</div>"
            f"  <div class='metric-delta' style='color:#e1b12c;'>Warning Cat</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col5:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Critical</div>"
            f"  <div class='metric-value'>{critical_count:,}</div>"
            f"  <div class='metric-delta' style='color:#f85149;'>Critical Cat</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col6:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Avg CSAT</div>"
            f"  <div class='metric-value'>{kpi_meta['engagement']['avg_satisfaction_score']:.2f}</div>"
            f"  <div class='metric-delta' style='color:#8b949e;'>Out of 5</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with col7:
        st.markdown(
            f"<div class='metric-card'>"
            f"  <div class='metric-title'>Complaints</div>"
            f"  <div class='metric-value'>{kpi_meta['engagement']['complaint_rate_pct']:.1f}%</div>"
            f"  <div class='metric-delta' style='color:#f85149;'>Log Rate</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =============================================================================
    # HEALTH VISUALIZATIONS (2 Columns Grid)
    # =============================================================================
    st.markdown("### HEALTH INDEX DISTRIBUTION ANALYSIS")

    row1_col1, row1_col2 = st.columns(2)
    row2_col1, row2_col2 = st.columns(2)
    row3_col1, row3_col2 = st.columns(2)

    # Chart 1: Health Category Distribution
    with row1_col1:
        st.markdown("#### Health Category Distribution")
        df_cat = pd.DataFrame(list(cat_counts.items()), columns=["Category", "Count"]).set_index("Category")
        st.bar_chart(df_cat, color="#1f6feb")

    # Chart 2: Health Grade Distribution
    with row1_col2:
        st.markdown("#### Health Grade Distribution (Letter Grades A to F)")
        grade_dist = df_proc["Health_Grade"].value_counts()
        df_grade = pd.DataFrame(list(grade_dist.items()), columns=["Grade", "Count"]).set_index("Grade")
        df_grade = df_grade.reindex(["A", "B", "C", "D", "E", "F"])
        st.bar_chart(df_grade, color="#8a63d2")

    # Chart 3: Risk Level Distribution (Health Risk Score Segmentations)
    with row2_col1:
        st.markdown("#### Operational Risk Level Distribution")
        risk_dist = df_proc["Risk_Level"].value_counts()
        df_risk = pd.DataFrame(list(risk_dist.items()), columns=["Risk Level", "Count"]).set_index("Risk Level")
        df_risk = df_risk.reindex(["Low Risk", "Medium Risk", "High Risk", "Critical Risk"])
        st.bar_chart(df_risk, color="#e1b12c")

    # Chart 4: Service Quality Score Distribution
    with row2_col2:
        st.markdown("#### Service Quality Index Score Distribution")
        sq_bins = pd.cut(
            df_proc["Service_Quality_Score"],
            bins=[0, 20, 40, 60, 80, 101],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
        )
        df_sq = pd.DataFrame(sq_bins.value_counts(), columns=["Count"])
        st.bar_chart(df_sq, color="#2ea44f")

    # Chart 5: Purchase Momentum Index Distribution
    with row3_col1:
        st.markdown("#### Purchase Momentum Index Score Distribution")
        pm_bins = pd.cut(
            df_proc["Purchase_Momentum_Score"],
            bins=[-1, 20, 40, 60, 80, 101],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
        )
        df_pm = pd.DataFrame(pm_bins.value_counts(), columns=["Count"])
        st.bar_chart(df_pm, color="#34d058")

    # Chart 6: App Engagement Index Distribution
    with row3_col2:
        st.markdown("#### App Engagement Index Score Distribution")
        ae_bins = pd.cut(
            df_proc["App_Engagement_Score"],
            bins=[-1, 20, 40, 60, 80, 101],
            labels=["0-20", "21-40", "41-60", "61-80", "81-100"],
        )
        df_ae = pd.DataFrame(ae_bins.value_counts(), columns=["Count"])
        st.bar_chart(df_ae, color="#1f6feb")

    st.markdown("---")

    # =============================================================================
    # HEALTH AUDIT TABLE (Customers Needing Attention)
    # =============================================================================
    st.markdown("### RETENTION INTELLIGENCE AUDIT GRID")
    st.caption("Showing Top 10 Critical Risk customers with the lowest health scores, flagged for CS outreach.")

    # Filter for Warning or Critical risk segments, sorted ascending by health score
    df_flagged = df_proc[df_proc["Prediction_Risk_Level"].isin(["High", "Critical"])].sort_values(
        by="CustomerHealthScore", ascending=True
    )

    display_cols = [
        "CustomerID",
        "CustomerHealthScore",
        "Health_Grade",
        "Risk_Level",
        "Churn_Probability",
        "Prediction_CRM_Action",
    ]

    st.dataframe(
        df_flagged[display_cols]
        .head(10)
        .rename(
            columns={
                "CustomerID": "Customer ID",
                "CustomerHealthScore": "Health Score",
                "Health_Grade": "Grade",
                "Risk_Level": "Health Risk",
                "Churn_Probability": "ML Churn Prob",
                "Prediction_CRM_Action": "CS Retention Playbook Action",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("---")

    # =============================================================================
    # RETENTION STRATEGIES PANEL
    # =============================================================================
    st.markdown("### 🩺 CUSTOMER SUCCESS RETENTION PLAYBOOKS")

    left_panel, right_panel = st.columns(2)

    with left_panel:
        st.markdown("#### 🚨 Retention Focus Areas")
        st.markdown(
            f"- **Service Quality Failures:** There are **{critical_count:,}** customers mapped to the **Critical Health** cohort. These accounts are driven by high complaints (complaint rate: {kpi_meta['engagement']['complaint_rate_pct']:.1f}%) and low CSAT scores."
            f"\n- **High Risk Warning:** Warning category accounts (**{warning_count:,}** customers) are losing purchase momentum and require proactive engagement."
        )

    with right_panel:
        st.markdown("#### 💡 Playbook Recommendations")
        st.markdown(
            "1. **Critical Playbook (Health Grade F):** Route accounts directly to the Escalations Desk. Offer compensation, resolve open issues, and schedule follow-ups within 24 hours."
            "\n2. **Warning Playbook (Health Grade D):** Auto-trigger loyalty cashback vouchers (e.g. $10 target reward) and send product recommendations based on preferred order categories."
            "\n3. **Engagement Playbook (Health Grade C):** Push app push-notifications and product catalogs to revive week-over-week active usage."
        )

    st.markdown("---")
    st.markdown("#### 🤖 AI Customer Health Summary")
    recommendations = health_meta.get("remediation_strategies", [])
    if recommendations:
        for idx, rec in enumerate(recommendations, 1):
            st.markdown(f"**Strategy {idx}:** {rec}")
    else:
        st.markdown(
            "AI Health Assessment: System indicators stable. General customer base shows healthy service indexes."
        )
