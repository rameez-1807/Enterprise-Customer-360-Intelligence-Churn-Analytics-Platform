"""
Flagship Customer 360 Dashboard Page
=====================================
Renders a comprehensive, 360-degree interactive profile of an individual customer.
Enables Customer Success, Sales, and CRM teams to search and audit customer accounts,
reviewing raw attributes, activity histories, behavioral RFM cohorts, health scores,
and explainable ML churn risk reports.

Author: Principal UI Architect
Version: 1.0.0
"""

from typing import Any, Dict

import streamlit as st

from src.dashboard.data_service import DashboardDataService


def render_customer_360_page(service: DashboardDataService) -> None:
    """Renders the Customer 360-degree profiling page layout.

    Args:
        service: Initialized dashboard data service facade.
    """
    # 1. Customer Selection Header
    st.markdown("### CUSTOMER AUDIT SEARCH")

    # Get active Customer ID from state or default to 50001
    current_id = st.session_state.get("selected_customer_id", 50001)

    selected_id = st.number_input(
        label="Enter Customer ID to Audit (50001 to 55630)",
        min_value=50001,
        max_value=55630,
        value=current_id,
        step=1,
        key="c360_search_input",
    )
    st.session_state["selected_customer_id"] = selected_id

    # 2. Fetch Customer 360 Data Profile
    try:
        profile_data = service.get_customer_360(selected_id)
    except Exception as e:
        st.error(f"Error querying profile for CustomerID {selected_id}: {e}")
        return

    # Extract subgroups
    profile = profile_data["profile"]
    activity = profile_data["activity_metrics"]
    rfm = profile_data["rfm"]
    health = profile_data["health"]
    pred = profile_data["predictions"]
    xai = profile_data["explainability"]

    # Color code maps for health and prediction
    health_colors = {"Good": "#2ea44f", "Stable": "#58a6ff", "At Risk": "#e1b12c", "Critical": "#f85149"}
    h_color = health_colors.get(health["health_category"], "#c9d1d9")

    pred_colors = {"Low": "#2ea44f", "Medium": "#e1b12c", "High": "#da3633", "Critical": "#f85149"}
    p_color = pred_colors.get(pred["prediction_risk_tier"], "#c9d1d9")

    # =============================================================================
    # TOP HEADER PANEL (Executive Profile Summary Card)
    # =============================================================================
    st.markdown(
        f"<div style='background-color:#161b22; padding:20px; border-radius:8px; border:1px solid #30363d; margin-bottom:25px;'>"
        f"  <div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"    <div>"
        f"      <h2 style='margin:0; color:#ffffff;'>Customer ID: #{profile['customer_id']}</h2>"
        f"      <p style='margin:5px 0 0 0; color:#8b949e;'>Segment: <strong style='color:#8a63d2;'>{rfm['rfm_segment']}</strong> | Churn Prediction: "
        f"         <span style='color:{p_color}; font-weight:bold;'>{pred['prediction_risk_tier']} Risk ({pred['churn_probability']:.1%})</span>"
        f"      </p>"
        f"    </div>"
        f"    <div style='text-align:right;'>"
        f"      <span style='background-color:{h_color}; color:#ffffff; padding:6px 12px; border-radius:4px; font-weight:bold; font-size:1.1rem;'>"
        f"        Health Grade: {health['health_grade']}"
        f"      </span>"
        f"    </div>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # =============================================================================
    # 3-COLUMN CORE AUDIT SECTIONS
    # =============================================================================
    col1, col2, col3 = st.columns(3)

    # COLUMN 1: Profile Demographics
    with col1:
        st.markdown("#### 👤 Customer Account Profile")
        st.markdown(
            f"<div style='background-color:#161b22; padding:15px; border-radius:6px; border:1px solid #30363d; min-height:300px;'>"
            f"  <p><strong>Gender:</strong> {profile['gender']}</p>"
            f"  <p><strong>Marital Status:</strong> {profile['marital_status']}</p>"
            f"  <p><strong>City Tier:</strong> Tier {profile['city_tier']}</p>"
            f"  <p><strong>Preferred Login Channel:</strong> {profile['preferred_login_device']}</p>"
            f"  <p><strong>Preferred Payment:</strong> {profile['preferred_payment_mode']}</p>"
            f"  <p><strong>Preferred Category:</strong> {profile['preferred_order_category']}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # COLUMN 2: Operational Activity Metrics
    with col2:
        st.markdown("#### 📈 Operations & Activity Metrics")
        st.markdown(
            f"<div style='background-color:#161b22; padding:15px; border-radius:6px; border:1px solid #30363d; min-height:300px;'>"
            f"  <p><strong>Customer Tenure:</strong> {activity['tenure_months']:.1f} months</p>"
            f"  <p><strong>Order Recency:</strong> {activity['days_since_last_order']:.0f} days since last order</p>"
            f"  <p><strong>Lifetime Orders:</strong> {activity['order_count']:.0f} orders</p>"
            f"  <p><strong>App Engagement Hours:</strong> {activity['hour_spend_on_app']:.1f} hrs/week</p>"
            f"  <p><strong>Registered Addresses:</strong> {activity['number_of_address']} addresses</p>"
            f"  <p><strong>Total Cashback Earned:</strong> ${activity['cashback_amount']:.2f}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # COLUMN 3: Customer Health & Behavioral RFM
    with col3:
        st.markdown("#### 🩺 Health Index & RFM Quintiles")
        st.markdown(
            f"<div style='background-color:#161b22; padding:15px; border-radius:6px; border:1px solid #30363d; min-height:300px;'>"
            f"  <p><strong>Health Score:</strong> {health['health_score']:.1f}/100 (<span style='color:{h_color}; font-weight:bold;'>{health['health_category']}</span>)</p>"
            f"  <p><strong>Service Quality:</strong> {health['service_quality_score']:.1f}/100</p>"
            f"  <p><strong>App Engagement Index:</strong> {health['app_engagement_score']:.1f}/100</p>"
            f"  <p><strong>Recency Score:</strong> {rfm['recency_score']}/5</p>"
            f"  <p><strong>Frequency Score:</strong> {rfm['frequency_score']}/5</p>"
            f"  <p><strong>Monetary Score:</strong> {rfm['monetary_score']}/5</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =============================================================================
    # MIDDLE SECTION: PREDICTION CRITERIA & CRM PLAYBOOK
    # =============================================================================
    st.markdown("### 🤖 MACHINE LEARNING CHURN INSIGHT")

    p_col1, p_col2 = st.columns([1, 2])

    with p_col1:
        st.markdown("#### Model Inference Metrics")
        st.markdown(
            f"<div style='background-color:#161b22; padding:20px; border-radius:6px; border:1px solid #30363d; text-align:center; min-height:180px;'>"
            f"  <div style='font-size:0.9rem; color:#8b949e; text-transform:uppercase;'>Churn Probability</div>"
            f"  <div style='font-size:2.5rem; color:{p_color}; font-weight:bold; margin:10px 0;'>{pred['churn_probability']:.1%}</div>"
            f"  <div style='font-size:0.85rem; color:#8b949e;'>Confidence Score: {pred['confidence_score']:.1%}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    with p_col2:
        st.markdown("#### Recommended CS CRM Retention Action")
        st.markdown(
            f"<div style='background-color:rgba(218, 54, 51, 0.08); border-left: 4px solid {p_color}; padding:20px; border-radius:4px; min-height:180px;'>"
            f"  <h4 style='margin-top:0; color:{p_color};'>PLAYBOOK ACTION PLAN</h4>"
            f"  <p style='font-size:1.1rem; color:#f0f6fc; line-height:1.4;'>{pred['crm_retention_action']}</p>"
            f"  <p style='font-size:0.8rem; color:#8b949e; margin-bottom:0;'>Model Engine: LightGBM v{pred['model_version']} | Run: {pred['prediction_timestamp'][:16]} UTC</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # =============================================================================
    # LOWER SECTION: SHAP EXPLANATIONS PANEL (XAI)
    # =============================================================================
    st.markdown("### 🔍 EXPLAINABLE AI (XAI) DECISION EXPLANATIONS")

    st.markdown(
        f"<div class='info-box'>"
        f"<strong>Executive AI Summary:</strong><br/>{xai['executive_summary'].replace('\n', '<br/>')}"
        f"</div>",
        unsafe_allow_html=True,
    )

    x_col1, x_col2 = st.columns(2)

    with x_col1:
        st.markdown("#### 🚨 Churn Risk Drivers (Positive SHAP Values)")
        pos_drivers = xai["drivers"].get("positive_churn_drivers", [])
        if pos_drivers:
            for idx, driver in enumerate(pos_drivers, 1):
                st.markdown(
                    f"- **{driver['feature'].replace('_', ' ')}** "
                    f"(Raw: `{driver['value']:.2f}` | Influence: **+{driver['shap_contribution']:.4f}**)"
                )
        else:
            st.markdown("*No significant positive risk drivers detected.*")

    with x_col2:
        st.markdown("#### 🛡️ Retention Boosters (Negative SHAP Values)")
        neg_drivers = xai["drivers"].get("negative_churn_drivers", [])
        if neg_drivers:
            for idx, driver in enumerate(neg_drivers, 1):
                st.markdown(
                    f"- **{driver['feature'].replace('_', ' ')}** "
                    f"(Raw: `{driver['value']:.2f}` | Influence: **{driver['shap_contribution']:.4f}**)"
                )
        else:
            st.markdown("*No significant retention signals detected.*")

    st.markdown("---")

    # =============================================================================
    # TIMELINE / JOURNEY SECTION
    # =============================================================================
    st.markdown("### 📅 CUSTOMER ACCOUNT LIFECYCLE TIMELINE")

    st.markdown(
        f"<div style='border-left: 2px solid #30363d; padding-left: 20px; margin-left: 10px;'>"
        f"  <div style='margin-bottom:20px; position:relative;'>"
        f"    <span style='position:absolute; left:-27px; top:3px; background-color:#1f6feb; border-radius:50%; width:12px; height:12px; display:inline-block;'></span>"
        f"    <h5 style='margin:0; color:#ffffff;'>Account Registration</h5>"
        f"    <p style='margin:5px 0 0 0; color:#8b949e; font-size:0.85rem;'>Customer tenure cycle established at {activity['tenure_months']:.1f} months ago.</p>"
        f"  </div>"
        f"  <div style='margin-bottom:20px; position:relative;'>"
        f"    <span style='position:absolute; left:-27px; top:3px; background-color:#8a63d2; border-radius:50%; width:12px; height:12px; display:inline-block;'></span>"
        f"    <h5 style='margin:0; color:#ffffff;'>CDP Segment Assignment</h5>"
        f"    <p style='margin:5px 0 0 0; color:#8b949e; font-size:0.85rem;'>Allocated to cohort: <strong>{rfm['rfm_segment']}</strong> (RFM Profile: {rfm['composite_rfm_score']}).</p>"
        f"  </div>"
        f"  <div style='margin-bottom:20px; position:relative;'>"
        f"    <span style='position:absolute; left:-27px; top:3px; background-color:#e1b12c; border-radius:50%; width:12px; height:12px; display:inline-block;'></span>"
        f"    <h5 style='margin:0; color:#ffffff;'>Last Transaction Activity</h5>"
        f"    <p style='margin:5px 0 0 0; color:#8b949e; font-size:0.85rem;'>Ordered {activity['order_count']:.0f} item(s) lifetime. Days since last order: {activity['days_since_last_order']:.0f} day(s).</p>"
        f"  </div>"
        f"  <div style='position:relative;'>"
        f"    <span style='position:absolute; left:-27px; top:3px; background-color:{p_color}; border-radius:50%; width:12px; height:12px; display:inline-block;'></span>"
        f"    <h5 style='margin:0; color:#ffffff;'>Model Health Evaluation</h5>"
        f"    <p style='margin:5px 0 0 0; color:#8b949e; font-size:0.85rem;'>Assessed Churn Risk level: <strong>{pred['prediction_risk_tier']} Risk</strong>. Engagement Index is {health['health_score']:.1f}/100.</p>"
        f"  </div>"
        f"</div>",
        unsafe_allow_html=True,
    )
