"""
Enterprise Dashboard Entry Point & State Controller
======================================================
Primary application launcher for the Customer 360 Intelligence platform.
Sets up the Streamlit page configuration, initializes the global state store,
injects custom enterprise styling (CSS), renders the sidebar navigation menu,
and routes requests to page controller stubs.

All data queries are routed through the DashboardDataService.

Author: Principal UI Architect
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add project root directory to sys.path to allow importing from src
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import streamlit as st

from src.core.logger import get_logger
from src.dashboard.data_service import DashboardDataService
from src.dashboard.design_system import inject_theme_styling

logger = get_logger(__name__)


# Design system styles are loaded dynamically from design_system.py


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================


def initialize_session_state() -> None:
    """Sets up the global Session State values for the dashboard application."""
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "Executive Overview"
    if "selected_customer_id" not in st.session_state:
        st.session_state["selected_customer_id"] = 50001
    if "filters" not in st.session_state:
        st.session_state["filters"] = {"gender": "All", "marital_status": "All", "city_tier": "All"}
    if "data_service" not in st.session_state:
        # Load and cache dashboard service instance
        st.session_state["data_service"] = DashboardDataService()


# =============================================================================
# MAIN APPLICATION CONTROLLER
# =============================================================================


def main() -> None:
    """Dashboard primary layout and navigation router."""
    # 1. Page Configuration (Wide Layout Default)
    st.set_page_config(
        page_title="Enterprise Customer 360 Analytics Platform", layout="wide", initial_sidebar_state="expanded"
    )

    # 2. Inject Design Custom Style CSS
    inject_theme_styling()

    # 3. State initialization
    try:
        initialize_session_state()
        service: DashboardDataService = st.session_state["data_service"]
    except Exception as e:
        import traceback
        st.error("### 🚨 Critical Application Startup Error")
        st.write("An exception occurred while initializing the Dashboard Data Service:")
        st.exception(e)
        st.code(traceback.format_exc(), language="python")
        return

    # 4. Sidebar Navigation
    st.sidebar.markdown(
        "<div style='text-align: center; padding: 10px 0px;'>"
        "<h2 style='color:#ffffff; margin:0;'>ENTERPRISE</h2>"
        "<p style='color:#58a6ff; font-weight:600; margin:0; letter-spacing:0.05em;'>CUSTOMER 360 PLATFORM</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    st.sidebar.markdown("### NAVIGATION")
    pages = [
        "Executive Overview",
        "Customer 360",
        "Customer Health",
        "Behavioral RFM",
        "Customer Segmentation",
        "Churn Prediction",
        "Explainable AI",
        "Settings",
        "About Project",
    ]

    # Store page selection directly to session state
    selected_page = st.sidebar.radio(label="Select Dashboard Tab", options=pages, label_visibility="collapsed")
    st.session_state["current_page"] = selected_page

    st.sidebar.markdown("---")

    # Display service status indicator in sidebar
    svc_meta = service.get_dashboard_metadata()
    st.sidebar.markdown("### PLATFORM METADATA")
    st.sidebar.caption(f"Status: **{svc_meta['service_status']}**")
    st.sidebar.caption(f"Engineered Records: **{svc_meta['processed_records_count']:,}**")
    st.sidebar.caption(f"Last Pipeline Run: **{svc_meta['pipeline_last_run_timestamp'][:16]}**")

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "<div style='text-align: center; font-size: 0.8rem; color: #8b949e; padding-top: 5px;'>"
        "Created by:<br>"
        "<span style='color: #58a6ff; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.03em;'>Md Rameez Ahmad</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # 5. Header Section
    st.markdown(f"<h1>{st.session_state['current_page']}</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color: #8b949e; margin-bottom: 20px;'>Enterprise Customer 360 Intelligence & Churn Analytics Platform</p>",
        unsafe_allow_html=True,
    )

    # 6. Page Routing
    page = st.session_state["current_page"]

    # Import actual page renderers
    from src.dashboard.views.about import render_about_page
    from src.dashboard.views.churn_analytics import render_churn_analytics_page
    from src.dashboard.views.customer_360 import render_customer_360_page
    from src.dashboard.views.customer_health import render_customer_health_page
    from src.dashboard.views.executive_view import render_executive_view
    from src.dashboard.views.explainable_ai import render_explainable_ai_page
    from src.dashboard.views.rfm_segmentation import render_rfm_segmentation_page
    from src.dashboard.views.segmentation import render_segmentation_page
    from src.dashboard.views.settings import render_settings_page

    if page == "Executive Overview":
        render_executive_view(service)

    elif page == "Customer 360":
        render_customer_360_page(service)

    elif page == "Customer Health":
        render_customer_health_page(service)

    elif page == "Behavioral RFM":
        render_rfm_segmentation_page(service)

    elif page == "Customer Segmentation":
        render_segmentation_page(service)

    elif page == "Churn Prediction":
        render_churn_analytics_page(service)

    elif page == "Explainable AI":
        render_explainable_ai_page(service)

    elif page == "Settings":
        render_settings_page(service)

    elif page == "About Project":
        render_about_page()

    # 7. Global Page Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #8b949e; font-size: 0.85rem; padding: 15px 0 5px 0;'>"
        "Enterprise Churn Analytics Platform &copy; 2026 | "
        "Created with ♥ by <span style='color: #58a6ff; font-weight: 600;'>Md Rameez Ahmad</span>"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
