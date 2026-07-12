"""
Reusable Dashboard UI Components
===================================
Factory functions for creating standardized KPI cards, filter header
panels, interactive data tables, and chart wrapper components that
enforce uniform enterprise design tokens across all dashboard pages.

Author: Principal Python Engineer
Version: 1.0.0
"""

from typing import Any, Dict

import streamlit as st

from src.dashboard.design_system import THEME_COLORS


def render_kpi_card(title: str, value: Any, subtitle: str, trend: str = "neutral") -> None:
    """Renders a standard design token-compliant metric card.

    Args:
        title: The name/label of the metric.
        value: The main metric value to display.
        subtitle: Supplementary text or percentage change.
        trend: Trend direction ('up', 'down', or 'neutral') to apply color coding.
    """
    trend_color_map = {
        "up": THEME_COLORS["success"],
        "down": THEME_COLORS["danger"],
        "neutral": THEME_COLORS["text_muted"],
    }
    sub_color = trend_color_map.get(trend, THEME_COLORS["text_muted"])

    st.markdown(
        f"<div class='kpi-card'>"
        f"  <div class='kpi-title'>{title}</div>"
        f"  <div class='kpi-value'>{value}</div>"
        f"  <div class='kpi-subtitle' style='color:{sub_color};'>{subtitle}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_info_banner(text: str, category: str = "info") -> None:
    """Renders a clean, styled callout banner for notifications or summaries.

    Args:
        text: Main text message inside the banner.
        category: Semantic category ('info', 'warning', 'danger').
    """
    class_map = {"info": "info-banner", "warning": "warning-banner", "danger": "danger-banner"}
    css_class = class_map.get(category, "info-banner")

    st.markdown(f"<div class='{css_class}'>" f"  {text}" f"</div>", unsafe_allow_html=True)


def start_glass_card(title: str = "") -> None:
    """Starts a glassmorphic container card. Must be closed with end_glass_card()."""
    header_html = f"<h4 style='margin-top:0; color:{THEME_COLORS['text_header']};'>{title}</h4>" if title else ""
    st.markdown(f"<div class='glass-card'>" f"  {header_html}", unsafe_allow_html=True)


def end_glass_card() -> None:
    """Closes the glassmorphic card container."""
    st.markdown("</div>", unsafe_allow_html=True)


def get_status_badge_html(label: str, status_type: str = "info") -> str:
    """Generates the HTML string for a colored status badge.

    Args:
        label: Text to display in the badge.
        status_type: Color category ('success', 'info', 'warning', 'danger').
    """
    color_map = {
        "success": (THEME_COLORS["success"], "rgba(46, 164, 79, 0.15)"),
        "info": (THEME_COLORS["info"], "rgba(88, 166, 255, 0.15)"),
        "warning": (THEME_COLORS["warning"], "rgba(210, 153, 34, 0.15)"),
        "danger": (THEME_COLORS["danger"], "rgba(248, 81, 73, 0.15)"),
    }
    color, bg = color_map.get(status_type, (THEME_COLORS["text_muted"], "rgba(139, 148, 158, 0.15)"))

    return (
        f"<span class='status-badge' style='color:{color}; background-color:{bg}; border:1px solid {color}33;'>"
        f"  {label}"
        f"</span>"
    )
