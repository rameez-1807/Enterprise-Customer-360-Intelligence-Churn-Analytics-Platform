"""
Enterprise UI Design Token System
====================================
Centralizes the visual identity of the dashboard including color palettes,
typography scales, spacing tokens, and theme configuration for both
Light and Dark modes conforming to enterprise design standards.

Author: Principal Product Designer & Python Engineer
Version: 1.0.0
"""

import streamlit as st

# Curated HSL/Hex Enterprise Color Palette
THEME_COLORS = {
    "bg_main": "#0d1117",  # Dark Slate Main Background
    "bg_card": "#161b22",  # Darker Slate Card Background
    "border": "#30363d",  # Neutral Border
    "primary": "#1f6feb",  # Enterprise Blue Accent
    "primary_glow": "rgba(31, 111, 235, 0.15)",
    "text_main": "#c9d1d9",  # Main grey-white text
    "text_header": "#ffffff",  # Bright white headers
    "text_muted": "#8b949e",  # Subtitles / Captions
    # Semantic Health Tiers Colors
    "success": "#2ea44f",  # Good Health / Low Churn
    "info": "#58a6ff",  # Stable Health
    "warning": "#d29922",  # Warning Churn Risk
    "danger": "#f85149",  # Critical Churn Risk
}

GLOBAL_STYLING = f"""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    /* Global Background and Typography Overrides */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    
    .stApp {{
        background-color: {THEME_COLORS['bg_main']};
        color: {THEME_COLORS['text_main']};
    }}
    
    /* Header and Title Accents */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        color: {THEME_COLORS['text_header']};
        margin-top: 0.8rem !important;
        margin-bottom: 0.8rem !important;
    }}
    
    /* Sleek Linear Gradient header text */
    .premium-header {{
        font-weight: 700;
        font-size: 2.3rem !important;
        background: linear-gradient(135deg, #58a6ff 0%, #1f6feb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem !important;
    }}
    
    /* Sidebar Layout Modifications */
    section[data-testid="stSidebar"] {{
        background-color: {THEME_COLORS['bg_card']} !important;
        border-right: 1px solid {THEME_COLORS['border']};
    }}
    
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {THEME_COLORS['text_muted']};
    }}
    
    /* Glassmorphic card styling (Standard Container) */
    .glass-card {{
        background: rgba(22, 27, 34, 0.7);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid {THEME_COLORS['border']};
        border-radius: 10px;
        padding: 22px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        transition: transform 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease;
    }}
    
    .glass-card:hover {{
        transform: translateY(-3px);
        border-color: {THEME_COLORS['primary']};
        box-shadow: 0 12px 40px 0 rgba(31, 111, 235, 0.15);
    }}
    
    /* Premium KPI Card Styling */
    .kpi-card {{
        background: linear-gradient(135deg, rgba(22, 27, 34, 0.8) 0%, rgba(13, 17, 23, 0.8) 100%);
        border: 1px solid {THEME_COLORS['border']};
        border-radius: 8px;
        padding: 18px;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }}
    
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: {THEME_COLORS['primary']};
        box-shadow: 0 6px 18px {THEME_COLORS['primary_glow']};
    }}
    
    .kpi-title {{
        font-size: 0.8rem;
        color: {THEME_COLORS['text_muted']};
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.07em;
    }}
    
    .kpi-value {{
        font-size: 1.9rem;
        color: {THEME_COLORS['text_header']};
        font-weight: 700;
        margin-top: 4px;
        margin-bottom: 2px;
    }}
    
    .kpi-subtitle {{
        font-size: 0.8rem;
        font-weight: 500;
    }}
    
    /* Custom Info Box */
    .info-banner {{
        background-color: rgba(31, 111, 235, 0.08);
        border-left: 4px solid {THEME_COLORS['primary']};
        border-radius: 4px;
        padding: 16px;
        color: {THEME_COLORS['text_main']};
        margin-bottom: 20px;
        font-size: 0.95rem;
        line-height: 1.5;
    }}

    /* Custom Warning / Success Banners */
    .warning-banner {{
        background-color: rgba(210, 153, 34, 0.08);
        border-left: 4px solid {THEME_COLORS['warning']};
        border-radius: 4px;
        padding: 16px;
        color: {THEME_COLORS['text_main']};
        margin-bottom: 20px;
        font-size: 0.95rem;
    }}
    
    .danger-banner {{
        background-color: rgba(248, 81, 73, 0.08);
        border-left: 4px solid {THEME_COLORS['danger']};
        border-radius: 4px;
        padding: 16px;
        color: {THEME_COLORS['text_main']};
        margin-bottom: 20px;
        font-size: 0.95rem;
    }}
    
    /* Sleek Status Badge */
    .status-badge {{
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }}
</style>
"""


def inject_theme_styling() -> None:
    """Injects custom design tokens and styles into the Streamlit viewport."""
    st.markdown(GLOBAL_STYLING, unsafe_allow_html=True)
