# app_header.py
"""
Hero component (final) - UPDATED FOR WHITE BACKGROUND
- Renders full-width hero via streamlit.components.v1.html so HTML isn't printed.
- Removes top padding and hides Streamlit header & toolbar but KEEPS sidebar.
- Feature pills are in one horizontal row; each pill wraps its text inside.
"""
from __future__ import annotations
from typing import Optional, List
import streamlit as st
from streamlit.components.v1 import html as components_html

DEFAULT_LOGO_SVG = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' width='84' height='84' viewBox='0 0 84 84'>"
    "<rect rx='12' width='84' height='84' fill='%23DD6B20'/>"  # Changed to orange
    "<g fill='white'>"
    "<rect x='18' y='48' width='8' height='18' rx='2'/>"
    "<rect x='34' y='38' width='8' height='28' rx='2'/>"
    "<rect x='50' y='22' width='8' height='44' rx='2'/>"
    "</g></svg>"
)

DEFAULT_FEATURES = [
    "Real-time Alerts",
    "Live + Historical Data",
    "Sentiment Analysis",
    "Predictive Signals",
    "Portfolio Tracking",
    "Trade Simulator",
    "Custom Visual Tools",
    "API Access",
]

def render_starting_hero(
    title_main: str = "Stock Market Intelligence Platform",
    emph_text: str = "More Trades, Less Spreadsheets",
    description: str = "You've got instinct. Back it up with real-time alerts, live & historical data, custom visual analysis tools in one platform built for how you trade.",
    features: Optional[List[str]] = None,
    logo_svg: Optional[str] = None,
    hide_sidebar: bool = False,  # CHANGED TO FALSE TO SHOW SIDEBAR
    show_chat: bool = True,
    height: int = 380,  # Further reduced height to remove space
):
    """
    Render full hero.
    - Call this at the very start of app.py (before other UI) so it is flush to top.
    """
    logo = logo_svg or DEFAULT_LOGO_SVG
    features = features or DEFAULT_FEATURES

    # Updated CSS - KEEP SIDEBAR VISIBLE
    # Updated CSS - KEEP SIDEBAR VISIBLE
    st.markdown(
        """
        <style>
        /* Make app background match hero so no white strip at top */
        .stApp { 
            background-color: #FFFFFF !important; 
            color: #1A365D !important; 
        }
        /* Remove ALL padding and margins from the main container */
        .main > div.block-container { 
            padding-top: 0px !important; 
            padding-bottom: 0px !important;
            padding-left: 2% !important;  /* Reduced for sidebar */
            padding-right: 2% !important; 
        }
        /* Remove any padding from the main block */
        .main .block-container {
            padding-top: 0rem !important;
            padding-bottom: 0rem !important;
        }
        /* hide streamlit header/toolbar but KEEP SIDEBAR */
        [data-testid="stHeader"], 
        [data-testid="stToolbar"] { 
            display: none !important; 
        }
        /* REMOVED THE LINE THAT HIDES SIDEBAR - THIS IS CRITICAL */
        /* [data-testid="stSidebar"] { display: none !important; } */
        
        /* Remove any additional margins */
        .stMarkdown {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    # Full hero HTML (rendered inside iframe)
    hero_html = f"""
    <!doctype html>
    <html>
    <head>
      <meta charset="utf-8"/>
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <style>
        :root{{
          --blue: #3182CE;
          --orange: #DD6B20;
          --hero-bg: #FFFFFF;
          --text: #1A365D;
          --text-muted: #718096;
        }}
        html,body{{margin:0;padding:0;background:transparent; overflow-x: hidden;}}
        .hero-landing{{
            width:100%;
            height:100%;
            box-sizing:border-box;
            padding: 0px 0px 24px 0px !important;  /* COMPLETELY REMOVED TOP PADDING */
            background: linear-gradient(135deg, #FFFFFF 0%, #F7FAFC 100%);
            color: var(--text);
            border-bottom: 1px solid #E2E8F0;
        }}
        .hero-inner{{max-width:1200px;margin:0 auto;display:flex;align-items:flex-start;position:relative}}
        .hero-content{{width:58%;padding-left:28px; padding-top: 0px;}}  /* Removed top padding */
        .hero-brand img{{height:52px;width:52px;border-radius:10px;margin-bottom:8px;display:block}}  /* Reduced margin */
        .hero-title{{
            font-size:56px;
            font-weight:900;
            margin:0px 0 4px 0;  /* Reduced margins */
            color: var(--text);
            line-height: 1.1;
        }}
        .hero-emph{{
            color: var(--orange);
            font-size:36px;
            font-weight:900;
            font-style:italic;
            margin:4px 0 12px;  /* Reduced margins */
        }}
        .hero-desc{{
            color: var(--text-muted);
            font-size:16px;
            max-width:720px;
            line-height:1.45;
            margin-bottom:12px;  /* Reduced margin */
        }}
        /* Features: two horizontal rows without scrolling */
        .features-container {{
            margin-top: 16px;  /* Reduced margin */
            display: flex;
            flex-direction: column;
            gap: 10px;  /* Reduced gap */
        }}
        .features-row{{
            display: flex;
            gap: 10px;  /* Reduced gap */
            flex-wrap: nowrap;
            justify-content: flex-start;
            width: 100%;
        }}
        .feature-pill{{
            flex: 1;
            min-width: 140px;
            background: var(--orange) !important;  /* CHANGED TO ORANGE */
            color: #fff;
            padding: 10px 12px;  /* Reduced padding */
            border-radius: 6px;
            font-weight: 600;
            white-space: nowrap;
            text-align: center;
            line-height: 1.18;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            transition: all 0.2s;
            font-size: 14px;  /* Slightly smaller font */
        }}
        .feature-pill:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.15);
            background: #C05621 !important;  /* Darker orange on hover */
        }}
        .hero-right{{
            position:absolute;
            right:-6%;
            top:0;
            width:44%;
            height:100%;
            background: url('data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000"><defs><linearGradient id="g" x1="0" x2="1" y1="0" y2="1"><stop offset="0" stop-color="%23DD6B20" stop-opacity="0.06"/><stop offset="1" stop-color="%23DD6B20" stop-opacity="0.04"/></linearGradient></defs><polygon points="500,0 1000,1000 0,1000" fill="url(%23g)"/></svg>') no-repeat right center/contain;
            pointer-events:none;
        }}
        @media(max-width:900px){{
            .hero-title{{font-size:34px; margin:0 0 4px 0;}}
            .hero-emph{{font-size:22px; margin:4px 0 8px;}}
            .hero-content{{width:100%;padding-left:14px; padding-top: 0px;}}
            .hero-right{{display:none}}
            .features-row{{
                flex-wrap: wrap;
            }}
            .feature-pill{{
                min-width: 120px;
                padding: 8px 10px;
                font-size: 13px;
            }}
        }}
      </style>
    </head>
    <body style="overflow-x: hidden;">
      <div class="hero-landing" role="banner" aria-label="Hero">
        <div class="hero-inner">
          <div class="hero-content">
            <div class="hero-brand"><img src="{logo}" alt="logo"/></div>
            <div class="hero-title">{title_main}</div>
            <div class="hero-emph">{emph_text}</div>
            <div class="hero-desc">{description}</div>
            <div class="features-container" role="list">
              <div class="features-row">
                <div class="feature-pill">Real-time Alerts</div>
                <div class="feature-pill">Live + Historical Data</div>
                <div class="feature-pill">Sentiment Analysis</div>
                <div class="feature-pill">Predictive Signals</div>
              </div>
              <div class="features-row">
                <div class="feature-pill">Portfolio Tracking</div>
                <div class="feature-pill">Trade Simulator</div>
                <div class="feature-pill">Custom Visual Tools</div>
                <div class="feature-pill">API Access</div>
              </div>
            </div>
          </div>
          <div class="hero-right" aria-hidden="true"></div>
        </div>
      </div>
      <script>
        // postMessage hooks (if you want to capture clicks via JS, not required)
        window.addEventListener('message', function(){{}}, false);
      </script>
    </body>
    </html>
    """

    components_html(hero_html, height=height, scrolling=False)