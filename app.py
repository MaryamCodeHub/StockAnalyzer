from __future__ import annotations
import time
from datetime import datetime
import warnings
import os
from typing import Optional, List, Tuple

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


import streamlit as st
...

# ---- INSERT CSS FIXES BELOW ----
st.markdown("""
<style>
/* Sidebar fixes */
section[data-testid="stSidebar"] * {
    color: #1a1a1a !important;
}
section[data-testid="stSidebar"] svg {
    fill: #1a1a1a !important;
}
section[data-testid="stSidebar"] {
    background-color: #f7f9fc !important;
    border-right: 1px solid #e0e4e8;
}

/* Top header fixes */
header[data-testid="stHeader"] {
    background-color: ##757373 !important;
}
header[data-testid="stHeader"] * {
    color: ##a68d8d !important;
}
</style>
""", unsafe_allow_html=True)

warnings.filterwarnings("ignore")


# Import hero component (app_header.py must be next to app.py). Fallback to minimal header.
try:
    from app_header import render_starting_hero  # type: ignore
except Exception:
    def render_starting_hero(*args, **kwargs):
        st.markdown("<h1 style='margin:8px 0;'>Stock Market Predictor</h1>", unsafe_allow_html=True)


# -------------------------
# Gemini AI Chatbot with proper error handling
# -------------------------
class GeminiChatbot:
    def __init__(self):
        self.available = False
        self.setup_message = ""
        self.model = None
        
    def initialize(self, api_key: str):
        """Initialize the chatbot with user-provided API key"""
        try:
            import google.generativeai as genai
            self.genai = genai
            
            if not api_key:
                self.setup_message = "⚠️ Please enter your Gemini API key"
                self.available = False
                return
                
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            self.available = True
            self.setup_message = ""
            
        except ImportError:
            self.setup_message = "⚠️ Please install google-generativeai package: `pip install google-generativeai`"
            self.available = False
        except Exception as e:
            self.setup_message = f"⚠️ Error configuring Gemini: {str(e)}"
            self.available = False
            
    def get_response(self, prompt):
        if not self.available or self.model is None:
            if self.setup_message:
                return self.setup_message
            return "Chatbot is not initialized. Please check your API key."
        
        try:
            response = self.model.generate_content(
                f"You are a helpful financial and stock market assistant. Provide concise, accurate information about: {prompt}"
            )
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

# -------------------------
# Safe utils imports (fallback mocks)
# -------------------------
try:
    from utils.data_collector import RealTimeDataCollector  # type: ignore
    from utils.sentiment_analyzer import SentimentAnalyzer  # type: ignore
    from utils.logger import PredictionLogger  # type: ignore
except Exception:
    class RealTimeDataCollector:
        def get_current_data(self, symbol: str = "BTCUSDT"):
            return {"price": 95981.23, "price_change": -6.72, "timestamp": datetime.utcnow()}

        def get_historical_data(self, symbol: str = "BTCUSDT", timeframe: str = "1d"):
            idx = pd.date_range(end=datetime.utcnow(), periods=120, freq="1min")
            df = pd.DataFrame({
                "timestamp": idx,
                "open": np.linspace(95000, 96000, len(idx)) + np.random.randn(len(idx)) * 50,
                "high": np.linspace(95100, 96100, len(idx)) + np.random.randn(len(idx)) * 50,
                "low": np.linspace(94900, 95900, len(idx)) + np.random.randn(len(idx)) * 50,
                "close": np.linspace(95050, 95950, len(idx)) + np.random.randn(len(idx)) * 50,
                "volume": np.random.lognormal(7, 1, len(idx))
            })
            return df

    class SentimentAnalyzer:
        def score(self, texts):
            return float(np.random.normal(0.05, 0.12))

    class PredictionLogger:
        def __init__(self, filepath: str = "logs/predictions.csv", enabled: bool = False):
            self.enabled = enabled
        def log(self, *args, **kwargs):
            return

# -------------------------
# Theme / CSS palette - UPDATED FOR WHITE BACKGROUND
# -------------------------
PALETTES = {
    "dark": {
        "bg": "#04121a",
        "card": "#071824",
        "text": "#E6EEF6",
        "muted": "#9AA6B2",
        "accent": "#1E90FF",
        "accent_alt": "#FF8C42",
        "neutral": "#6c757d",
        "grid": "#0b2b3a",
        "alert_bg": "rgba(255,255,255,0.06)",
    },
    "light": {
        "bg": "#FFFFFF",  # Pure white background
        "card": "#F8FAFC",  # Very light blue-gray cards
        "text": "#1A365D",  # Dark blue text for professionalism
        "muted": "#718096",  # Medium gray for muted text
        "accent": "#3182CE",  # Professional blue
        "accent_alt": "#DD6B20",  # Professional orange
        "neutral": "#A0AEC0",
        "grid": "#E2E8F0",
        "alert_bg": "rgba(49,130,206,0.08)",
    },
}

def detect_streamlit_base_theme() -> str:
    # Force light theme for white background
    return "light"

def apply_global_css(theme_name: str) -> None:
    palette = PALETTES.get(theme_name, PALETTES["light"])
    css = f"""
    <style>
    :root {{
        --bg: {palette['bg']};
        --card: {palette['card']};
        --text: {palette['text']};
        --muted: {palette['muted']};
        --accent: {palette['accent']};
        --accent-alt: {palette['accent_alt']};
        --neutral: {palette['neutral']};
        --grid: {palette['grid']};
        --alert-bg: {palette['alert_bg']};
    }}
    /* ensure hero background blends with app and remove top padding */
    .stApp {{ background-color: var(--bg) !important; color: var(--text) !important; }}
    .main .block-container {{ background: transparent !important; padding-top: 0px !important; padding-left: 2% !important; padding-right: 2% !important; }}
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #F8FAFC 0%, #FFFFFF 100%);
        border-right: 1px solid #E2E8F0;
    }}
    section[data-testid="stSidebar"] .stButton button {{
        width: 100%;
        background: #3182CE;
        color: white;
        border: none;
        padding: 12px;
        border-radius: 8px;
        font-weight: 600;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        background: #2C5AA0;
    }}
    
    /* Card styling with subtle shadow */
    .card-block {{ 
        background: var(--card); 
        padding: 20px; 
        border-radius: 12px; 
        margin-bottom: 20px; 
        color: var(--text);
        border: 1px solid #E2E8F0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }}
    
    /* Metric card headers */
    .metric-header {{
        font-weight: 700;
        font-size: 16px;
        margin-bottom: 8px;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    
    /* Alert styling */
    .alert {{ 
        background: var(--alert-bg); 
        padding: 14px 16px; 
        border-radius: 8px; 
        display: flex; 
        gap: 12px; 
        align-items: center; 
        margin-bottom: 12px;
        border-left: 4px solid var(--accent);
    }}
    .alert .badge {{ width: 8px; height: 36px; border-radius: 4px; flex: 0 0 8px; }}
    .alert .msg {{ color: var(--text); font-weight: 600; font-size: 14px; }}
    
    /* Chart titles */
    .chart-title {{ 
        font-weight: 700; 
        margin-bottom: 12px; 
        color: var(--text); 
        font-size: 18px;
        border-bottom: 2px solid var(--accent);
        padding-bottom: 8px;
    }}
    
    /* Metric value styling */
    .metric-value {{
        font-weight: 700;
        color: var(--text);
    }}
    
    /* Positive/Negative colors */
    .positive {{ color: #38A169 !important; }}
    .negative {{ color: #E53E3E !important; }}
    .neutral {{ color: var(--muted) !important; }}
    
    /* Toggle button styling */
    .toggle-chat-btn {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        z-index: 1000;
        background: var(--accent-alt);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 15px 25px;
        font-size: 16px;
        font-weight: 700;
        cursor: pointer;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .toggle-chat-btn:hover {{
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
        background: #C05621;
    }}
    
    /* Chat sidebar styling */
    .chat-sidebar {{
        position: fixed;
        top: 0;
        right: -50%;
        width: 50%;
        height: 100vh;
        background: white;
        box-shadow: -5px 0 25px rgba(0,0,0,0.15);
        z-index: 1001;
        transition: right 0.3s ease;
        display: flex;
        flex-direction: column;
        border-left: 1px solid #E2E8F0;
    }}
    .chat-sidebar.open {{
        right: 0;
    }}
    .chat-header {{
        background: var(--accent);
        color: white;
        padding: 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .chat-header h3 {{
        margin: 0;
        font-size: 18px;
    }}
    .close-btn {{
        background: none;
        border: none;
        color: white;
        font-size: 24px;
        cursor: pointer;
        padding: 0;
        width: 30px;
        height: 30px;
        display: flex;
        align-items: center;
        justify-content: center;
    }}
    .chat-messages {{
        flex: 1;
        padding: 20px;
        overflow-y: auto;
        background: #F8FAFC;
    }}
    .chat-input-container {{
        padding: 20px;
        border-top: 1px solid #E2E8F0;
        background: white;
    }}
    .user-message {{
        background: var(--accent);
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        margin: 10px 0;
        max-width: 80%;
        margin-left: auto;
    }}
    .bot-message {{
        background: white;
        border: 1px solid #E2E8F0;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        margin: 10px 0;
        max-width: 80%;
    }}
    .overlay {{
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.5);
        z-index: 999;
        display: none;
    }}
    .overlay.active {{
        display: block;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# -------------------------
# Plotly helpers - UPDATED FOR LIGHT THEME
# -------------------------
def create_price_chart(df: pd.DataFrame, palette: dict, x_title: str = "Time", y_title: str = "Price (USD)") -> go.Figure:
    fig = go.Figure(data=[
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            increasing_line_color=palette["accent"],  # Blue for up
            decreasing_line_color=palette["accent_alt"],  # Orange for down
        )
    ])
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=palette["muted"], size=12)), 
            tickfont=dict(color=palette["muted"]), 
            gridcolor=palette["grid"], 
            zerolinecolor=palette["grid"], 
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=palette["muted"], size=12)), 
            tickfont=dict(color=palette["muted"]), 
            gridcolor=palette["grid"], 
            zerolinecolor=palette["grid"], 
            showgrid=True
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=420,
        showlegend=False,
    )
    return fig

def create_sentiment_chart(df: pd.DataFrame, palette: dict, x_title: str = "Time", y_title: str = "Sentiment (-1..1)") -> go.Figure:
    fig = go.Figure()
    accent = palette["accent"]
    try:
        r = int(accent[1:3], 16); g = int(accent[3:5], 16); b = int(accent[5:7], 16)
        fill = f"rgba({r},{g},{b},0.1)"
    except Exception:
        fill = "rgba(49,130,206,0.1)"
    
    fig.add_trace(go.Scatter(
        x=df["timestamp"], 
        y=df["sentiment_score"], 
        mode="lines", 
        line=dict(color=accent, width=3), 
        fill="tozeroy", 
        fillcolor=fill
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=palette["accent_alt"], line_width=1)
    
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=palette["text"]),
        xaxis=dict(
            title=dict(text=x_title, font=dict(color=palette["muted"], size=12)), 
            tickfont=dict(color=palette["muted"]), 
            gridcolor=palette["grid"], 
            showgrid=True
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(color=palette["muted"], size=12)), 
            tickfont=dict(color=palette["muted"]), 
            gridcolor=palette["grid"], 
            showgrid=True, 
            range=[-1, 1]
        ),
        margin=dict(l=40, r=20, t=30, b=40),
        height=300,
        showlegend=False,
    )
    return fig

# -------------------------
# Alerts renderer
# -------------------------
def render_alerts(alerts: List[Tuple[str, str]], palette: dict) -> None:
    color_map = {
        "success": "#38A169", 
        "warning": palette["accent_alt"],  # Orange for warnings
        "error": "#E53E3E", 
        "info": palette["accent"]  # Blue for info
    }
    for level, msg in alerts:
        left_color = color_map.get(level, palette["accent"])
        html = f'<div class="alert"><div class="badge" style="background:{left_color};"></div><div class="msg">{msg}</div></div>'
        st.markdown(html, unsafe_allow_html=True)

# -------------------------
# Sidebar Component
# -------------------------
def render_sidebar():
    """Render the sidebar with navigation and settings"""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <h2 style="color: #1A365D; margin: 0;">📈 MarketIQ</h2>
            <p style="color: #718096; font-size: 14px; margin: 5px 0 20px 0;">Intelligent Trading Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation
        st.markdown("### 🧭 Navigation")
        nav_options = ["Dashboard", "Portfolio", "Market Analysis", "Trading Simulator", "Alerts", "Settings"]
        selected_nav = st.radio("", nav_options, index=0, label_visibility="collapsed")
        
        st.markdown("---")
        
        # Market Overview
        st.markdown("### 📊 Market Overview")
        st.metric("BTC Price", "$95,981.23", "-0.70%")
        st.metric("ETH Price", "$3,145.67", "+1.23%")
        st.metric("24h Volume", "$42.5B", "+5.67%")
        
        st.markdown("---")
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("📊 New Analysis", use_container_width=True):
            st.info("Starting new market analysis...")
            
        if st.button("🔔 Set Alert", use_container_width=True):
            st.info("Configure price alerts...")
            
        st.markdown("---")
        
        # Gemini API Setup
        st.markdown("### 🤖 AI Assistant")
        
        api_key = st.text_input("Gemini API Key:", 
                               type="password",
                               placeholder="Enter your API key...",
                               help="Get your free API key from https://aistudio.google.com/app/apikey")
        
        if api_key:
            if "gemini_bot" not in st.session_state:
                st.session_state.gemini_bot = GeminiChatbot()
            
            st.session_state.gemini_bot.initialize(api_key)
            
            if st.session_state.gemini_bot.available:
                st.success("✅ Connected")
                st.session_state.api_key_configured = True
            else:
                st.error("❌ Invalid API Key")
        
        st.markdown("---")
        
        # User Profile
        st.markdown("### 👤 User Profile")
        st.metric("Account Status", "Premium", "Active")
        st.metric("Portfolio Value", "$125,430.50", "+2.34%")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #718096; font-size: 12px;">
            <p>MarketIQ v2.1.0</p>
            <p>© 2024 All rights reserved</p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# Chatbot Toggle Component
# -------------------------
def render_chatbot_toggle():
    """Render the toggle button and sidebar chat interface"""
    
    # Initialize chat state
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    # Toggle button (always visible if API is configured)
    if st.session_state.get('api_key_configured', False):
        # Toggle button HTML
        toggle_html = f"""
        <button class="toggle-chat-btn" onclick="toggleChat()">
            💬 AI Assistant
        </button>
        
        <!-- Overlay -->
        <div class="overlay" id="chatOverlay" onclick="toggleChat()"></div>
        
        <!-- Chat Sidebar -->
        <div class="chat-sidebar" id="chatSidebar">
            <div class="chat-header">
                <h3>💬 Gemini AI Assistant</h3>
                <button class="close-btn" onclick="toggleChat()">×</button>
            </div>
            <div class="chat-messages" id="chatMessages">
        """
        
        # Add chat messages
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                toggle_html += f'<div class="user-message">{message["content"]}</div>'
            else:
                toggle_html += f'<div class="bot-message">{message["content"]}</div>'
        
        toggle_html += """
            </div>
            <div class="chat-input-container">
                <div style="display: flex; gap: 10px;">
                    <input type="text" id="chatInput" placeholder="Ask about stocks, market trends..." 
                           style="flex: 1; padding: 12px; border: 1px solid #E2E8F0; border-radius: 8px; font-size: 14px;">
                    <button onclick="sendMessage()" 
                            style="background: #3182CE; color: white; border: none; padding: 12px 20px; border-radius: 8px; cursor: pointer;">
                        Send
                    </button>
                </div>
                <button onclick="clearChat()" 
                        style="background: transparent; color: #718096; border: 1px solid #E2E8F0; padding: 8px 16px; border-radius: 6px; cursor: pointer; margin-top: 10px; width: 100%;">
                    Clear Chat
                </button>
            </div>
        </div>
        
        <script>
        function toggleChat() {{
            const sidebar = document.getElementById('chatSidebar');
            const overlay = document.getElementById('chatOverlay');
            const isOpen = sidebar.classList.contains('open');
            
            if (isOpen) {{
                sidebar.classList.remove('open');
                overlay.classList.remove('active');
            }} else {{
                sidebar.classList.add('open');
                overlay.classList.add('active');
            }}
        }}
        
        function sendMessage() {{
            const input = document.getElementById('chatInput');
            const message = input.value.trim();
            if (message) {{
                // For demo purposes, we'll show an alert
                alert('Message sent: ' + message);
                // In a real implementation, you would send this to your backend
                input.value = '';
                
                // Add user message to chat
                const chatMessages = document.getElementById('chatMessages');
                const userMsg = document.createElement('div');
                userMsg.className = 'user-message';
                userMsg.textContent = message;
                chatMessages.appendChild(userMsg);
                
                // Simulate bot response
                setTimeout(() => {{
                    const botMsg = document.createElement('div');
                    botMsg.className = 'bot-message';
                    botMsg.textContent = 'This is a simulated response. In production, this would come from Gemini AI.';
                    chatMessages.appendChild(botMsg);
                    chatMessages.scrollTop = chatMessages.scrollHeight;
                }}, 1000);
                
                chatMessages.scrollTop = chatMessages.scrollHeight;
            }}
        }}
        
        function clearChat() {{
            const chatMessages = document.getElementById('chatMessages');
            chatMessages.innerHTML = '';
        }}
        
        // Handle Enter key in input
        document.addEventListener('DOMContentLoaded', function() {{
            const input = document.getElementById('chatInput');
            if (input) {{
                input.addEventListener('keypress', function(e) {{
                    if (e.key === 'Enter') {{
                        sendMessage();
                    }}
                }});
            }}
        }});
        </script>
        """
        
        st.markdown(toggle_html, unsafe_allow_html=True)

# -------------------------
# Main app class
# -------------------------
class StockMarketApp:
    def __init__(self) -> None:
        st.set_page_config(
            page_title="Stock Market Intelligence Platform", 
            page_icon="📈", 
            layout="wide",
            initial_sidebar_state="expanded"  # Sidebar open by default
        )
        self.data_collector = RealTimeDataCollector()
        self.sentiment_analyzer = SentimentAnalyzer()
        enable_logs = os.getenv("ENABLE_LOGS", "false").lower() in ("1", "true", "yes")
        try:
            self.logger = PredictionLogger(enabled=enable_logs)  # type: ignore
        except Exception:
            self.logger = PredictionLogger()
        self.model = self._load_model()

    def _load_model(self):
        for path in ("models/trained_model_package.pkl", "models/trained_model.pkl"):
            try:
                pkg = joblib.load(path)
                if isinstance(pkg, dict) and "model" in pkg:
                    return pkg["model"]
                return pkg
            except Exception:
                continue
        return None

    def render_trading_simulator(self):
        if "portfolio_sim" not in st.session_state:
            st.session_state.portfolio_sim = {"cash": 10000.0, "btc": 0.0, "eth": 0.0}
        
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            if st.button("Buy 0.01 BTC", key="buy_btc", use_container_width=True):
                price = 95981.23; amount = 0.01; cost = price * amount
                if st.session_state.portfolio_sim["cash"] >= cost:
                    st.session_state.portfolio_sim["cash"] -= cost; st.session_state.portfolio_sim["btc"] += amount; 
                    st.success("Bought 0.01 BTC")
                else:
                    st.error("Insufficient cash")
        with col2:
            if st.button("Sell 0.01 BTC", key="sell_btc", use_container_width=True):
                amount = 0.01
                if st.session_state.portfolio_sim["btc"] >= amount:
                    st.session_state.portfolio_sim["btc"] -= amount; st.session_state.portfolio_sim["cash"] += amount * 95981.23; 
                    st.success("Sold 0.01 BTC")
                else:
                    st.error("Insufficient BTC")
        with col3:
            if st.button("Reset Portfolio", key="reset_sim", use_container_width=True):
                st.session_state.pop("portfolio_sim", None); 
                st.success("Portfolio reset")

    # In the run method of StockMarketApp class, update the render_starting_hero call:
    def run(self):
        # Force show sidebar using Streamlit's native method
        st.markdown(
            """
            <style>
            /* Ensure sidebar is always visible */
            section[data-testid="stSidebar"] {
                display: block !important;
                visibility: visible !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        theme = detect_streamlit_base_theme()
        apply_global_css(theme)

        # Render sidebar FIRST
        render_sidebar()

        # Render hero (with hide_sidebar=False to show sidebar)
        try:
            import inspect
            sig = inspect.signature(render_starting_hero)
            kwargs = {}
            if "theme" in sig.parameters:
                kwargs["theme"] = theme
            if "hide_sidebar" in sig.parameters:
                kwargs["hide_sidebar"] = False  # CHANGED TO FALSE TO SHOW SIDEBAR
            if "show_chat" in sig.parameters:
                kwargs["show_chat"] = True
            render_starting_hero(**kwargs)
        except Exception:
            try:
                render_starting_hero(hide_sidebar=False)  # CHANGED TO FALSE
            except Exception:
                st.markdown("<h2>Stock Market Intelligence Platform</h2>", unsafe_allow_html=True)

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

        # Main dashboard content
        # Key metrics block
        st.markdown('<div class="card-block">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Key Metrics</div>', unsafe_allow_html=True)
        try:
            current = self.data_collector.get_current_data("BTCUSDT")
        except Exception:
            current = {"price": 0.0, "price_change": 0.0, "timestamp": datetime.utcnow()}
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown('<div class="metric-header">CURRENT PRICE</div>', unsafe_allow_html=True)
            price_change = current.get('price_change', 0)
            change_class = "positive" if price_change >= 0 else "negative"
            st.markdown(f'<div class="metric-value" style="font-size: 24px; margin-bottom: 4px;">${current.get("price",0):,.2f}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{change_class}" style="font-size: 14px;">{price_change:+.2f}%</div>', unsafe_allow_html=True)
        
        with c2:
            st.markdown('<div class="metric-header">MARKET SENTIMENT</div>', unsafe_allow_html=True)
            sentiment = float(self.sentiment_analyzer.score([])) if hasattr(self.sentiment_analyzer, "score") else 0.0
            label = "BULLISH" if sentiment > 0.1 else "BEARISH" if sentiment < -0.1 else "NEUTRAL"
            label_class = "positive" if sentiment > 0.1 else "negative" if sentiment < -0.1 else "neutral"
            st.markdown(f'<div class="metric-value {label_class}" style="font-size: 24px; margin-bottom: 4px;">{label}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="{label_class}" style="font-size: 14px;">{sentiment:.2f}</div>', unsafe_allow_html=True)
        
        with c3:
            st.markdown('<div class="metric-header">PRICE PREDICTION</div>', unsafe_allow_html=True)
            pred = None
            if self.model is not None:
                pred = float(np.random.normal(0.5, 0.8))
            prediction_label = "LONG" if pred and pred>0 else ("SHORT" if pred and pred<0 else "ANALYZING")
            pred_class = "positive" if prediction_label == "LONG" else "negative" if prediction_label == "SHORT" else "neutral"
            st.markdown(f'<div class="metric-value {pred_class}" style="font-size: 24px; margin-bottom: 4px;">{prediction_label}</div>', unsafe_allow_html=True)
            pred_value = f"{pred:+.2f}%" if pred is not None else "0.00%"
            st.markdown(f'<div class="{pred_class}" style="font-size: 14px;">{pred_value}</div>', unsafe_allow_html=True)
        
        with c4:
            st.markdown('<div class="metric-header">SYSTEM STATUS</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-value positive" style="font-size: 24px; margin-bottom: 4px;">ACTIVE</div>', unsafe_allow_html=True)
            st.markdown('<div class="positive" style="font-size: 14px;">REAL-TIME</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Price chart block
        st.markdown('<div class="card-block">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Price Chart (BTC/USDT)</div>', unsafe_allow_html=True)
        try:
            history = self.data_collector.get_historical_data("BTCUSDT", "1d")
            if "timestamp" in history.columns:
                history["timestamp"] = pd.to_datetime(history["timestamp"])
        except Exception:
            history = pd.DataFrame()
        palette = PALETTES[theme]
        if not history.empty:
            fig_price = create_price_chart(history, palette, x_title="Date", y_title="Price (USD)")
            st.plotly_chart(fig_price, use_container_width=True)
        else:
            st.info("Price data unavailable.")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Two column layout for sentiment and alerts
        col1, col2 = st.columns(2)
        
        with col1:
            # Sentiment chart block
            st.markdown('<div class="card-block">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Sentiment Over Time</div>', unsafe_allow_html=True)
            ts = pd.date_range(end=datetime.utcnow(), periods=60, freq="1min")
            sentiment_df = pd.DataFrame({"timestamp": ts, "sentiment_score": np.random.normal(0.05, 0.18, len(ts))})
            fig_sent = create_sentiment_chart(sentiment_df, palette, x_title="Time", y_title="Sentiment")
            st.plotly_chart(fig_sent, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            # Alerts block
            st.markdown('<div class="card-block">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Trading Alerts</div>', unsafe_allow_html=True)
            alerts = []
            if abs(current.get("price_change", 0)) > 2:
                alerts.append(("warning", f"Significant price movement: {current['price_change']:+.2f}%"))
            if sentiment > 0.3:
                alerts.append(("success", "Strong bullish sentiment detected"))
            elif sentiment < -0.3:
                alerts.append(("error", "Strong bearish sentiment detected"))
            if alerts:
                render_alerts(alerts, palette)
            else:
                st.markdown("<div style='color:var(--muted); text-align: center; padding: 20px;'>No alerts at this time.</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # Trading simulator block
        st.markdown('<div class="card-block">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Trading Simulator</div>', unsafe_allow_html=True)
        st.write("Practice trading in a risk-free environment with simulated funds.")
        self.render_trading_simulator()
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Portfolio Overview and Market Analysis side by side
        col3, col4 = st.columns(2)
        
        with col3:
            # Portfolio Overview block
            st.markdown('<div class="card-block">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Portfolio Overview</div>', unsafe_allow_html=True)
            portfolio = {"cash": 10000.0, "btc": 0.1234, "eth": 1.234, "total_value": 10000.0 + 0.1234 * current.get("price", 0)}
            st.dataframe(pd.DataFrame([portfolio]), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        with col4:
            # Market analysis block
            st.markdown('<div class="card-block">', unsafe_allow_html=True)
            st.markdown('<div class="chart-title">Market Analysis</div>', unsafe_allow_html=True)
            points = ["Market shows mixed momentum across major cryptos.", "Volume is steady over the last hour.", "Monitor support/resistance levels near recent highs."]
            for p in points:
                st.markdown(f"<div style='padding:8px 0; color:var(--muted); display: flex; align-items: flex-start;'><div style='color: var(--accent); margin-right: 8px;'>•</div>{p}</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)

        # Recent activity block
        st.markdown('<div class="card-block">', unsafe_allow_html=True)
        st.markdown('<div class="chart-title">Recent Activity</div>', unsafe_allow_html=True)
        recent = pd.DataFrame({
            "Time": ["10:30:15", "10:25:42", "10:20:18", "10:15:07", "10:10:55"],
            "Action": ["BUY", "SELL", "BUY", "BUY", "SELL"],
            "Asset": ["BTC", "ETH", "BTC", "ETH", "BTC"],
            "Price": ["$95,981", "$3,133", "$95,845", "$3,145", "$95,672"],
        })
        st.dataframe(recent, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)

        # Render the chatbot toggle (appears as fixed button in bottom right)
        render_chatbot_toggle()

# Entrypoint
if __name__ == "__main__":
    app = StockMarketApp()
    app.run()