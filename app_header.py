from __future__ import annotations
from typing import Optional
import streamlit as st
from streamlit.components.v1 import html as components_html


def render_hero(
    title: str = "Stock Market Intelligence Platform",
    tagline: Optional[str] = "More Trades, Less Spreadsheets",
    height: int = 380
):
    hero_html = f"""
    <div style="
        width: 100%;
        padding: 60px 20px;
        background-color: white;
        box-sizing: border-box;
        border-bottom: 1px solid #eee;
        margin-left: 0;
    ">
        <h1 style="
            font-size: 40px; 
            margin: 0; 
            font-weight: 700; 
            color: #0A2540;">
            {title}
        </h1>

        <h3 style="
            margin-top: 8px; 
            color: #C55A11; 
            font-style: italic;">
            {tagline}
        </h3>

        <p style="
            max-width: 800px; 
            margin-top: 10px;
            line-height: 1.5;
            font-size: 16px;
            color: #333;">
            You've got instinct. Back it up with real-time alerts, live & historical data,
            custom visual analysis tools—all in one platform built for how you trade.
        </p>

        <div style="margin-top: 25px; display: flex; flex-wrap: wrap; gap: 12px;">
            <button class="hero-btn">Real-time Alerts</button>
            <button class="hero-btn">Live + Historical Data</button>
            <button class="hero-btn">Sentiment Analysis</button>
            <button class="hero-btn">Predictive Signals</button>
            <button class="hero-btn">Portfolio Tracking</button>
            <button class="hero-btn">Trade Simulator</button>
            <button class="hero-btn">Custom Visual Tools</button>
            <button class="hero-btn">API Access</button>
        </div>
    </div>

    <style>
        /* Buttons inside hero */
        .hero-btn {{
            background-color: #C55A11;
            color: white;
            border: none;
            padding: 10px 18px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
        }}

        .hero-btn:hover {{
            background-color: #A64A0C;
        }}

        /* Remove Streamlit default spacing so hero sticks to top */
        section.main > div:first-child {{
            padding-top: 0 !important;
        }}
    </style>
    """

    components_html(hero_html, height=height, scrolling=False)
