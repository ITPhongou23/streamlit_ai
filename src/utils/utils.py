import streamlit as st
import os
from PIL import Image


def init_page(title):
    current_dir = os.path.dirname(__file__) 
    src_dir = os.path.dirname(current_dir)
    icon_path = os.path.join(src_dir, "assets", "icons.png")
    
    icon = Image.open(icon_path)

    st.set_page_config(
        page_title=title,
        page_icon=icon,
        layout="wide"
    )

    try:
        with open("assets/css/styles.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def render_footer():
    st.markdown("""
        <div class="footer-minimal">
            <div class="footer-content">
                <span>© 2026 OUEL PROJECT</span>
                <span class="dot">•</span>
                <span>DEVELOPED BY OU STUDENTS</span>
            </div>
        </div>
    """, unsafe_allow_html=True)
