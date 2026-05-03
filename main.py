import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
hf_token = os.getenv("HF_TOKEN")

pages = st.navigation([
    st.Page("src/pages/home.py", title="Trang chủ"),
    st.Page("src/pages/faq.py", title="Hướng dẫn sử dụng"),
    st.Page("src/pages/detector.py", title="Kiểm tra AI"),
])

pages.run()
