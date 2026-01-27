import streamlit as st
import pandas as pd
import numpy as np
import time

prompt = st.chat_input("Nhập ở đây...")

with st.chat_message("user"):
    st.write("Hello 👋")
