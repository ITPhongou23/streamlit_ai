import streamlit as st
import pandas as pd
import numpy as np
import time
from transformers import pipeline
from pypdf import PdfReader


MODEL_NAME = "Juner/AI-generated-text-detection"

@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model=MODEL_NAME
    )

clf = load_model()


st.title("Chương trình kiểm tra AI")
text = ""
data = st.file_uploader("Tải file pdf", type="pdf")

if data:
    reader = PdfReader(data)

    for i, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    st.text_area(label="Nhập ở đây:",value=text)

else:
    text = st.text_area("Nhập ở đây:")

if st.button("Kiểm tra ngay"):
    if text.strip():
        with st.spinner("Đang phân tích dữ liệu..."):  
            try:
                raw = clf(text)

                result = raw[0][0] if isinstance(raw[0], list) else raw[0]

                label_map = {
                    "LABEL_1": "Văn bản do AI tạo",
                    "LABEL_0": "Văn bản do con người viết"
                }

                label = label_map.get(result["label"], result["label"])
                confidence = result["score"] * 100

                st.divider()
                st.subheader("Kết quả phân tích")

                if result["label"] == "LABEL_1":
                    st.error(f"⚠️ **{label}**")
                    st.progress(result["score"])
                else:
                    st.success(f"✅ **{label}**")
                    st.progress(result["score"])

                st.write(f"Độ tin cậy: **{confidence:.2f}%**")

            except Exception as e:
                st.error(f"Hệ thống đang bị lỗi vui lòng quay lại sau!")
    else:
        st.warning("Vui lòng nhập văn bản trước khi kiểm tra.")