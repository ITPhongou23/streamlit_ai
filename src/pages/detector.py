import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

from src.services.generate_docx_services import generate_docx
from src.services.pdf_reader_services import PdfReaderManager
from src.services.pipeline import PipelineManager
from src.utils.utils import init_page, render_footer
from huggingface_hub import hf_hub_download
import joblib

#Biến.
init_page("AI Detector")
options = ["Phobert-large-VietNamese-news-ai-detection", "Phobert-v2-VietNamese-news-ai-detection"]
model_name = ["JuniorThanh/phobert-large-vietnamese-news-ai-detection","JuniorThanh/phobert-v2-vietnamese-news-ai-detection"]
n = 0
err_msg = ""
label1 = ""
label2 = ""
text_kq = ""



#Hàm.
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer
    )


def get_model(status, options, model_name):
    if status == options[0]:
        return load_model(model_name[0]), 0
    else:
        return load_model(model_name[1]), 1

def set_label(result_label):
    if result_label == 'AI':
        return 'AI','human', 'Văn bản này là do AI tạo ra'
    else:
        return 'human', 'AI', 'Văn bản này là do con người tạo ra'

def normalize_result(result):
    label_map = {
        "LABEL_0": "human",
        "LABEL_1": "AI"
    }

    raw_label = result.get("label", "Unknown")
    score = result.get("score", 0)

    label = label_map.get(raw_label, raw_label)

    return label, score, raw_label

def render_score(label1, label2, score):
    st.markdown(f"""
    <div class="badge-container">
        <div class="badge-green">
            {label1} {score * 100:.1f}%
        </div>
        <div class="badge-blue">
            {label2} {100 - score * 100:.1f}%
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_circle_score(label, score):
    st.markdown(f"""
    <div class="circle" style="background: conic-gradient(#2e8b57 {score * 100}%, #e6e6e6 {score * 100}%);">
        <div class="circle-inner">
            {label}
        </div>
    </div>
    """, unsafe_allow_html=True)



#Header.
if "status" not in st.session_state:
    st.session_state.status = options[0]

st.selectbox("Model Deep Learning:", options, key="status")

status = st.session_state.status

clf, n = get_model(status, options, model_name)

st.markdown(f"""
    <div class="header-split">
        <p class="eyebrow_1">{model_name[n]}</p>
        <div class="status-badge">● Mô hình {n+1}:</div>
    </div>
""", unsafe_allow_html=True)



#Body.
col_left, col_right = st.columns(2, gap="medium")


with col_left:
    st.markdown('<p class="label-v3">INPUT TẠI ĐÂY</p>', unsafe_allow_html=True)
    data = st.file_uploader("Upload PDF file", type="pdf", label_visibility="collapsed")

    if data:
        data_text = PdfReaderManager.load_file(data)
        write_permission = True
    else:
        data_text = ""
        write_permission = False

    text = st.text_area(
        label="Input Area",
        value=data_text,
        height=300,
        placeholder="Dán văn bản hoặc tải file PDF",
        disabled=write_permission,
        label_visibility="collapsed"
    )

    word_count = len(text.strip().split())

    check_btn = st.button("PHÂN TÍCH NGAY", use_container_width=True)

    if check_btn:
        if 200 <= word_count <= 600:
            pass
        else:
            check_btn = False
            err_msg = "Vui lòng nhập từ 200 đến 600 từ trước khi phân tích"



with col_right:
    st.markdown('<p class="label-v3">KẾT QUẢ PHÂN TÍCH</p>', unsafe_allow_html=True)
    result_placeholder = st.container(border=True)

    with result_placeholder:
        if check_btn:
            processed_text = PipelineManager.input_processing(text)

            if not processed_text.strip():
                st.info("Vui lòng nhập nội dung ở bên trái.")
            else:
                with st.spinner("Đang xử lý yêu cầu"):
                    try:
                        result = clf(processed_text)

                        if isinstance(result, list):
                            result = result[0]

                        result_label, score, label = normalize_result(result)

                        label1, label2, text_kq = set_label(result_label)

                        docx_file = generate_docx(processed_text, result_label, score)


                        #render UI.
                        st.write("INPUT:", processed_text)
                        st.write("MODEL TYPE:", type(clf))
                        if hasattr(clf, "tokenizer"):
                            st.write("VOCAB:", len(clf.tokenizer))

                        render_circle_score(result_label, score)

                        render_score(label1, label2, score)

                        st.markdown('<div style="margin-top:20px;"></div>',unsafe_allow_html=True)

                        st.markdown(f"""<div class="result-text">{text_kq}</div>""", unsafe_allow_html=True)

                        st.download_button(label="XUẤT BÁO CÁO (.DOCX)", data=docx_file, file_name="Kết quả kiểm tra.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)

                    except Exception as e:
                        st.error(f"Lỗi kết nối mô hình: {e}")

        else:
            if err_msg:
                st.error(err_msg)
            else:
                st.markdown(
                    '<div class="empty-state"><p>Kết quả sẽ hiển thị tại đây.</p></div>',
                    unsafe_allow_html=True
                )

render_footer()
