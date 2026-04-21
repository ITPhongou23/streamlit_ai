import streamlit as st
from transformers import pipeline
from yaml import unsafe_load

from services.generate_docx_services import generate_docx
from services.pdf_reader_services import PdfReaderManager
from services.pipeline import PipelineManager
from utils.utils import init_page, render_footer


@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="ITPhongou23/AINewsIndicator",
        subfolder="phobert_model"
    )


clf = load_model()

init_page("AI Detector")

st.markdown("""
<div class="header-split">
    <p class="eyebrow_1">AI DETECTOR</p>
    <div class="status-badge">Trạng thái: Hoạt động</div>
</div>
""", unsafe_allow_html=True)

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
        height=500,
        placeholder="Dán văn bản hoặc tải file PDF",
        disabled=write_permission,
        label_visibility="collapsed"
    )

    check_btn = st.button("PHÂN TÍCH NGAY", use_container_width=True)

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

                        label = result.get("label", "Unknown")
                        score = result.get("score", 0)

                        label_map = {
                            "LABEL_0": "human",
                            "LABEL_1": "AI"
                        }

                        label = label_map.get(label, label)
                        result_label = label

                        st.markdown(f"""
                        <div style="
                            width:120px;
                            height:120px;
                            border-radius:50%;
                            background: conic-gradient(#2e8b57 {score*100}%, #e6e6e6 {score*100}%);
                            display:flex;
                            justify-content:center;
                            align-items:center;
                            margin:auto;
                        ">
                            <div style="
                                width:105px;
                                height:105px;
                                border-radius:50%;
                                background:white;
                                display:flex;
                                justify-content:center;
                                align-items:center;
                                font-weight:500;
                                color:#2e8b57;
                            ">
                                {label}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        label1 = ''
                        label2 = ''
                        text_kq = ''

                        if label == 'AI':
                            label1 = 'AI'
                            label2 = 'human'
                            text_kq = 'Văn bản này là do AI tạo ra'
                        else:
                            label1 = 'human'
                            label2 = 'AI'
                            text_kq = 'Văn bản này là do con người tạo ra'

                        st.markdown(f"""
                        <div style="width:100%; margin-top:10px; display:flex; padding: 10px; justify-content: center;">
                            <div style="
                                border: 2px solid #2e8b57;
                                background:#f0f9f4;
                                padding:10px 15px;
                                border-radius:999px;
                                color:#2e8b57;
                                font-weight:600;
                                text-align:center;
                                display:inline-block;
                                margin-right:15px;
                            ">
                                {label1}   {score * 100:.1f}%
                            </div>
                            <div style="
                                border: 2px solid #1e3a8a;
                                background:#f0f9f4;
                                padding:10px 15px;
                                border-radius:999px;
                                color:#1e3a8a;
                                font-weight:600;
                                text-align:center;
                                display:inline-block;
                            ">
                                {label2}   {100-score * 100:.1f}%
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown(
                            '<div style="margin-top:20px;"></div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(f"""
                        <div style="font-family: 'Inter'; max-width: 600px; margin: 5px auto; padding: 20px 24px; text-align: center;">
                            <div style=" font-size: 22px; color: #111827; line-height: 1.6;">
                                {text_kq}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        docx_file = generate_docx(
                            processed_text,
                            result_label,
                            score
                        )

                        st.download_button(
                            label="XUẤT BÁO CÁO (.DOCX)",
                            data=docx_file,
                            file_name="Kết quả kiểm tra.docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                            use_container_width=True
                        )

                    except Exception as e:
                        st.error(f"Lỗi kết nối mô hình: {e}")

        else:
            st.markdown(
                '<div class="empty-state"><p>Kết quả sẽ hiển thị tại đây.</p></div>',
                unsafe_allow_html=True
            )

render_footer()
