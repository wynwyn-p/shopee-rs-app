# File chính gọi các components
import streamlit as st
import pandas as pd

# Import các module
from components import sidebar, home, introduction, exploration, visualization, demo, link_code
from utils.data_loader import load_data, load_stop_words
from utils.download_models import download_models          
from utils.model_loader import load_all_models

# ========================== Tải mô hình trước khi vào cache ==========================
download_models()  # Gọi tải mô hình 1 lần, trước khi cache load_all_models()

# Cấu hình giao diện trang
st.set_page_config(page_title="Shopee Recommender", page_icon="🛍️", layout="wide")

# Hiển thị thanh bên
sidebar.show()

# ========================== Load dữ liệu và mô hình ==========================
with st.spinner("📁 Đang tải dữ liệu và mô hình..."):
    # Load dữ liệu
    file_1, file_2, df = load_data()

    # Load stopword
    stop_words = load_stop_words("utils/vietnamese-stopwords.txt")

    # Load toàn bộ mô hình (có cache)
    dictionary, tfidf_model, similarity_index, df_final, baseline_model = load_all_models()

# ========================== Điều hướng theo menu ==========================
selected = st.session_state.get("selected_page", "Introduction")

if selected == "Home":
    home.show()
elif selected == "Introduction":
    introduction.show()
elif selected == "Data Exploration":
    exploration.render_exploration(file_1)
elif selected == "Data Visualization":
    visualization.show(df)
elif selected == "Demo App":
    demo.show(
        file_2=file_2,
        file_1=file_1,
        df=df,
        df_final=df_final,
        dictionary=dictionary,
        tfidf_model=tfidf_model,
        similarity_index=similarity_index,
        stop_words=stop_words,
        baseline_model=baseline_model
    )
elif selected == "Link Code":
    link_code.show()
