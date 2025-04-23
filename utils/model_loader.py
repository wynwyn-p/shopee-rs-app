import os
import joblib
import pandas as pd
from gensim import corpora, models, similarities
import streamlit as st
from utils.download_models import download_models

# Gọi hàm tải nếu thiếu
download_models()

@st.cache_resource(show_spinner="🔄 Đang tải mô hình và dữ liệu...")
def load_all_models():
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

    print("📁 Đường dẫn models:", base_path)

    df_final = pd.read_parquet(os.path.join(base_path, "df_final_new.parquet"))
    dictionary = corpora.Dictionary.load(os.path.join(base_path, "dictionary_tokenized.dict"))
    tfidf_model = models.TfidfModel.load(os.path.join(base_path, "tfidf_model_gensim.pkl"))
    similarity_index = similarities.Similarity.load(os.path.join(base_path, "gensim_index_merged"))
    baseline_model = joblib.load(os.path.join(base_path, "baseline_only_model.pkl"))

    return dictionary, tfidf_model, similarity_index, df_final, baseline_model
