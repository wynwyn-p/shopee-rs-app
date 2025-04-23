import os
import pandas as pd
import joblib
import streamlit as st
from gensim import corpora, models, similarities

@st.cache_resource(show_spinner="🧠 Đang tải mô hình và dữ liệu...")
def load_all_models(base_path="models"):
    print("📂 Load mô hình từ thư mục:", base_path)

    try:
        df_final = pd.read_parquet(os.path.join(base_path, "df_final_new.parquet"))
        print("✅ df_final loaded")
    except Exception as e:
        print("❌ Lỗi df_final:", e)
        df_final = None

    try:
        dictionary = corpora.Dictionary.load(os.path.join(base_path, "dictionary_tokenized.dict"))
        print("✅ dictionary loaded")
    except Exception as e:
        print("❌ Lỗi dictionary:", e)
        dictionary = None

    try:
        tfidf_model = models.TfidfModel.load(os.path.join(base_path, "tfidf_model_gensim.pkl"))
        print("✅ tfidf_model loaded")
    except Exception as e:
        print("❌ Lỗi tfidf_model:", e)
        tfidf_model = None

    try:
        similarity_index = similarities.Similarity.load(os.path.join(base_path, "gensim_index_merged"))
        print("✅ similarity_index loaded")
    except Exception as e:
        print("❌ Lỗi similarity_index:", e)
        similarity_index = None

    try:
        baseline_model = joblib.load(os.path.join(base_path, "baseline_only_model.pkl"))
        print("✅ baseline_only_model loaded")
    except Exception as e:
        print("❌ Lỗi baseline_only_model:", e)
        baseline_model = None

    return dictionary, tfidf_model, similarity_index, df_final, baseline_model
