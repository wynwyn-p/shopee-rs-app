import os
import gdown
import time

def download_models(folder_url="https://drive.google.com/drive/u/1/folders/19qxil5Adf9YlQYlnetbDsRW8QsThNvZj", model_dir="models"):
    """
    Tải toàn bộ mô hình từ Google Drive folder về thư mục models/
    và in danh sách file. Chỉ tải nếu thiếu.
    """
    os.makedirs(model_dir, exist_ok=True)

    # Danh sách các file cần kiểm tra
    required_files = [
        "df_final_new.parquet",
        "baseline_only_model.pkl",
        "dictionary_tokenized.dict",
        "tfidf_model_gensim.pkl",
        "gensim_index_merged",  # index sẽ gồm .0, .1,... nên chỉ cần kiểm tra tên gốc
    ]

    # Kiểm tra file nào còn thiếu
    missing_files = []
    for file in required_files:
        path = os.path.join(model_dir, file)
        if not os.path.exists(path):
            missing_files.append(file)

    if not missing_files:
        print("✅ Toàn bộ mô hình đã tồn tại, bỏ qua tải lại.")
    else:
        print("📥 Thiếu các file sau, bắt đầu tải từ Google Drive:")
        for f in missing_files:
            print("  -", f)

        try:
            gdown.download_folder(
                url=folder_url,
                output=model_dir,
                quiet=False,
                use_cookies=False
            )
            print("✅ Tải thành công toàn bộ mô hình!")
            print("🕒 Đợi 3 giây để đảm bảo ghi xong file...")
            time.sleep(3)
        except Exception as e:
            print(f"❌ Lỗi khi tải mô hình: {e}")
            return

    # In toàn bộ file hiện có
    print("📂 Danh sách file trong models/:")
    for f in sorted(os.listdir(model_dir)):
        print("  -", f)

if __name__ == "__main__":
    download_models()
