import os
import gdown
import time

def download_models(model_dir="models"):
    """
    Tải từng file mô hình cần thiết từ Google Drive.
    Không dùng download_folder để tránh tải file không mong muốn.
    """

    os.makedirs(model_dir, exist_ok=True)

    # Danh sách các file cần tải và ID tương ứng từ Google Drive
    file_links = {
        "df_final_new.parquet": "1rEJ8_TfUmxCHpV6E3wPYdYQwbWjj2Qxh",
        "baseline_only_model.pkl": "1Br2V6iE7UYyie9aZQXa0EZwH1-6OTo1F",
        "dictionary_tokenized.dict": "1Y2ZcSeTfWUJphW1ScGpSpPyecbJe8G9K",
        "tfidf_model_gensim.pkl": "1Bv2cRTAAKw3n5Rg5ErXL3TKD9vxUYquD",
        "gensim_index_merged": "1-jRVyfG2maGrTWRSHeoEiCDMMPiHz1db",  # File chính .0/.1/.2 sẽ kèm theo
    }

    print("📥 Bắt đầu kiểm tra và tải từng file mô hình...")

    for filename, file_id in file_links.items():
        save_path = os.path.join(model_dir, filename)
        if not os.path.exists(save_path):
            print(f"🔽 Tải {filename}...")
            url = f"https://drive.google.com/uc?id={file_id}"
            gdown.download(url, save_path, quiet=False)
        else:
            print(f"✅ {filename} đã tồn tại, bỏ qua tải.")

    print("\n✅ Toàn bộ file cần thiết đã sẵn sàng!")
    time.sleep(1)

    # In danh sách file hiện có
    print("📂 Danh sách file trong models/:")
    for f in sorted(os.listdir(model_dir)):
        print("  -", f)

if __name__ == "__main__":
    download_models()
